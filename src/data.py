"""PublicHearingBR-NLI: download, flattening, data checks, pilot sample and raw-output I/O.

The Hugging Face repository ``unicamp-dl/PublicHearingBR`` holds two JSONL files with
different schemas, so ``datasets.load_dataset`` cannot load it. The NLI file is
downloaded with ``huggingface_hub.hf_hub_download`` at a pinned revision and parsed line
by line.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ID = "unicamp-dl/PublicHearingBR"
FILENAME = "PublicHearingBR_NLI.jsonl"

# The twelve judges released with the dataset (Fernandes et al., 2024): 4 models x 3 prompts.
PAPER_MODELS = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "deepseek-chat", "sabia-3.1-2025-05-08"]
PAPER_PROMPTS = [1, 2, 3]


def paper_col(prompt: int, model: str) -> str:
    """Column name of a dataset judge, e.g. ``paper_P1_gpt-4o-2024-08-06``."""
    return f"paper_P{prompt}_{model}"


PAPER_COLS = [paper_col(p, m) for m in PAPER_MODELS for p in PAPER_PROMPTS]


def download(revision: str, local_dir: str | Path) -> Path:
    """Download the NLI JSONL at a fixed ``revision`` and return its local path."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset",
                                revision=revision, local_dir=str(local_dir)))


def iter_opinions(path: str | Path):
    """Yield ``(hearing, speaker_idx, speaker, opinion_idx, opinion)`` for every opinion."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            hearing = json.loads(line)
            for ie, env in enumerate(hearing["metadados_extraidos"]["envolvidos"]):
                for io, op in enumerate(env.get("opinioes") or []):
                    yield hearing, ie, env, io, op


def polarity_report(path: str | Path) -> dict:
    """Raw value counts of the human label and of every dataset judge's verdict.

    Used to determine which boolean value denotes a hallucination (the human label
    has 504 hallucinations) and to detect missing or non-boolean verdicts.
    """
    manual, paper = {}, {}
    for *_, op in iter_opinions(path):
        va = op["verificacao_alucinacao"]
        v = repr(va.get("verificacao_manual"))
        manual[v] = manual.get(v, 0) + 1
        for k, sub in va.items():
            if k != "verificacao_manual":
                a = repr(sub.get("alucinacao")) if isinstance(sub, dict) else "<non-dict>"
                paper.setdefault(k, {})
                paper[k][a] = paper[k].get(a, 0) + 1
    return {"verificacao_manual": manual, "paper_alucinacao": paper}


def build_texto(chunks: list[str]) -> str:
    """The ``{TEXTO}`` placeholder: the stored chunks joined by a blank line."""
    return "\n\n".join(chunks)


def flatten(path: str | Path, halluc_value_manual: bool, halluc_value_paper: bool) -> pd.DataFrame:
    """One row per opinion.

    Columns: ``item_uid`` (``{hearing}-{speaker}-{opinion}``), ``hearing_id``, ``nome``,
    ``cargo``, ``opiniao``, ``chunks``, ``label_halluc`` (1 = hallucination),
    ``no_context`` (1 if the chunks are missing or whitespace-only; such items are kept)
    and the twelve dataset judges as 0/1 columns (``PAPER_COLS``).
    """
    rows = []
    for hearing, ie, env, io, op in iter_opinions(path):
        va = op["verificacao_alucinacao"]
        row = {
            "item_uid": f"{hearing['id']}-{ie}-{io}",
            "hearing_id": hearing["id"],
            "nome": env.get("nome"),
            "cargo": env.get("cargo"),
            "opiniao": op["opiniao"],
            "chunks": list(op["chunks_proximos"]),
            "label_halluc": int(va["verificacao_manual"] == halluc_value_manual),
            "no_context": int(not build_texto(op["chunks_proximos"]).strip()),
        }
        for m in PAPER_MODELS:
            for p in PAPER_PROMPTS:
                v = va.get(f"prompt_{p}_{m}", {}).get("alucinacao")
                row[paper_col(p, m)] = float("nan") if v is None else int(v == halluc_value_paper)
        rows.append(row)
    df = pd.DataFrame(rows)
    assert df["item_uid"].is_unique
    return df


def confusion(df: pd.DataFrame, col: str) -> dict:
    """Confusion counts of a 0/1 judge column against ``label_halluc`` (1 = hallucination)."""
    sub = df[["label_halluc", col]].dropna()
    y, p = sub["label_halluc"].astype(int), sub[col].astype(int)
    return {"judge": col,
            "TP": int(((y == 1) & (p == 1)).sum()), "FN": int(((y == 1) & (p == 0)).sum()),
            "FP": int(((y == 0) & (p == 1)).sum()), "TN": int(((y == 0) & (p == 0)).sum()),
            "missing": int(df[col].isna().sum())}


def draw_pilot(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """Balanced pilot sample: ``n_per_class`` hallucinations and valid opinions."""
    rng = np.random.default_rng(seed)
    picks = []
    for lab in (1, 0):
        uids = df.loc[df["label_halluc"] == lab, "item_uid"].sort_values().to_numpy()
        picks += list(rng.choice(uids, size=n_per_class, replace=False))
    return df.set_index("item_uid").loc[picks, ["hearing_id", "label_halluc", "no_context"]].reset_index()


def load_items(root: Path, split: str = "full") -> pd.DataFrame:
    """Flattened items (``data/items.parquet``), optionally restricted to the pilot sample."""
    path = root / "data" / "items.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python scripts/prepare_data.py` first.")
    items = pd.read_parquet(path)
    if split == "pilot":
        uids = pd.read_csv(root / "data" / "pilot_items.csv")["item_uid"]
        items = items.set_index("item_uid").loc[uids].reset_index()
    elif split != "full":
        raise ValueError(f"unknown split {split!r} (expected 'full' or 'pilot')")
    return items


def raw_file(raw_dir: str | Path, model_short: str, prompt_id: str) -> Path:
    """Path of a raw-output file ``{model}__{prompt}.jsonl``; a gzipped copy is accepted.

    Generation writes plain ``.jsonl``; the released outputs are gzipped. Returns the
    plain file if it exists, otherwise the ``.gz`` file if it exists, otherwise the
    plain path (callers check ``exists()`` or let :func:`read_raw` raise).
    """
    p = Path(raw_dir) / f"{model_short}__{prompt_id}.jsonl"
    gz = p.with_name(p.name + ".gz")
    return p if p.exists() or not gz.exists() else gz


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a raw-output JSONL (plain or gzipped), keeping ``item_uid`` as string."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"raw output not found: {path} (.jsonl or .jsonl.gz)")
    return pd.read_json(path, lines=True, dtype={"item_uid": str})
