"""Run one judge model on all prompts and write raw outputs.

    python scripts/generate.py --model qwen35-2b                      # all 4,238 items -> outputs/main
    python scripts/generate.py --model qwen35-2b --split pilot        # 300-item pilot -> outputs/pilot
    python scripts/generate.py --model qwen35-9b --tensor-parallel 2  # large model on 2 GPUs

Exploratory runs:
    --out outputs/replication --logprobs                    second run, verdict log-probabilities
    --out outputs/recovery --only-missing outputs/main --no-whitespace
                                                            regenerate missing verdicts only

The model is loaded once and every (prompt, item) request is sent to vLLM together.
Writes {out}/{model}__{prompt}.jsonl and {out}/run_{model}.json (diagnostics, timing).
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import build_texto, load_items, raw_file, read_raw  # noqa: E402
from generation import make_llm, run_model  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    models = {m["short"]: m for m in cfg["models"] + cfg["exploratory_models"]}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(models))
    ap.add_argument("--split", choices=["full", "pilot"], default="full")
    ap.add_argument("--out", type=Path, help="output directory (default: outputs/main or outputs/pilot)")
    ap.add_argument("--logprobs", action="store_true", help="record verdict log-probabilities")
    ap.add_argument("--only-missing", type=Path, metavar="DIR",
                    help="regenerate only the (prompt, item) pairs whose verdict is missing in DIR")
    ap.add_argument("--no-whitespace", action="store_true",
                    help="forbid free whitespace in the JSON grammar")
    ap.add_argument("--tensor-parallel", type=int, default=1, help="number of GPUs for one model")
    ap.add_argument("--limit", type=int, help="use only the first N items (smoke test)")
    a = ap.parse_args()

    m = models[a.model]
    out = a.out or ROOT / "outputs" / ("pilot" if a.split == "pilot" else "main")
    items_df = load_items(ROOT, a.split)
    if a.limit:
        items_df = items_df.head(a.limit)
    items = {r.item_uid: {"item_uid": r.item_uid, "hearing_id": int(r.hearing_id),
                          "texto": build_texto(list(r.chunks)), "opiniao": r.opiniao}
             for r in items_df.itertuples()}

    prompts = list(cfg["prompts"])
    if a.only_missing:
        requests = []
        for p in prompts:
            f = raw_file(a.only_missing, a.model, p)
            if f.exists():
                r = read_raw(f)
                requests += [(p, items[u]) for u in r.loc[r["flag"].isna(), "item_uid"] if u in items]
        if not requests:
            print(f"no missing verdicts for {a.model} in {a.only_missing}")
            return
    else:
        requests = [(p, it) for p in prompts for it in items.values()]

    llm = make_llm(m, cfg["generation"], cfg["seed"], a.tensor_parallel, a.no_whitespace)
    run = run_model(llm, m, cfg["prompts"], requests, cfg["generation"], cfg["seed"], out,
                    logprobs=cfg["logprobs"]["top_k"] if a.logprobs else None,
                    chunk_size=cfg["logprobs"]["chunk_size"] if a.logprobs else None)
    run.update(split=a.split, logprobs=a.logprobs, no_whitespace=a.no_whitespace,
               only_missing=str(a.only_missing) if a.only_missing else None,
               tensor_parallel=a.tensor_parallel)
    json.dump(run, open(out / f"run_{a.model}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in run.items() if k != "per_prompt"}, indent=2))


if __name__ == "__main__":
    main()
