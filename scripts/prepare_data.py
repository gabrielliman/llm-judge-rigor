"""Download PublicHearingBR-NLI, flatten it, run the data checks and draw the pilot sample.

    python scripts/prepare_data.py

Writes data/items.parquet (one row per opinion), data/pilot_items.csv and
results/data_checks/{checks.json, paper_confusion.csv}. Exits with an error if a check fails.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import PAPER_COLS, confusion, download, draw_pilot, flatten, polarity_report  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    path = download(cfg["data"]["revision"], ROOT / "data" / "raw")
    out = ROOT / "results" / "data_checks"
    out.mkdir(parents=True, exist_ok=True)

    # Label polarity: the human label has 504 hallucinations; the dataset judges'
    # `alucinacao` field is True for a hallucination (confirmed by the confusion matrices).
    pol = polarity_report(path)
    halluc_values = [v for v, n in pol["verificacao_manual"].items() if n == 504]
    if len(halluc_values) != 1:
        sys.exit(f"no single value of verificacao_manual counts 504 hallucinations: {pol['verificacao_manual']}")
    df = flatten(path, halluc_values[0] == "True", True)

    n_chunks = df["chunks"].map(len)
    checks = {
        "n_items": len(df), "n_hearings": int(df["hearing_id"].nunique()),
        "n_hallucinations": int(df["label_halluc"].sum()),
        "label_value_for_hallucination": halluc_values[0],
        "chunk_count_distribution": {int(k): int(v) for k, v in n_chunks.value_counts().items()},
        "items_without_context": df.loc[df["no_context"] == 1, "item_uid"].tolist(),
        "raw_value_counts": pol,
    }

    # The twelve dataset judges against the published Tables 5-8 (arXiv:2410.07495v2).
    ref = pd.read_csv(ROOT / "data" / "paper_confusion_tables.csv").set_index("judge")
    rows = []
    for col in PAPER_COLS:
        c, r = confusion(df, col), ref.loc[col]
        c.update({f"paper_{k}": int(r[k]) for k in ["TN", "FP", "FN", "TP"]})
        c["cells_match"] = all(c[k] == r[k] for k in ["TN", "FP", "FN", "TP"])
        c["printed_col_totals_match"] = (c["TN"] + c["FN"] == r.pred_valid_total
                                         and c["TP"] + c["FP"] == r.pred_halluc_total)
        rows.append(c)
    cm = pd.DataFrame(rows)
    cm.to_csv(out / "paper_confusion.csv", index=False)
    checks["paper_cells_all_match"] = bool(cm["cells_match"].all())
    json.dump(checks, open(out / "checks.json", "w"), indent=2)

    ok = len(df) == 4238 and checks["n_hallucinations"] == 504 and checks["paper_cells_all_match"]
    if not ok:
        sys.exit(f"data checks failed: {json.dumps({k: v for k, v in checks.items() if k != 'raw_value_counts'})}")

    df.to_parquet(ROOT / "data" / "items.parquet", index=False)
    draw_pilot(df, cfg["data"]["pilot_per_class"], cfg["seed"]).to_csv(ROOT / "data" / "pilot_items.csv", index=False)
    print(f"{len(df)} opinions, {checks['n_hallucinations']} hallucinations, {checks['n_hearings']} hearings; "
          f"paper matrices match: {checks['paper_cells_all_match']}; "
          f"items without context: {checks['items_without_context']}")


if __name__ == "__main__":
    main()
