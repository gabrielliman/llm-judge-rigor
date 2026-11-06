"""Pilot: inclusion criterion and generation diagnostics.

    python scripts/pilot_inclusion.py

A judge is included if the lower bound of the 95% hearing-bootstrap CI of d′ under the
neutral prompt N is above 0. Also reports, per configuration, missing verdicts, retries,
thinking markers and output-length quantiles. Reads outputs/pilot, writes results/pilot/.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bootstrap import hearing_weights, percentile_ci, resample_counts  # noqa: E402
from data import raw_file, read_raw  # noqa: E402
from measures import per_hearing_counts, sdt  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    raw, out = ROOT / "outputs" / "pilot", ROOT / "results" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT / "data" / "pilot_items.csv")
    hearings = np.sort(labels["hearing_id"].unique())
    W = hearing_weights(len(hearings), cfg["analysis"]["bootstrap_B"], cfg["seed"])
    markers = re.compile("|".join(map(re.escape, cfg["generation"]["thinking_markers"])))

    inclusion, diag, tokens = [], [], []
    for m in cfg["models"]:
        for p in cfg["prompts"]:
            d = read_raw(raw_file(raw, m["short"], p)).merge(labels[["item_uid", "label_halluc"]], on="item_uid")
            counts = per_hearing_counts(d.astype({"flag": float})).reindex(hearings, fill_value=0.0)
            est = sdt(*counts.sum().to_numpy())
            diag.append({"model": m["short"], "prompt_id": p, "d_prime": float(est["d_prime"]),
                         "R_raw": float(est["R_raw"]), "R_c": float(est["R_c"]),
                         "n_missing": int(d["flag"].isna().sum()), "n_retried": int((d["attempt"] == 2).sum()),
                         "n_thinking_marker": int(d["raw_text"].str.contains(markers).sum())})
            tokens.append(d.assign(model=m["short"], prompt_id=p)[["model", "prompt_id", "n_output_tokens"]])
            if p == "N":
                lo, hi = percentile_ci(sdt(*resample_counts(counts.to_numpy(), W).T)["d_prime"], cfg["analysis"]["ci"])
                inclusion.append({"model": m["short"], "d_prime": float(est["d_prime"]), "ci_low": float(lo),
                                  "ci_high": float(hi), "included": bool(lo > 0)})

    inc = pd.DataFrame(inclusion)
    inc.to_csv(out / "inclusion.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "diagnostics.csv", index=False)
    tq = pd.concat(tokens).groupby(["model", "prompt_id"])["n_output_tokens"].describe(percentiles=[.5, .95, .99])
    tq.to_csv(out / "output_tokens.csv")
    print(inc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
