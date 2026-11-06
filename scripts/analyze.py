"""Confirmatory analyses (H1-H5, variance decomposition, reliability, Rasch fit, dataset
judges) and their sensitivity to missing verdicts. CPU only.

    python scripts/analyze.py

Reads outputs/main and writes results/confirmatory/ (plus sensitivity_missing_as_{1,0}/,
where missing verdicts are imputed as flag = 1 and flag = 0).
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis import confirmatory_analysis, load_design  # noqa: E402
from data import load_items  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    prompts, B, ci, seed = list(cfg["prompts"]), cfg["analysis"]["bootstrap_B"], cfg["analysis"]["ci"], cfg["seed"]
    items = load_items(ROOT)
    raw, out = ROOT / "outputs" / "main", ROOT / "results" / "confirmatory"

    d = load_design(raw, items, cfg["models"], prompts)
    s = confirmatory_analysis(d, items, prompts, B, seed, ci, out)
    for v in (1, 0):
        dv = load_design(raw, items, cfg["models"], prompts, impute=v)
        confirmatory_analysis(dv, items, prompts, B, seed, ci, out / f"sensitivity_missing_as_{v}",
                              with_paper_judges=False)
    print(json.dumps({h: [r["verdict"] for r in s[h] if isinstance(r.get("verdict"), str)] for h in ["H1", "H2", "H3", "H4", "H5"]},
                     indent=1))


if __name__ == "__main__":
    main()
