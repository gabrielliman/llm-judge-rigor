"""Exploratory analyses. CPU only.

    python scripts/analyze_exploratory.py

  reproducibility   outputs/main vs. outputs/replication (identical decoding)
  recovered         confirmatory analyses with missing verdicts filled from outputs/recovery
  logprobs          verdict log-probability scores (outputs/replication)
  size_ladder       all models of config.yaml (outputs/main + outputs/size_ladder): measures
                    with a Rasch refit on all configurations, H1/H2-type contrasts

Writes results/exploratory/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analysis as A  # noqa: E402
from data import load_items  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    prompts, B, ci, seed = list(cfg["prompts"]), cfg["analysis"]["bootstrap_B"], cfg["analysis"]["ci"], cfg["seed"]
    items = load_items(ROOT)
    out_dir, raw = ROOT / "results" / "exploratory", ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    rep = A.reproducibility(raw / "main", raw / "replication", items, cfg["models"], prompts)
    rep.to_csv(out_dir / "reproducibility.csv", index=False)
    summary["reproducibility"] = {"agree_min": rep.agree.min(), "agree_median": rep.agree.median(),
                                  "kappa_min": rep.kappa.min(), "identical_text_median": rep.identical_text.median(),
                                  "max_abs_diff_R_raw": rep.abs_diff_R_raw.max()}

    d = A.load_design(raw / "main", items, cfg["models"], prompts)
    d_rec, filled, still = A.fill_recovered(d, raw / "recovery")
    s = A.confirmatory_analysis(d_rec, items, prompts, B, seed, ci, out_dir / "recovered", with_paper_judges=False)
    summary["recovered"] = {"n_filled": filled, "n_still_missing": still,
                            "verdicts": {h: [r["verdict"] for r in s[h] if isinstance(r.get("verdict"), str)] for h in ["H1", "H2", "H3", "H4", "H5"]}}

    soft = A.soft_scores(raw / "replication", items, cfg["models"], prompts)
    T = pd.read_csv(ROOT / "results" / "confirmatory" / "measures_by_config.csv")
    soft = soft.merge(T[["model", "prompt_id", "d_prime", "R_c", "R_raw"]], on=["model", "prompt_id"])
    soft.to_csv(out_dir / "logprob_scores.csv", index=False)
    summary["logprobs"] = {"argmax_matches_verdict_min": soft.argmax_matches_verdict.min(),
                           "rho_meanP_Rc": spearmanr(soft.mean_p_halluc, soft.R_c).statistic,
                           "rho_AUC_dprime": spearmanr(soft.AUC, soft.d_prime).statistic,
                           "delta_AUC_by_model": soft.groupby("model").delta_AUC.mean().to_dict(),
                           "frac_p_extreme_by_model": soft.groupby("model").frac_p_extreme.mean().to_dict()}

    extra = cfg["exploratory_models"]
    d_main = A.load_design(raw / "main", items, cfg["models"], prompts)
    d_ext = A.load_design(raw / "size_ladder", items, extra, prompts)
    d_all = A.Design(Y=np.hstack([d_main.Y, d_ext.Y]), label=d_main.label, hearing=d_main.hearing,
                     configs=pd.concat([d_main.configs, d_ext.configs], ignore_index=True),
                     item_uid=d_main.item_uid)
    res = A.compute_all(d_all, B, seed, ci)
    params = {m["short"]: m["params_b"] for m in cfg["models"] + extra}
    pt = res["point"].assign(params_b=lambda x: x.model.map(params))
    pt.to_csv(out_dir / "size_ladder_measures.csv", index=False)
    con = pd.concat([A.contrasts(res, [("S", "N"), ("N", "L")], ci, "H1-type"),
                     A.contrasts(res, [("P2", "P1"), ("P3", "P2")], ci, "H2-type")])
    con.to_csv(out_dir / "size_ladder_contrasts.csv", index=False)
    taus = {}
    for fam in ["qwen", "gemma"]:
        for meas in ["R_raw", "R_c", "R_rasch", "d_prime"]:
            t = [kendalltau(g.params_b, g[meas]).statistic
                 for _, g in pt[pt.family == fam].sort_values("params_b").groupby("prompt_id")]
            taus[f"{fam}|{meas}"] = float(np.nanmean(t))
    summary["size_ladder"] = {"mean_kendall_size_vs_measure": taus,
                              "contrasts_against_prediction": con[con.ci_lt_0][["model", "measure", "contrast"]]
                              .to_dict("records")}

    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
