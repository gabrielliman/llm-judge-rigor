"""Rigor measures with a joint cluster bootstrap, hypothesis tests H1-H5, variance
decomposition, ranking stability, Rasch fit, the dataset's judges, and the exploratory
analyses (reproducibility, recovered verdicts, verdict log-probabilities, size ladder).

Sign convention everywhere: larger = stricter (more hallucination flags).

Decision rules:
  * contrast hypotheses (H1, H2, H5): supported if every required CI lies entirely on
    the predicted side; not supported if at least one CI lies entirely on the opposite
    side (for H5b: below 0); inconclusive otherwise;
  * H3: supported if rho >= 0.7 and its CI lower bound > 0; not supported if the CI
    upper bound < 0.7; inconclusive otherwise;
  * H4: supported if the CI of |rho(R_raw, d')| - |rho(R_c, d')| is > 0; not supported
    if it is < 0; inconclusive otherwise.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, kendalltau, rankdata, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from bootstrap import hearing_weights, percentile_ci
from data import PAPER_COLS, raw_file, read_raw
from measures import sdt
from rasch import decompose_two_way, extreme_items, fit_jml, fit_stats, separation_reliability

MEASURES = ["R_raw", "R_c", "R_rasch"]
NEUTRAL = ["N", "N1", "N2", "N3"]
TERMS = ["family", "size", "prompt", "family:size", "family:prompt", "size:prompt", "residual"]
SUPPORTED, NOT_SUPPORTED, INCONCLUSIVE = "supported", "not supported", "inconclusive"


# ------------------------------------------------------------------------------ data
@dataclass
class Design:
    """Response matrix (items x configurations) with labels and design factors."""
    Y: np.ndarray            # (I, J) flags: 1 = hallucination, 0 = not, NaN = missing
    label: np.ndarray        # (I,) human label, 1 = hallucination
    hearing: np.ndarray      # (I,) hearing index 0..H-1 (bootstrap cluster)
    configs: pd.DataFrame    # J rows: model, prompt_id, family, size
    item_uid: np.ndarray


def load_design(raw_dir: Path, items: pd.DataFrame, models: list[dict], prompts: list[str],
                impute: float | None = None) -> Design:
    """Read ``{raw_dir}/{model}__{prompt}.jsonl[.gz]`` into a Design (rows in ``items`` order).

    ``impute`` = None keeps missing verdicts as NaN; 0 or 1 imputes them.
    """
    cols, rows = [], []
    for m in models:
        for p in prompts:
            r = read_raw(raw_file(raw_dir, m["short"], p)).set_index("item_uid").reindex(items["item_uid"])
            cols.append(r["flag"].astype(float).to_numpy())
            rows.append({"model": m["short"], "prompt_id": p, "family": m["family"], "size": m["size"]})
    Y = np.column_stack(cols)
    if impute is not None:
        Y = np.where(np.isnan(Y), float(impute), Y)
    _, hidx = np.unique(items["hearing_id"].to_numpy(), return_inverse=True)
    return Design(Y=Y, label=items["label_halluc"].to_numpy().astype(int), hearing=hidx,
                  configs=pd.DataFrame(rows), item_uid=items["item_uid"].to_numpy())


def sdt_hearing_counts(d: Design) -> np.ndarray:
    """(H, J, 4) per-hearing counts over valid verdicts: hits, n_pos, false alarms, n_neg."""
    v, f = ~np.isnan(d.Y), np.nan_to_num(d.Y)
    pos = (d.label == 1)[:, None]
    out = np.zeros((d.hearing.max() + 1, d.Y.shape[1], 4))
    for k, a in enumerate([v & (f == 1) & pos, v & pos, v & (f == 1) & ~pos, v & ~pos]):
        np.add.at(out[:, :, k], d.hearing, a.astype(float))
    return out


def sdt_from_counts(C: np.ndarray) -> dict:
    """SDT measures from (..., 4) count arrays."""
    return sdt(C[..., 0], C[..., 1], C[..., 2], C[..., 3])


def rasch_fit(Y: np.ndarray, w_items=None, keep=None) -> dict:
    """Interaction-model JML (one facet level per configuration) on non-extreme items."""
    keep = ~extreme_items(Y) if keep is None else keep
    return fit_jml(Y[keep], [np.arange(Y.shape[1])], None if w_items is None else w_items[keep])


# --------------------------------------------------------------------- measures
def compute_all(d: Design, B: int, seed: int, ci: float) -> dict:
    """Point estimates and joint hearing-bootstrap replicates of H, F, d′ and the three
    rigor measures. The same resampling weights are used for every measure, so contrasts
    and correlations between measures are bootstrapped jointly; the Rasch model is
    refitted in every replicate."""
    C = sdt_hearing_counts(d)
    W = hearing_weights(d.hearing.max() + 1, B, seed)
    keep = ~extreme_items(d.Y)
    rf = rasch_fit(d.Y, keep=keep)
    pt = sdt_from_counts(C.sum(0))
    point = d.configs.copy()
    for k in ["H", "F", "d_prime", "R_raw", "R_c"]:
        point[k] = pt[k]
    point["R_rasch"] = rf["delta"]
    point["n_missing"] = np.isnan(d.Y).sum(0)

    bs = sdt_from_counts(np.einsum("bh,hjk->bjk", W, C))
    boot = {k: bs[k] for k in ["H", "F", "d_prime", "R_raw", "R_c"]}
    boot["R_rasch"] = np.stack([rasch_fit(d.Y, w_items=W[b, d.hearing], keep=keep)["delta"]
                                for b in range(B)])
    for k in ["d_prime", *MEASURES]:
        point[f"{k}_lo"], point[f"{k}_hi"] = percentile_ci(boot[k], ci)
        point[f"{k}_se"] = np.nanstd(boot[k], axis=0, ddof=1)
    return {"point": point, "boot": boot, "W": W, "rasch_full": rf, "keep": keep}


# ----------------------------------------------------------- variance decomposition
def ss_decomposition(values: np.ndarray, configs: pd.DataFrame, prompts: list[str]) -> np.ndarray:
    """Share of the total sum of squares per term (``TERMS``) for the balanced
    family x size x prompt design: main effects and two-way interactions, three-way
    interaction as residual. ``values`` is (J,) or (B, J)."""
    v = np.atleast_2d(values)
    fam, siz = sorted(configs["family"].unique()), sorted(configs["size"].unique())
    sel = configs["prompt_id"].isin(prompts).to_numpy()
    X = np.full((v.shape[0], len(fam), len(siz), len(prompts)), np.nan)
    for col, (_, r) in zip(np.where(sel)[0], configs[sel].iterrows()):
        X[:, fam.index(r.family), siz.index(r["size"]), prompts.index(r.prompt_id)] = v[:, col]
    assert not np.isnan(X).any(), "design is not complete"
    g = X.mean((1, 2, 3), keepdims=True)
    mF, mS, mP = (X.mean(ax, keepdims=True) - g for ax in [(2, 3), (1, 3), (1, 2)])
    mFS = X.mean(3, keepdims=True) - g - mF - mS
    mFP = X.mean(2, keepdims=True) - g - mF - mP
    mSP = X.mean(1, keepdims=True) - g - mS - mP
    res = X - g - mF - mS - mP - mFS - mFP - mSP
    ss = [(np.broadcast_to(e, X.shape) ** 2).reshape(X.shape[0], -1).sum(1)
          for e in (mF, mS, mP, mFS, mFP, mSP, res)]
    out = np.stack(ss, 1) / ((X - g) ** 2).reshape(X.shape[0], -1).sum(1)[:, None]
    return out if values.ndim == 2 else out[0]


def decomposition_table(res: dict, prompts_all: list[str], ci: float) -> pd.DataFrame:
    """Variance decomposition over (a) all prompts and (b) the neutral wordings only."""
    rows, cfg = [], res["point"]
    for version, prompts in (("a_all9", prompts_all), ("b_neutral4", NEUTRAL)):
        for m in MEASURES:
            p = ss_decomposition(cfg[m].to_numpy(), cfg, prompts)
            lo, hi = percentile_ci(ss_decomposition(res["boot"][m], cfg, prompts), ci)
            rows += [{"version": version, "measure": m, "term": t, "prop_ss": a, "lo": l, "hi": h}
                     for t, a, l, h in zip(TERMS, p, lo, hi)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- hypotheses
def _col(cfg, model, prompt):
    return int(np.where((cfg["model"] == model) & (cfg["prompt_id"] == prompt))[0][0])


def contrasts(res: dict, pairs: list[tuple[str, str]], ci: float, tag: str) -> pd.DataFrame:
    """R(a) − R(b) per model and measure, with bootstrap CIs, for each pair (a, b)."""
    cfg, rows = res["point"], []
    for m in cfg["model"].unique():
        for meas in MEASURES:
            for pa, pb in pairs:
                a, b = _col(cfg, m, pa), _col(cfg, m, pb)
                lo, hi = percentile_ci(res["boot"][meas][:, a] - res["boot"][meas][:, b], ci)
                rows.append({"hypothesis": tag, "model": m, "measure": meas, "contrast": f"{pa}-{pb}",
                             "estimate": cfg[meas].iat[a] - cfg[meas].iat[b], "lo": lo, "hi": hi,
                             "ci_gt_0": bool(lo > 0), "ci_lt_0": bool(hi < 0)})
    return pd.DataFrame(rows)


def verdict_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    """Outcome per measure for a contrast hypothesis (rule in the module docstring)."""
    out = []
    for meas, g in df.groupby("measure", sort=False):
        v = SUPPORTED if g["ci_gt_0"].all() else NOT_SUPPORTED if g["ci_lt_0"].any() else INCONCLUSIVE
        out.append({"measure": meas, "n_ci_gt_0": int(g["ci_gt_0"].sum()),
                    "n_ci_lt_0": int(g["ci_lt_0"].sum()), "n": len(g), "verdict": v})
    return pd.DataFrame(out)


def _spearman_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise Spearman correlation of two (B, J) arrays."""
    rx = np.apply_along_axis(rankdata, 1, X)
    ry = np.apply_along_axis(rankdata, 1, Y)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def _kendall_rows(X, Y) -> np.ndarray:
    return np.array([kendalltau(x, y).statistic for x, y in zip(X, Y)])


def convergence(res: dict, ci: float) -> pd.DataFrame:
    """H3: Spearman (and Kendall) between each pair of measures across configurations."""
    cfg, bt, rows = res["point"], res["boot"], []
    for a, b in itertools.combinations(MEASURES, 2):
        rho = spearmanr(cfg[a], cfg[b]).statistic
        lo, hi = percentile_ci(_spearman_rows(bt[a], bt[b]), ci)
        tlo, thi = percentile_ci(_kendall_rows(bt[a], bt[b]), ci)
        v = SUPPORTED if (rho >= 0.7 and lo > 0) else NOT_SUPPORTED if hi < 0.7 else INCONCLUSIVE
        rows.append({"pair": f"{a}~{b}", "spearman": rho, "lo": lo, "hi": hi,
                     "kendall": kendalltau(cfg[a], cfg[b]).statistic, "kendall_lo": tlo,
                     "kendall_hi": thi, "verdict": v})
    return pd.DataFrame(rows)


def ability_independence(res: dict, ci: float) -> pd.DataFrame:
    """H4: Spearman of each measure with d′, and |rho(R_raw,d′)| − |rho(R_c,d′)|."""
    cfg, bt, rows, rb = res["point"], res["boot"], [], {}
    for m in MEASURES:
        rb[m] = _spearman_rows(bt[m], bt["d_prime"])
        lo, hi = percentile_ci(rb[m], ci)
        rows.append({"quantity": f"rho({m}, d')", "estimate": spearmanr(cfg[m], cfg["d_prime"]).statistic,
                     "lo": lo, "hi": hi})
    est = abs(rows[0]["estimate"]) - abs(rows[1]["estimate"])
    lo, hi = percentile_ci(np.abs(rb["R_raw"]) - np.abs(rb["R_c"]), ci)
    rows.append({"quantity": "|rho(R_raw,d')| - |rho(R_c,d')|", "estimate": est, "lo": lo, "hi": hi,
                 "verdict": SUPPORTED if lo > 0 else NOT_SUPPORTED if hi < 0 else INCONCLUSIVE})
    return pd.DataFrame(rows)


def wording(res: dict, ci: float) -> pd.DataFrame:
    """H5: range over the neutral wordings (a) and S−L amplitude minus that range (b)."""
    cfg, bt, rows = res["point"], res["boot"], []
    for m in cfg["model"].unique():
        idx = [_col(cfg, m, p) for p in NEUTRAL]
        s, l = _col(cfg, m, "S"), _col(cfg, m, "L")
        for meas in MEASURES:
            v, b = cfg[meas].to_numpy(), bt[meas]
            rng_pt, rng_b = v[idx].max() - v[idx].min(), b[:, idx].max(1) - b[:, idx].min(1)
            amp_pt, amp_b = v[s] - v[l], b[:, s] - b[:, l]
            lo_a, hi_a = percentile_ci(rng_b, ci)
            lo_b, hi_b = percentile_ci(amp_b - rng_b, ci)
            rows.append({"model": m, "measure": meas, "range_neutral": rng_pt, "range_lo": lo_a,
                         "range_hi": hi_a, "amp_S_minus_L": amp_pt, "amp_minus_range": amp_pt - rng_pt,
                         "amp_minus_range_lo": lo_b, "amp_minus_range_hi": hi_b,
                         "a_ok": bool(lo_a > 0), "b_ok": bool(lo_b > 0), "b_contrary": bool(hi_b < 0)})
    return pd.DataFrame(rows)


def verdict_wording(df: pd.DataFrame) -> pd.DataFrame:
    """Outcome per measure for H5."""
    out = []
    for meas, g in df.groupby("measure", sort=False):
        ok = (g["a_ok"] & g["b_ok"]).all()
        v = SUPPORTED if ok else NOT_SUPPORTED if g["b_contrary"].any() else INCONCLUSIVE
        out.append({"measure": meas, "n_a_ok": int(g["a_ok"].sum()), "n_b_ok": int(g["b_ok"].sum()),
                    "n": len(g), "verdict": v})
    return pd.DataFrame(out)


# ------------------------------------------------------------------- reliability
def stability(res: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kendall tau between each replicate's ranking and the full-sample ranking, and the
    fraction of replicates that keep the order of every pair of configurations."""
    cfg, taus, pairs = res["point"], [], []
    for m in MEASURES:
        full, b = cfg[m].to_numpy(), res["boot"][m]
        t = _kendall_rows(b, np.broadcast_to(full, b.shape))
        taus.append(pd.DataFrame({"measure": m, "replicate": np.arange(len(t)), "tau": t}))
        for j, k in itertools.combinations(range(len(full)), 2):
            pairs.append({"measure": m, "config_a": f"{cfg.model[j]}|{cfg.prompt_id[j]}",
                          "config_b": f"{cfg.model[k]}|{cfg.prompt_id[k]}",
                          "frac_order_kept": float((np.sign(b[:, j] - b[:, k]) == np.sign(full[j] - full[k])).mean())})
    return pd.concat(taus), pd.DataFrame(pairs)


def separation(res: dict) -> pd.DataFrame:
    """Separation reliability of the configurations for each measure (SE = bootstrap SD)."""
    cfg = res["point"]
    return pd.DataFrame([{"measure": m, "R_sep": separation_reliability(cfg[m].to_numpy(),
                                                                       cfg[f"{m}_se"].to_numpy())}
                         for m in MEASURES])


def rasch_report(d: Design, res: dict) -> dict:
    """Likelihood-ratio test (interaction vs. main effects), facet decomposition, infit/outfit."""
    keep, cfg, fi = res["keep"], d.configs, res["rasch_full"]
    Y = d.Y[keep]
    mi = pd.Categorical(cfg["model"], categories=cfg["model"].unique()).codes.astype(int)
    pi = pd.Categorical(cfg["prompt_id"], categories=cfg["prompt_id"].unique()).codes.astype(int)
    fm = fit_jml(Y, [mi, pi])
    lr = 2 * (fi["loglik"] - fm["loglik"])
    df_ = (Y.shape[1] - 1) - (mi.max() + pi.max())
    dec = decompose_two_way(fi["delta"], mi, pi)
    fs = fit_stats(Y, fi["P"])
    return {"lr_stat": lr, "lr_df": int(df_), "lr_p": float(chi2.sf(lr, df_)),
            "converged": bool(fi["converged"] and fm["converged"]),
            "alpha_model": dict(zip(cfg["model"].unique(), dec["alpha"])),
            "gamma_prompt": dict(zip(cfg["prompt_id"].unique(), dec["gamma"])),
            "n_items_used": int(keep.sum()), "n_extreme": int((~keep).sum()),
            "config_fit": cfg.assign(infit=fs["infit_config"], outfit=fs["outfit_config"], R_rasch=fi["delta"]),
            "item_fit": pd.DataFrame({"item_uid": d.item_uid[keep], "theta": fi["theta"],
                                      "infit": fs["infit_item"], "outfit": fs["outfit_item"]})}


def paper_judges(items: pd.DataFrame, d: Design, res: dict, ci: float) -> tuple[pd.DataFrame, dict]:
    """The dataset's twelve judges: (i) R_raw, R_c, d′ with hearing-bootstrap CIs
    (descriptive); (ii) a Rasch refit with them as extra configurations, reported on the
    scale of our configurations (shifted so that ours average to zero), plus Kendall tau
    of our severities with vs. without them."""
    P = items[PAPER_COLS].to_numpy(float)
    dp = Design(Y=P, label=d.label, hearing=d.hearing, configs=pd.DataFrame({"judge": PAPER_COLS}),
                item_uid=d.item_uid)
    C = sdt_hearing_counts(dp)
    pt, bs = sdt_from_counts(C.sum(0)), sdt_from_counts(np.einsum("bh,hjk->bjk", res["W"], C))
    out = pd.DataFrame({"judge": PAPER_COLS, "prompt_id": [c.split("_")[1] for c in PAPER_COLS],
                        "paper_model": ["_".join(c.split("_")[2:]) for c in PAPER_COLS]})
    for k in ["H", "F", "d_prime", "R_raw", "R_c"]:
        out[k] = pt[k]
        if k not in ("H", "F"):
            out[f"{k}_lo"], out[f"{k}_hi"] = percentile_ci(bs[k], ci)
    J = d.Y.shape[1]
    f_all = rasch_fit(np.hstack([d.Y, P]))
    out["R_rasch"] = f_all["delta"][J:] - f_all["delta"][:J].mean()
    tau = kendalltau(res["point"]["R_rasch"], f_all["delta"][:J]).statistic
    return out, {"kendall_tau_ours_with_vs_without": float(tau)}


# --------------------------------------------------------------- full pipeline
def confirmatory_analysis(d: Design, items: pd.DataFrame, prompts: list[str], B: int, seed: int,
                          ci: float, out: Path, with_paper_judges: bool = True) -> dict:
    """Every confirmatory analysis on ``d``; writes CSV tables and ``summary.json`` to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    res = compute_all(d, B, seed, ci)
    res["point"].to_csv(out / "measures_by_config.csv", index=False)
    s = {"B": B, "n_extreme_items": int((~res["keep"]).sum())}

    h1 = contrasts(res, [("S", "N"), ("N", "L")], ci, "H1")
    h2 = contrasts(res, [("P2", "P1"), ("P3", "P2")], ci, "H2")
    e3 = contrasts(res, [("P1", "N")], ci, "E3")
    pd.concat([h1, h2, e3]).to_csv(out / "contrasts.csv", index=False)
    s["H1"] = verdict_contrasts(h1).to_dict("records")
    s["H2"] = verdict_contrasts(h2).to_dict("records")
    h3 = convergence(res, ci)
    h3.to_csv(out / "convergence.csv", index=False)
    s["H3"] = h3.to_dict("records")
    h4 = ability_independence(res, ci)
    h4.to_csv(out / "ability.csv", index=False)
    s["H4"] = h4.to_dict("records")
    h5 = wording(res, ci)
    h5.to_csv(out / "wording.csv", index=False)
    s["H5"] = verdict_wording(h5).to_dict("records")

    decomposition_table(res, prompts, ci).to_csv(out / "variance_decomposition.csv", index=False)
    taus, pairs = stability(res)
    taus.to_csv(out / "stability_kendall_tau.csv", index=False)
    pairs.to_csv(out / "stability_pairwise.csv", index=False)
    s["stability_tau_median"] = taus.groupby("measure")["tau"].median().to_dict()
    sep = separation(res)
    sep.to_csv(out / "separation_reliability.csv", index=False)
    s["separation"] = sep.to_dict("records")
    rr = rasch_report(d, res)
    rr["config_fit"].to_csv(out / "rasch_config_fit.csv", index=False)
    rr["item_fit"].to_csv(out / "rasch_item_fit.csv", index=False)
    s["rasch"] = {k: v for k, v in rr.items() if k not in ("config_fit", "item_fit")}
    if with_paper_judges:
        pj, info = paper_judges(items, d, res, ci)
        pj.to_csv(out / "paper_judges.csv", index=False)
        s["paper_judges"] = info
    json.dump(s, open(out / "summary.json", "w"), indent=2, default=float)
    return s


# ---------------------------------------------------------- exploratory analyses
def reproducibility(main_dir: Path, rep_dir: Path, items: pd.DataFrame, models: list[dict],
                    prompts: list[str]) -> pd.DataFrame:
    """Agreement between two runs with identical decoding, per configuration."""
    rows = []
    for m in models:
        for p in prompts:
            a = read_raw(raw_file(main_dir, m["short"], p)).set_index("item_uid").reindex(items["item_uid"])
            b = read_raw(raw_file(rep_dir, m["short"], p)).set_index("item_uid").reindex(items["item_uid"])
            both = a["flag"].notna() & b["flag"].notna()
            rows.append({"model": m["short"], "prompt_id": p,
                         "agree": float((a.flag[both] == b.flag[both]).mean()),
                         "kappa": float(cohen_kappa_score(a.flag[both], b.flag[both])),
                         "identical_text": float((a.raw_text == b.raw_text).mean()),
                         "abs_diff_R_raw": float(abs(a.flag.mean() - b.flag.mean()))})
    return pd.DataFrame(rows)


def fill_recovered(d: Design, recovery_dir: Path) -> tuple[Design, int, int]:
    """Copy of ``d`` with missing verdicts replaced by regenerated ones where available."""
    Y = d.Y.copy()
    pos = {u: i for i, u in enumerate(d.item_uid)}
    filled = still = 0
    for j, (m, p) in enumerate(zip(d.configs.model, d.configs.prompt_id)):
        f = raw_file(recovery_dir, m, p)
        if not f.exists():
            continue
        for u, fl in zip(*read_raw(f)[["item_uid", "flag"]].to_numpy().T):
            if pd.notna(fl):
                Y[pos[u], j], filled = fl, filled + 1
            else:
                still += 1
    return Design(Y=Y, label=d.label, hearing=d.hearing, configs=d.configs, item_uid=d.item_uid), filled, still


def soft_scores(raw_dir: Path, items: pd.DataFrame, models: list[dict], prompts: list[str]) -> pd.DataFrame:
    """Verdict log-probability scores per configuration: AUC of P(hallucination), AUC of
    the binary verdict ((H − F + 1)/2), their difference, mean P(hallucination), the share
    of saturated probabilities, and agreement of the log-probability argmax with the verdict."""
    lab, rows = items.set_index("item_uid")["label_halluc"], []
    for m in models:
        for p in prompts:
            r = read_raw(raw_file(raw_dir, m["short"], p)).set_index("item_uid").reindex(items["item_uid"])
            ph = r["verdict_logprobs"].map(lambda v: v["p_halluc"] if isinstance(v, dict) else np.nan)
            ok, fl = ph.notna(), r["flag"]
            both = ok & fl.notna()
            auc = roc_auc_score(lab[ph.index][ok], ph[ok])
            auc_bin = roc_auc_score(lab[fl.index][fl.notna()], fl.dropna())
            rows.append({"model": m["short"], "prompt_id": p, "n_with_logprob": int(ok.sum()),
                         "AUC": auc, "AUC_binary": auc_bin, "delta_AUC": auc - auc_bin,
                         "mean_p_halluc": float(ph[ok].mean()),
                         "frac_p_extreme": float(((ph[ok] < 0.01) | (ph[ok] > 0.99)).mean()),
                         "argmax_matches_verdict": float(((ph[both] > 0.5).astype(int) == fl[both]).mean())})
    return pd.DataFrame(rows)
