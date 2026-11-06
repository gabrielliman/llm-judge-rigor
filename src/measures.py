"""Raw flag rate and signal-detection measures. Larger rigor values = stricter judge.

Functions take count arrays, so they vectorise over configurations and bootstrap
replicates: ``hits`` = #(flag 1, hallucination), ``n_pos`` = #hallucinations,
``fas`` = #(flag 1, valid opinion), ``n_neg`` = #valid opinions (valid verdicts only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def hautus_rates(hits, n_pos, fas, n_neg):
    """Log-linear corrected hit and false-alarm rates (Hautus, 1995), used everywhere."""
    H = (np.asarray(hits) + 0.5) / (np.asarray(n_pos) + 1)
    F = (np.asarray(fas) + 0.5) / (np.asarray(n_neg) + 1)
    return H, F


def sdt(hits, n_pos, fas, n_neg) -> dict:
    """Uncorrected H and F, ability d′ = z(H) − z(F), rigor R_c = (z(H) + z(F))/2 = −c,
    and the raw flag rate R_raw."""
    zH, zF = (norm.ppf(r) for r in hautus_rates(hits, n_pos, fas, n_neg))
    hits, n_pos, fas, n_neg = map(np.asarray, (hits, n_pos, fas, n_neg))
    return {"H": hits / n_pos, "F": fas / n_neg, "d_prime": zH - zF, "R_c": (zH + zF) / 2,
            "R_raw": (hits + fas) / (n_pos + n_neg)}


def per_hearing_counts(df: pd.DataFrame, flag_col: str = "flag", label_col: str = "label_halluc",
                       hearing_col: str = "hearing_id") -> pd.DataFrame:
    """SDT counts aggregated per hearing (the resampling unit of the cluster bootstrap)."""
    d = df[[hearing_col, flag_col, label_col]].copy()
    v = d[flag_col].notna()
    d["hits"] = (v & (d[flag_col] == 1) & (d[label_col] == 1)).astype(float)
    d["n_pos"] = (v & (d[label_col] == 1)).astype(float)
    d["fas"] = (v & (d[flag_col] == 1) & (d[label_col] == 0)).astype(float)
    d["n_neg"] = (v & (d[label_col] == 0)).astype(float)
    return d.groupby(hearing_col)[["hits", "n_pos", "fas", "n_neg"]].sum()
