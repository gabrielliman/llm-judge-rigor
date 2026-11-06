"""Cluster (hearing-level) bootstrap."""
from __future__ import annotations

import numpy as np


def hearing_weights(n_hearings: int, B: int, seed: int) -> np.ndarray:
    """``B × n_hearings`` matrix of resampling counts.

    Row b gives how many times each hearing is drawn (with replacement, n draws) in
    replicate b. Applying the same matrix to every measure keeps replicates aligned
    across configurations, so contrasts and correlations are bootstrapped jointly.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_hearings, size=(B, n_hearings))
    w = np.zeros((B, n_hearings))
    for b in range(B):
        w[b] = np.bincount(idx[b], minlength=n_hearings)
    return w


def resample_counts(per_hearing: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bootstrap totals: ``weights (B×H) @ per_hearing (H×k)`` → ``B×k``."""
    return weights @ per_hearing


def percentile_ci(x: np.ndarray, level: float = 0.95, axis: int = 0):
    """Percentile CI (lower, upper) along ``axis``, ignoring NaNs."""
    a = (1 - level) / 2
    return np.nanquantile(x, a, axis=axis), np.nanquantile(x, 1 - a, axis=axis)
