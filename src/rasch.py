"""Many-facet Rasch model by joint maximum likelihood.

    logit P(flag_ij = 1) = θ_i + Σ_f η_f[level_f(j)]

θ_i: suspicion of item i (fixed effect). Facets are effects of configuration j; the
"interaction" model uses a single facet with one level per configuration (δ_j =
α_m + γ_p + (αγ)_mp), the "main-effects" model uses two facets (model, prompt).
Identification: sum-to-zero within each facet. Estimation is unpenalised JML (the
approach of FACETS), implemented as block-coordinate Newton–Raphson on the dense
item × configuration matrix: within a block (θ, or one facet) the Hessian is
diagonal, so each block update is exact 1-D Newton per parameter. This is the
same MLE as an unpenalised sparse logistic regression with item and facet dummies.
Items with extreme scores (all 0 or all 1 over their non-missing responses) are
not identifiable under JML and must be excluded by the caller (see
:func:`extreme_items`). Missing responses (NaN) are skipped in the likelihood.
Item weights implement the hearing-cluster bootstrap (a duplicated item has the
same data, hence the same θ, so weighting is equivalent to duplication).
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit


def extreme_items(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of items whose non-missing responses are all 0 or all 1."""
    obs = ~np.isnan(Y)
    s = np.nansum(Y, axis=1)
    n = obs.sum(axis=1)
    return (s == 0) | (s == n)


def fit_jml(Y: np.ndarray, facets: list[np.ndarray], w: np.ndarray | None = None,
            tol: float = 1e-8, max_iter: int = 5000, max_step: float = 2.0) -> dict:
    """Fit the many-facet Rasch model by JML.

    Parameters
    ----------
    Y : (I, J) array of 0/1/NaN (items × configurations), extreme items removed.
    facets : list of length-J integer arrays; ``facets[f][j]`` is the level of
        configuration j in facet f (levels 0..L_f-1).
    w : optional (I,) item weights (bootstrap counts); 0 removes an item.

    Returns dict with ``theta`` (I,), ``eta`` (list of per-facet level effects,
    each summing to zero), ``delta`` (J,) = Σ_f η_f[level_f(j)], ``loglik``,
    ``n_iter``, ``converged``.
    """
    I, J = Y.shape
    obs = ~np.isnan(Y)
    Y0 = np.where(obs, Y, 0.0)
    w = np.ones(I) if w is None else np.asarray(w, float)
    W = obs * w[:, None]                      # per-cell weight (0 if missing)
    theta = np.zeros(I)
    eta = [np.zeros(int(f.max()) + 1) for f in facets]
    delta = np.zeros(J)
    converged = False
    for it in range(max_iter):
        max_change = 0.0
        # θ block
        P = expit(theta[:, None] + delta[None, :])
        g = ((Y0 - P) * W).sum(1)
        h = (P * (1 - P) * W).sum(1)
        step = np.clip(np.divide(g, h, out=np.zeros_like(g), where=h > 0), -max_step, max_step)
        theta += step
        max_change = max(max_change, np.abs(step).max())
        # facet blocks
        for f, lev in enumerate(facets):
            P = expit(theta[:, None] + delta[None, :])
            gj = ((Y0 - P) * W).sum(0)
            hj = (P * (1 - P) * W).sum(0)
            L = len(eta[f])
            gl = np.bincount(lev, weights=gj, minlength=L)
            hl = np.bincount(lev, weights=hj, minlength=L)
            st = np.clip(np.divide(gl, hl, out=np.zeros_like(gl), where=hl > 0), -max_step, max_step)
            eta[f] += st
            shift = eta[f].mean()             # sum-to-zero; absorb shift into θ
            eta[f] -= shift
            theta += shift
            max_change = max(max_change, np.abs(st).max())
            delta = sum(e[l] for e, l in zip(eta, facets))
        if max_change < tol:
            converged = True
            break
    P = expit(theta[:, None] + delta[None, :])
    ll = float((W * (Y0 * np.log(P + 1e-300) + (1 - Y0) * np.log(1 - P + 1e-300))).sum())
    return {"theta": theta, "eta": eta, "delta": delta, "loglik": ll,
            "n_iter": it + 1, "converged": converged, "P": P}


def decompose_two_way(delta: np.ndarray, model_idx: np.ndarray, prompt_idx: np.ndarray) -> dict:
    """Split configuration effects δ_mp into α_m, γ_p, (αγ)_mp (all sum-to-zero).

    Exact for a complete crossed design (every model × prompt present).
    """
    M, Pn = model_idx.max() + 1, prompt_idx.max() + 1
    D = np.full((M, Pn), np.nan)
    D[model_idx, prompt_idx] = delta
    g = np.nanmean(D)
    a = np.nanmean(D, 1) - g
    c = np.nanmean(D, 0) - g
    ac = D - g - a[:, None] - c[None, :]
    return {"grand": g, "alpha": a, "gamma": c, "alphagamma": ac}


def fit_stats(Y: np.ndarray, P: np.ndarray) -> dict:
    """Infit and outfit mean squares per configuration (columns) and per item (rows)."""
    obs = ~np.isnan(Y)
    V = P * (1 - P)
    R2 = np.where(obs, (np.nan_to_num(Y) - P) ** 2, 0.0)
    Vo = np.where(obs, V, 0.0)
    Z2 = np.divide(R2, Vo, out=np.zeros_like(R2), where=obs)
    out = {}
    for name, ax in (("config", 0), ("item", 1)):
        n = obs.sum(ax)
        out[f"outfit_{name}"] = Z2.sum(ax) / n
        out[f"infit_{name}"] = R2.sum(ax) / Vo.sum(ax)
    return out


def separation_reliability(sev: np.ndarray, se: np.ndarray) -> float:
    """R_sep = (Var(sev) − mean(SE²)) / Var(sev)."""
    v = np.var(sev, ddof=1)
    return float((v - np.mean(se ** 2)) / v)
