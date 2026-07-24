"""Shared core for the lag-order (M1) and ridge-alpha (I3) sensitivity analyses.

Does NOT reimplement the diagnostic: it CALLS the primary code (scripts/test_b.py:
fit_var1, bootstrap_site, select_top_regions) and applies the identical
per-coefficient two-sample Wald z + BH-FDR + >20%-of-pairs rule used by
test_b_marginal.py / condition_sms/test_condition_sms.py. The only extension is
VAR(1) -> VAR(p): the lag design is stacked and p=1 is routed through the exact
fit_var1, so the primary Table-3 numbers reproduce at (p=1, alpha=1).

GENERALIZED PARTITION at higher lags (stated in every output):
  with lag matrices A_1..A_p (each R x R, A_k[i,j] = effect of region j at lag k
  on region i), the coefficient grid is (R, p*R).
    DIAGONAL      = all self terms across every lag  {A_k[i,i]}           (p*R)
    OFF-DIAGONAL  = all cross terms across every lag  {A_k[i,j], i!=j}    (p*R*(R-1))
  The diagonal/off-diagonal counts and the >20%-of-pairs unstable rule apply to
  these generalized sets.

Standardization: NONE added here. ABIDE (CC200, abide_harmonized.npz) is
ComBat-harmonized and the masker used zscore_sample; AOMIC (Schaefer-200) is
masker-standardized + WM task-regressed. Both are used exactly as the primary
pipelines use them (no extra z-scoring).
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import norm, ttest_rel
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_b import fit_var1, select_top_regions           # reused primary code

N_TOP = 50
N_BOOT = 100
FDR_Q = 0.05
UNSTABLE_THRESH = 0.20
RNG_SEED = 20260628                                        # primary bootstrap seed


# --------------------------------------------------------------------------- #
# data loaders (identical selection to the primary pipelines)
# --------------------------------------------------------------------------- #
def load_abide():
    """(X_top (742,116,50), site_ids). CC200, ComBat-harmonized; top-50 by pooled
    variance -- exactly scripts/test_b.py."""
    z = np.load(ROOT / "data/processed/abide_harmonized.npz", allow_pickle=True)
    X = z["X"].astype(np.float64)
    top = select_top_regions(X, N_TOP)
    return X[:, :, top], z["site_ids"]


def load_condition():
    """(rest_top, wm_top) each (222,160,50). Schaefer-200, WM task-regressed;
    top-50 by pooled variance across BOTH conditions -- exactly
    scripts/condition_sms/test_condition_sms.py."""
    TSD = ROOT / "data/condition_sms/ts"
    subs = sorted(p.name[:-9] for p in TSD.glob("*_rest.npy")
                  if (TSD / f"{p.name[:-9]}_wm.npy").exists())
    rest = np.stack([np.load(TSD / f"{s}_rest.npy") for s in subs]).astype(np.float64)
    wm = np.stack([np.load(TSD / f"{s}_wm.npy") for s in subs]).astype(np.float64)
    top = select_top_regions(np.concatenate([rest, wm], axis=0), N_TOP)
    return rest[:, :, top], wm[:, :, top]


# --------------------------------------------------------------------------- #
# VAR(p) fit + bootstrap (p=1 == the primary fit_var1 exactly)
# --------------------------------------------------------------------------- #
def var_design(X_site, p):
    """(n,T,R) -> (rows, p*R) stacked-lag predictors and (rows, R) targets, per
    subject, without bridging across subjects. Column block k (0-based) is lag
    k+1, so coef_[:, k*R:(k+1)*R] = A_{k+1}."""
    n, T, R = X_site.shape
    Y = X_site[:, p:, :].reshape(-1, R)
    lags = [X_site[:, p - k:T - k, :].reshape(-1, R) for k in range(1, p + 1)]
    return np.concatenate(lags, axis=1), Y


def fit_varp(X_site, p, alpha):
    """Ridge VAR(p); returns (R, p*R). p=1 uses the primary fit_var1 verbatim."""
    if p == 1:
        return fit_var1(X_site, alpha=alpha)
    Xd, Y = var_design(X_site, p)
    return Ridge(alpha=alpha, fit_intercept=True).fit(Xd, Y).coef_


def bootstrap_varp(X_site, p, n_boot, rng, alpha):
    """Subject-resampling bootstrap (mirrors test_b.bootstrap_site, VAR(p) +
    explicit alpha). Returns (n_boot, R, p*R)."""
    n = X_site.shape[0]
    boots = None
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = fit_varp(X_site[idx], p, alpha)
        if boots is None:
            boots = np.empty((n_boot,) + c.shape, dtype=np.float64)
        boots[b] = c
    return boots


def diag_mask(R, p):
    """(R, p*R) boolean: True on the generalized diagonal {A_k[i,i]}."""
    m = np.zeros((R, p * R), dtype=bool)
    for k in range(p):
        for i in range(R):
            m[i, k * R + i] = True
    return m


# --------------------------------------------------------------------------- #
# the three diagnostic runs
# --------------------------------------------------------------------------- #
def site_diagnostic(X, site_ids, p, alpha, n_boot=N_BOOT, seed=RNG_SEED):
    """ABIDE site: per-pair two-sample Wald z, BH-FDR jointly over all pairs,
    >20%-of-pairs unstable rule. Returns counts on the generalized partition."""
    R = X.shape[-1]
    sites = sorted(np.unique(site_ids).tolist())
    rng = np.random.default_rng(seed)
    mean_s, var_s = {}, {}
    for s in sites:
        b = bootstrap_varp(X[site_ids == s], p, n_boot, rng, alpha)
        mean_s[s], var_s[s] = b.mean(0), b.var(0, ddof=1)
    pairs = list(combinations(range(len(sites)), 2))
    Pmat = np.empty((len(pairs), R, p * R))
    absz = np.empty((len(pairs), R, p * R))
    for k, (a, b) in enumerate(pairs):
        d = mean_s[sites[a]] - mean_s[sites[b]]
        se = np.sqrt(var_s[sites[a]] + var_s[sites[b]])
        z = np.where(se > 0, d / se, 0.0)
        absz[k] = np.abs(z)
        Pmat[k] = 2.0 * norm.sf(np.abs(z))
    reject = multipletests(Pmat.ravel(), alpha=FDR_Q,
                           method="fdr_bh")[0].reshape(Pmat.shape)
    unstable = reject.mean(0) > UNSTABLE_THRESH
    dm = diag_mask(R, p)
    return _counts(reject.mean(), float(np.median(absz)), unstable, dm,
                   n_pairs=len(pairs))


def condition_bootstrap_diagnostic(rest, wm, p, alpha, n_boot=N_BOOT, seed=RNG_SEED):
    """AOMIC condition (PRIMARY at higher lags): unpaired bootstrap two-sample
    Wald z on the single rest-vs-WM contrast, BH-FDR over the grid."""
    R = rest.shape[-1]
    rng = np.random.default_rng(seed)
    b_rest = bootstrap_varp(rest, p, n_boot, rng, alpha)
    b_wm = bootstrap_varp(wm, p, n_boot, rng, alpha)
    d = b_rest.mean(0) - b_wm.mean(0)
    se = np.sqrt(b_rest.var(0, ddof=1) + b_wm.var(0, ddof=1))
    z = np.where(se > 0, d / se, 0.0)
    reject = multipletests((2.0 * norm.sf(np.abs(z))).ravel(), alpha=FDR_Q,
                           method="fdr_bh")[0].reshape(z.shape)
    dm = diag_mask(R, p)
    return _counts(reject.mean(), float(np.median(np.abs(z))), reject, dm)


def condition_paired_diagnostic(rest, wm, p, alpha):
    """AOMIC condition PAIRED sensitivity (per-subject VAR, paired t, BH-FDR) --
    reproduces network_localize.py Step 3, generalized to VAR(p). Per-subject
    fits at p>1 are underdetermined (T~160 vs p*R predictors): report but FLAG."""
    R = rest.shape[-1]
    N = rest.shape[0]
    rc = np.stack([fit_varp(rest[k:k + 1], p, alpha) for k in range(N)])
    wc = np.stack([fit_varp(wm[k:k + 1], p, alpha) for k in range(N)])
    _t, pval = ttest_rel(rc, wc, axis=0)
    pflat = np.nan_to_num(pval.ravel(), nan=1.0)          # zero-variance coefs -> p=1
    reject = multipletests(pflat, alpha=FDR_Q,
                           method="fdr_bh")[0].reshape(pval.shape)
    dm = diag_mask(R, p)
    return _counts(reject.mean(), float("nan"), reject, dm)


def _counts(overall, median_absz, mask_or_reject, dm, n_pairs=None):
    m = mask_or_reject
    out = dict(overall_pct=float(overall * 100),
               median_absz=median_absz,
               diag_unstable=int(m[dm].sum()), diag_total=int(dm.sum()),
               off_unstable=int(m[~dm].sum()), off_total=int((~dm).sum()))
    if n_pairs is not None:
        out["n_pairs"] = n_pairs
    return out


# --------------------------------------------------------------------------- #
# AIC / BIC lag-order selection (OLS pooled per group, standard ML criteria)
# --------------------------------------------------------------------------- #
def aic_bic(X_group, p):
    """Lutkepohl information criteria for a pooled OLS VAR(p) fit on one group.
    Returns (aic, bic, n_obs). Uses OLS (ML) as is standard for order selection;
    the diagnostic itself uses ridge."""
    Xd, Y = var_design(X_group, p)
    Xd1 = np.hstack([Xd, np.ones((len(Xd), 1))])
    beta, *_ = np.linalg.lstsq(Xd1, Y, rcond=None)
    resid = Y - Xd1 @ beta
    Nobs, R = Y.shape
    Sigma = resid.T @ resid / Nobs
    # log|Sigma| via eigenvalues: numerically clean and warning-free (numpy's
    # slogdet emits benign overflow warnings from its internal det/sign step even
    # though its log-domain result is correct). Positive floor guards a rare
    # singular Sigma. Here Sigma is full-rank/well-conditioned for every group.
    w = np.linalg.eigvalsh(Sigma)
    w = np.clip(w, 1e-12 * float(w.max()), None)
    logdet = float(np.sum(np.log(w)))
    kparams = p * R * R                                    # AR coefficients
    aic = logdet + 2.0 * kparams / Nobs
    bic = logdet + np.log(Nobs) * kparams / Nobs
    return float(aic), float(bic), int(Nobs)


def select_orders(groups, ps=(1, 2, 3)):
    """Sum AIC/BIC across a dataset's groups per lag; return the argmin order for
    each criterion and the per-order totals."""
    aic_tot = {p: 0.0 for p in ps}
    bic_tot = {p: 0.0 for p in ps}
    for g in groups:
        for p in ps:
            a, b, _ = aic_bic(g, p)
            aic_tot[p] += a
            bic_tot[p] += b
    aic_sel = min(ps, key=lambda p: aic_tot[p])
    bic_sel = min(ps, key=lambda p: bic_tot[p])
    return aic_sel, bic_sel, aic_tot, bic_tot


# --------------------------------------------------------------------------- #
# atomic, resumable cell IO
# --------------------------------------------------------------------------- #
def save_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)
