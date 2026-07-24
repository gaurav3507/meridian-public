"""Ground-truth generators for the VAR-stability diagnostic validation.

Family 1 (DIRECT-VAR): a 50-region VAR(1) is generated DIRECTLY at the observed
level, so the injected coefficient matrix IS the ground truth (the diagnostic
fits VAR on these same 50 regions). This is deliberately different from the main
simulator, which generates a d=10 latent VAR and mixes up to 50 regions -- there,
the observed-level coefficients are not the injected ones. The main simulator's
DAG generator is also strictly triangular with NO self-loops, so it cannot
produce the diagonal (autocorrelation) structure the diagnostic keys on.

We REUSE the simulator where it applies:
  * dynamics.spectral_radius / dynamics.ensure_stationary  -> stationarity guard
  * hrf.glover_kernel / hrf._convolve_causal (TR=2.0)       -> Family 2 HRF path
The VAR recurrence below mirrors dynamics.simulate_latents (burn-in + Gaussian
innovations), specialised to a raw coefficient matrix M with self-loops.

Convention (matches the diagnostic): x[t] = M @ x[t-1] + noise, so M[i, j] is the
effect of region j (source, column) on region i (target, row). Diagonal M[i, i]
= self-loop (autocorrelation, "measurement"); off-diagonal = cross-region
influence ("mechanism"). This is the same (target=row i, source=col j) layout the
diagnostic's fitted A_hat and the marginal CSVs use.

NOTE on standardisation: the real pipeline z-scores each region to unit variance
(masker standardize="zscore_sample"). Per-region scaling leaves the diagonal
A[i,i] invariant (the i/i scale factors cancel) and only rescales off-diagonal
values within a row, so it never MOVES an injected shift between the diagonal and
off-diagonal partitions. Localisation ground truth is therefore preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parent.parent
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
from dynamics import spectral_radius, ensure_stationary      # reused
from hrf import glover_kernel, _convolve_causal, TR           # reused


# --------------------------------------------------------------------------- #
# base coefficient matrix
# --------------------------------------------------------------------------- #
def make_base_A(p=50, seed=0, diag_lo=0.20, diag_hi=0.40, off_density=0.10,
                off_lo=0.05, off_hi=0.15, target_rho=0.70):
    """Sparse stationary VAR(1) matrix with diagonal self-loops.

    Returns (M, rho, offdiag_support_mask). ensure_stationary (reused) shrinks
    uniformly if needed, which preserves the diagonal/off-diagonal STRUCTURE.
    """
    rng = np.random.default_rng(seed)
    M = np.zeros((p, p))
    np.fill_diagonal(M, rng.uniform(diag_lo, diag_hi, p))
    off_flat = np.flatnonzero(~np.eye(p, dtype=bool).ravel())
    n_off = int(round(off_density * p * (p - 1)))
    chosen = rng.choice(off_flat, size=n_off, replace=False)
    M.flat[chosen] = rng.choice([-1.0, 1.0], n_off) * rng.uniform(off_lo, off_hi, n_off)
    A = M[None, :, :].copy()                       # (1, p, p) for the reused check
    A, rho, _ = ensure_stationary(A, target=target_rho)
    M = A[0]
    support = np.zeros((p, p), dtype=bool)
    support.flat[chosen] = True
    return M, float(rho), support


# --------------------------------------------------------------------------- #
# VAR(1) simulation (mirrors dynamics.simulate_latents, raw-matrix form)
# --------------------------------------------------------------------------- #
def simulate_var(M, n_subj, T, noise_scale, rng, burn_in=50):
    """Return (n_subj, T, p): independent VAR(1) realisations x[t]=M x[t-1]+e."""
    p = M.shape[0]
    out = np.empty((n_subj, T, p), dtype=np.float64)
    for s in range(n_subj):
        total = burn_in + T
        x = np.zeros((total, p))
        e = noise_scale * rng.standard_normal((total, p))
        for t in range(1, total):
            x[t] = M @ x[t - 1] + e[t]
        out[s] = x[burn_in:]
    return out


def zscore_subjects(X):
    """Per-subject, per-region z-score over time (matches zscore_sample)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


# --------------------------------------------------------------------------- #
# shift supports and per-condition matrices
# --------------------------------------------------------------------------- #
def pick_support(p, seed, n_diag, n_off):
    """Fixed reproducible set of injected coefficients: diagonal (i,i) and
    off-diagonal (i,j) locations, with fixed random signs."""
    rng = np.random.default_rng(seed)
    diag = rng.choice(p, size=n_diag, replace=False)
    off_pool = [(i, j) for i in range(p) for j in range(p) if i != j]
    off = [off_pool[k] for k in rng.choice(len(off_pool), size=n_off, replace=False)]
    diag_sign = rng.choice([-1.0, 1.0], size=n_diag)
    off_sign = rng.choice([-1.0, 1.0], size=n_off)
    return dict(diag=diag, off=off, diag_sign=diag_sign, off_sign=off_sign)


def make_condition_matrix(M_base, condition, magnitude, support):
    """Build (M_A, M_B, injected_mask) for a 2-level contrast.

    C1 diagonal-only shift; C2 off-diagonal-only; C3 both; C4 identical (null).
    injected_mask marks the coefficients that DIFFER between the two levels
    (empty for C4). Raises if M_B is non-stationary at this magnitude.
    """
    p = M_base.shape[0]
    M_A = M_base.copy()
    M_B = M_base.copy()
    injected = np.zeros((p, p), dtype=bool)
    if condition in ("C1", "C3"):
        for idx, s in zip(support["diag"], support["diag_sign"]):
            M_B[idx, idx] += magnitude * s
            injected[idx, idx] = True
    if condition in ("C2", "C3"):
        for (i, j), s in zip(support["off"], support["off_sign"]):
            M_B[i, j] += magnitude * s
            injected[i, j] = True
    # C4: no coefficient change (nuisance handled by the caller: noise + N)
    rho_B = spectral_radius(M_B[None, :, :])
    if rho_B >= 0.995:
        raise ValueError(f"{condition} mag={magnitude}: M_B non-stationary "
                         f"(rho={rho_B:.3f}); reduce magnitude range")
    return M_A, M_B, injected


def make_multilevel_matrices(M_base, n_levels, magnitude, support, seed):
    """10-level site-like design: a FIXED sparse support S (diag+off) varies
    across levels; off-S coefficients are identical across levels.

    Returns (list of M_level, injected_mask). injected_mask marks S (the
    coefficients that are unstable across levels); the '>20% of pairs' rule
    should recover exactly this support.
    """
    p = M_base.shape[0]
    rng = np.random.default_rng(seed)
    injected = np.zeros((p, p), dtype=bool)
    for idx in support["diag"]:
        injected[idx, idx] = True
    for (i, j) in support["off"]:
        injected[i, j] = True
    coords = list(zip(*np.where(injected)))
    mats = []
    for _ in range(n_levels):
        delta = np.zeros((p, p))
        for (i, j) in coords:                      # per-level perturbation on S
            delta[i, j] = magnitude * rng.standard_normal()
        # scale the perturbation down if this level would be non-stationary; this
        # preserves the support LOCATION (still nonzero exactly on S), so the
        # instability ground truth is intact.
        scale = 1.0
        while spectral_radius((M_base + scale * delta)[None, :, :]) >= 0.95:
            scale *= 0.7
            if scale < 1e-3:
                break
        mats.append(M_base + scale * delta)
    return mats, injected


# --------------------------------------------------------------------------- #
# Family 2: HRF forward path (reuses the simulator's Glover kernel at TR=2.0)
# --------------------------------------------------------------------------- #
def apply_hrf(X):
    """Convolve each region of each subject with the simulator's Glover HRF
    (TR=2.0), matching simulator/hrf.py's forward model. Input/return (n,T,p).

    The 50-region series is already at the observed level, so (unlike
    hrf.forward_model) we do NOT mix -- mixing would destroy the direct-VAR
    ground truth. This isolates the haemodynamic-filtering confound: does HRF
    low-pass filtering at TR create spurious off-diagonal instability or wash out
    true off-diagonal shifts?
    """
    kernel = glover_kernel(tr=TR)
    out = np.empty_like(X)
    n, T, p = X.shape
    for s in range(n):
        for r in range(p):
            out[s, :, r] = _convolve_causal(X[s, :, r], kernel)
    return out
