"""Part 4 - HRF forward model + mixing for the MERIDIAN fMRI simulator.

Pipeline: latents (T, d)  --mix-->  region signals (T, N)  --HRF-->  BOLD (T, N)

  * Mixing: a fixed random matrix W (d, N) maps d latents up to N observed
    regions, so each region is a linear combination of latents. This is the
    map an encoder must invert.
  * HRF: each region signal is convolved with a hemodynamic response. The
    canonical shared kernel is nilearn's glover_hrf (TR = 2.0 s). With
    region_specific_hrf=True each region gets a slightly jittered double-gamma
    HRF (latency/dispersion), a known confound (default off).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import gamma as _gamma

from generate_dag import generate_dag
from dynamics import simulate_latents

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"

TR = 2.0


def make_mixing_matrix(d: int, N: int, seed: int | None = None) -> np.ndarray:
    """Fixed random mixing matrix W of shape (d, N), columns ~unit norm."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, N))
    W /= np.linalg.norm(W, axis=0, keepdims=True)   # each region: unit-norm mix
    return W


def glover_kernel(tr: float = TR, time_length: float = 32.0) -> np.ndarray:
    """Canonical Glover HRF from nilearn, sampled at TR, normalised to sum 1."""
    from nilearn.glm.first_level import glover_hrf
    k = glover_hrf(tr, oversampling=1, time_length=time_length)
    return k / k.sum()


def double_gamma_kernel(
    tr: float = TR, time_length: float = 32.0,
    peak_delay: float = 6.0, under_delay: float = 16.0,
    dispersion: float = 1.0, ratio: float = 6.0,
) -> np.ndarray:
    """Parametric SPM-style double-gamma HRF, normalised to sum 1.
    Used only for the region-specific (jittered) HRF path."""
    t = np.arange(0, time_length, tr)
    peak = _gamma.pdf(t, peak_delay / dispersion, scale=dispersion)
    under = _gamma.pdf(t, under_delay / dispersion, scale=dispersion)
    k = peak - under / ratio
    return k / k.sum()


def _convolve_causal(sig: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a (T,) signal with an HRF kernel, truncated to length T."""
    T = sig.shape[0]
    return np.convolve(sig, kernel, mode="full")[:T]


def forward_model(
    latents: np.ndarray,
    W: np.ndarray,
    tr: float = TR,
    region_specific_hrf: bool = False,
    hrf_time_length: float = 32.0,
    hrf_jitter_seed: int | None = None,
) -> np.ndarray:
    """Map latents (T, d) to observed BOLD (T, N) via mixing + HRF convolution."""
    T, d = latents.shape
    assert W.shape[0] == d, f"mixing matrix expects d={d}, got {W.shape}"
    N = W.shape[1]
    mixed = latents @ W                                  # (T, N)

    bold = np.empty_like(mixed)
    if not region_specific_hrf:
        kernel = glover_kernel(tr, time_length=hrf_time_length)
        for n in range(N):
            bold[:, n] = _convolve_causal(mixed[:, n], kernel)
    else:
        rng = np.random.default_rng(hrf_jitter_seed)
        for n in range(N):
            kernel = double_gamma_kernel(
                tr, time_length=hrf_time_length,
                peak_delay=float(rng.normal(6.0, 0.5)),
                dispersion=float(rng.normal(1.0, 0.1)),
            )
            bold[:, n] = _convolve_causal(mixed[:, n], kernel)
    return bold


def _selftest() -> None:
    print("=" * 70)
    print("PART 4 TEST  -  HRF forward model + mixing")
    print("=" * 70)
    dag = generate_dag(d=10, edge_density=1.5, max_lag=2, seed=0)
    latents = simulate_latents(dag, T=200, noise_scale=1.0, burn_in=50, seed=1)

    W = make_mixing_matrix(d=10, N=50, seed=42)
    print(f"mixing matrix W       = {W.shape}  (d x N)")
    print(f"  column norms        = {W[:, :3].__class__.__name__}, "
          f"mean {np.linalg.norm(W, axis=0).mean():.3f} (≈1)")

    kernel = glover_kernel()
    print(f"glover HRF kernel     = {kernel.shape}, sum {kernel.sum():.3f}, "
          f"peak at t={kernel.argmax() * TR:.0f}s")

    bold = forward_model(latents, W, region_specific_hrf=False)
    print(f"\nobserved BOLD shape   = {bold.shape}  (expect (200, 50))")
    print(f"all finite            = {np.isfinite(bold).all()}")
    print(f"per-region std range  = [{bold.std(0).min():.3f}, {bold.std(0).max():.3f}]")

    # BOLD should be smoother (higher lag-1 autocorr) than the mixed signal.
    mixed = latents @ W
    def lag1(a):
        a = a - a.mean(0)
        num = (a[:-1] * a[1:]).sum(0); den = (a * a).sum(0)
        return np.where(den > 0, num / den, 0.0)
    print(f"lag-1 autocorr  mixed  mean = {lag1(mixed).mean():.3f}")
    print(f"lag-1 autocorr  BOLD   mean = {lag1(bold).mean():.3f}   "
          f"(HRF low-pass => higher)")

    # region-specific HRF path
    bold_rs = forward_model(latents, W, region_specific_hrf=True,
                            hrf_jitter_seed=3)
    print(f"\nregion-specific HRF   : shape {bold_rs.shape}, finite "
          f"{np.isfinite(bold_rs).all()}, "
          f"differs from shared-HRF: {not np.allclose(bold_rs, bold)}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(bold.shape[0]) * TR
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

    sp = 3.5 * np.median(bold.std(0))
    for n in range(5):
        ax1.plot(t, bold[:, n] + n * sp, lw=0.9)
        ax1.text(-4, n * sp, f"reg{n}", ha="right", va="center", fontsize=8)
    ax1.set_title("Observed BOLD (5 regions) — smooth, HRF-shaped")
    ax1.set_xlabel("time (s)"); ax1.set_yticks([])

    # latent driver vs a resulting BOLD region, to show delay + smoothing
    reg = int(np.argmax(np.abs(W[0])))          # region most driven by latent 0
    zl = (latents[:, 0] - latents[:, 0].mean()) / latents[:, 0].std()
    zb = (bold[:, reg] - bold[:, reg].mean()) / bold[:, reg].std()
    ax2.plot(t, zl, lw=0.9, alpha=0.7, label="latent 0 (z)")
    ax2.plot(t, zb, lw=1.3, label=f"BOLD region {reg} (z)")
    ax2.set_title("Latent vs its BOLD region — BOLD is delayed & smoothed")
    ax2.set_xlabel("time (s)"); ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig_path = FIGURES / "example_bold.png"
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    print(f"\nBOLD plot saved -> {fig_path}")

    assert bold.shape == (200, 50)
    assert np.isfinite(bold).all()
    assert lag1(bold).mean() > lag1(mixed).mean()      # HRF adds smoothness
    print("\nALL PART 4 CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
