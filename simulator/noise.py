"""Part 5 - Noise model for the MERIDIAN fMRI simulator.

Adds two noise components to clean BOLD (T, N), calibrated to a target SNR:

  * thermal      : additive Gaussian white noise (flat spectrum)
  * physiological: low-frequency drift (sum of a few slow sinusoids), the
                   scanner/physiology confound that dominates low frequencies

SNR is defined per region as  var(signal) / var(noise).  For each region the
total noise variance is set to var(signal) / snr_target, then split between
thermal and drift by ``thermal_frac`` (fraction of noise variance that is
thermal). ``thermal_frac`` also lets Part 7 trade off autocorrelation: drift
is highly autocorrelated, thermal is white.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from generate_dag import generate_dag
from dynamics import simulate_latents
from hrf import make_mixing_matrix, forward_model, TR

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"


def generate_drift(
    T: int, N: int, tr: float = TR,
    n_components: int = 3, f_high: float = 0.04,
    seed: int | None = None,
) -> np.ndarray:
    """Low-frequency drift (T, N): sum of slow sinusoids, unit variance/region.

    Frequencies are drawn in [1/(T*tr), f_high] Hz (a few cycles per scan up to
    ~0.04 Hz), with random per-region amplitude and phase.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) * tr
    f_low = 1.0 / (T * tr)
    drift = np.zeros((T, N))
    for _ in range(n_components):
        f = rng.uniform(f_low, f_high, size=N)
        phase = rng.uniform(0, 2 * np.pi, size=N)
        amp = rng.uniform(0.5, 1.0, size=N)
        drift += amp[None, :] * np.sin(2 * np.pi * f[None, :] * t[:, None]
                                       + phase[None, :])
    drift -= drift.mean(axis=0, keepdims=True)
    std = drift.std(axis=0, keepdims=True)
    return drift / np.where(std > 0, std, 1.0)          # unit variance per region


def add_noise(
    bold: np.ndarray,
    snr_target: float = 2.0,
    thermal_frac: float = 0.5,
    tr: float = TR,
    n_drift_components: int = 3,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Add SNR-calibrated thermal + drift noise to clean BOLD (T, N)."""
    T, N = bold.shape
    # Independent streams for thermal and drift; accepts int or SeedSequence.
    ss = seed if isinstance(seed, np.random.SeedSequence) \
        else np.random.SeedSequence(seed)
    thermal_ss, drift_ss = ss.spawn(2)
    rng = np.random.default_rng(thermal_ss)

    sig_var = bold.var(axis=0)                           # (N,)
    noise_var = sig_var / snr_target                     # target per region
    thermal_var = thermal_frac * noise_var
    drift_var = (1.0 - thermal_frac) * noise_var

    thermal = rng.standard_normal((T, N)) * np.sqrt(thermal_var)[None, :]
    drift_unit = generate_drift(T, N, tr, n_drift_components, seed=drift_ss)
    drift = drift_unit * np.sqrt(drift_var)[None, :]

    noise = thermal + drift
    noisy = bold + noise

    achieved = sig_var / noise.var(axis=0)
    info = {
        "snr_target": snr_target,
        "achieved_snr_mean": float(np.mean(achieved)),
        "achieved_snr_median": float(np.median(achieved)),
        "thermal_frac_target": thermal_frac,
        "thermal_frac_achieved": float(
            np.mean(thermal.var(0) / noise.var(0))),
    }
    return noisy, info


def _lag1(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(0)
    num = (a[:-1] * a[1:]).sum(0)
    den = (a * a).sum(0)
    return np.where(den > 0, num / den, 0.0)


def _selftest() -> None:
    print("=" * 70)
    print("PART 5 TEST  -  thermal + physiological noise")
    print("=" * 70)
    dag = generate_dag(d=10, edge_density=1.5, max_lag=2, seed=0)
    latents = simulate_latents(dag, T=200, noise_scale=1.0, burn_in=50, seed=1)
    W = make_mixing_matrix(10, 50, seed=42)
    clean = forward_model(latents, W)

    noisy, info = add_noise(clean, snr_target=2.0, thermal_frac=0.5, seed=5)
    print(f"clean BOLD shape      = {clean.shape}")
    print(f"noisy BOLD shape      = {noisy.shape}  finite={np.isfinite(noisy).all()}")
    print(f"\nSNR target            = {info['snr_target']}")
    print(f"SNR achieved (mean)   = {info['achieved_snr_mean']:.3f}  "
          f"(median {info['achieved_snr_median']:.3f})")
    print(f"thermal frac target   = {info['thermal_frac_target']}")
    print(f"thermal frac achieved = {info['thermal_frac_achieved']:.3f}")

    corr = np.mean([np.corrcoef(clean[:, n], noisy[:, n])[0, 1]
                    for n in range(clean.shape[1])])
    print(f"\nmean corr(clean, noisy) per region = {corr:.3f}  "
          f"(signal preserved, not destroyed)")
    print(f"lag-1 autocorr  clean = {_lag1(clean).mean():.3f}")
    print(f"lag-1 autocorr  noisy = {_lag1(noisy).mean():.3f}")

    # sweep thermal_frac to show the autocorr lever for Part 7
    print("\nthermal_frac sweep (effect on noisy lag-1 autocorr):")
    for tf in (0.2, 0.5, 0.8):
        ny, _ = add_noise(clean, snr_target=2.0, thermal_frac=tf, seed=5)
        print(f"  thermal_frac={tf:.1f}  ->  noisy lag-1 AC = {_lag1(ny).mean():.3f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(clean.shape[0]) * TR
    n = 0
    thermal_only = add_noise(clean, 2.0, thermal_frac=1.0, seed=5)[0]
    drift_only = add_noise(clean, 2.0, thermal_frac=0.0, seed=5)[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(t, clean[:, n], lw=1.6, color="k", label="clean BOLD")
    ax1.plot(t, noisy[:, n], lw=0.9, color="crimson", alpha=0.8,
             label="noisy (SNR=2, 50/50)")
    ax1.set_title(f"Region {n}: clean vs noisy — signal visible through noise")
    ax1.legend(loc="upper right", fontsize=9); ax1.set_ylabel("BOLD")

    ax2.plot(t, (drift_only - clean)[:, n], lw=1.1, color="teal",
             label="drift component")
    ax2.plot(t, (thermal_only - clean)[:, n], lw=0.7, color="gray", alpha=0.8,
             label="thermal component")
    ax2.set_title("Noise components: low-freq drift vs white thermal")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("noise")
    fig.tight_layout()
    fig_path = FIGURES / "example_noisy_bold.png"
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    print(f"\nnoisy BOLD plot saved -> {fig_path}")

    assert noisy.shape == clean.shape
    assert np.isfinite(noisy).all()
    assert 1.5 < info["achieved_snr_mean"] < 2.5        # calibration works
    assert corr > 0.5                                    # signal not destroyed
    print("\nALL PART 5 CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
