"""Part 7 - Validate simulated BOLD against real ABIDE-I.

Compares three marginal properties of simulated vs real (harmonized) BOLD:
  1. Power spectrum          (both should peak at low frequency)
  2. Lag-1 autocorrelation   (distribution over subject x region)
  3. Value distribution      (after per-series z-scoring)

Real ABIDE is read (never written) from ../data/processed/abide_harmonized.npz.
Because ABIDE TR varies by site and the two datasets have different lengths,
spectra are compared on a NORMALISED frequency axis (cycles per sample, 0-0.5).

If the lag-1 autocorrelation is far from real, a small noise-parameter sweep
(reusing the saved latents + mixing matrix) recommends thermal_frac / SNR
settings that bring the simulator closer, per the Part 5 autocorr lever.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hrf import forward_model, TR
from noise import add_noise

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
REAL_NPZ = ROOT.parent / "data" / "processed" / "abide_harmonized.npz"
SIM_NPZ = ROOT / "data" / "simulated" / "sim_sparse_crosssite_d10_seed0.npz"


def lag1_ac(X: np.ndarray) -> np.ndarray:
    """Per (subject, region) lag-1 autocorrelation for X (S, T, R) -> (S*R,)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    num = (Xc[:, :-1, :] * Xc[:, 1:, :]).sum(axis=1)
    den = (Xc * Xc).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.where(den > 1e-12, num / den, np.nan)
    return ac.ravel()


def avg_periodogram(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average normalised power spectrum over subjects+regions.
    Returns (normalised_freq in [0,0.5], power summing to 1)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    T = Xc.shape[1]
    fft = np.fft.rfft(Xc, axis=1)
    power = (np.abs(fft) ** 2).mean(axis=(0, 2))         # avg over subj, region
    freq = np.fft.rfftfreq(T, d=1.0)                     # cycles / sample
    power = power / power.sum()
    return freq, power


def zscored_values(X: np.ndarray, max_series: int = 40000,
                   seed: int = 0) -> np.ndarray:
    """Pool z-scored values across a subsample of subject-region series."""
    S, T, R = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1)
    flat = Xc.transpose(0, 2, 1).reshape(S * R, T)
    sd = sd.ravel()
    good = np.where(sd > 1e-12)[0]
    rng = np.random.default_rng(seed)
    if len(good) > max_series:
        good = rng.choice(good, size=max_series, replace=False)
    z = flat[good] / sd[good][:, None]
    return z.ravel()


def _summ(name: str, ac: np.ndarray) -> dict:
    ac = ac[np.isfinite(ac)]
    return dict(name=name, median=float(np.median(ac)),
                mean=float(np.mean(ac)),
                p10=float(np.percentile(ac, 10)),
                p90=float(np.percentile(ac, 90)))


def noise_param_sweep(sim_file, real_median_ac, n_subj=120):
    """Recompute clean BOLD from saved latents+W, re-add noise across a grid,
    and report which (thermal_frac, snr) best matches real median lag-1 AC."""
    f = np.load(sim_file)
    Z = f["latents"][:n_subj]           # (n, T, d) ground-truth latents
    W = f["mixing_W"]
    clean = np.stack([forward_model(Z[i], W, tr=TR) for i in range(len(Z))])
    clean_ac = np.nanmedian(lag1_ac(clean))

    rows = []
    for tf in (0.0, 0.2, 0.35, 0.5):
        for snr in (1.0, 2.0, 4.0, 8.0):
            noisy = np.stack([
                add_noise(clean[i], snr_target=snr, thermal_frac=tf,
                          seed=1000 + i)[0]
                for i in range(len(clean))
            ])
            med = float(np.nanmedian(lag1_ac(noisy)))
            rows.append((tf, snr, med, abs(med - real_median_ac)))
    rows.sort(key=lambda r: r[-1])
    return clean_ac, rows


def main() -> None:
    print("=" * 72)
    print("PART 7  -  validation vs real ABIDE-I")
    print("=" * 72)
    if not REAL_NPZ.exists():
        raise FileNotFoundError(f"real ABIDE not found at {REAL_NPZ}")

    real = np.load(REAL_NPZ)["X"]                        # (742, 116, 200)
    sim = np.load(SIM_NPZ)["X"]                          # (1000, 200, 50)
    print(f"real ABIDE X : {real.shape}   (subjects, T, regions)")
    print(f"sim   BOLD X : {sim.shape}   (from {SIM_NPZ.name})")

    # subsample for the pooled distributions
    rng = np.random.default_rng(0)
    real_s = real[rng.choice(real.shape[0], 200, replace=False)]
    sim_s = sim[rng.choice(sim.shape[0], 300, replace=False)]

    # --- metric 1: power spectrum ---
    f_real, p_real = avg_periodogram(real_s)
    f_sim, p_sim = avg_periodogram(sim_s)
    lf = lambda f, p: float(p[f <= 0.1].sum())          # fraction below 0.1 cyc/samp
    print(f"\n[power spectrum]  fraction of power at f<=0.1 cyc/sample:")
    print(f"  real = {lf(f_real, p_real):.3f}   sim = {lf(f_sim, p_sim):.3f}   "
          f"(both high => low-freq dominated)")

    # --- metric 2: lag-1 autocorrelation ---
    ac_real = lag1_ac(real_s)
    ac_sim = lag1_ac(sim_s)
    sr = _summ("real", ac_real); ss = _summ("sim", ac_sim)
    print(f"\n[lag-1 autocorrelation]")
    print(f"  {'':6s} {'median':>8s} {'mean':>8s} {'p10':>8s} {'p90':>8s}")
    for s in (sr, ss):
        print(f"  {s['name']:6s} {s['median']:8.3f} {s['mean']:8.3f} "
              f"{s['p10']:8.3f} {s['p90']:8.3f}")
    ac_gap = ss["median"] - sr["median"]
    print(f"  median gap (sim - real) = {ac_gap:+.3f}")

    # --- metric 3: z-scored value distribution ---
    z_real = zscored_values(real_s); z_sim = zscored_values(sim_s)
    print(f"\n[z-scored values]  (per-series standardised)")
    print(f"  real: kurtosis={_kurt(z_real):.3f}  sim: kurtosis={_kurt(z_sim):.3f}"
          f"   (0 = Gaussian)")

    # --- paper-ready comparison table ---
    lfr, lfs = lf(f_real, p_real), lf(f_sim, p_sim)
    kr, ksi = _kurt(z_real), _kurt(z_sim)
    print("\n" + "-" * 60)
    print("PAPER TABLE  (copy/paste)")
    print("-" * 60)
    print(f"{'Metric':38s}{'Real':>10s}{'Simulated':>12s}")
    print(f"{'Power fraction (f<=0.1 cyc/sample)':38s}"
          f"{lfr:>10.3f}{lfs:>12.3f}")
    print(f"{'Lag-1 autocorr (median)':38s}"
          f"{sr['median']:>10.3f}{ss['median']:>12.3f}")
    print(f"{'Lag-1 autocorr (mean)':38s}"
          f"{sr['mean']:>10.3f}{ss['mean']:>12.3f}")
    print(f"{'Lag-1 autocorr (p10-p90)':38s}"
          f"{sr['p10']:>4.2f}-{sr['p90']:<4.2f} {ss['p10']:>5.2f}-{ss['p90']:<4.2f}")
    print(f"{'Z-scored value kurtosis (0=Gaussian)':38s}"
          f"{kr:>10.3f}{ksi:>12.3f}")
    print("-" * 60)

    # --- figure ---
    _figure(f_real, p_real, f_sim, p_sim, ac_real, ac_sim, z_real, z_sim)

    # --- verdict + suggestions ---
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    spec_ok = abs(lf(f_real, p_real) - lf(f_sim, p_sim)) < 0.25
    ac_ok = abs(ac_gap) < 0.10
    print(f"  power spectrum shape  : {'MATCH' if spec_ok else 'DIFFERENT'} "
          f"(both low-freq dominated: {spec_ok})")
    print(f"  lag-1 autocorrelation : {'MATCH' if ac_ok else 'OFF'} "
          f"(|gap|={abs(ac_gap):.3f}, threshold 0.10)")

    if not ac_ok:
        print(f"\n  Simulated autocorr is "
              f"{'LOWER' if ac_gap < 0 else 'HIGHER'} than real. "
              f"Running noise-parameter sweep to recommend settings...")
        clean_ac, rows = noise_param_sweep(SIM_NPZ, sr["median"])
        print(f"  (clean, noiseless sim median AC = {clean_ac:.3f}; "
              f"real target = {sr['median']:.3f})")
        print(f"  best (thermal_frac, snr) combos by |median AC - real|:")
        print(f"    {'thermal_frac':>12s} {'snr':>6s} {'sim_median_AC':>14s} {'|gap|':>8s}")
        for tf, snr, med, gap in rows[:5]:
            print(f"    {tf:12.2f} {snr:6.1f} {med:14.3f} {gap:8.3f}")
        tf, snr, med, gap = rows[0]
        print(f"\n  RECOMMENDATION: regenerate with "
              f"--thermal_frac {tf} --snr_target {snr}  "
              f"(-> sim median AC {med:.3f} vs real {sr['median']:.3f})")
        if clean_ac < sr["median"]:
            print(f"  NOTE: even noiseless sim AC ({clean_ac:.3f}) < real "
                  f"({sr['median']:.3f}); to close the rest, lengthen/strengthen "
                  f"the HRF (e.g. time_length 40-50s) or add mild AR structure.")
    print("\nPART 7 VALIDATION COMPLETE")


def _kurt(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    m, s = x.mean(), x.std()
    return float(((x - m) ** 4).mean() / s ** 4 - 3.0)


def _figure(f_real, p_real, f_sim, p_sim, ac_real, ac_sim, z_real, z_sim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.plot(f_real, p_real, label="real ABIDE", lw=1.8, color="k")
    ax.plot(f_sim, p_sim, label="simulated", lw=1.8, color="crimson")
    ax.set_xlabel("normalised frequency (cycles/sample)")
    ax.set_ylabel("normalised power")
    ax.set_title("1. Power spectrum")
    ax.legend(fontsize=9)

    ax = axes[1]
    bins = np.linspace(-0.5, 1.0, 60)
    ax.hist(ac_real[np.isfinite(ac_real)], bins=bins, density=True,
            alpha=0.55, color="k", label="real ABIDE")
    ax.hist(ac_sim[np.isfinite(ac_sim)], bins=bins, density=True,
            alpha=0.55, color="crimson", label="simulated")
    ax.axvline(np.nanmedian(ac_real), color="k", ls="--", lw=1)
    ax.axvline(np.nanmedian(ac_sim), color="crimson", ls="--", lw=1)
    ax.set_xlabel("lag-1 autocorrelation")
    ax.set_ylabel("density")
    ax.set_title("2. Lag-1 autocorrelation")
    ax.legend(fontsize=9)

    ax = axes[2]
    bins = np.linspace(-4, 4, 80)
    ax.hist(z_real, bins=bins, density=True, alpha=0.55, color="k",
            label="real ABIDE")
    ax.hist(z_sim, bins=bins, density=True, alpha=0.55, color="crimson",
            label="simulated")
    g = np.exp(-bins ** 2 / 2) / np.sqrt(2 * np.pi)
    ax.plot(bins, g, "b-", lw=1, label="N(0,1)")
    ax.set_xlabel("z-scored BOLD value")
    ax.set_ylabel("density")
    ax.set_title("3. Value distribution (z-scored)")
    ax.legend(fontsize=9)

    fig.suptitle("Part 7: simulated vs real ABIDE-I BOLD", fontsize=13)
    fig.tight_layout()
    path = FIGURES / "validation_vs_abide.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nvalidation figure saved -> {path}")


if __name__ == "__main__":
    main()
