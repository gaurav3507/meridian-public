"""Per-coefficient marginal version of Test B.

For each VAR(1) coefficient entry (i, j) with i, j in [0..49] (2500 entries)
and each site pair (91 pairs), run a two-sample Wald z-test on the per-site
bootstrap distributions of that single coefficient:

    z = (mean_s1[i,j] - mean_s2[i,j]) / sqrt(var_s1[i,j] + var_s2[i,j])
    p = 2 * Phi(-|z|)                       (two-sided normal)

This is the per-dimension shadow of the Hotelling T^2 in test_b.py. BH-FDR
correct across all 2500 * 91 = 227,500 tests jointly.

Writes:
  results/test_b_marginal.csv
  figures/test_b_marginal_distribution.png
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

N_TOP_REGIONS = 50
N_BOOT = 100
FDR_Q = 0.05


def main() -> None:
    boot = np.load(DATA / "test_b_bootstrap.npz")
    sites = boot["sites"].tolist()
    top_idx = boot["top_region_idx"]
    n_sites = len(sites)
    print(f"Loaded bootstrap: {n_sites} sites, n_boot={N_BOOT}, "
          f"top-{N_TOP_REGIONS} regions")

    # Per-site bootstrap mean and variance per (i, j) entry.
    site_mean: dict[str, np.ndarray] = {}
    site_var: dict[str, np.ndarray] = {}
    for s in sites:
        arr = boot[f"boot_{s}"].astype(np.float64)        # (n_boot, R, R)
        assert arr.shape == (N_BOOT, N_TOP_REGIONS, N_TOP_REGIONS), \
            f"unexpected boot shape for {s}: {arr.shape}"
        site_mean[s] = arr.mean(axis=0)                   # (R, R)
        site_var[s] = arr.var(axis=0, ddof=1)             # (R, R)

    pairs = list(combinations(range(n_sites), 2))
    n_pairs = len(pairs)
    n_tests = N_TOP_REGIONS * N_TOP_REGIONS * n_pairs
    print(f"Running {n_tests:,} marginal Wald z-tests "
          f"({N_TOP_REGIONS**2:,} coefficients x {n_pairs} site pairs)...")

    # Flat index grids used to label each (i, j) in the long DataFrame.
    i_grid, j_grid = np.meshgrid(np.arange(N_TOP_REGIONS),
                                 np.arange(N_TOP_REGIONS), indexing="ij")
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()
    i_atlas_flat = top_idx[i_flat]
    j_atlas_flat = top_idx[j_flat]

    frames: list[pd.DataFrame] = []
    for s1_i, s2_i in pairs:
        s1, s2 = sites[s1_i], sites[s2_i]
        d = site_mean[s1] - site_mean[s2]
        se = np.sqrt(site_var[s1] + site_var[s2])
        z = np.where(se > 0, d / se, 0.0)
        p_raw = 2.0 * norm.sf(np.abs(z))
        frames.append(pd.DataFrame({
            "region_i": i_flat,
            "region_j": j_flat,
            "region_i_atlas": i_atlas_flat,
            "region_j_atlas": j_atlas_flat,
            "site_1": s1,
            "site_2": s2,
            "d": d.ravel(),
            "se": se.ravel(),
            "abs_z": np.abs(z).ravel(),
            "p_raw": p_raw.ravel(),
        }))
    df = pd.concat(frames, ignore_index=True)
    print(f"Assembled long DataFrame: {len(df):,} rows")

    reject, p_fdr, _, _ = multipletests(df["p_raw"].values,
                                        alpha=FDR_Q, method="fdr_bh")
    df["p_fdr"] = p_fdr
    df["reject_fdr"] = reject

    out_csv = RESULTS / "test_b_marginal.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({out_csv.stat().st_size/1e6:.1f} MB)")

    # =================== console reports ===================
    print()
    print("=" * 64)
    print("MARGINAL TEST B SUMMARY")
    print("=" * 64)
    print(f"Total tests:                            {len(df):,}")
    print(f"FDR-reject rate (q={FDR_Q}):                "
          f"{df['reject_fdr'].mean()*100:6.2f}%  "
          f"({df['reject_fdr'].sum():,} / {len(df):,})")
    n_unique_raw = (df['p_raw'] < FDR_Q).sum()
    print(f"Uncorrected reject rate (p_raw<0.05):   "
          f"{n_unique_raw/len(df)*100:6.2f}%  ({n_unique_raw:,} / {len(df):,})")

    z = df["abs_z"].values
    print()
    print("Distribution of |d|/SE:")
    for label, val in [("min",       z.min()),
                       ("10th pct",  np.percentile(z, 10)),
                       ("median",    np.median(z)),
                       ("mean",      z.mean()),
                       ("90th pct",  np.percentile(z, 90)),
                       ("95th pct",  np.percentile(z, 95)),
                       ("99th pct",  np.percentile(z, 99)),
                       ("max",       z.max())]:
        print(f"  {label:10s}  {val:7.3f}")
    pct_above = (z > 1.96).mean() * 100
    print(f"  % above |z|=1.96 (uncorrected two-sided 0.05): {pct_above:.2f}%")

    rej_per_coef = df.groupby(["region_i", "region_j"])["reject_fdr"].mean()
    print()
    print(f"Per-coefficient rejection rate over the {n_pairs} site pairs "
          f"({len(rej_per_coef):,} coefficients):")
    print(f"  median:                           "
          f"{rej_per_coef.median()*100:6.2f}%")
    print(f"  mean:                             "
          f"{rej_per_coef.mean()*100:6.2f}%")
    print(f"  90th percentile:                  "
          f"{np.percentile(rej_per_coef, 90)*100:6.2f}%")
    for thresh in (0.05, 0.20, 0.50, 0.80):
        frac = (rej_per_coef > thresh).mean() * 100
        n_coef = int((rej_per_coef > thresh).sum())
        print(f"  coeffs rejecting in >{int(thresh*100):>2d}% of site pairs: "
              f"{n_coef:5d} / {len(rej_per_coef)}  ({frac:5.2f}%)")

    # =================== figure ===================
    sns.set_theme(style="whitegrid", context="talk")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    xmax_left = max(6.0, float(np.percentile(z, 99.5)))
    bins = np.linspace(0, xmax_left, 200)
    above = z > 1.96
    axL.hist(z[~above], bins=bins, color="steelblue", alpha=0.85,
             edgecolor="none", label="|z| ≤ 1.96")
    axL.hist(z[above], bins=bins, color="crimson", alpha=0.85,
             edgecolor="none", label="|z| > 1.96")
    axL.axvline(1.96, ls="--", color="black", lw=1.2)
    axL.set_xlim(0, xmax_left)
    axL.set_xlabel(r"$|d| / \mathrm{SE}$  (per-coefficient Wald z)")
    axL.set_ylabel("count")
    axL.set_title(f"Marginal effect sizes — {len(df):,} tests\n"
                  f"{pct_above:.1f}% above 1.96  ·  "
                  f"FDR reject {df['reject_fdr'].mean()*100:.1f}%")
    axL.legend(frameon=False, fontsize=11)

    rej_vals = rej_per_coef.values * 100
    axR.hist(rej_vals, bins=np.linspace(0, 100, 26),
             color="slategray", edgecolor="white", alpha=0.9)
    axR.set_xlabel("Per-coefficient rejection rate across 91 site pairs (%)")
    axR.set_ylabel("count of coefficients (of 2,500)")
    axR.set_title("Site-pair rejection rate per coefficient")
    for x in (20, 50, 80):
        axR.axvline(x, ls=":", color="black", lw=0.8, alpha=0.4)

    fig.tight_layout()
    fig_out = FIGURES / "test_b_marginal_distribution.png"
    fig.savefig(fig_out, dpi=150)
    print(f"\nWrote {fig_out}")


if __name__ == "__main__":
    main()
