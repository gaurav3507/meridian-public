"""Test A: marginal site effects on observed BOLD summaries.

For each of 200 cc200 regions we compute three per-subject summary statistics
(mean, std, lag-1 autocorrelation), then run a one-way ANOVA with SITE_ID as
the predictor and report eta-squared (SS_between / SS_total) as the site
effect size. We do this for both the raw and ComBat-GAM-harmonized arrays so
the harmonization's residual site signature can be read off the distribution
shift.

Writes: results/test_a_results.csv, figures/test_a_site_effects.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

STATS = ("mean", "std", "lag1_ac")
LARGE_EFFECT = 0.10


def per_subject_summaries(X: np.ndarray) -> dict[str, np.ndarray]:
    """X: (N, T, R) -> dict of (N, R) per-subject per-region summary stats.

    Upcast to float64 before mean subtraction. In float32, summing T copies of
    the same constant accumulates a rounding residual; the demeaned series is
    then [-eps/T,...] instead of exactly zero, and lag-1 AC of any
    masked-out (constant) region collapses to (T-1)/T, polluting the eta^2.
    """
    X = X.astype(np.float64, copy=False)
    mean_ = X.mean(axis=1)
    std_ = X.std(axis=1, ddof=1)
    xc = X - X.mean(axis=1, keepdims=True)
    num = (xc[:, :-1, :] * xc[:, 1:, :]).sum(axis=1)
    den = (xc * xc).sum(axis=1)
    valid = std_ > 1e-10
    lag1 = np.zeros_like(den)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(num, den, out=lag1, where=valid)
    return {"mean": mean_, "std": std_, "lag1_ac": lag1}


def eta_squared_per_region(S: np.ndarray, site: np.ndarray) -> np.ndarray:
    """One-way ANOVA SS_between / SS_total per column of S (vectorized over R).

    S: (N, R) per-subject statistic.  site: (N,) site labels.
    Returns (R,) eta-squared.
    """
    grand = S.mean(axis=0, keepdims=True)             # (1, R)
    ss_total = ((S - grand) ** 2).sum(axis=0)         # (R,)
    ss_between = np.zeros(S.shape[1], dtype=np.float64)
    for s in np.unique(site):
        mask = site == s
        n = int(mask.sum())
        group_mean = S[mask].mean(axis=0)             # (R,)
        ss_between += n * (group_mean - grand[0]) ** 2
    return ss_between / np.where(ss_total > 0, ss_total, np.nan)


def analyse(npz_path: Path) -> np.ndarray:
    """Returns (R, len(STATS)) eta-squared matrix for the given dataset."""
    f = np.load(npz_path)
    X = f["X"]
    site = f["site_ids"]
    summaries = per_subject_summaries(X)
    return np.stack([eta_squared_per_region(summaries[k], site) for k in STATS],
                    axis=1)


def main() -> None:
    raw_eta = analyse(DATA / "abide_raw.npz")
    harm_eta = analyse(DATA / "abide_harmonized.npz")
    n_regions, _ = raw_eta.shape

    rows = []
    for r in range(n_regions):
        for i, stat in enumerate(STATS):
            rows.append({
                "region": r,
                "statistic": stat,
                "eta_squared_raw": float(raw_eta[r, i]),
                "eta_squared_harmonized": float(harm_eta[r, i]),
            })
    df = pd.DataFrame(rows)
    out_csv = RESULTS / "test_a_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df)} rows)")

    # Console summary -- overall
    print()
    print(f"{'metric':38s} {'raw':>10s} {'harmonized':>12s}")
    print("-" * 62)
    print(f"{'median eta^2':38s} "
          f"{df.eta_squared_raw.median():10.4f} "
          f"{df.eta_squared_harmonized.median():12.4f}")
    print(f"{'90th-percentile eta^2':38s} "
          f"{np.percentile(df.eta_squared_raw, 90):10.4f} "
          f"{np.percentile(df.eta_squared_harmonized, 90):12.4f}")
    pct_raw = (df.eta_squared_raw > LARGE_EFFECT).mean() * 100
    pct_harm = (df.eta_squared_harmonized > LARGE_EFFECT).mean() * 100
    print(f"{'% (region,stat) with eta^2 > 0.10':38s} "
          f"{pct_raw:9.2f}% {pct_harm:11.2f}%")

    # Console summary -- per statistic
    print()
    print(f"{'statistic':9s} {'med raw':>10s} {'med harm':>10s} "
          f"{'p90 raw':>9s} {'p90 harm':>10s} {'%>.10 raw':>11s} {'%>.10 harm':>12s}")
    for s in STATS:
        sub = df[df.statistic == s]
        print(f"{s:9s} {sub.eta_squared_raw.median():10.4f} "
              f"{sub.eta_squared_harmonized.median():10.4f} "
              f"{np.percentile(sub.eta_squared_raw, 90):9.4f} "
              f"{np.percentile(sub.eta_squared_harmonized, 90):10.4f} "
              f"{(sub.eta_squared_raw > LARGE_EFFECT).mean()*100:10.2f}% "
              f"{(sub.eta_squared_harmonized > LARGE_EFFECT).mean()*100:11.2f}%")

    # Figure: overlaid KDEs of eta^2 across all (region, statistic) pairs.
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.kdeplot(df.eta_squared_raw, ax=ax, label="Raw",
                fill=True, alpha=0.35, clip=(0, None), bw_adjust=0.8)
    sns.kdeplot(df.eta_squared_harmonized, ax=ax, label="ComBat-GAM harmonized",
                fill=True, alpha=0.35, clip=(0, None), bw_adjust=0.8)
    ax.axvline(LARGE_EFFECT, ls="--", color="k", lw=1.2,
               label=r"$\eta^2 = 0.10$ (large effect)")
    ax.set_xlabel(r"Site effect size  $\eta^{2}$  (one-way ANOVA, SITE_ID)")
    ax.set_ylabel("Density over (region, statistic) pairs")
    ax.set_title("Test A: marginal site effects on per-subject BOLD summaries")
    ax.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig_out = FIGURES / "test_a_site_effects.png"
    fig.savefig(fig_out, dpi=150)
    print(f"\nWrote {fig_out}")


if __name__ == "__main__":
    main()
