"""Test B: VAR(1) mechanism stability across sites.

For the 50 highest-variance cc200 regions of the ComBat-harmonized dataset,
fit a per-site VAR(1) (ridge, alpha=1.0), bootstrap 100 times by resampling
subjects within site, and for each (region, site-pair) compare the row-wise
bootstrap distribution of A[region, :] between sites via a Wald-style
approximate Hotelling T-squared with chi-squared(p) calibration. Apply
Benjamini-Hochberg FDR at q=0.05.

Writes:
  data/processed/test_b_bootstrap.npz     (n_boot, R, R) per site
  results/test_b_results.csv              one row per (region, site-pair)
  figures/test_b_heatmap.png              -log10(p_fdr) heatmap
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

N_TOP_REGIONS = 50
N_BOOT = 100
ALPHA_RIDGE = 1.0
FDR_Q = 0.05
RNG_SEED = 20260628


def select_top_regions(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of top-k regions by pooled across-(subject, time) variance."""
    var_per_region = X.reshape(-1, X.shape[-1]).var(axis=0, ddof=1)
    return np.argsort(var_per_region)[::-1][:k]


def site_pairs_xy(X_site: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_subj, T, R) for one site -> (N_pairs, R) Xlag and (N_pairs, R) Xnext,
    without bridging consecutive subjects' time series."""
    R = X_site.shape[-1]
    return X_site[:, :-1, :].reshape(-1, R), X_site[:, 1:, :].reshape(-1, R)


def fit_var1(X_site: np.ndarray, alpha: float = ALPHA_RIDGE) -> np.ndarray:
    """Ridge VAR(1) fit; returns (R, R) coefficient matrix."""
    Xlag, Xnxt = site_pairs_xy(X_site)
    return Ridge(alpha=alpha, fit_intercept=True).fit(Xlag, Xnxt).coef_


def bootstrap_site(X_site: np.ndarray, n_boot: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Subject-resampling bootstrap. Returns (n_boot, R, R)."""
    n_subj = X_site.shape[0]
    R = X_site.shape[-1]
    boots = np.empty((n_boot, R, R), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n_subj, size=n_subj)
        boots[b] = fit_var1(X_site[idx])
    return boots


def approx_hotelling_t2(a1_boot: np.ndarray, a2_boot: np.ndarray) -> float:
    """Wald-style two-sample T^2 on the difference of two bootstrap distributions.

    a1_boot, a2_boot: (n_boot, p) bootstrap samples of the SAME coefficient
    row from sites 1 and 2. The bootstrap sample covariance directly estimates
    Var(â_site), so

        T^2 = d^T [Var(â_1) + Var(â_2)]^{-1} d,   ~chi^2(p) under H0.

    Tikhonov regularization on the combined covariance keeps the solve stable
    when 100 reps estimate a 50x50 cov.
    """
    d = a1_boot.mean(axis=0) - a2_boot.mean(axis=0)
    S = (np.cov(a1_boot, rowvar=False, ddof=1)
         + np.cov(a2_boot, rowvar=False, ddof=1))
    p = S.shape[0]
    eps = 1e-6 * np.trace(S) / p
    S_reg = S + eps * np.eye(p)
    return float(d @ np.linalg.solve(S_reg, d))


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    f = np.load(DATA / "abide_harmonized.npz")
    X_full = f["X"]
    site_ids = f["site_ids"]
    print(f"Loaded X: shape={X_full.shape}, "
          f"sites={len(np.unique(site_ids))}, subjects={X_full.shape[0]}")

    top_idx = select_top_regions(X_full, N_TOP_REGIONS)
    X = X_full[:, :, top_idx].astype(np.float32, copy=False)
    print(f"Selected top-{N_TOP_REGIONS} regions by pooled variance "
          f"(atlas indices, sorted): {sorted(top_idx.tolist())}")

    sites = sorted(np.unique(site_ids).tolist())
    n_sites = len(sites)
    print(f"Sites ({n_sites}): {sites}")

    # ---- Bootstrap VAR(1) per site ----------------------------------------
    site_boots: dict[str, np.ndarray] = {}
    print(f"\nBootstrap ridge-VAR(1) per site, {N_BOOT} iters each "
          f"(alpha={ALPHA_RIDGE})...")
    for s in tqdm(sites, desc="sites"):
        Xs = X[site_ids == s]
        site_boots[s] = bootstrap_site(Xs, N_BOOT, rng)

    boot_out = DATA / "test_b_bootstrap.npz"
    np.savez_compressed(
        boot_out,
        top_region_idx=top_idx.astype(np.int32),
        sites=np.asarray(sites, dtype="U32"),
        **{f"boot_{s}": site_boots[s] for s in sites},
    )
    print(f"Saved bootstrap coefficients -> {boot_out} "
          f"({boot_out.stat().st_size/1e6:.1f} MB)")

    # ---- Hotelling T^2 per (region, site_pair) ----------------------------
    pairs = list(combinations(range(n_sites), 2))
    n_pairs = len(pairs)
    print(f"\nRunning {N_TOP_REGIONS * n_pairs} Hotelling T^2 tests "
          f"({N_TOP_REGIONS} regions x {n_pairs} site pairs)...")

    rows = []
    for region_i in tqdm(range(N_TOP_REGIONS), desc="regions"):
        for s1, s2 in pairs:
            sn1, sn2 = sites[s1], sites[s2]
            a1 = site_boots[sn1][:, region_i, :].astype(np.float64)
            a2 = site_boots[sn2][:, region_i, :].astype(np.float64)
            T2 = approx_hotelling_t2(a1, a2)
            rows.append({
                "region": region_i,
                "region_orig_atlas_idx": int(top_idx[region_i]),
                "site_1": sn1,
                "site_2": sn2,
                "T2": T2,
                "p_raw": float(chi2.sf(T2, df=N_TOP_REGIONS)),
            })

    df = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(df["p_raw"].values,
                                        alpha=FDR_Q, method="fdr_bh")
    df["p_fdr"] = p_fdr
    df["reject_fdr"] = reject

    out_csv = RESULTS / "test_b_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df)} rows)")

    # ---- Summary ----------------------------------------------------------
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"Total tests:                       {len(df)}")
    print(f"Rejection rate at FDR q={FDR_Q}:        "
          f"{df['reject_fdr'].mean()*100:6.2f}%  "
          f"({df['reject_fdr'].sum()} / {len(df)})")

    rej_per_region = df.groupby("region")["reject_fdr"].mean()
    print()
    print("Per-region rejection fraction (across the 91 site pairs):")
    print(f"  median:                           {rej_per_region.median()*100:6.2f}%")
    print(f"  90th percentile:                  "
          f"{np.percentile(rej_per_region, 90)*100:6.2f}%")
    n_stable = int((rej_per_region < 0.20).sum())
    print(f"  regions with <20% pairs rejecting: "
          f"{n_stable}/{N_TOP_REGIONS}  ({n_stable/N_TOP_REGIONS*100:.1f}%)")

    # ---- Heatmap ----------------------------------------------------------
    df["pair"] = df["site_1"] + "__" + df["site_2"]
    pair_order = sorted(df["pair"].unique())
    region_order = rej_per_region.sort_values(ascending=False).index.tolist()
    p_mat = (df.pivot(index="region", columns="pair", values="p_fdr")
               .loc[region_order, pair_order])
    sig_mat = (df.pivot(index="region", columns="pair", values="reject_fdr")
                 .loc[region_order, pair_order]).astype(bool)

    neglog = -np.log10(np.clip(p_mat.values, 1e-300, 1.0))
    vmax = max(float(np.percentile(neglog, 99)), -np.log10(FDR_Q) + 1)

    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(18, 10))
    im = ax.imshow(neglog, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    sy, sx = np.where(sig_mat.values)
    ax.scatter(sx, sy, marker="x", color="white",
               s=10, lw=0.5, alpha=0.55)
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels(
        [f"r{r} (atlas {int(top_idx[r])})" for r in region_order],
        fontsize=6)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, rotation=90, fontsize=5)
    ax.set_xlabel("Site pair")
    ax.set_ylabel(f"Region (top-{N_TOP_REGIONS}, sorted by # rejections desc.)")
    ax.set_title(f"Test B: VAR(1) row-wise stability  "
                 f"--  -log10(p_FDR);  x = reject at FDR {FDR_Q}")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(r"$-\log_{10}(p_{\mathrm{FDR}})$")
    fig.tight_layout()
    fig_out = FIGURES / "test_b_heatmap.png"
    fig.savefig(fig_out, dpi=140)
    print(f"\nWrote {fig_out}")


if __name__ == "__main__":
    main()
