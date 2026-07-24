"""Schaefer-200 site VAR-stability rerun — IDENTICAL method to test_b/test_b_marginal,
only the parcellation changed (CC200 -> Schaefer-200).

Pipeline:
  1. Load 742 per-subject (116,200) Schaefer-200 timeseries (manifest order).
  2. ComBat-GAM on per-region temporal mean (batch=SITE, protect AGE/SEX/DX/FD,
     smooth AGE), additive correction broadcast across time  -- exactly as
     scripts/build_dataset.py.
  3. Top-50 regions by pooled across-(subject,time) variance (ddof=1).
  4. Per-site ridge VAR(1) (alpha=1.0, intercept), 100 subject-resampling
     bootstraps, seed 20260628, sites in sorted order  -- exactly as test_b.py.
  5. Per-coefficient two-sample Wald z over all 91 site pairs, BH-FDR q=0.05
     jointly across 2500*91 = 227,500 tests  -- exactly as test_b_marginal.py.
  6. Report overall reject %, median |d|/SE, diagonal (of 50) / off-diagonal
     (of 2450) unstable counts (rejection_fraction > 0.20), and Yeo-7 network
     composition of the selected top-50.

Writes results/site_schaefer/{results.md, marginal.csv}.
"""
from __future__ import annotations
import csv, os, warnings
from itertools import combinations
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests
from neuroHarmonize import harmonizationLearn
from nilearn.datasets import fetch_atlas_schaefer_2018

BASE = os.environ.get("MERIDIAN_SCRATCH", os.path.expanduser("~/meridian_schaefer"))
TS   = os.path.join(BASE, "timeseries")
MANIFEST = os.path.join(BASE, "manifest.csv")
ROOT = Path(__file__).resolve().parent.parent.parent
OUT  = ROOT / "results" / "site_schaefer"
OUT.mkdir(parents=True, exist_ok=True)

N_TOP_REGIONS = 50
N_BOOT = 100
ALPHA_RIDGE = 1.0
FDR_Q = 0.05
RNG_SEED = 20260628           # identical to test_b.py
UNSTABLE_THRESH = 0.20        # identical to annotate_unstable.py
T_CROP = 116

# ---- functions copied verbatim from test_b.py -----------------------------
def select_top_regions(X, k):
    var_per_region = X.reshape(-1, X.shape[-1]).var(axis=0, ddof=1)
    return np.argsort(var_per_region)[::-1][:k]

def site_pairs_xy(X_site):
    R = X_site.shape[-1]
    return X_site[:, :-1, :].reshape(-1, R), X_site[:, 1:, :].reshape(-1, R)

def fit_var1(X_site, alpha=ALPHA_RIDGE):
    Xlag, Xnxt = site_pairs_xy(X_site)
    return Ridge(alpha=alpha, fit_intercept=True).fit(Xlag, Xnxt).coef_

def bootstrap_site(X_site, n_boot, rng):
    n_subj = X_site.shape[0]; R = X_site.shape[-1]
    boots = np.empty((n_boot, R, R), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, n_subj, size=n_subj)
        boots[b] = fit_var1(X_site[idx])
    return boots

def yeo7_of(label):
    # Schaefer label e.g. '7Networks_LH_Cont_Par_1' -> 'Cont'
    s = label.decode() if isinstance(label, (bytes, bytearray)) else str(label)
    parts = s.split("_")
    return parts[2] if len(parts) > 2 else "Unknown"

def main():
    # ---- 1. load timeseries in manifest order ----
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    fids  = [r["FILE_ID"] for r in rows]
    sites_col = np.array([r["SITE_ID"] for r in rows])
    missing = [fid for fid in fids if not os.path.exists(os.path.join(TS, fid + ".npy"))]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} timeseries missing, e.g. {missing[:5]}")
    arrays = [np.load(os.path.join(TS, fid + ".npy")) for fid in fids]
    for fid, a in zip(fids, arrays):
        if a.shape != (T_CROP, 200):
            raise SystemExit(f"ABORT: {fid} shape {a.shape} != (116,200)")
    X_raw = np.stack(arrays, axis=0).astype(np.float32)   # (742,116,200)
    print(f"[1] X_raw {X_raw.shape}  sites={len(set(sites_col))}")

    # ---- 2. ComBat-GAM on per-region temporal mean (as build_dataset.py) ----
    mean_signal = X_raw.mean(axis=1).astype(np.float64)   # (N,200)
    covars = pd.DataFrame({
        "SITE":         np.array([r["SITE_ID"] for r in rows]),
        "AGE_AT_SCAN":  np.array([float(r["AGE"]) for r in rows]),
        "SEX":          np.array([int(r["SEX"]) for r in rows]),
        "DX_GROUP":     np.array([int(r["DX_GROUP"]) for r in rows]),
        "func_mean_fd": np.array([float(r["func_mean_fd"]) for r in rows]),
    })
    _m, mean_harm = harmonizationLearn(mean_signal, covars, smooth_terms=["AGE_AT_SCAN"])
    correction = (mean_harm - mean_signal).astype(np.float32)
    X = (X_raw + correction[:, None, :]).astype(np.float32)
    print(f"[2] ComBat done, mean|correction|={np.abs(correction).mean():.4f}")

    # ---- 3. top-50 by pooled variance ----
    top_idx = select_top_regions(X, N_TOP_REGIONS)
    Xtop = X[:, :, top_idx].astype(np.float32, copy=False)
    print(f"[3] top-50 atlas idx (sorted): {sorted(top_idx.tolist())}")

    # ---- 4. per-site bootstrap VAR(1) (identical seed/order) ----
    rng = np.random.default_rng(RNG_SEED)
    sites = sorted(np.unique(sites_col).tolist())
    site_boots = {}
    for s in sites:
        site_boots[s] = bootstrap_site(Xtop[sites_col == s], N_BOOT, rng)
    print(f"[4] bootstrapped {len(sites)} sites x {N_BOOT}")

    # ---- 5. per-coefficient Wald z + BH-FDR (identical to test_b_marginal) ----
    site_mean = {s: site_boots[s].astype(np.float64).mean(axis=0) for s in sites}
    site_var  = {s: site_boots[s].astype(np.float64).var(axis=0, ddof=1) for s in sites}
    pairs = list(combinations(range(len(sites)), 2))
    i_grid, j_grid = np.meshgrid(np.arange(N_TOP_REGIONS), np.arange(N_TOP_REGIONS), indexing="ij")
    i_flat, j_flat = i_grid.ravel(), j_grid.ravel()
    frames = []
    for a, b in pairs:
        s1, s2 = sites[a], sites[b]
        d = site_mean[s1] - site_mean[s2]
        se = np.sqrt(site_var[s1] + site_var[s2])
        z = np.where(se > 0, d / se, 0.0)
        frames.append(pd.DataFrame({
            "region_i": i_flat, "region_j": j_flat,
            "region_i_atlas": top_idx[i_flat], "region_j_atlas": top_idx[j_flat],
            "site_1": s1, "site_2": s2,
            "d": d.ravel(), "se": se.ravel(), "abs_z": np.abs(z).ravel(),
            "p_raw": (2.0 * norm.sf(np.abs(z))).ravel(),
        }))
    df = pd.concat(frames, ignore_index=True)
    reject, p_fdr, _, _ = multipletests(df["p_raw"].values, alpha=FDR_Q, method="fdr_bh")
    df["p_fdr"] = p_fdr; df["reject_fdr"] = reject
    df.to_csv(OUT / "marginal.csv", index=False)
    print(f"[5] {len(df):,} tests -> {OUT/'marginal.csv'}")

    # ---- 6. metrics ----
    overall = df["reject_fdr"].mean() * 100
    median_z = float(df["abs_z"].median())
    agg = df.groupby(["region_i", "region_j"])["reject_fdr"].mean()
    unstable = agg[agg > UNSTABLE_THRESH]
    diag_unstable = sum(1 for (i, j) in unstable.index if i == j)
    off_unstable  = sum(1 for (i, j) in unstable.index if i != j)

    # Yeo-7 composition of the selected top-50
    atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    labels = atlas.labels
    nets = [yeo7_of(labels[k]) for k in top_idx]
    order = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default"]
    comp = {n: nets.count(n) for n in order}
    for n in set(nets):
        if n not in comp: comp[n] = nets.count(n)

    # ---- side-by-side table ----
    lines = []
    lines.append("# ABIDE-I site VAR(1) stability — Schaefer-200 rerun\n")
    lines.append(f"Method identical to the CC200 site test (test_b_marginal.py): top-50 by "
                 f"pooled variance, ridge VAR(1) alpha=1.0 + intercept, {N_BOOT}-bootstrap "
                 f"subject resampling per site (seed {RNG_SEED}), per-coefficient two-sample "
                 f"Wald z over 91 site pairs, BH-FDR q={FDR_Q} across {len(df):,} tests. "
                 f"Only the parcellation changed (CC200 -> Schaefer-200). "
                 f"t_r inert (detrend/low_pass/high_pass off); nominal 2.0 used.\n")
    lines.append("## Result\n")
    lines.append(f"- Subjects: {X.shape[0]}  |  sites: {len(sites)}  |  site pairs: {len(pairs)}  |  T={T_CROP}")
    lines.append(f"- Overall FDR-reject: **{overall:.2f}%**  ({int(df['reject_fdr'].sum()):,} / {len(df):,})")
    lines.append(f"- Median |d|/SE: **{median_z:.2f}**")
    lines.append(f"- Diagonal unstable (>20% of pairs): **{diag_unstable}/50**")
    lines.append(f"- Off-diagonal unstable (>20% of pairs): **{off_unstable}/2450**\n")
    lines.append("## Side by side\n")
    lines.append("| Metric | CC200 site | **Schaefer-200 site** | AOMIC condition (Schaefer-200) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Overall reject % | 1.06% | **{overall:.2f}%** | — |")
    lines.append(f"| Median \\|d\\|/SE | 0.75 | **{median_z:.2f}** | — |")
    lines.append(f"| Diagonal reject (/50) | 49/50 | **{diag_unstable}/50** | — |")
    lines.append(f"| Off-diagonal reject (/2450) | 0/2450 | **{off_unstable}/2450** | — |\n")
    lines.append("## Yeo-7 composition of selected top-50 (Schaefer-200)\n")
    lines.append("| Network | Site top-50 | AOMIC condition top-50 |")
    lines.append("|---|---|---|")
    aomic = {"Default":23,"Cont":8,"SalVentAttn":7,"DorsAttn":6,"Vis":4,"Limbic":2,"SomMot":0}
    for n in order:
        lines.append(f"| {n} | {comp.get(n,0)} | {aomic.get(n,0)} |")
    extra = [n for n in comp if n not in order]
    for n in extra:
        lines.append(f"| {n} | {comp[n]} | 0 |")
    lines.append(f"\nSelected top-50 networks (in selection order): {nets}\n")
    (OUT / "results.md").write_text("\n".join(lines))

    print("\n" + "=" * 60)
    print(f"Overall reject:      {overall:.2f}%")
    print(f"Median |d|/SE:       {median_z:.2f}")
    print(f"Diagonal unstable:   {diag_unstable}/50")
    print(f"Off-diagonal unstable: {off_unstable}/2450   <-- HONESTY: reported as-is")
    print(f"Yeo-7 top-50:        {comp}")
    print("=" * 60)
    print(f"wrote {OUT/'results.md'} and {OUT/'marginal.csv'}")

if __name__ == "__main__":
    main()
