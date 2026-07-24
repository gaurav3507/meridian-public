"""ADHD-200 (Athena) SMS replication: VAR(1) per-site bootstrap + per-coefficient
Wald test + diagonal-vs-off-diagonal breakdown. Mirrors scripts/test_b.py and
scripts/test_b_marginal.py exactly (ridge VAR(1) alpha=1, subject-resampling
bootstrap N_BOOT=100, per-coefficient two-sample Wald z, BH-FDR q=0.05), applied
to data/adhd200/processed/adhd200_harmonized.npz.

Two region variants (both reported):
  PRIMARY  : top-50 regions by pooled variance (method-faithful; regions differ
             from ABIDE by construction).
  MATCHED  : ABIDE's exact top-50 CC200 labels intersected with the 190 ADHD-200
             common parcels (controls for "top-50 coincidentally picked similar
             regions").

Pre-registered reading (see scripts/adhd200/README.md): the TARGET is cross-region
(off-diagonal) coefficients stable, instability concentrated on diagonal self-
loops. A higher overall rate than ABIDE's 1.06% that STAYS diagonal-concentrated
CONFIRMS the TR/measurement interpretation. Substantial off-diagonal instability
is a genuine non-replication, read cautiously (disorder+engine+GSR confound).

Writes results/adhd200/{variant}_marginal.csv and results/adhd200/results.md.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent.parent
ADHD = ROOT / "data/adhd200/processed/adhd200_harmonized.npz"
ABIDE = ROOT / "data/processed/abide_harmonized.npz"
OUT = ROOT / "results/adhd200"
OUT.mkdir(parents=True, exist_ok=True)

N_TOP_REGIONS = 50
N_BOOT = 100
ALPHA_RIDGE = 1.0
FDR_Q = 0.05
RNG_SEED = 20260628          # same as ABIDE test_b for comparability
UNSTABLE_PAIR_FRAC = 0.20    # ABIDE's ">20% of site pairs" threshold


def select_top_regions(X, k):
    v = X.reshape(-1, X.shape[-1]).var(axis=0, ddof=1)
    return np.argsort(v)[::-1][:k]


def fit_var1(X_site, alpha=ALPHA_RIDGE):
    R = X_site.shape[-1]
    Xlag = X_site[:, :-1, :].reshape(-1, R)
    Xnxt = X_site[:, 1:, :].reshape(-1, R)
    return Ridge(alpha=alpha, fit_intercept=True).fit(Xlag, Xnxt).coef_


def bootstrap_site(X_site, n_boot, rng):
    n, R = X_site.shape[0], X_site.shape[-1]
    boots = np.empty((n_boot, R, R), dtype=np.float32)
    for b in range(n_boot):
        boots[b] = fit_var1(X_site[rng.integers(0, n, size=n)])
    return boots


def run_variant(name, X_sel, site_ids, sites, atlas_labels):
    """X_sel: (N, T, 50). Returns dict of headline numbers + writes marginal CSV."""
    rng = np.random.default_rng(RNG_SEED)
    R = X_sel.shape[-1]
    boots = {s: bootstrap_site(X_sel[site_ids == s], N_BOOT, rng) for s in sites}
    mean = {s: boots[s].astype(np.float64).mean(0) for s in sites}
    var = {s: boots[s].astype(np.float64).var(0, ddof=1) for s in sites}

    pairs = list(combinations(range(len(sites)), 2))
    i_grid, j_grid = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    frames = []
    for a, b in pairs:
        s1, s2 = sites[a], sites[b]
        d = mean[s1] - mean[s2]
        se = np.sqrt(var[s1] + var[s2])
        z = np.where(se > 0, d / se, 0.0)
        frames.append(pd.DataFrame({
            "region_i": i_grid.ravel(), "region_j": j_grid.ravel(),
            "site_1": s1, "site_2": s2, "abs_z": np.abs(z).ravel(),
            "p_raw": (2.0 * norm.sf(np.abs(z))).ravel(),
            "is_diag": (i_grid == j_grid).ravel()}))
    df = pd.concat(frames, ignore_index=True)
    df["reject_fdr"] = multipletests(df["p_raw"].values, alpha=FDR_Q,
                                     method="fdr_bh")[0]
    df.to_csv(OUT / f"{name}_marginal.csv", index=False)

    n_pairs = len(pairs)
    rej_rate = df["reject_fdr"].mean()
    med_z = float(df["abs_z"].median())
    # per-coefficient rejection fraction across pairs; unstable = > threshold
    per = (df.groupby(["region_i", "region_j"])
             .agg(frac=("reject_fdr", "mean"),
                  is_diag=("is_diag", "first")).reset_index())
    unstable = per[per["frac"] > UNSTABLE_PAIR_FRAC]
    n_unstable = len(unstable)
    n_unstable_diag = int(unstable["is_diag"].sum())
    n_unstable_off = n_unstable - n_unstable_diag
    n_off_total = int((~per["is_diag"]).sum())      # 2450 for R=50
    n_diag_total = int(per["is_diag"].sum())        # 50
    return dict(
        name=name, R=R, n_pairs=n_pairs, n_tests=len(df),
        rej_rate=rej_rate, n_reject=int(df["reject_fdr"].sum()),
        med_z=med_z, pct_z_above196=float((df["abs_z"] > 1.96).mean() * 100),
        n_unstable=n_unstable, n_unstable_diag=n_unstable_diag,
        n_unstable_off=n_unstable_off, n_off_total=n_off_total,
        n_diag_total=n_diag_total,
        atlas_labels=atlas_labels.tolist())


def main():
    z = np.load(ADHD, allow_pickle=True)
    X = z["X"].astype(np.float32)
    site_ids = z["site_ids"]
    labels = z["region_labels"]                     # (190,) CC200 labels present
    sites = sorted(np.unique(site_ids).tolist())
    perN = pd.Series(site_ids).value_counts().to_dict()
    print(f"ADHD-200: X={X.shape}, sites={sites}, N/site={perN}")

    # PRIMARY: top-50 by variance among the 190
    top = select_top_regions(X, N_TOP_REGIONS)
    prim = run_variant("primary_top50", X[:, :, top], site_ids, sites, labels[top])

    # MATCHED: ABIDE top-50 labels (col+1) intersected with ADHD-200 labels
    za = np.load(ABIDE, allow_pickle=True)
    abide_top = select_top_regions(za["X"].astype(np.float32), N_TOP_REGIONS)
    abide_labels = set((abide_top + 1).tolist())    # ABIDE col c -> CC200 label c+1
    lab_to_col = {int(l): k for k, l in enumerate(labels)}
    matched_cols = [lab_to_col[l] for l in sorted(abide_labels) if l in lab_to_col]
    n_matched = len(matched_cols)
    matched = run_variant("matched_region", X[:, :, matched_cols], site_ids, sites,
                          labels[matched_cols])
    print(f"[matched] ABIDE top-50 labels present in ADHD-200: {n_matched}/50")

    _report([prim, matched], sites, perN, n_matched)


def _verdict(r):
    if r["n_unstable"] == 0:
        return "no coefficient unstable in >20% of pairs (very stable overall)"
    diag_frac = r["n_unstable_diag"] / r["n_unstable"]
    if r["n_unstable_off"] == 0:
        return ("REPLICATES: all unstable coefficients are diagonal self-loops "
                f"({r['n_unstable_diag']}/{r['n_unstable']}), 0 off-diagonal")
    if diag_frac >= 0.8:
        return (f"MOSTLY REPLICATES: {r['n_unstable_diag']}/{r['n_unstable']} "
                f"unstable are diagonal; {r['n_unstable_off']} off-diagonal "
                f"of {r['n_off_total']} (read cautiously)")
    return (f"DOES NOT CLEANLY REPLICATE: {r['n_unstable_off']} off-diagonal "
            f"cross-region coefficients unstable (of {r['n_off_total']}); "
            "cross-region mechanisms differ across sites (disorder+engine+GSR)")


def _report(results, sites, perN, n_matched):
    lines = ["# ADHD-200 (Athena) SMS replication results", "",
             f"5 sites, N/site: {perN} (total {sum(perN.values())}), T=120, "
             f"190 common CC200 parcels, {results[0]['n_pairs']} site pairs. "
             "Ridge VAR(1) alpha=1, 100-boot subject resampling, per-coefficient "
             "Wald z, BH-FDR q=0.05. ABIDE reference: 1.06% reject, median "
             "|d|/SE=0.75, 49/49 unstable coeffs diagonal.", "",
             f"Matched-region variant uses {n_matched}/50 ABIDE top-50 CC200 "
             "labels present in ADHD-200.", "",
             "| Variant | tests | reject@FDR | median |d|/SE | %|z|>1.96 | "
             "unstable(>20% pairs) | diagonal | off-diagonal |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['n_tests']:,} | "
            f"{r['rej_rate']*100:.2f}% ({r['n_reject']}) | {r['med_z']:.3f} | "
            f"{r['pct_z_above196']:.2f}% | {r['n_unstable']} | "
            f"{r['n_unstable_diag']} (of {r['n_diag_total']}) | "
            f"{r['n_unstable_off']} (of {r['n_off_total']}) |")
    lines += ["", "## Reading against pre-registration", ""]
    for r in results:
        lines.append(f"- **{r['name']}**: {_verdict(r)}")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")

    print("\n" + "=" * 72)
    print("ADHD-200 SMS REPLICATION RESULTS")
    print("=" * 72)
    print(f"per-site N: {perN}  (total {sum(perN.values())})")
    for r in results:
        print(f"\n[{r['name']}]  ({r['R']} regions, {r['n_pairs']} pairs, "
              f"{r['n_tests']:,} tests)")
        print(f"  rejection @ FDR {FDR_Q}      : {r['rej_rate']*100:.2f}%  "
              f"({r['n_reject']} / {r['n_tests']:,})   [ABIDE 1.06%]")
        print(f"  median |d|/SE            : {r['med_z']:.3f}   [ABIDE 0.75]")
        print(f"  unstable (>20% of pairs) : {r['n_unstable']}  "
              f"= {r['n_unstable_diag']} diagonal (of {r['n_diag_total']}) "
              f"+ {r['n_unstable_off']} off-diagonal (of {r['n_off_total']})")
        print(f"  --> {_verdict(r)}")
    print(f"\nwrote {OUT/'results.md'} + per-variant marginal CSVs")


if __name__ == "__main__":
    main()
