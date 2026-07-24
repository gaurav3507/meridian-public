"""ABIDE-I within-dataset robustness check for the SMS finding.

Question: is the ABIDE-I result (residual site instability concentrates on
diagonal self-loops; cross-region VAR(1) mechanisms are site-stable) a sampling
fluke or driven by a particular site? Two analyses on the SAME 742-subject
harmonized data and the SAME method that produced the original 227,500-test
result (1.06% reject at FDR 0.05, 49/49 unstable coefficients diagonal).

N NOTE: the analyzed cohort is 742 subjects across 14 sites (91 site pairs,
227,500 tests). The "845" in some handoff docs is a pre-final-QC count. Quote 742.

METHOD (reused exactly from test_b.py / test_b_marginal.py, imported, not
re-implemented): N_TOP_REGIONS=50, ridge VAR(1) alpha=1.0 fit_intercept, N_BOOT=100
subject-resampling bootstrap, per-coefficient two-sample Wald z, BH-FDR q=0.05
applied jointly across all (coefficient x site-pair) tests. Region set is FIXED to
the original full-data top-50 (test_b_bootstrap.npz top_region_idx) for every
subset, so diagonal/off-diagonal identities are comparable across subsets.
"unstable" = rejects in > threshold of that subset's site pairs; diagonal = i==j.

ANALYSES
  1. Leave-one-site-out: drop each of 14 sites, recompute on the remaining 13
     (C(13,2)=78 pairs). Reuses the saved original per-site bootstrap (the per-
     site bootstrap does not depend on which other sites are present), so this is
     just cheap Wald recomputes. A full-14-site REFERENCE ROW runs through the
     same helper first and MUST reproduce ~1.06% / 49-diagonal-0-off as a sanity
     check before any variant is trusted.
  2. Split-half: 50 splits, each partitioning subjects 50/50 STRATIFIED within
     site (both halves keep all 14 sites and the full 91-pair structure). Each
     half re-bootstraps all 14 sites on the fixed 50 regions, then the Wald test.

PRE-REGISTERED QUANTITATIVE BAR (committed before running):
  - ROBUST if the diagonal concentration holds (unstable coefficients
    predominantly diagonal, off-diagonal count ZERO at the strict >50%-of-pairs
    threshold) in ALL 14 leave-one-out subsets AND in at least 90% (45/50) of
    split-half resamples (a split counts as holding only if BOTH halves hold).
  - If any single leave-one-out subset collapses the finding, NAME the site and
    quantify the change (rejection rate and off-diagonal count).
  - Report raw counts alongside percentages throughout.

Reference for comparison (full data): 1.06% reject, median |d|/SE 0.75,
49/50 diagonal self-loops unstable at >20% of pairs, 0/2450 off-diagonal.

Usage:
  python abide_robustness.py loo        # leave-one-out + reference row (fast)
  python abide_robustness.py splithalf  # 50 stratified split-half resamples (slow)
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from test_b import fit_var1, bootstrap_site          # noqa: E402  (exact reuse)

ROOT = HERE.parent
DATA = ROOT / "data/processed"
OUT = ROOT / "results/abide_robustness"
OUT.mkdir(parents=True, exist_ok=True)

N_TOP_REGIONS = 50
N_BOOT = 100
FDR_Q = 0.05
THR_LOOSE = 0.20            # ABIDE's ">20% of site pairs" unstable threshold
THR_STRICT = 0.50          # strict threshold for the pre-registered bar
BASE_SEED = 20260628
N_SPLITS = 50


def marginal_wald(boots: dict, sites: list) -> dict:
    """Reproduces test_b_marginal.py exactly on a given set of sites' bootstraps.
    boots[s]: (n_boot, R, R). Returns headline metrics + diagonal breakdown."""
    R = boots[sites[0]].shape[1]
    mean = {s: boots[s].astype(np.float64).mean(0) for s in sites}
    var = {s: boots[s].astype(np.float64).var(0, ddof=1) for s in sites}
    pairs = list(combinations(sites, 2))
    Z = np.empty((len(pairs), R, R))
    for k, (s1, s2) in enumerate(pairs):
        se = np.sqrt(var[s1] + var[s2])
        Z[k] = np.where(se > 0, (mean[s1] - mean[s2]) / se, 0.0)
    P = 2.0 * norm.sf(np.abs(Z))
    reject = multipletests(P.ravel(), alpha=FDR_Q,
                           method="fdr_bh")[0].reshape(P.shape)
    frac = reject.mean(axis=0)                        # (R,R) reject-fraction/coef
    eye = np.eye(R, dtype=bool)

    def brk(thr):
        u = frac > thr
        return int((u & eye).sum()), int((u & ~eye).sum())
    d20, o20 = brk(THR_LOOSE)
    d50, o50 = brk(THR_STRICT)
    return dict(n_pairs=len(pairs), n_tests=int(P.size),
                n_reject=int(reject.sum()), rej_rate=float(reject.mean()),
                median_z=float(np.median(np.abs(Z))),
                d20=d20, o20=o20, d50=d50, o50=o50,
                n_diag=R, n_off=R * R - R)


def holds(m: dict) -> bool:
    """Pre-registered: diagonal concentration holds iff no off-diagonal coef is
    unstable at the strict >50%-of-pairs threshold, and the diagonal carries it."""
    return m["o50"] == 0 and m["d20"] > 0


# --------------------------------------------------------------------------- #
def run_loo():
    b = np.load(DATA / "test_b_bootstrap.npz", allow_pickle=True)
    sites = b["sites"].tolist()
    boots = {s: b[f"boot_{s}"] for s in sites}
    assert len(sites) == 14 and boots[sites[0]].shape == (N_BOOT, N_TOP_REGIONS,
                                                          N_TOP_REGIONS)

    ref = marginal_wald(boots, sites)
    print("=" * 74)
    print("REFERENCE ROW (full 14 sites) , sanity check vs original")
    print("=" * 74)
    print(f"  N_sites=14  pairs={ref['n_pairs']}  tests={ref['n_tests']:,}")
    print(f"  reject@FDR {FDR_Q} : {ref['rej_rate']*100:.2f}%  "
          f"({ref['n_reject']} / {ref['n_tests']:,})   [original 1.06%]")
    print(f"  median |d|/SE    : {ref['median_z']:.3f}   [original 0.75]")
    print(f"  unstable >20%    : {ref['d20']} diagonal (of 50) + {ref['o20']} "
          f"off-diag (of 2450)   [original 49 + 0]")
    print(f"  unstable >50%    : {ref['d50']} diagonal + {ref['o50']} off-diag")
    ok = abs(ref["rej_rate"] - 0.0106) < 0.004 and ref["o20"] == 0
    print(f"  SANITY {'PASS' if ok else 'CHECK'}: helper reproduces the original\n")

    rows = [{"dropped": "(none) full-14", **ref, "holds": holds(ref)}]
    print("=" * 74)
    print("LEAVE-ONE-SITE-OUT (13 sites each, 78 pairs)")
    print("=" * 74)
    print(f"  {'dropped':12s} {'reject%':>8} {'diag>20%':>9} {'off>20%':>8} "
          f"{'off>50%':>8}  holds")
    for s in sites:
        sub = [x for x in sites if x != s]
        m = marginal_wald(boots, sub)
        rows.append({"dropped": s, **m, "holds": holds(m)})
        print(f"  {s:12s} {m['rej_rate']*100:7.2f}% {m['d20']:6d}/50 "
              f"{m['o20']:6d} {m['o50']:6d}    {'yes' if holds(m) else 'NO'}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "leave_one_out.csv", index=False)
    n_hold = int(df.iloc[1:]["holds"].sum())
    collapsed = df.iloc[1:][~df.iloc[1:]["holds"]]["dropped"].tolist()
    print(f"\n  leave-one-out subsets where finding holds: {n_hold}/14")
    if collapsed:
        print(f"  COLLAPSED when dropping: {collapsed}")
    else:
        print("  finding holds in ALL 14 leave-one-out subsets "
              "(no single site drives it)")
    print(f"\n  wrote {OUT/'leave_one_out.csv'}")
    return ref, df


# --------------------------------------------------------------------------- #
def stratified_half(site_ids: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    a, bb = [], []
    for s in np.unique(site_ids):
        idx = np.where(site_ids == s)[0]
        rng.shuffle(idx)
        cut = len(idx) // 2
        a.append(idx[:cut]); bb.append(idx[cut:])
    return np.concatenate(a), np.concatenate(bb)


def run_splithalf():
    h = np.load(DATA / "abide_harmonized.npz", allow_pickle=True)
    b = np.load(DATA / "test_b_bootstrap.npz", allow_pickle=True)
    top = b["top_region_idx"]
    X = h["X"][:, :, top].astype(np.float32)          # (N, T, 50) fixed regions
    site_ids = h["site_ids"]
    sites = sorted(np.unique(site_ids).tolist())

    def analyze_half(idx, seed):
        rng = np.random.default_rng(seed)
        boots = {}
        for s in sites:
            sidx = idx[site_ids[idx] == s]
            boots[s] = bootstrap_site(X[sidx], N_BOOT, rng)
        return marginal_wald(boots, sites)

    rows = []
    print(f"split-half: {N_SPLITS} stratified splits (background)...", flush=True)
    for sp in range(N_SPLITS):
        ia, ib = stratified_half(site_ids, BASE_SEED + sp)
        for hlabel, idx in (("A", ia), ("B", ib)):
            m = analyze_half(idx, BASE_SEED + 1000 * sp + (0 if hlabel == "A" else 1))
            rows.append({"split": sp, "half": hlabel, "n_subj": len(idx),
                         **m, "holds": holds(m)})
        pd.DataFrame(rows).to_csv(OUT / "split_half.csv", index=False)   # checkpoint
        if (sp + 1) % 10 == 0:
            print(f"  {sp+1}/{N_SPLITS} splits done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "split_half.csv", index=False)
    per_half_hold = df["holds"].mean()
    split_both = df.groupby("split")["holds"].all()
    n_split_both = int(split_both.sum())
    _write_splithalf_md(df, per_half_hold, n_split_both)
    print("\n" + "=" * 74)
    print("SPLIT-HALF RESULTS")
    print("=" * 74)
    print(f"  halves analyzed: {len(df)} (2 x {N_SPLITS})")
    print(f"  rejection rate across halves: mean {df['rej_rate'].mean()*100:.2f}% "
          f"sd {df['rej_rate'].std()*100:.2f}% "
          f"range [{df['rej_rate'].min()*100:.2f}, {df['rej_rate'].max()*100:.2f}]%")
    print(f"  off-diagonal unstable >50% across halves: "
          f"max {int(df['o50'].max())}, mean {df['o50'].mean():.2f}")
    print(f"  halves where finding holds: {int(df['holds'].sum())}/{len(df)} "
          f"({per_half_hold*100:.0f}%)")
    print(f"  splits where BOTH halves hold: {n_split_both}/{N_SPLITS} "
          f"({n_split_both/N_SPLITS*100:.0f}%)   [pre-registered bar: >=45/50]")
    print(f"  VERDICT: {'ROBUST' if n_split_both >= 45 else 'BELOW BAR'} "
          "on the split-half criterion")
    print(f"\n  wrote {OUT/'split_half.csv'} + results.md")


def _write_splithalf_md(df, per_half_hold, n_split_both):
    with open(OUT / "results.md", "a") as f:
        f.write("\n## Split-half resampling (50 stratified splits, 100 halves)\n\n")
        f.write(f"- rejection rate across halves: mean "
                f"{df['rej_rate'].mean()*100:.2f}%, sd {df['rej_rate'].std()*100:.2f}%, "
                f"range [{df['rej_rate'].min()*100:.2f}, "
                f"{df['rej_rate'].max()*100:.2f}]% (reference full-data 1.06%)\n")
        f.write(f"- off-diagonal unstable at >50% of pairs: mean "
                f"{df['o50'].mean():.2f}, max {int(df['o50'].max())} (per half)\n")
        f.write(f"- finding holds in {int(df['holds'].sum())}/{len(df)} halves "
                f"({per_half_hold*100:.0f}%)\n")
        f.write(f"- splits where BOTH halves hold: {n_split_both}/{N_SPLITS} "
                f"({n_split_both/N_SPLITS*100:.0f}%); pre-registered bar >=45/50 "
                f"=> {'ROBUST' if n_split_both>=45 else 'BELOW BAR'}\n")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "loo"
    if arg in ("loo", "all"):
        run_loo()
    if arg in ("splithalf", "all"):
        run_splithalf()
