"""Region-selection sensitivity analysis (paper robustness).

Primary analyses pick 50 regions by pooled variance, which after z-scoring is
near-arbitrary and returned zero somatomotor regions. This tests whether the
double dissociation (site = diagonal only; condition = off-diagonal present)
survives a principled, network-balanced region set, and whether it is stable
across random region subsets.

Both datasets are Schaefer-200 (identical parcel indexing), so ONE region set is
applied to both halves.

Pipeline is byte-for-byte the existing one (reused from test_b / condition test):
  ridge VAR(1) alpha=1.0 + intercept; 100-bootstrap subject resampling
  (global RNG seed 20260628, same as primary); per-coefficient two-sample Wald z;
  BH-FDR q=0.05. Site instability = rejects in >20% of 91 pairs. Condition =
  significant at FDR 0.05 on the single rest-vs-WM contrast.

NOTHING is tuned. One run, reported.

Writes results/region_sensitivity/*.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from nilearn.datasets import fetch_atlas_schaefer_2018

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_b import fit_var1, bootstrap_site, select_top_regions  # exact reuse

ABIDE_SCHAEFER = ROOT / "data/processed/abide_schaefer_harmonized.npz"
COND_TS = ROOT / "data/condition_sms/ts"
OUT = ROOT / "results/region_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 100
FDR_Q = 0.05
RNG_SEED = 20260628          # SAME base seed as the primary analyses
BAL_SEED = 20260713          # network-balanced selection seed (fixed, reported)
RANDOM_SEEDS = [101, 202, 303, 404, 505]   # RUN B subset seeds (reported)
NET_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


# --------------------------------------------------------------------------- #
# data + atlas
# --------------------------------------------------------------------------- #
def load_abide():
    assert ABIDE_SCHAEFER.exists(), f"missing {ABIDE_SCHAEFER}"
    z = np.load(ABIDE_SCHAEFER, allow_pickle=True)
    print(f"  LOADED SITE FILE: {ABIDE_SCHAEFER}")
    print(f"    X shape {z['X'].shape}, sites {len(np.unique(z['site_ids']))}, "
          f"subjects {z['X'].shape[0]}")
    return z["X"].astype(np.float32), z["site_ids"]


def load_condition():
    subs = sorted(p.name[:-9] for p in COND_TS.glob("*_rest.npy")
                  if (COND_TS / f"{p.name[:-9]}_wm.npy").exists())
    rest = np.stack([np.load(COND_TS / f"{s}_rest.npy") for s in subs]).astype(np.float32)
    wm = np.stack([np.load(COND_TS / f"{s}_wm.npy") for s in subs]).astype(np.float32)
    print(f"  LOADED CONDITION: {len(subs)} subjects, rest{rest.shape} wm{wm.shape}")
    return rest, wm


def schaefer_networks():
    a = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    labs = [x.decode() if isinstance(x, bytes) else str(x) for x in a.labels]
    if labs[0].lower() == "background":
        labs = labs[1:]
    return np.array([l.split("_")[2] for l in labs])      # (200,) short net names


# --------------------------------------------------------------------------- #
# region selection
# --------------------------------------------------------------------------- #
def balanced_regions(net200, k, seed):
    """k regions, allocated proportional to Yeo-7 network size (Hamilton
    largest-remainder), every network represented, random within-network pick at
    fixed seed. Deterministic tie-break: remainder desc, then network size desc,
    then NET_ORDER index."""
    sizes = np.array([(net200 == n).sum() for n in NET_ORDER], float)
    raw = k * sizes / sizes.sum()
    alloc = np.maximum(np.floor(raw).astype(int), 1)   # every network >= 1
    remaining = k - int(alloc.sum())
    rem = raw - np.floor(raw)
    if remaining > 0:                                   # hand out leftover seats once
        order = sorted(range(7), key=lambda i: (-rem[i], -sizes[i], i))
        for i in order[:remaining]:
            alloc[i] += 1
    elif remaining < 0:                                # min-1 overshot: trim largest
        order = sorted(range(7), key=lambda i: (-alloc[i], -sizes[i], i))
        for i in order[:-remaining]:
            alloc[i] -= 1
    assert alloc.sum() == k and (alloc >= 1).all() and (alloc <= sizes).all()
    rng = np.random.default_rng(seed)
    sel = []
    for n, a in zip(NET_ORDER, alloc):
        idx = np.where(net200 == n)[0]
        sel.extend(rng.choice(idx, size=int(a), replace=False).tolist())
    sel = np.array(sorted(sel))
    assert len(sel) == k and len(set(sel.tolist())) == k
    return sel, dict(zip(NET_ORDER, alloc.tolist()))


def random_regions(k, seed, n_total=200):
    rng = np.random.default_rng(seed)
    return np.array(sorted(rng.choice(n_total, size=k, replace=False).tolist()))


# --------------------------------------------------------------------------- #
# analyses (identical machinery, region set swapped)
# --------------------------------------------------------------------------- #
def run_site(X, site_ids, sel):
    """Returns dict with overall reject %, median |d|/SE, diag/off unstable, and
    the 50x50 rejection-fraction matrix."""
    R = len(sel)
    Xs = X[:, :, sel]
    sites = sorted(np.unique(site_ids).tolist())
    rng = np.random.default_rng(RNG_SEED)
    mean_s, var_s = {}, {}
    for s in sites:                                  # sorted order == primary
        b = bootstrap_site(Xs[site_ids == s], N_BOOT, rng).astype(np.float64)
        mean_s[s] = b.mean(0)
        var_s[s] = b.var(0, ddof=1)
    pairs = list(combinations(range(len(sites)), 2))
    P = np.empty((len(pairs), R, R))
    Zabs = np.empty((len(pairs), R, R))
    for k, (a, b) in enumerate(pairs):
        d = mean_s[sites[a]] - mean_s[sites[b]]
        se = np.sqrt(var_s[sites[a]] + var_s[sites[b]])
        z = np.where(se > 0, d / se, 0.0)
        Zabs[k] = np.abs(z)
        P[k] = 2.0 * norm.sf(np.abs(z))
    reject = multipletests(P.ravel(), alpha=FDR_Q, method="fdr_bh")[0].reshape(P.shape)
    rej_frac = reject.mean(0)                         # (R,R) over 91 pairs
    unstable = rej_frac > 0.20
    eye = np.eye(R, dtype=bool)
    return dict(
        overall_reject_pct=float(reject.mean() * 100),
        median_absz=float(np.median(Zabs)),
        diag_unstable=int(unstable[eye].sum()),
        off_unstable=int(unstable[~eye].sum()),
        n_pairs=len(pairs), rej_frac=rej_frac)


def run_condition(rest, wm, sel):
    R = len(sel)
    rest_s, wm_s = rest[:, :, sel], wm[:, :, sel]
    rng = np.random.default_rng(RNG_SEED)
    b_rest = bootstrap_site(rest_s, N_BOOT, rng).astype(np.float64)
    b_wm = bootstrap_site(wm_s, N_BOOT, rng).astype(np.float64)
    d = b_rest.mean(0) - b_wm.mean(0)
    se = np.sqrt(b_rest.var(0, ddof=1) + b_wm.var(0, ddof=1))
    z = np.where(se > 0, d / se, 0.0)
    p = 2.0 * norm.sf(np.abs(z))
    reject = multipletests(p.ravel(), alpha=FDR_Q, method="fdr_bh")[0].reshape(R, R)
    eye = np.eye(R, dtype=bool)
    return dict(
        overall_reject_pct=float(reject.mean() * 100),
        median_absz=float(np.median(np.abs(z))),
        diag_sig=int(reject[eye].sum()),
        off_sig=int(reject[~eye].sum()),
        reject=reject, absz=np.abs(z))


def enrichment(sel, cond, net200):
    """Yeo-7 obs/exp enrichment on the significant OFF-diagonal condition shifts."""
    R = len(sel)
    net50 = net200[sel]
    eye = np.eye(R, dtype=bool)
    sig_off = cond["reject"] & ~eye
    ii, jj = np.where(sig_off)                        # i = target, j = source
    idx = {n: k for k, n in enumerate(NET_ORDER)}
    obs = np.zeros((7, 7))
    for i, j in zip(ii, jj):
        obs[idx[net50[j]], idx[net50[i]]] += 1        # source -> target
    n = np.array([(net50 == x).sum() for x in NET_ORDER], float)
    avail = np.outer(n, n).copy()
    for k in range(7):
        avail[k, k] = n[k] * (n[k] - 1)
    n_sig = int(sig_off.sum())
    exp = n_sig * avail / avail.sum() if avail.sum() > 0 else np.zeros((7, 7))
    with np.errstate(divide="ignore", invalid="ignore"):
        enr = np.where(exp > 0, obs / exp, np.nan)
    return net50, n, obs, exp, enr, n_sig


# --------------------------------------------------------------------------- #
def main():
    print("=" * 72)
    print("REGION-SELECTION SENSITIVITY")
    print("=" * 72)
    X, site_ids = load_abide()
    rest, wm = load_condition()
    net200 = schaefer_networks()
    print(f"  Schaefer-200 network sizes: "
          + ", ".join(f"{n}:{int((net200==n).sum())}" for n in NET_ORDER))

    # ---- validation: reproduce the variance-based Schaefer baselines --------
    print("\n[VALIDATION] variance-based top-50 (should match stated baselines)")
    var_site_sel = np.sort(select_top_regions(X, 50))
    var_cond_sel = np.sort(select_top_regions(np.concatenate([rest, wm], 0), 50))
    v_site = run_site(X, site_ids, var_site_sel)
    v_cond = run_condition(rest, wm, var_cond_sel)
    print(f"  site  var-based : {v_site['overall_reject_pct']:.2f}%  "
          f"med {v_site['median_absz']:.3f}  diag {v_site['diag_unstable']}/50  "
          f"off {v_site['off_unstable']}/2450   [stated 1.02%, 0.727, 49, 0]")
    print(f"  cond  var-based : {v_cond['overall_reject_pct']:.2f}%  "
          f"med {v_cond['median_absz']:.3f}  diag {v_cond['diag_sig']}/50  "
          f"off {v_cond['off_sig']}/2450   [stated 7.16%, 1.02, 50, 129]")

    # ---- RUN A: network-balanced ------------------------------------------
    print("\n" + "=" * 72)
    print(f"RUN A: network-balanced 50 (seed {BAL_SEED})")
    print("=" * 72)
    sel_bal, alloc = balanced_regions(net200, 50, BAL_SEED)
    print(f"  per-network allocation: {alloc}")
    print(f"  region IDs (0-based Schaefer): {sel_bal.tolist()}")
    a_site = run_site(X, site_ids, sel_bal)
    a_cond = run_condition(rest, wm, sel_bal)
    net50, ncnt, obs, exp, enr, n_sig = enrichment(sel_bal, a_cond, net200)
    print(f"  SITE      : {a_site['overall_reject_pct']:.2f}%  "
          f"med {a_site['median_absz']:.3f}  diag {a_site['diag_unstable']}/50  "
          f"off {a_site['off_unstable']}/2450")
    print(f"  CONDITION : {a_cond['overall_reject_pct']:.2f}%  "
          f"med {a_cond['median_absz']:.3f}  diag {a_cond['diag_sig']}/50  "
          f"off {a_cond['off_sig']}/2450")

    # save balanced region table
    pd.DataFrame({"region_id": sel_bal, "network": net50}).to_csv(
        OUT / "balanced_regions.csv", index=False)

    # save enrichment long table
    rows = []
    for si, s in enumerate(NET_ORDER):
        for ti, t in enumerate(NET_ORDER):
            rows.append(dict(source=s, target=t, observed=int(obs[si, ti]),
                             expected=round(float(exp[si, ti]), 3),
                             obs_over_exp=(round(float(enr[si, ti]), 3)
                                           if not np.isnan(enr[si, ti]) else None)))
    pd.DataFrame(rows).to_csv(OUT / "runA_condition_enrichment.csv", index=False)

    # ---- RUN B: random subsets --------------------------------------------
    print("\n" + "=" * 72)
    print(f"RUN B: 5 random subsets (seeds {RANDOM_SEEDS})")
    print("=" * 72)
    brows = []
    for seed in RANDOM_SEEDS:
        sel = random_regions(50, seed)
        rs = run_site(X, site_ids, sel)
        rc = run_condition(rest, wm, sel)
        brows.append(dict(seed=seed,
                          site_off_unstable=rs["off_unstable"],
                          site_diag_unstable=rs["diag_unstable"],
                          site_overall_pct=round(rs["overall_reject_pct"], 3),
                          cond_off_sig=rc["off_sig"],
                          cond_diag_sig=rc["diag_sig"],
                          cond_overall_pct=round(rc["overall_reject_pct"], 3)))
        print(f"  seed {seed}: site off {rs['off_unstable']}/2450 "
              f"(diag {rs['diag_unstable']}) | cond off {rc['off_sig']}/2450 "
              f"(diag {rc['diag_sig']})")
    bdf = pd.DataFrame(brows)
    bdf.to_csv(OUT / "runB_random_subsets.csv", index=False)
    so = bdf["site_off_unstable"]; co = bdf["cond_off_sig"]
    print(f"\n  SITE off-diagonal unstable  min/median/max: "
          f"{so.min()}/{int(so.median())}/{so.max()}")
    print(f"  COND off-diagonal sig       min/median/max: "
          f"{co.min()}/{int(co.median())}/{co.max()}")

    # ---- summary json -----------------------------------------------------
    summary = dict(
        site_file=str(ABIDE_SCHAEFER),
        seeds=dict(bootstrap=RNG_SEED, balanced=BAL_SEED, random=RANDOM_SEEDS),
        schaefer_network_sizes={n: int((net200 == n).sum()) for n in NET_ORDER},
        validation=dict(
            site=dict(overall_pct=round(v_site["overall_reject_pct"], 3),
                      median_absz=round(v_site["median_absz"], 3),
                      diag=v_site["diag_unstable"], off=v_site["off_unstable"]),
            condition=dict(overall_pct=round(v_cond["overall_reject_pct"], 3),
                           median_absz=round(v_cond["median_absz"], 3),
                           diag=v_cond["diag_sig"], off=v_cond["off_sig"])),
        runA=dict(
            allocation=alloc, region_ids=sel_bal.tolist(),
            site=dict(overall_pct=round(a_site["overall_reject_pct"], 3),
                      median_absz=round(a_site["median_absz"], 3),
                      diag=a_site["diag_unstable"], off=a_site["off_unstable"]),
            condition=dict(overall_pct=round(a_cond["overall_reject_pct"], 3),
                           median_absz=round(a_cond["median_absz"], 3),
                           diag=a_cond["diag_sig"], off=a_cond["off_sig"]),
            condition_n_sig_off=n_sig),
        runB=dict(
            per_draw=brows,
            site_off_min_med_max=[int(so.min()), int(so.median()), int(so.max())],
            cond_off_min_med_max=[int(co.min()), int(co.median()), int(co.max())]),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/  (balanced_regions.csv, runA_condition_enrichment.csv, "
          "runB_random_subsets.csv, summary.json)")


if __name__ == "__main__":
    main()
