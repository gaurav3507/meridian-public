"""M2 STEP 2 -- re-run the ABIDE-I SITE diagnostic on CCS derivatives.

Cross-pipeline robustness: identical machinery, identical subjects, identical
covariates; the ONLY change is the input time series (CCS instead of C-PAC).
Kills the "the site->diagonal dissociation is a C-PAC artifact" objection.

REUSED VERBATIM (not reimplemented):
  scripts/test_b.py        select_top_regions, fit_var1 (ridge alpha=1.0,
                           fit_intercept), bootstrap_site (100 subject resamples)
  scripts/test_b_marginal  per-pair two-sample Wald z (d / sqrt(var1+var2)),
                           p = 2*Phi(-|z|), BH-FDR q=0.05 JOINTLY over all
                           pairs x coefficients, >20%-of-pairs unstable rule
  scripts/build_dataset.py the build: retained-subject set, 116-timepoint crop,
                           ComBat-GAM (neuroHarmonize harmonizationLearn on the
                           per-region time mean; batch=SITE, smooth AGE_AT_SCAN,
                           covars SEX/DX_GROUP/func_mean_fd; additive correction
                           broadcast to every timepoint)

MATCHING NOTES (verified against the real code, not the prose spec):
  * The C-PAC pipeline applies NO per-region standardization -- the PCP .1D files
    are raw ROI means and test_b fits them directly. CCS is treated identically.
  * The retained subject set and every ComBat covariate are read from
    data/processed/abide_harmonized.npz, so QC is NOT redone and site membership,
    N and TR-per-site are identical by construction.
  * Crop follows build_dataset.py step 4 VERBATIM: every subject is cropped to
    the shortest T across the cohort. MEASURED FACT: CCS ships exactly one fewer
    volume than C-PAC for every subject (UCLA_1/MAX_MUN 115 vs 116, NYU 175 vs
    176, PITT 195 vs 196), so the CCS cohort minimum is T=115 vs C-PAC's 116.
    This 1-volume difference is an unavoidable pipeline difference and is
    reported. (Hardcoding 116 would silently drop every UCLA_1 and MAX_MUN
    subject -- i.e. 2 of 14 sites -- and break the subject matching.)
  * select_top_regions is re-run on the CCS data (the same pipeline applied to
    the new input); the overlap with the C-PAC top-50 is reported.

PRE-REGISTERED EXPECTATION (report honestly even if it fails):
  CCS should reproduce near-0 off-diagonal site instability and a
  diagonal-dominant pattern. Small numeric differences from C-PAC are fine and
  expected; a qualitative flip (off-diagonal lighting up) is a real finding to
  report, not hide.

Atomic writes + skip-existing resume. New folders only.
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_b import select_top_regions, bootstrap_site        # reused verbatim

NPZ = ROOT / "data/processed/abide_harmonized.npz"
CCS_DIR = ROOT / "data/raw/abide_ccs_cc200"
OUT = ROOT / "results/cross_pipeline"
CACHE = ROOT / "data/processed/abide_ccs_harmonized.npz"

T_CPAC = 116                 # C-PAC cohort minimum, for reference/comparison only

# MEASURED input-convention mismatch between the two PCP derivative products:
#   C-PAC filt_global .1D : demeaned  (grand mean 0.001; per-region |mean| 0.20;
#                           pooled variance is 99.9% within-subject temporal)
#   CCS   filt_global .1D : NOT demeaned (grand mean ~8896; pooled variance is
#                           ~100% between-subject DC offset, temporal signal is
#                           0.05% of it)
# Left uncorrected this is a SECOND difference on top of the pipeline: it makes
# select_top_regions rank regions by DC offset rather than temporal signal (only
# 17/50 regions overlapped C-PAC) and swamps the pooled VAR fit with
# subject-specific offsets. Demeaning each subject x region over the full series
# reproduces C-PAC's own convention so that the ONLY remaining difference is the
# preprocessing pipeline. This is a matching correction, not a tuned knob; the
# uncorrected ("as-shipped") result is retained below and reported alongside.
DEMEAN_TO_MATCH_CPAC = True
# as-shipped (uncorrected) CCS result, measured on this same data, kept for the
# transparency column in the report:
CCS_AS_SHIPPED = dict(overall_pct=0.23, median_absz=0.521, diag_unstable=9,
                      off_unstable=0, top50_overlap=17)
N_TOP = 50
N_BOOT = 100
FDR_Q = 0.05
UNSTABLE_THRESH = 0.20
RNG_SEED = 20260628          # same bootstrap seed as the primary run
MIN_SITE_N = 30

# primary C-PAC numbers for the side-by-side table (from the paper / results)
CPAC = dict(overall_pct=1.06, median_absz=0.750, diag_unstable=49, diag_total=50,
            off_unstable=0, off_total=2450, n_subjects=742, n_sites=14, n_pairs=91)


def _atomic_write(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    print(f"  wrote {path}")


def build_ccs(smoke=False):
    """Load CCS .1D for the retained subjects, crop to T=116, ComBat-GAM with the
    identical covariates. Returns (X_harm, site_ids, info)."""
    if CACHE.exists() and not smoke:
        z = np.load(CACHE, allow_pickle=True)
        print(f"  [cache] {CACHE.name}: X={z['X'].shape}")
        return z["X"], np.asarray(z["site_ids"], dtype=str), json.loads(str(z["info"]))

    z = np.load(NPZ, allow_pickle=True)
    fids = np.asarray(z["subject_ids"], dtype=str)
    meta = pd.DataFrame({
        "FILE_ID": fids, "SITE_ID": np.asarray(z["site_ids"], dtype=str),
        "AGE_AT_SCAN": z["age"].astype(float), "SEX": z["sex"].astype(int),
        "DX_GROUP": z["dx_group"].astype(int),
        "func_mean_fd": z["func_mean_fd"].astype(float)})
    meta["_path"] = meta.FILE_ID.apply(lambda f: CCS_DIR / f"{f}_rois_cc200.1D")
    present = meta[meta._path.apply(lambda p: p.is_file())].reset_index(drop=True)
    n_missing = len(meta) - len(present)
    print(f"  CCS files present: {len(present)}/{len(meta)} (missing {n_missing})")

    arrays, lengths = [], []
    for _, r in present.iterrows():
        a = np.loadtxt(r._path, dtype=np.float32)
        if DEMEAN_TO_MATCH_CPAC:
            # match C-PAC's convention: per-region demean over the full series
            a = a - a.mean(axis=0, keepdims=True)
        lengths.append(a.shape[0])
        arrays.append(a)
    # build_dataset.py step 4 verbatim: crop to the shortest T across the cohort.
    # No subject is dropped -- CCS simply ships 1 fewer volume than C-PAC, so the
    # CCS cohort minimum lands at 115 vs C-PAC's 116 (reported, not hidden).
    T_min = int(min(lengths))
    X_raw = np.stack([a[:T_min] for a in arrays], 0).astype(np.float32)
    print(f"  CCS T: min={T_min} median={int(np.median(lengths))} "
          f"max={max(lengths)}; cropped to cohort min T={T_min} "
          f"(C-PAC was {T_CPAC}) -> X{X_raw.shape}")
    if T_min != T_CPAC:
        print(f"  NOTE: CCS cohort min T={T_min} differs from C-PAC T={T_CPAC} "
              "(CCS discards one extra volume). Unavoidable pipeline difference; "
              "all sites/subjects retained.")

    # site N check against the C-PAC threshold
    counts = present.SITE_ID.value_counts()
    below = counts[counts < MIN_SITE_N]
    if len(below) and not smoke:
        print(f"  *** FLAG: site(s) below N>={MIN_SITE_N} under CCS: "
              f"{below.to_dict()} -- comparison no longer subject-matched.")

    info = dict(n_subjects=int(len(present)), n_sites=int(present.SITE_ID.nunique()),
                n_missing=int(n_missing), T=int(T_min), T_cpac=int(T_CPAC),
                ccs_T_median=int(np.median(lengths)))

    if smoke or present.SITE_ID.nunique() < 3:
        print("  [smoke] skipping ComBat-GAM (needs the full cohort/GAM df); "
              "proving download->load->crop->VAR->bootstrap->Wald->FDR path only")
        return X_raw, present.SITE_ID.to_numpy(), info

    # ComBat-GAM -- identical call to scripts/build_dataset.py
    from neuroHarmonize import harmonizationLearn
    mean_signal = X_raw.mean(axis=1).astype(np.float64)
    covars = pd.DataFrame({
        "SITE": present.SITE_ID.astype(str).values,
        "AGE_AT_SCAN": present.AGE_AT_SCAN.astype(float).values,
        "SEX": present.SEX.astype(int).values,
        "DX_GROUP": present.DX_GROUP.astype(int).values,
        "func_mean_fd": present.func_mean_fd.astype(float).values})
    _m, mean_harm = harmonizationLearn(mean_signal, covars,
                                       smooth_terms=["AGE_AT_SCAN"])
    correction = (mean_harm - mean_signal).astype(np.float32)
    X_harm = (X_raw + correction[:, None, :]).astype(np.float32)
    print(f"  ComBat-GAM done; mean |correction| = {np.abs(correction).mean():.4f}")

    np.savez_compressed(str(CACHE) + ".tmp.npz", X=X_harm,
                        site_ids=present.SITE_ID.to_numpy().astype("U32"),
                        subject_ids=present.FILE_ID.to_numpy().astype("U64"),
                        info=json.dumps(info))
    Path(str(CACHE) + ".tmp.npz").replace(CACHE)
    print(f"  cached -> {CACHE}")
    return X_harm, present.SITE_ID.to_numpy(), info


def site_diagnostic(X, site_ids, n_boot=N_BOOT):
    """Verbatim primary machinery: bootstrap_site per site, per-pair Wald z,
    BH-FDR jointly over all pairs, >20%-of-pairs unstable rule."""
    top = select_top_regions(X, N_TOP)
    Xs = X[:, :, top]
    sites = sorted(np.unique(site_ids).tolist())
    rng = np.random.default_rng(RNG_SEED)
    mean_s, var_s = {}, {}
    for s in sites:
        b = bootstrap_site(Xs[site_ids == s], n_boot, rng).astype(np.float64)
        mean_s[s], var_s[s] = b.mean(0), b.var(0, ddof=1)
    pairs = list(combinations(range(len(sites)), 2))
    P = np.empty((len(pairs), N_TOP, N_TOP))
    Z = np.empty((len(pairs), N_TOP, N_TOP))
    for k, (a, b) in enumerate(pairs):
        d = mean_s[sites[a]] - mean_s[sites[b]]
        se = np.sqrt(var_s[sites[a]] + var_s[sites[b]])
        z = np.where(se > 0, d / se, 0.0)
        Z[k], P[k] = np.abs(z), 2.0 * norm.sf(np.abs(z))
    reject = multipletests(P.ravel(), alpha=FDR_Q,
                           method="fdr_bh")[0].reshape(P.shape)
    unstable = reject.mean(0) > UNSTABLE_THRESH
    eye = np.eye(N_TOP, dtype=bool)
    return dict(overall_pct=float(reject.mean() * 100),
                median_absz=float(np.median(Z)),
                diag_unstable=int(unstable[eye].sum()), diag_total=N_TOP,
                off_unstable=int(unstable[~eye].sum()),
                off_total=N_TOP * N_TOP - N_TOP,
                n_pairs=len(pairs), n_sites=len(sites),
                top50=top.tolist())


def report(res, info):
    z = np.load(NPZ, allow_pickle=True)
    cpac_top = set(select_top_regions(z["X"], N_TOP).tolist())
    overlap = len(cpac_top & set(res["top50"]))
    off_ok = res["off_unstable"] <= 5
    diag_ok = res["diag_unstable"] >= 40
    verdict = ("REPRODUCES: CCS is diagonal-dominant with near-zero off-diagonal "
               "site instability, matching C-PAC. The dissociation is not a "
               "C-PAC artifact." if (off_ok and diag_ok) else
               "DOES NOT reproduce the C-PAC pattern -- see the table; this is a "
               "real finding and must be reported, not hidden.")
    S = CCS_AS_SHIPPED
    rows = pd.DataFrame([
        dict(metric="subjects", cpac=CPAC["n_subjects"],
             ccs_as_shipped=info["n_subjects"], ccs_matched=info["n_subjects"]),
        dict(metric="sites", cpac=CPAC["n_sites"],
             ccs_as_shipped=res["n_sites"], ccs_matched=res["n_sites"]),
        dict(metric="site pairs", cpac=CPAC["n_pairs"],
             ccs_as_shipped=res["n_pairs"], ccs_matched=res["n_pairs"]),
        dict(metric="timepoints T (cohort min crop)", cpac=T_CPAC,
             ccs_as_shipped=info["T"], ccs_matched=info["T"]),
        dict(metric="overall reject %", cpac=CPAC["overall_pct"],
             ccs_as_shipped=S["overall_pct"], ccs_matched=round(res["overall_pct"], 2)),
        dict(metric="median |d|/SE", cpac=CPAC["median_absz"],
             ccs_as_shipped=S["median_absz"], ccs_matched=round(res["median_absz"], 3)),
        dict(metric="diagonal unstable (of 50)", cpac=CPAC["diag_unstable"],
             ccs_as_shipped=S["diag_unstable"], ccs_matched=res["diag_unstable"]),
        dict(metric="off-diagonal unstable (of 2450)", cpac=CPAC["off_unstable"],
             ccs_as_shipped=S["off_unstable"], ccs_matched=res["off_unstable"]),
        dict(metric="top-50 region overlap with C-PAC", cpac=50,
             ccs_as_shipped=S["top50_overlap"], ccs_matched=overlap),
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT / "ccs_site_diagnostic.csv") + ".tmp")
    rows.to_csv(tmp, index=False); tmp.replace(OUT / "ccs_site_diagnostic.csv")

    md = ["# M2: cross-pipeline robustness -- ABIDE-I site diagnostic on CCS", "",
          "Identical machinery, identical retained subjects and covariates; the "
          "ONLY change is the input time series (CCS `filt_global/rois_cc200` "
          "instead of C-PAC). Ridge VAR(1) alpha=1.0, 100-bootstrap subject "
          "resampling, per-coefficient two-sample Wald z, BH-FDR q=0.05, "
          "unstable = rejects in >20% of site pairs. ComBat-GAM with the same "
          "covariates (batch=SITE, smooth AGE, SEX/DX_GROUP/func_mean_fd). "
          "Crop = cohort minimum, exactly as build_dataset.py step 4. No "
          "per-region standardization in either pipeline (PCP .1D are raw ROI "
          "means).", "",
          "## Input-convention mismatch between the two PCP products (measured)", "",
          "C-PAC `filt_global` .1D are **demeaned** (grand mean 0.001; pooled "
          "variance 99.9% within-subject temporal). CCS `filt_global` .1D are "
          "**not** (grand mean ~8896; pooled variance ~100% between-subject DC "
          "offset, temporal signal 0.05% of it). Left uncorrected that is a "
          "SECOND difference on top of the pipeline: `select_top_regions` then "
          "ranks regions by DC offset rather than temporal signal, and the pooled "
          "VAR fit is swamped by subject-specific offsets. The matched column "
          "demeans each subject x region over the full series, reproducing "
          "C-PAC's own convention so the ONLY remaining difference is the "
          "pipeline. Both columns are shown; the as-shipped column is confounded "
          "and is reported for transparency, not as the cross-pipeline result.", "",
          "## C-PAC vs CCS", "",
          "| metric | C-PAC (primary) | CCS as-shipped (confounded) | CCS matched (valid test) |",
          "|---|---|---|---|"]
    for _, r in rows.iterrows():
        md.append(f"| {r['metric']} | {r['cpac']} | {r['ccs_as_shipped']} | "
                  f"{r['ccs_matched']} |")
    md += ["", f"CCS volumes: cohort min T={info['T']} (median "
           f"{info['ccs_T_median']}) vs C-PAC T={info['T_cpac']} -- CCS discards "
           "one extra volume per subject; all sites/subjects retained. Missing "
           f"FILE_IDs under CCS: {info['n_missing']}.", "",
           "## Verdict", "", verdict, ""]
    _atomic_write(OUT / "ccs_site_diagnostic.md", "\n".join(md) + "\n")
    print("\n" + verdict)


def main(smoke=False):
    X, sids, info = build_ccs(smoke=smoke)
    n_boot = 20 if smoke else N_BOOT
    res = site_diagnostic(X, sids, n_boot=n_boot)
    print(f"\n  overall {res['overall_pct']:.2f}%  median|z| {res['median_absz']:.3f}  "
          f"diag {res['diag_unstable']}/{res['diag_total']}  "
          f"off {res['off_unstable']}/{res['off_total']}  "
          f"({res['n_sites']} sites, {res['n_pairs']} pairs)")
    if smoke:
        print("SMOKE OK -- download->load->crop->VAR->bootstrap->Wald->FDR path "
              "proven end-to-end (no ComBat, no report written).")
        return
    report(res, info)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
