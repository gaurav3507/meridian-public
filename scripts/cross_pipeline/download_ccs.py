"""M2 STEP 1 -- download ABIDE-I CCS derivatives (rois_cc200, filt_global).

Cross-pipeline robustness check: the same site diagnostic re-run on a DIFFERENT
preprocessing pipeline (CCS instead of C-PAC), to kill the "it's a C-PAC
artifact" objection. Public PCP S3, no credentials, CPU only.

Downloads ONLY the FILE_IDs already retained by the C-PAC analysis, read from
data/processed/abide_harmonized.npz['subject_ids'] (742 subjects, 14 sites).
That set is already QC'd (mean FD<0.2, %FD<30, site N>=30) by
scripts/build_dataset.py, so we do NOT re-QC: site membership, N and TR-per-site
stay identical and the ONLY difference is the pipeline.

Skip-existing + atomic write (.part -> rename) so re-runs resume. Missing
FILE_IDs are logged, and any site that drops below N>=30 under CCS is FLAGGED
explicitly rather than silently proceeding.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent.parent
NPZ = ROOT / "data/processed/abide_harmonized.npz"
OUT_DIR = ROOT / "data/raw/abide_ccs_cc200"
REPORT_DIR = ROOT / "results/cross_pipeline"
BASE = ("https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
        "Outputs/ccs/filt_global/rois_cc200")
MIN_SITE_N = 30                     # same threshold as scripts/build_dataset.py
N_WORKERS = 8


def _session():
    s = requests.Session()
    r = Retry(total=4, backoff_factor=1.0,
              status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=r, pool_maxsize=N_WORKERS * 2))
    return s


def retained_subjects():
    """(FILE_IDs, SITE_IDs) exactly as retained by the C-PAC analysis."""
    z = np.load(NPZ, allow_pickle=True)
    return (np.asarray(z["subject_ids"], dtype=str),
            np.asarray(z["site_ids"], dtype=str))


def fetch_one(fid, sess):
    out = OUT_DIR / f"{fid}_rois_cc200.1D"
    if out.exists() and out.stat().st_size > 0:
        return ("skip", fid)
    part = out.with_suffix(".1D.part")
    try:
        r = sess.get(f"{BASE}/{fid}_rois_cc200.1D", timeout=120)
        if r.status_code == 404:
            return ("missing", fid)
        r.raise_for_status()
        part.write_bytes(r.content)
        part.replace(out)                       # atomic
        return ("ok", fid)
    except Exception as e:
        if part.exists():
            part.unlink(missing_ok=True)
        return (f"error:{type(e).__name__}", fid)


def main(limit_sites=None, limit_per_site=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fids, sites = retained_subjects()
    df = pd.DataFrame({"FILE_ID": fids, "SITE_ID": sites})

    if limit_sites is not None:                 # smoke mode
        keep = sorted(df.SITE_ID.unique())[:limit_sites]
        df = (df[df.SITE_ID.isin(keep)].groupby("SITE_ID", group_keys=False)
                .head(limit_per_site).reset_index(drop=True))
        print(f"SMOKE: {len(df)} subjects from {keep}")

    print(f"target: {len(df)} subjects, {df.SITE_ID.nunique()} sites -> {OUT_DIR}")
    sess = _session()
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fetch_one, f, sess): f for f in df.FILE_ID}
        for k, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if k % 50 == 0:
                print(f"  {k}/{len(df)}")
    status = pd.DataFrame(results, columns=["status", "FILE_ID"]).merge(df, on="FILE_ID")
    n_ok = int((status.status == "ok").sum())
    n_skip = int((status.status == "skip").sum())
    missing = status[status.status == "missing"]
    errors = status[status.status.str.startswith("error")]
    print(f"\ndownloaded {n_ok}, already-present {n_skip}, "
          f"missing(404) {len(missing)}, errors {len(errors)}")

    # ---- availability per site + N>=30 flag ---------------------------------
    have = status[status.status.isin(["ok", "skip"])]
    per_site = (have.groupby("SITE_ID").size()
                .reindex(sorted(df.SITE_ID.unique())).fillna(0).astype(int))
    orig = df.groupby("SITE_ID").size().reindex(per_site.index)
    avail = pd.DataFrame({"n_cpac_retained": orig, "n_ccs_available": per_site})
    avail["dropped"] = avail.n_cpac_retained - avail.n_ccs_available
    avail["below_min_N"] = avail.n_ccs_available < MIN_SITE_N
    print("\nper-site availability:")
    print(avail.to_string())
    flagged = avail[avail.below_min_N]
    if len(flagged):
        print(f"\n*** FLAG: {len(flagged)} site(s) fall below N>={MIN_SITE_N} under "
              f"CCS: {flagged.index.tolist()} -- do NOT silently proceed; the "
              "cross-pipeline comparison would no longer be subject-matched.")
    else:
        print(f"\nAll sites retain N>={MIN_SITE_N} under CCS "
              "(subject-matched comparison intact).")

    if limit_sites is None:                     # only write reports for full runs
        _atomic_csv(avail.reset_index(), REPORT_DIR / "ccs_download_availability.csv")
        if len(missing):
            _atomic_csv(missing[["FILE_ID", "SITE_ID"]],
                        REPORT_DIR / "ccs_missing_subjects.csv")
    return avail


def _atomic_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    print(f"  wrote {path}")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        main(limit_sites=2, limit_per_site=5)
    else:
        main()
