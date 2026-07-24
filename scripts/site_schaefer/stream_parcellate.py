"""Stream-parcellate ABIDE-I func_preproc -> Schaefer-200, one subject at a time.

Architecture (non-negotiable):
  download ONE volume -> parcellate -> save (T,200) -> DELETE volume -> next.
  Peak disk = (n_workers) volumes only, never the full 79 GB.

Resumable across power loss:
  * skip subjects whose output .npy already exists
  * write output atomically (tmp file -> os.replace) so a crash cannot leave a
    truncated file that looks complete
  * append one line per completed subject to progress.log
  * transient volumes go to VOL_DIR and are wiped at startup; partial downloads
    use a .part suffix and are never mistaken for complete

t_r is INERT here (verified: parcellating the same volume with t_r=1.5 vs 3.0 is
bit-identical when detrend=False/low_pass=None/high_pass=None). We pass a nominal
t_r=2.0 and do NOT source per-subject TR.
"""
from __future__ import annotations
import csv, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker

BASE   = os.environ.get("MERIDIAN_SCRATCH", os.path.expanduser("~/meridian_schaefer"))
VOL    = os.path.join(BASE, "volumes")
TS     = os.path.join(BASE, "timeseries")
LOGDIR = os.path.join(BASE, "logs")
MANIFEST = os.path.join(BASE, "manifest.csv")
PROG   = os.path.join(LOGDIR, "progress.log")
PRE = ("https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
       "Outputs/cpac/filt_global/func_preproc/")
T_CROP = 116
NOMINAL_TR = 2.0          # inert; documented
N_WORKERS = 4             # peak = 4 volumes on disk (~0.3-1 GB), never 79 GB
DL_RETRY = 4

for d in (VOL, TS, LOGDIR):
    os.makedirs(d, exist_ok=True)

_atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
ATLAS_MAPS = _atlas.maps

def _session():
    s = requests.Session()
    r = Retry(total=DL_RETRY, backoff_factor=1.5,
              status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=r))
    return s

def _logline(msg: str):
    with open(PROG, "a") as f:
        f.write(msg + "\n")

def process(row, sess):
    fid, site = row["FILE_ID"], row["SITE_ID"]
    out = os.path.join(TS, fid + ".npy")
    if os.path.exists(out):
        print(f"SKIP {fid} ({site}) already done", flush=True)
        return ("SKIP", fid)
    vol  = os.path.join(VOL, fid + "_func_preproc.nii.gz")
    part = vol + ".part"
    try:
        # --- download atomically ---
        with sess.get(PRE + fid + "_func_preproc.nii.gz", stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        os.replace(part, vol)
        # --- parcellate (fresh masker; nominal inert t_r) ---
        m = NiftiLabelsMasker(ATLAS_MAPS, standardize="zscore_sample", t_r=NOMINAL_TR,
                              resampling_target="data", detrend=False,
                              low_pass=None, high_pass=None, verbose=0)
        ts = m.fit_transform(vol)
        if ts.shape[0] < T_CROP:
            raise ValueError(f"only {ts.shape[0]} timepoints < {T_CROP}")
        ts = ts[:T_CROP].astype(np.float32)
        # --- atomic save ---
        tmp = out + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, ts)
        os.replace(tmp, out)
        print(f"OK   {fid} ({site}) shape={ts.shape}", flush=True)
        _logline(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tOK\t{fid}\t{site}\t{ts.shape[0]}x{ts.shape[1]}")
        return ("OK", fid)
    except Exception as e:
        print(f"FAIL {fid} ({site}) {type(e).__name__}: {e}", flush=True)
        _logline(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tFAIL\t{fid}\t{site}\t{type(e).__name__}: {e}")
        return ("FAIL", fid)
    finally:
        for p in (part, vol):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

def main():
    # wipe transient volumes left by any prior crash
    for f in os.listdir(VOL):
        try: os.remove(os.path.join(VOL, f))
        except OSError: pass
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    done0 = sum(os.path.exists(os.path.join(TS, r["FILE_ID"] + ".npy")) for r in rows)
    print(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S')} total={total} already_done={done0} "
          f"remaining={total-done0} workers={N_WORKERS} t_r={NOMINAL_TR}(inert)", flush=True)
    _logline(f"# RUN START {time.strftime('%Y-%m-%d %H:%M:%S')} done={done0}/{total}")
    todo = [r for r in rows if not os.path.exists(os.path.join(TS, r["FILE_ID"] + ".npy"))]
    counts = {"OK": 0, "FAIL": 0, "SKIP": 0}
    sessions = [_session() for _ in range(N_WORKERS)]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process, r, sessions[i % N_WORKERS]): r for i, r in enumerate(todo)}
        n = 0
        for fut in as_completed(futs):
            status, _ = fut.result()
            counts[status] += 1
            n += 1
            done = done0 + counts["OK"]
            if n % 10 == 0 or status == "FAIL":
                print(f"[PROG] processed={n}/{len(todo)}  done_total={done}/{total}  "
                      f"OK={counts['OK']} FAIL={counts['FAIL']}", flush=True)
    final_done = sum(os.path.exists(os.path.join(TS, r["FILE_ID"] + ".npy")) for r in rows)
    print(f"[END] {time.strftime('%Y-%m-%d %H:%M:%S')} done={final_done}/{total} "
          f"OK={counts['OK']} FAIL={counts['FAIL']} SKIP={counts['SKIP']}", flush=True)
    _logline(f"# RUN END {time.strftime('%Y-%m-%d %H:%M:%S')} done={final_done}/{total} "
             f"FAIL={counts['FAIL']}")
    # exit non-zero if anything still missing, so a supervisor could re-launch
    sys.exit(0 if final_done == total else 1)

if __name__ == "__main__":
    main()
