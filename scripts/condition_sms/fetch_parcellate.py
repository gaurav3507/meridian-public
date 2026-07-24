"""Resumable download-parcellate-delete loop for the condition-based SMS test.

For each AOMIC PIOP2 subject that has BOTH rest and working-memory runs:
  download the fmriprep MNI preproc BOLD + confounds (+ WM events), parcellate to
  Schaefer-200 with confound regression (and, for WM, task-design regression),
  crop to 160 volumes, save the (160, 200) region timeseries, DELETE the BOLD.
Never holds more than one subject's BOLD on disk. Skips subjects already done, so
it is safely resumable after interruption.

Usage:
  python fetch_parcellate.py [N]     # process first N subjects (smoke test), or all
"""
from __future__ import annotations

import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.maskers import NiftiLabelsMasker

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data/condition_sms/ts"
TMP = ROOT / "data/condition_sms/_tmp"
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
SUBJ_CACHE = ROOT / "data/condition_sms/subjects_both.txt"

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DS = "ds002790"
TR = 2.0
CROP_T = 160
N_ROIS = 200
MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
CONF_BASE = MOTION + [m + "_derivative1" for m in MOTION] \
    + [f"a_comp_cor_{i:02d}" for i in range(5)]

_ATLAS = fetch_atlas_schaefer_2018(n_rois=N_ROIS, yeo_networks=7, resolution_mm=2)


def _get(url: str, dest: Path):
    urllib.request.urlretrieve(url, dest)


def _list_subjects_with(task: str) -> set:
    suffix = f"task-{task}_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    subs, token = set(), None
    while True:
        q = {"list-type": "2", "prefix": f"{DS}/derivatives/fmriprep/",
             "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(BUCKET + "/?" + urllib.parse.urlencode(q),
                                     timeout=90).read().decode()
        for c in xml.split("<Key>")[1:]:
            k = c.split("</Key>")[0]
            if k.endswith(suffix):
                m = re.search(r"(sub-\d+)_task", k)
                if m:
                    subs.add(m.group(1))
        if "<IsTruncated>true</IsTruncated>" in xml:
            token = xml.split("<NextContinuationToken>")[1].split(
                "</NextContinuationToken>")[0]
        else:
            break
    return subs


def subjects_both() -> list:
    if SUBJ_CACHE.exists():
        return SUBJ_CACHE.read_text().split()
    inter = sorted(_list_subjects_with("restingstate")
                   & _list_subjects_with("workingmemory"))
    SUBJ_CACHE.write_text("\n".join(inter))
    return inter


def _select_confounds(tsv: Path, n_vol: int) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    cols = [c for c in CONF_BASE if c in df.columns]
    cols += [c for c in df.columns if c.startswith("cosine")]
    return df[cols].iloc[:n_vol].fillna(0.0).reset_index(drop=True)


def _task_design(events: Path, n_vol: int) -> pd.DataFrame:
    ev = pd.read_csv(events, sep="\t")[["onset", "duration", "trial_type"]]
    ev = ev.dropna(subset=["onset", "trial_type"])
    frame_times = TR * np.arange(n_vol)
    dm = make_first_level_design_matrix(
        frame_times, ev, hrf_model="spm + derivative", drift_model=None)
    return dm.drop(columns=[c for c in dm.columns if c == "constant"]
                   ).reset_index(drop=True)


def _process_run(sub: str, task: str, add_task: bool) -> np.ndarray:
    base = f"{sub}_task-{task}_acq-seq"
    bold = TMP / f"{base}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    conf = TMP / f"{base}_desc-confounds_regressors.tsv"
    _get(f"{BUCKET}/{DS}/derivatives/fmriprep/{sub}/func/{conf.name}", conf)
    _get(f"{BUCKET}/{DS}/derivatives/fmriprep/{sub}/func/{bold.name}", bold)
    img = nib.load(str(bold))
    n_vol = img.shape[3]
    confounds = _select_confounds(conf, n_vol)
    if add_task:
        ev = TMP / f"{base}_events.tsv"
        _get(f"{BUCKET}/{DS}/{sub}/func/{ev.name}", ev)
        confounds = pd.concat([confounds, _task_design(ev, n_vol)], axis=1)
        ev.unlink()
    masker = NiftiLabelsMasker(_ATLAS.maps, standardize="zscore_sample", t_r=TR,
                               resampling_target="data", verbose=0)
    ts = masker.fit_transform(img, confounds=confounds.values)   # (n_vol, 200)
    bold.unlink(); conf.unlink()
    if ts.shape[0] < CROP_T:
        raise ValueError(f"{task}: only {ts.shape[0]} volumes (< {CROP_T})")
    return ts[:CROP_T].astype(np.float32)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    subs = subjects_both()
    print(f"{len(subs)} subjects have both rest and working memory", flush=True)
    if limit:
        subs = subs[:limit]
    done = ok = fail = 0
    for sub in subs:
        rp, wp = OUT / f"{sub}_rest.npy", OUT / f"{sub}_wm.npy"
        if rp.exists() and wp.exists():
            done += 1
            continue
        try:
            ts_r = _process_run(sub, "restingstate", add_task=False)
            ts_w = _process_run(sub, "workingmemory", add_task=True)
            np.save(rp, ts_r); np.save(wp, ts_w)
            ok += 1
            print(f"{sub}: rest{ts_r.shape} wm{ts_w.shape} OK", flush=True)
        except Exception as e:
            fail += 1
            print(f"{sub}: FAIL {type(e).__name__}: {e}", flush=True)
    n_complete = len(list(OUT.glob("*_rest.npy")))
    print(f"\nnew ok={ok} fail={fail} already={done} | "
          f"subjects with both saved = {n_complete}", flush=True)


if __name__ == "__main__":
    main()
