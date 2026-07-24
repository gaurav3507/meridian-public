"""Build raw + ComBat-GAM-harmonized ADHD-200 (Athena) cc200 dataset for the SMS
replication. Mirrors scripts/build_dataset.py (ABIDE-I) as closely as honestly
possible; every deviation is logged. See scripts/adhd200/README.md for the
pre-registration and the full list of ABIDE differences.

Source: Neuro Bureau ADHD-200 Preprocessed, Athena pipeline (AFNI+FSL), filtered
cc200 region timeseries (sfnwmrda*_cc200_TCs.1D), read directly from
data/adhd/ADHD200_CC200_TCs_filtfix.tar (no full extraction).

Key deviations from ABIDE-I (all documented):
  - Engine: Athena (AFNI+FSL), not C-PAC.
  - Nuisance: WM+CSF+6motion+poly3, band-pass 0.009-0.08 Hz, NO GSR (ABIDE
    filt_global has GSR). Athena filtered == ABIDE filt_NOglobal.
  - Motion QC: max displacement < 3 mm (no per-frame data -> no mean FD). Max
    displacement is also the ComBat motion covariate (in place of func_mean_fd).
  - Regions: 3dROIstats drops empty parcels -> ~190 regions, not 200. Analysis
    on the common CC200 labels present across ALL retained subjects.
  - Peking pooled (Peking_1/2/3 -> PEKING), same scanner.

Pipeline: parse phenotype+motion per site -> QC (max motion < 3, one scan per
subject) -> load filtered cc200 TCs -> align to common labels -> crop to shortest
common T -> ComBat-GAM on per-region time mean -> save adhd200_harmonized.npz.
Stops after reporting post-QC per-site counts (the gate before the bootstrap).
"""
from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn

ROOT = Path(__file__).resolve().parent.parent.parent
TAR = ROOT / "data/adhd/ADHD200_CC200_TCs_filtfix.tar"
OUT = ROOT / "data/adhd200/processed"
OUT.mkdir(parents=True, exist_ok=True)

SITE_FOLDERS = ["KKI", "NeuroIMAGE", "NYU", "OHSU",
                "Peking_1", "Peking_2", "Peking_3", "Pittsburgh", "WashU"]
MIN_SITE_N = 30                           # QC = QC_Rest_1==pass (no motion data)
MIN_PROTOCOL_T = 100                       # drop short-protocol sites (only OHSU,
#   T=74): short-series lag-1 AC bias would contaminate the diagonal self-loop
#   test AND crop the whole cohort. Pre-specified rule, not a hand-named drop.

TC_RE = re.compile(r"^([A-Za-z_]+\w*)/(\d+)/sfnwmrda(\d+)_session_1_rest_1_cc200_TCs\.1D$")
MEAN_RE = re.compile(r"Mean_(\d+)")


def pooled_site(folder: str) -> str:
    return "PEKING" if folder.startswith("Peking") else folder.upper()


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize phenotype column names across site variants (e.g. WashU uses
    'ScanDirID' with no space)."""
    ren = {}
    for c in df.columns:
        n = re.sub(r"[^a-z0-9]", "", str(c).lower())
        ren[c] = {"scandirid": "ScanDir ID", "age": "Age", "gender": "Gender",
                  "dx": "DX", "qcrest1": "QC_Rest_1"}.get(n, c)
    return df.rename(columns=ren)


def parse_tcs(text: str):
    """3dROIstats output -> (labels list[int], arr (T, R) float64).

    Header: File, Sub-brick, Mean_<label>...  Data rows: filename, subbrick,
    then R region means. Parcels with no voxels are absent from the header.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split()
    labels = [int(MEAN_RE.match(t).group(1)) for t in header if MEAN_RE.match(t)]
    R = len(labels)
    rows = []
    for ln in lines[1:]:
        parts = ln.split()
        vals = parts[-R:]                 # last R fields = region means
        rows.append([float(x) for x in vals])
    arr = np.asarray(rows, dtype=np.float64)
    assert arr.shape[1] == R, f"row width {arr.shape[1]} != {R} labels"
    return labels, arr


def main() -> None:
    assert TAR.is_file(), f"missing tar: {TAR}"
    tf = tarfile.open(TAR)
    names = tf.getnames()
    nameset = set(names)

    # ---- 1. per-site phenotype + motion lookups --------------------------- #
    pheno_by_site, motion_by_site = {}, {}
    for folder in SITE_FOLDERS:
        pfn, mfn = f"{folder}/{folder}_phenotypic.csv", f"{folder}/{folder}_motion.csv"
        if pfn in nameset:
            df = pd.read_csv(io.StringIO(tf.extractfile(pfn).read().decode()))
            df.columns = [str(c).strip() for c in df.columns]
            df = _canon_cols(df)
            pheno_by_site[folder] = df.set_index(
                df["ScanDir ID"].map(lambda v: str(int(float(v))).zfill(7)))
        if mfn in nameset:
            dm = pd.read_csv(io.StringIO(tf.extractfile(mfn).read().decode()))
            dm.columns = [c.strip() for c in dm.columns]
            mm = {}
            for _, r in dm.iterrows():
                m = re.match(r"rp_(\d+)_session_1_rest_1", str(r["File"]))
                if m:
                    mm[m.group(1).zfill(7)] = float(r["Max Motion (mm)"])
            motion_by_site[folder] = mm

    # ---- 2. enumerate filtered session_1 rest_1 subjects, join meta ------- #
    recs = []
    for name in names:
        m = TC_RE.match(name)
        if not m:
            continue
        folder, folder_id, _ = m.groups()
        sid = folder_id.zfill(7)
        ph = pheno_by_site.get(folder)
        prow = ph.loc[sid] if (ph is not None and sid in ph.index) else None
        maxmot = motion_by_site.get(folder, {}).get(sid, np.nan)
        if prow is None:
            continue                       # no phenotype row -> cannot covary
        recs.append(dict(
            member=name, site=pooled_site(folder), sample=folder, id=sid,
            age=pd.to_numeric(prow.get("Age"), errors="coerce"),
            gender=pd.to_numeric(prow.get("Gender"), errors="coerce"),
            dx_raw=pd.to_numeric(prow.get("DX"), errors="coerce"),
            qc_rest1=pd.to_numeric(prow.get("QC_Rest_1"), errors="coerce"),
            max_motion=maxmot))          # KKI-only, informational; not used in QC
    sub = pd.DataFrame(recs).drop_duplicates(subset=["id"]).reset_index(drop=True)
    n_found = len(sub)

    # ---- 3. QC (QC_Rest_1 == pass; no motion data across sites) ------------ #
    sub["dx_bin"] = np.where(sub["dx_raw"] == 0, 0,
                             np.where(sub["dx_raw"].isin([1, 2, 3]), 1, np.nan))
    complete = (sub["age"].notna() & sub["gender"].notna()
                & sub["dx_bin"].notna())
    n_incomplete = int((~complete).sum())
    sub = sub[complete]
    n_before_qc = len(sub)
    sub = sub[sub["qc_rest1"] == 1]                 # consortium resting-run pass
    n_after_qc = len(sub)

    site_counts = sub["site"].value_counts()
    keep_sites = sorted(site_counts[site_counts >= MIN_SITE_N].index.tolist())
    dropped_sites = sorted(site_counts[site_counts < MIN_SITE_N].index.tolist())
    sub = (sub[sub["site"].isin(keep_sites)]
           .sort_values(["site", "id"]).reset_index(drop=True))

    # ---- 4. load TCs, drop short-protocol sites (T<100), align, crop ------- #
    print(f"[load] reading {len(sub)} filtered cc200 TC files from tar...")
    labels_list, arrays, native_T = [], [], []
    for member in sub["member"]:
        lbls, arr = parse_tcs(tf.extractfile(member).read().decode())
        labels_list.append(set(lbls))
        arrays.append((lbls, arr))
        native_T.append(arr.shape[0])
    sub = sub.assign(T_native=native_T)

    # pre-specified: drop sites whose protocol length (site min T) < MIN_PROTOCOL_T
    site_minT = sub.groupby("site")["T_native"].min()
    shortT_sites = sorted(site_minT[site_minT < MIN_PROTOCOL_T].index.tolist())
    keep_mask = (~sub["site"].isin(shortT_sites)).to_numpy()
    sub = sub[keep_mask].reset_index(drop=True)
    labels_list = [l for l, k in zip(labels_list, keep_mask) if k]
    arrays = [a for a, k in zip(arrays, keep_mask) if k]
    keep_sites = sorted(sub["site"].unique().tolist())

    common = sorted(set.intersection(*labels_list)) if labels_list else []
    per_subj_R = [len(s) for s in labels_list]
    print(f"[regions] per-subject label counts: min={min(per_subj_R)}, "
          f"median={int(np.median(per_subj_R))}, max={max(per_subj_R)}; "
          f"COMMON across all retained = {len(common)}")

    aligned = []
    for lbls, arr in arrays:
        idx = {l: k for k, l in enumerate(lbls)}
        aligned.append(arr[:, [idx[l] for l in common]])   # (T, R_common)
    T_min = min(a.shape[0] for a in aligned)
    X_raw = np.stack([a[:T_min] for a in aligned], axis=0).astype(np.float32)
    print(f"[crop] shortest common T = {T_min}  -> X_raw {X_raw.shape}")

    # ---- GATE: post-QC per-site counts ------------------------------------ #
    print("\n" + "=" * 64)
    print("POST-QC PER-SITE COUNTS (gate before bootstrap)")
    print("=" * 64)
    print(f"  filtered session_1 rest_1 subjects found : {n_found}")
    print(f"  dropped, incomplete covariates (age/sex/dx): {n_incomplete}")
    print(f"  dropped, QC_Rest_1 != pass                 : "
          f"{n_before_qc - n_after_qc}")
    print(f"  sites dropped (N<{MIN_SITE_N}, incl. WashU no-QC): {dropped_sites}")
    print(f"  sites dropped (protocol T<{MIN_PROTOCOL_T}, AC-bias): {shortT_sites}")
    print("  ---- retained ----")
    final = sub["site"].value_counts().sort_values(ascending=False)
    for s, n in final.items():
        print(f"    {s:12s} {int(n)}")
    print(f"  TOTAL retained: {len(sub)}  across {len(keep_sites)} sites")
    peking = sub[sub["site"] == "PEKING"]["sample"].value_counts().to_dict()
    print(f"  Peking pooled from (audit): {peking}")
    print(f"  site pairs: {len(keep_sites)*(len(keep_sites)-1)//2}")
    print(f"  common regions: {len(common)} (of 200) | T={T_min}")
    if T_min < 100:
        print(f"  [FLAG] T={T_min} is short; VAR(1) fit weaker than ABIDE (T=116).")

    # ---- 5. ComBat-GAM on per-region time mean ---------------------------- #
    print("\n[ComBat-GAM] harmonizing per-region time mean...")
    mean_signal = X_raw.mean(axis=1).astype(np.float64)         # (N, R_common)
    covars = pd.DataFrame({          # no motion covariate (no per-subject motion)
        "SITE": sub["site"].values,
        "AGE_AT_SCAN": sub["age"].astype(float).values,
        "SEX": sub["gender"].astype(int).values,
        "DX_GROUP": sub["dx_bin"].astype(int).values,
    })
    _model, mean_harm = harmonizationLearn(mean_signal, covars,
                                           smooth_terms=["AGE_AT_SCAN"])
    correction = (mean_harm - mean_signal).astype(np.float32)
    X_harm = (X_raw + correction[:, None, :]).astype(np.float32)
    print(f"[ComBat-GAM] done; mean |correction| = {np.abs(correction).mean():.4f}")

    # ---- 6. save ----------------------------------------------------------- #
    meta = dict(
        subject_ids=np.asarray(sub["id"].values, dtype="U16"),
        site_ids=np.asarray(sub["site"].values, dtype="U16"),
        sample_ids=np.asarray(sub["sample"].values, dtype="U16"),
        age=sub["age"].astype(np.float32).values,
        sex=sub["gender"].astype(np.int8).values,
        dx_group=sub["dx_bin"].astype(np.int8).values,
        qc_rest1=sub["qc_rest1"].astype(np.float32).values,
        region_labels=np.asarray(common, dtype=np.int32),
        T=np.int32(T_min), n_regions=np.int32(len(common)),
    )
    harm_out = OUT / "adhd200_harmonized.npz"
    raw_out = OUT / "adhd200_raw.npz"
    np.savez_compressed(raw_out, X=X_raw, **meta)
    np.savez_compressed(harm_out, X=X_harm, **meta)
    print(f"\n[save] {raw_out.name} ({raw_out.stat().st_size/1e6:.1f} MB)")
    print(f"[save] {harm_out.name} ({harm_out.stat().st_size/1e6:.1f} MB)")
    print("\nGATE reached. Review per-site counts above before the bootstrap.")


if __name__ == "__main__":
    main()
