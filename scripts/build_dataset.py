"""Build raw + ComBat-GAM-harmonized ABIDE-I cc200 dataset for SMS verification.

Pipeline
--------
1. Load phenotype CSV.
2. Filter subjects:
     - drop FILE_ID == 'no_filename'
     - keep func_mean_fd < 0.2 AND func_perc_fd < 30
     - require .1D file on disk
     - drop sites with fewer than MIN_SITE_N retained subjects
3. Load each subject's (T x 200) cc200 time series from .1D.
4. Crop every subject to the shortest T across the cohort -> (N, T, 200).
5. ComBat-GAM via neuroHarmonize on the per-region time-averaged signal,
   with SITE_ID as batch and AGE_AT_SCAN (smooth), SEX, DX_GROUP,
   func_mean_fd as protected covariates. Broadcast the additive correction
   (harmonized_mean - raw_mean) to every timepoint.
6. Save data/processed/abide_raw.npz and abide_harmonized.npz with metadata.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PHENO_CSV = ROOT / "data/raw/Phenotypic_V1_0b_preprocessed1.csv"
TS_DIR = ROOT / "data/raw/abide_i_cc200/Outputs/cpac/filt_global/rois_cc200"
OUT_DIR = ROOT / "data/processed"

MEAN_FD_MAX = 0.2
PERC_FD_MAX = 30.0
MIN_SITE_N = 30


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load phenotype
    pheno = pd.read_csv(PHENO_CSV)
    print(f"[1] Loaded phenotype: {len(pheno)} rows")

    # 2. Filter
    n0 = len(pheno)
    pheno = pheno[pheno["FILE_ID"].astype(str) != "no_filename"]
    print(f"[2a] Drop FILE_ID==no_filename: {n0} -> {len(pheno)}")

    pheno = pheno[pheno["func_mean_fd"].notna() & pheno["func_perc_fd"].notna()]
    pheno = pheno[(pheno["func_mean_fd"] < MEAN_FD_MAX)
                  & (pheno["func_perc_fd"] < PERC_FD_MAX)]
    print(f"[2b] Motion filter (mean_fd<{MEAN_FD_MAX}, perc_fd<{PERC_FD_MAX}): {len(pheno)}")

    pheno = pheno.assign(_path=pheno["FILE_ID"].apply(
        lambda fid: TS_DIR / f"{fid}_rois_cc200.1D"))
    pheno = pheno[pheno["_path"].apply(lambda p: p.is_file())]
    print(f"[2c] Require .1D on disk:           {len(pheno)}")

    site_counts = pheno["SITE_ID"].value_counts()
    keep_sites = sorted(site_counts[site_counts >= MIN_SITE_N].index.tolist())
    pheno = (pheno[pheno["SITE_ID"].isin(keep_sites)]
             .sort_values(["SITE_ID", "FILE_ID"])
             .reset_index(drop=True))
    print(f"[3]  Keep sites with N>={MIN_SITE_N}: {len(keep_sites)} sites, "
          f"{len(pheno)} subjects")
    print(f"     sites: {keep_sites}")

    # 4. Load .1D files
    print(f"[4]  Loading {len(pheno)} .1D files...")
    arrays: list[np.ndarray] = []
    lengths: list[int] = []
    for path in tqdm(pheno["_path"].tolist(), unit="subj"):
        a = np.loadtxt(path, dtype=np.float32)
        arrays.append(a)
        lengths.append(a.shape[0])

    n_regions = arrays[0].shape[1]
    for i, a in enumerate(arrays):
        if a.shape[1] != n_regions:
            raise RuntimeError(
                f"Region count mismatch: subj {pheno.loc[i,'FILE_ID']} has {a.shape}")

    T_min = int(min(lengths))
    print(f"     n_regions={n_regions}, T min={T_min}, median={int(np.median(lengths))}, "
          f"max={int(max(lengths))}")

    X_raw = np.stack([a[:T_min] for a in arrays], axis=0).astype(np.float32)
    print(f"     stacked (N,T,R)={X_raw.shape}, dtype={X_raw.dtype}, "
          f"{X_raw.nbytes/1e6:.1f} MB in memory")

    # 5. ComBat-GAM on per-region time-averaged signal
    print("[5]  ComBat-GAM on per-region time mean...")
    mean_signal = X_raw.mean(axis=1).astype(np.float64)  # (N, R)

    covars = pd.DataFrame({
        "SITE":         pheno["SITE_ID"].astype(str).values,
        "AGE_AT_SCAN":  pheno["AGE_AT_SCAN"].astype(float).values,
        "SEX":          pheno["SEX"].astype(int).values,
        "DX_GROUP":     pheno["DX_GROUP"].astype(int).values,
        "func_mean_fd": pheno["func_mean_fd"].astype(float).values,
    })

    _model, mean_harm = harmonizationLearn(
        mean_signal, covars, smooth_terms=["AGE_AT_SCAN"])
    correction = (mean_harm - mean_signal).astype(np.float32)  # (N, R)
    X_harm = (X_raw + correction[:, None, :]).astype(np.float32)
    print(f"     harmonized shape={X_harm.shape}, "
          f"mean |correction|={np.abs(correction).mean():.4f}")

    # 6. Save
    metadata = dict(
        subject_ids=np.asarray(pheno["FILE_ID"].astype(str).values, dtype="U64"),
        sub_ids=pheno["SUB_ID"].astype(np.int64).values,
        site_ids=np.asarray(pheno["SITE_ID"].astype(str).values, dtype="U32"),
        age=pheno["AGE_AT_SCAN"].astype(np.float32).values,
        sex=pheno["SEX"].astype(np.int8).values,
        dx_group=pheno["DX_GROUP"].astype(np.int8).values,
        func_mean_fd=pheno["func_mean_fd"].astype(np.float32).values,
        func_perc_fd=pheno["func_perc_fd"].astype(np.float32).values,
        T=np.int32(T_min),
        n_regions=np.int32(n_regions),
    )

    raw_out = OUT_DIR / "abide_raw.npz"
    harm_out = OUT_DIR / "abide_harmonized.npz"
    np.savez_compressed(raw_out, X=X_raw, **metadata)
    np.savez_compressed(harm_out, X=X_harm, **metadata)
    print(f"[6]  Wrote {raw_out.name} ({raw_out.stat().st_size/1e6:.1f} MB)")
    print(f"     Wrote {harm_out.name} ({harm_out.stat().st_size/1e6:.1f} MB)")

    # 7. Summary
    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"Final N:             {len(pheno)}")
    print(f"Final n_sites:       {len(keep_sites)}")
    print(f"Sites:               {keep_sites}")
    print(f"Final array shape:   {X_harm.shape}  (subjects, T, regions)")
    print(f"abide_raw.npz:       {raw_out.stat().st_size/1e6:.1f} MB")
    print(f"abide_harmonized.npz:{harm_out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
