"""Precompute the Yeo-7 network label of each of the 50 regions used in each
panel of fig2_three_panel, so the figure can group its display axis by network.

This changes NO data and NO reported number. It only derives, for display
ordering, a network label per region using the SAME method the paper's labeling
scripts already use:
  ABIDE-I  (CC200): region_i_atlas -> CC200 centroid (data/raw/cc200_roi_atlas)
                    -> Yeo 2011 7-network  (mirrors scripts/annotate_unstable.py)
  ADHD-200 (CC200): recompute top-50 from adhd200_harmonized.npz -> CC200 label
                    -> ADHD release parcellation centroid -> Yeo 2011
                    (mirrors scripts/adhd200/label_selfloops_adhd200.py)
  AOMIC    (Schaefer-200): network is embedded in the Schaefer label string.

Output: results/panel_region_networks.csv
  columns: panel, index (0-49), atlas_id, network (canonical Yeo-7 or Unlabelled)
"""
from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from nilearn.datasets import fetch_atlas_schaefer_2018

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
# per-run temp dir for the extracted ADHD-200 parcellation volume (auto-cleaned
# by the OS); override with the MERIDIAN_SCRATCH env var if you prefer a fixed dir
SCRATCH = Path(os.environ["MERIDIAN_SCRATCH"]) if os.environ.get("MERIDIAN_SCRATCH") \
    else Path(tempfile.mkdtemp(prefix="meridian_"))

# canonical Yeo-7 order used for grouping (Unlabelled/subcortical goes last)
CANON = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
         "Unlabelled"]
# Yeo-2011 integer code (1..7) -> canonical short name (matches the paper scripts'
# YEO_7_NAMES / YEO7 dicts, then renamed to the Schaefer short convention)
YEO_CODE_TO_CANON = {0: "Unlabelled", 1: "Vis", 2: "SomMot", 3: "DorsAttn",
                     4: "SalVentAttn", 5: "Limbic", 6: "Cont", 7: "Default"}


def select_top(X, k):
    v = X.reshape(-1, X.shape[-1]).var(axis=0, ddof=1)
    return np.argsort(v)[::-1][:k]


def label_centroids(atlas_path):
    """{integer label L : mean-voxel MNI centroid} for every L>0 in the atlas."""
    img = nib.load(str(atlas_path))
    A = np.asarray(img.dataobj).astype(int)
    aff = img.affine
    cent = {}
    for L in np.unique(A[A > 0]):
        vox = np.argwhere(A == L).mean(0)
        cent[int(L)] = (aff @ np.append(vox, 1.0))[:3]
    return cent


def yeo_sampler():
    """MNI xyz -> Yeo-2011 7-network integer code (0 = none/unlabelled)."""
    y = datasets.fetch_atlas_yeo_2011()
    key = "thick_7" if "thick_7" in y else ("maps" if "maps" in y else None)
    yimg = nib.load(y[key]) if isinstance(y[key], str) else y[key]
    data = np.asarray(yimg.dataobj).astype(int)
    if data.ndim == 4:
        data = data[..., 0]
    inv = np.linalg.inv(yimg.affine)
    shape = np.array(data.shape)

    def f(mni):
        v = np.round(inv @ np.append(mni, 1.0))[:3].astype(int)
        if np.any(v < 0) or np.any(v >= shape):
            return 0
        return int(data[v[0], v[1], v[2]])
    return f


def abide_networks(yeo):
    """region index 0-49 -> canonical network, via CC200 centroid + Yeo."""
    df = pd.read_csv(RES / "test_b_marginal.csv")
    idx_to_atlas = (df.groupby("region_i")["region_i_atlas"].first()
                      .reindex(range(50)).astype(int))
    cent = label_centroids(ROOT / "data/raw/cc200_roi_atlas.nii.gz")  # keys 1..200
    rows = []
    for k in range(50):
        a0 = int(idx_to_atlas[k])          # 0-based CC200 index (as in marginal.csv)
        mni = cent[a0 + 1]                 # centroid dict is keyed by 1-based label
        net = YEO_CODE_TO_CANON[yeo(mni)]
        rows.append(dict(panel="ABIDE", index=k, atlas_id=a0, network=net))
    return rows


def adhd_networks(yeo):
    z = np.load(ROOT / "data/adhd200/processed/adhd200_harmonized.npz",
                allow_pickle=True)
    X = z["X"].astype(np.float32)
    labels = z["region_labels"]
    top = select_top(X, 50)
    prim_lab = labels[top].astype(int)     # region index 0-49 -> CC200 label
    tar = ROOT / "data/adhd/ADHD200_CC200_TCs_filtfix.tar"
    ap = SCRATCH / "ADHD200_parcellate_200.nii.gz"
    ap.parent.mkdir(parents=True, exist_ok=True)
    if not ap.exists():
        with tarfile.open(tar) as tf:
            ap.write_bytes(
                tf.extractfile("templates/ADHD200_parcellate_200.nii.gz").read())
    cent = label_centroids(ap)
    rows = []
    for k in range(50):
        L = int(prim_lab[k])
        net = YEO_CODE_TO_CANON[yeo(cent[L])] if L in cent else "Unlabelled"
        rows.append(dict(panel="ADHD200", index=k, atlas_id=L, network=net))
    return rows


def aomic_networks():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from test_b import select_top_regions
    TSD = ROOT / "data/condition_sms/ts"
    subs = sorted(p.name[:-9] for p in TSD.glob("*_rest.npy")
                  if (TSD / f"{p.name[:-9]}_wm.npy").exists())
    rest = np.stack([np.load(TSD / f"{s}_rest.npy") for s in subs])
    wm = np.stack([np.load(TSD / f"{s}_wm.npy") for s in subs])
    top = select_top_regions(np.concatenate([rest, wm], axis=0), 50)
    a = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    labs = [x.decode() if isinstance(x, bytes) else str(x) for x in a.labels]
    if labs[0].lower() == "background":
        labs = labs[1:]
    rows = []
    for k in range(50):
        net = labs[int(top[k])].split("_")[2]     # already canonical short name
        rows.append(dict(panel="AOMIC", index=k, atlas_id=int(top[k]), network=net))
    return rows


def main():
    yeo = yeo_sampler()
    rows = abide_networks(yeo) + adhd_networks(yeo) + aomic_networks()
    out = pd.DataFrame(rows)
    # sanity: canonical names only
    bad = set(out["network"]) - set(CANON)
    assert not bad, f"non-canonical network names: {bad}"
    out.to_csv(RES / "panel_region_networks.csv", index=False)
    print(f"wrote {RES/'panel_region_networks.csv'}  ({len(out)} rows)\n")
    for panel in ["ABIDE", "ADHD200", "AOMIC"]:
        d = out[out.panel == panel]
        comp = d["network"].value_counts().reindex(CANON).fillna(0).astype(int)
        print(f"{panel:8s} network composition (of 50):")
        print("   " + "  ".join(f"{n}:{comp[n]}" for n in CANON if comp[n] > 0))


if __name__ == "__main__":
    main()
