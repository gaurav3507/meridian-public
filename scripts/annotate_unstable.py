"""Annotate Test B marginal unstable VAR(1) coefficients with anatomical
labels and produce paper-ready table + figures.

Pipeline
--------
1. Recover the 50 CC200 region indices used in Test B (from the saved
   bootstrap npz) and verify by recomputing top-50 from abide_harmonized.
2. Aggregate test_b_marginal.csv to one row per (i, j) coefficient.
3. Filter to coefficients rejecting in > 20 % of the 91 site pairs.
4. Fetch CC200 (Craddock 2012), Harvard-Oxford, Yeo 2011 via nilearn.
5. For each of the 50 selected regions, look up MNI centroid +
   Harvard-Oxford label + Yeo 7-network membership.
6. Build results/test_b_unstable_coefficients.csv and two figures.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import datasets
from nilearn.image import coord_transform
from scipy.ndimage import center_of_mass
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

YEO_7_NAMES = {
    0: "None",
    1: "Visual",
    2: "Somatomotor",
    3: "DorsalAttn",
    4: "VentralAttn",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default",
}

UNSTABLE_THRESH = 0.20


def find_cc200_volume(craddock_bunch) -> nib.Nifti1Image:
    """Locate the K=200 volume across the Craddock 2012 variants/4D files."""
    variant_order = ("tcorr_mean", "scorr_mean",
                     "tcorr_2level", "scorr_2level", "random")
    for variant in variant_order:
        if variant not in craddock_bunch:
            continue
        path = craddock_bunch[variant]
        img = nib.load(path)
        data = np.asarray(img.dataobj).astype(int)
        if data.ndim == 3:
            n_uniq = len(np.unique(data)) - 1
            if n_uniq == 200:
                print(f"  CC200 -> variant '{variant}' (3D)")
                return img
        elif data.ndim == 4:
            for k in range(data.shape[3]):
                vol = data[..., k]
                n_uniq = len(np.unique(vol)) - 1
                if n_uniq == 200:
                    print(f"  CC200 -> variant '{variant}', K-index {k} "
                          f"(of {data.shape[3]} K levels)")
                    return img.slicer[..., k]
    raise RuntimeError("Could not find K=200 volume in Craddock 2012 atlas")


def lookup_voxel(mni_xyz, data, affine) -> int | None:
    inv = np.linalg.inv(affine)
    i, j, k = coord_transform(mni_xyz[0], mni_xyz[1], mni_xyz[2], inv)
    i, j, k = int(round(i)), int(round(j)), int(round(k))
    if (0 <= i < data.shape[0] and
            0 <= j < data.shape[1] and
            0 <= k < data.shape[2]):
        return int(data[i, j, k])
    return None


def lookup_ho(mni, ho_c_data, ho_c_aff, ho_c_lbl,
              ho_s_data, ho_s_aff, ho_s_lbl) -> str:
    v = lookup_voxel(mni, ho_c_data, ho_c_aff)
    if v is not None and v > 0:
        return ho_c_lbl[v]
    v = lookup_voxel(mni, ho_s_data, ho_s_aff)
    if v is not None and v > 0:
        return ho_s_lbl[v]
    return "Unlabelled"


def lookup_yeo(mni, yeo_data, yeo_affine) -> str:
    v = lookup_voxel(mni, yeo_data, yeo_affine)
    if v is None:
        return "OOB"
    return YEO_7_NAMES.get(v, f"Unknown({v})")


def main() -> None:
    # ------------------------------------------------------------ 1) indices
    print("=" * 72)
    print("1) Recover & verify top-50 region indices used in Test B")
    print("=" * 72)
    boot = np.load(DATA / "test_b_bootstrap.npz")
    top_idx = boot["top_region_idx"].astype(int)

    harm = np.load(DATA / "abide_harmonized.npz")
    var_pr = harm["X"].reshape(-1, harm["X"].shape[-1]).var(axis=0, ddof=1)
    top_idx_recomp = np.argsort(var_pr)[::-1][:50].astype(int)

    matches = sorted(top_idx.tolist()) == sorted(top_idx_recomp.tolist())
    print(f"  Stored top-50 == recomputed top-50: {matches}")
    if not matches:
        raise RuntimeError("Top-50 mismatch — investigate before continuing")
    print(f"  CC200 indices (0-based, sorted): {sorted(top_idx.tolist())}")

    # ----------------------------------------------------- 2) aggregate
    print()
    print("=" * 72)
    print("2) Aggregate per-coefficient stats")
    print("=" * 72)
    df = pd.read_csv(RESULTS / "test_b_marginal.csv")
    print(f"  Loaded {len(df):,} rows from test_b_marginal.csv")
    agg = (df.groupby(["region_i", "region_j"])
             .agg(rejection_fraction=("reject_fdr", "mean"),
                  mean_dz_se=("abs_z", "mean"),
                  median_p_fdr=("p_fdr", "median"),
                  region_i_atlas=("region_i_atlas", "first"),
                  region_j_atlas=("region_j_atlas", "first"))
             .reset_index())
    assert len(agg) == 2500, f"expected 2500 coefficient entries, got {len(agg)}"
    print(f"  Aggregated to {len(agg)} coefficient entries")

    # ----------------------------------------------------- 3) filter
    print()
    print("=" * 72)
    print(f"3) Filter to unstable (rejection_fraction > {UNSTABLE_THRESH})")
    print("=" * 72)
    unstable = (agg[agg["rejection_fraction"] > UNSTABLE_THRESH]
                  .sort_values("rejection_fraction", ascending=False)
                  .reset_index(drop=True))
    print(f"  Unstable coefficients: {len(unstable)}")

    # ----------------------------------------------------- 4) atlas fetch
    print()
    print("=" * 72)
    print("4) Fetch nilearn atlases (cached after first download)")
    print("=" * 72)

    # nilearn.datasets.fetch_atlas_craddock_2012() currently fails: the
    # upstream NITRC host (cluster_roi.projects.nitrc.org) has both a
    # certificate hostname mismatch and a 403 on the parcellation tar.gz.
    # We instead load the *exact* cc200 atlas PCP used to extract the .1D
    # files, downloaded from the FCP-INDI S3 bucket:
    #   https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative
    #     /Resources/cc200_roi_atlas.nii.gz
    cc200_path = ROOT / "data/raw/cc200_roi_atlas.nii.gz"
    print(f"  loading PCP cc200 atlas: {cc200_path.name}")
    cc200_img = nib.load(cc200_path)
    cc200_data = np.asarray(cc200_img.dataobj).astype(int)
    cc200_affine = cc200_img.affine

    label_values = np.unique(cc200_data)
    label_values = label_values[label_values > 0]
    n_labels = len(label_values)
    contig = bool((label_values == np.arange(1, n_labels + 1)).all())
    print(f"  CC200 labels: count={n_labels}, contiguous 1..{n_labels} = {contig}")
    assert n_labels == 200 and contig, "CC200 atlas labels are not 1..200 contiguous"

    print("  computing CC200 centroids in MNI...")
    centroids_vox = center_of_mass(
        np.ones_like(cc200_data, dtype=np.float64),
        cc200_data, list(range(1, 201)))
    centroids_mni = np.array(
        [nib.affines.apply_affine(cc200_affine, np.asarray(c))
         for c in centroids_vox])
    # centroids_mni[k] (k in 0..199) corresponds to atlas label k+1 = PCP column k
    print(f"  centroids_mni shape: {centroids_mni.shape}")

    def _as_img(x):
        # nilearn 0.13.x returns a Nifti1Image in ["maps"], older versions
        # returned a path. Accept either.
        return x if hasattr(x, "dataobj") else nib.load(x)

    print("  fetch_atlas_harvard_oxford(cort-maxprob-thr25-2mm)...")
    ho_c = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    ho_c_img = _as_img(ho_c["maps"])
    ho_c_data = np.asarray(ho_c_img.dataobj).astype(int)
    if ho_c_data.ndim == 4:
        ho_c_data = ho_c_data[..., 0]
    ho_c_aff = ho_c_img.affine
    ho_c_lbl = list(ho_c["labels"])

    print("  fetch_atlas_harvard_oxford(sub-maxprob-thr25-2mm)...")
    ho_s = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ho_s_img = _as_img(ho_s["maps"])
    ho_s_data = np.asarray(ho_s_img.dataobj).astype(int)
    if ho_s_data.ndim == 4:
        ho_s_data = ho_s_data[..., 0]
    ho_s_aff = ho_s_img.affine
    ho_s_lbl = list(ho_s["labels"])

    print("  fetch_atlas_yeo_2011()...")
    yeo = datasets.fetch_atlas_yeo_2011()
    # Recent nilearn returns just `maps` (the 7-network LiberalMask
    # at 1mm). Older versions had separate `thick_7` / `thin_7` keys.
    yeo_img = _as_img(yeo["maps" if "maps" in yeo else "thick_7"])
    yeo_data = np.asarray(yeo_img.dataobj).astype(int)
    if yeo_data.ndim == 4:
        yeo_data = yeo_data[..., 0]
    yeo_aff = yeo_img.affine

    # ----------------------------------------------------- 5) label 50 regions
    print()
    print("=" * 72)
    print("5) Label each of the 50 selected CC200 regions")
    print("=" * 72)
    region_info = {}
    for k in tqdm(range(50), desc="  labeling"):
        atlas_idx = int(top_idx[k])
        mni = centroids_mni[atlas_idx]
        region_info[atlas_idx] = {
            "label": lookup_ho(mni, ho_c_data, ho_c_aff, ho_c_lbl,
                               ho_s_data, ho_s_aff, ho_s_lbl),
            "network": lookup_yeo(mni, yeo_data, yeo_aff),
            "mni": (float(mni[0]), float(mni[1]), float(mni[2])),
        }

    # ----------------------------------------------------- 6) labeled CSV
    print()
    print("=" * 72)
    print("6) Write labeled unstable-coefficients CSV")
    print("=" * 72)
    rows = []
    for _, r in unstable.iterrows():
        ai, aj = int(r["region_i_atlas"]), int(r["region_j_atlas"])
        ii, jj = region_info[ai], region_info[aj]
        rows.append({
            "coefficient_i": int(r["region_i"]),
            "coefficient_j": int(r["region_j"]),
            "region_i_atlas": ai,
            "region_j_atlas": aj,
            "region_i_label": ii["label"],
            "region_j_label": jj["label"],
            "region_i_network": ii["network"],
            "region_j_network": jj["network"],
            "MNI_i_xyz": f"({ii['mni'][0]:+.0f}, {ii['mni'][1]:+.0f}, {ii['mni'][2]:+.0f})",
            "MNI_j_xyz": f"({jj['mni'][0]:+.0f}, {jj['mni'][1]:+.0f}, {jj['mni'][2]:+.0f})",
            "rejection_fraction": float(r["rejection_fraction"]),
            "mean_dz_se": float(r["mean_dz_se"]),
            "median_p_fdr": float(r["median_p_fdr"]),
        })
    out = pd.DataFrame(rows)
    out_csv = RESULTS / "test_b_unstable_coefficients.csv"
    out.to_csv(out_csv, index=False)
    print(f"  Wrote {out_csv}  ({len(out)} rows)")

    # ----------------------------------------------------- 7) heatmap
    print()
    print("=" * 72)
    print("7a) Figure: 50x50 rejection-fraction heatmap")
    print("=" * 72)
    rej_mat = (agg.pivot(index="region_i", columns="region_j",
                         values="rejection_fraction")
                  .reindex(index=range(50), columns=range(50)).values)
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(rej_mat, aspect="equal", cmap="magma", vmin=0, vmax=1)
    for _, r in unstable.iterrows():
        ax.plot(int(r["region_j"]), int(r["region_i"]),
                marker="s", mfc="none", mec="cyan", mew=1.2, ms=9)
    ax.set_xlabel("Input region j  (top-50 index)")
    ax.set_ylabel("Output region i  (top-50 index)")
    ax.set_title("Test B marginal — rejection fraction per VAR(1) coefficient\n"
                 f"(cyan squares: {len(unstable)} coeffs with > {int(UNSTABLE_THRESH*100)}% "
                 "of site pairs rejecting at FDR 0.05)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("rejection fraction over 91 site pairs")
    fig.tight_layout()
    fig_grid = FIGURES / "test_b_unstable_grid.png"
    fig.savefig(fig_grid, dpi=150)
    plt.close(fig)
    print(f"  Wrote {fig_grid}")

    # ----------------------------------------------------- 7b) network bars
    print()
    print("=" * 72)
    print("7b) Figure: source-network -> target-network bar chart")
    print("=" * 72)
    # VAR(1) convention: A[i, j] means input j at t -> output i at t+1.
    pair_counts: Counter = Counter()
    for _, r in out.iterrows():
        src = r["region_j_network"]
        tgt = r["region_i_network"]
        pair_counts[(src, tgt)] += 1
    pair_df = (pd.DataFrame([{"source": s, "target": t, "count": c}
                             for (s, t), c in pair_counts.items()])
                 .sort_values("count", ascending=False).reset_index(drop=True))

    fig2, ax2 = plt.subplots(figsize=(11, max(4, 0.4 * len(pair_df) + 2)))
    labels = [f"{r['source']}  ->  {r['target']}" for _, r in pair_df.iterrows()]
    ax2.barh(labels, pair_df["count"], color="steelblue", alpha=0.88)
    ax2.invert_yaxis()
    ax2.set_xlabel("# unstable VAR(1) coefficients")
    ax2.set_title("Unstable coefficients by Yeo 7-network pair "
                  "(source j -> target i)")
    for i, v in enumerate(pair_df["count"]):
        ax2.text(v + 0.1, i, str(int(v)), va="center")
    fig2.tight_layout()
    fig_net = FIGURES / "test_b_unstable_network_summary.png"
    fig2.savefig(fig_net, dpi=150)
    plt.close(fig2)
    print(f"  Wrote {fig_net}")

    # ----------------------------------------------------- 8) summary print
    print()
    print("=" * 72)
    print("8) SUMMARY REPORT")
    print("=" * 72)

    print(f"\n--- Top 10 most-unstable VAR(1) coefficients ---")
    cols = ["coefficient_i", "coefficient_j", "region_i_label",
            "region_j_label", "region_i_network", "region_j_network",
            "rejection_fraction", "mean_dz_se", "median_p_fdr"]
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.max_colwidth", 32):
        print(out[cols].head(10).to_string(index=False))

    print(f"\n--- Network-pair counts (source j -> target i) ---")
    for _, r in pair_df.iterrows():
        print(f"  {r['source']:14s} -> {r['target']:14s}  {int(r['count']):3d}")

    self_loops = out[out["region_i_atlas"] == out["region_j_atlas"]]
    print(f"\n--- Self-loops (i == j): {len(self_loops)} of {len(out)} ---")
    if len(self_loops):
        cols2 = ["coefficient_i", "region_i_label", "region_i_network",
                 "rejection_fraction", "mean_dz_se"]
        with pd.option_context("display.max_columns", None,
                               "display.width", 180,
                               "display.max_colwidth", 40):
            print(self_loops[cols2].to_string(index=False))

    signal_poor_terms = (
        "orbital", "frontal pole", "subcallosal",
        "temporal pole", "inferior temporal",
        "brain-stem", "brainstem",
    )
    is_poor = lambda lbl: any(t in lbl.lower() for t in signal_poor_terms)
    i_poor = int(out["region_i_label"].apply(is_poor).sum())
    j_poor = int(out["region_j_label"].apply(is_poor).sum())
    unlabelled_i = int((out["region_i_label"] == "Unlabelled").sum())
    unlabelled_j = int((out["region_j_label"] == "Unlabelled").sum())
    print(f"\n--- Anatomy of unstable coefficients ---")
    print(f"  signal-poor anatomy (OFC / frontal pole / temporal pole /")
    print(f"  inferior temporal / subcallosal / brainstem):")
    print(f"      output (i) side: {i_poor:3d} / {len(out)}")
    print(f"      input  (j) side: {j_poor:3d} / {len(out)}")
    print(f"  Unlabelled (centroid outside Harvard-Oxford masks; commonly")
    print(f"  white-matter / ventricle adjacent / OOB):")
    print(f"      output (i) side: {unlabelled_i:3d} / {len(out)}")
    print(f"      input  (j) side: {unlabelled_j:3d} / {len(out)}")


if __name__ == "__main__":
    main()
