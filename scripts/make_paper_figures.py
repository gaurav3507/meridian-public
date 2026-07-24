"""Build the FINAL publication figure set into figures/paper/ (300 dpi PNG + vector
PDF each). Works only from existing computed results (CSVs + saved .npy arrays);
NO analysis is re-run and NO reported number is altered.

Output file names follow the v10 manuscript numbering (4 main figs + supplement):
  fig1_diagnostic_schematic          Fig 1  (draw.io, NOT built here)
  fig2_three_panel                   Fig 2  ABIDE | ADHD-200 | AOMIC, one binary unstable scale
  figS3_abide_marginal_effects       Fig S3 two-panel marginal effect sizes
  figS4_condition_network_enrichment Fig S4 lighter hatch + SomMot 0/50 annotation
  figS1_test_a_site_effects          Fig S1 re-render
  figS2_test_b_multivariate          Fig S2 re-render, double-column
  figS5_unstable_by_network          Fig S5 re-render (nan bar -> Unlabelled/subcortical)
(Fig 3 = fig3_tr_vs_instability and Fig 4 = fig4_sim_power_curve are built by other scripts:
 scripts/tr_dependence and simulator/diagnostic_validation/render_power_curve.py.)

fig1_diagnostic_schematic() is retained below as a matplotlib fallback but is NOT
called by main(): the paper's Fig 1 is hand-built in draw.io, and re-running this
script must not overwrite it.
"""
from __future__ import annotations

import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, FancyArrowPatch
import seaborn as sns

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paper_style                                          # noqa: E402
from paper_style import SINGLE_COL, DOUBLE_COL              # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
OUT = ROOT / "figures/paper"
UNUSED = ROOT / "figures/unused"          # analysis figures not cited in the paper
OUT.mkdir(parents=True, exist_ok=True)
UNUSED.mkdir(parents=True, exist_ok=True)

LARGE_EFFECT = 0.10
UNSTABLE_THRESH = 0.20
NET_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
_written = []


def save(fig, name, out=OUT):
    png, pdf = out / f"{name}.png", out / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    _written.extend([png, pdf])
    print(f"  wrote {out.name}/{name}.png + .pdf")


def _lum_text(value, cmap, vmin=0, vmax=2):
    """White text on dark cells, black on light, by cell luminance."""
    t = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
    r, g, b, _ = cmap(t)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"


# --------------------------------------------------------------------------- #
# data loaders (rejection-fraction grids built identically for ABIDE & ADHD)
# --------------------------------------------------------------------------- #
def _site_grid(csv_path):
    """Return (50x50 rejection-fraction matrix, unstable DataFrame, n_pairs).

    Rejection fraction of a coefficient = fraction of site pairs in which that
    coefficient rejects at FDR 0.05. 'unstable' = fraction > UNSTABLE_THRESH.
    """
    df = pd.read_csv(csv_path)
    n_pairs = df.groupby(["site_1", "site_2"]).ngroups
    agg = (df.groupby(["region_i", "region_j"])
             .agg(rejection_fraction=("reject_fdr", "mean")).reset_index())
    agg["is_diag"] = agg["region_i"] == agg["region_j"]
    unstable = agg[agg["rejection_fraction"] > UNSTABLE_THRESH].copy()
    mat = (agg.pivot(index="region_i", columns="region_j",
                     values="rejection_fraction")
              .reindex(index=range(50), columns=range(50)).values)
    return mat, unstable, n_pairs


# --------------------------------------------------------------------------- #
# FIGURE 1  -  diagnostic workflow schematic (matplotlib fallback; NOT built by
#              main() -- the paper's Fig 1 is maintained in draw.io)
# --------------------------------------------------------------------------- #
def fig1_diagnostic_schematic():
    paper_style.apply()
    C_DIAG, C_OFF = "0.30", "0.90"        # dark grey diagonal, light grey off (greyscale-safe)
    C_BOX = "#f2f2f2"
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(cx, cy, w, h, text, fs=6.6, fc=C_BOX, weight="normal"):
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fc,
                               edgecolor="black", lw=0.7, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                zorder=3, weight=weight)

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                     mutation_scale=9, lw=1.0, color="black",
                                     zorder=1))

    # --- Stage 1: two environments (stacked) ---
    box(9, 68, 15, 12, "Environment 1\n(site A / rest)", weight="bold")
    box(9, 46, 15, 12, "Environment 2\n(site B / task)", weight="bold")

    # --- Stage 2: fit VAR(1) within each ---
    arrow(17.5, 25.5, 57)
    box(33, 57, 17, 20,
        "Fit VAR(1)\nwithin each\n\n" r"$x(t)=\mathbf{A}\,x(t{-}1)+c+e(t)$",
        fs=6.4)

    # --- Stage 3: compare coefficients of A (with the highlighted 6x6 matrix) ---
    arrow(42, 50.5, 57)
    ax.text(60.5, 78, "Compare every coefficient of "
            r"$\mathbf{A}$" "\nacross environments\n(per-coeff Wald test, FDR 0.05)",
            ha="center", va="center", fontsize=6.4)
    # 6x6 matrix A, diagonal highlighted
    nn = 6
    cw = 2.7
    mx0 = 60.5 - nn * cw / 2
    my_top = 68
    for r in range(nn):
        for c in range(nn):
            diag = (r == c)
            ax.add_patch(Rectangle((mx0 + c * cw, my_top - (r + 1) * cw), cw, cw,
                                   facecolor=C_DIAG if diag else C_OFF,
                                   edgecolor="black", lw=0.4, zorder=2))
    ax.text(60.5, my_top - nn * cw - 3.2, r"coefficient grid $\mathbf{A}$",
            ha="center", va="center", fontsize=6.0, style="italic")

    # --- Stage 4: the readout question + two branches ---
    arrow(73.5, 79.5, 57)
    box(89, 57, 19, 11, "Where does the\ninstability land?", fs=6.6,
        fc="#e9e9e9", weight="bold")
    # branch up = diagonal / measurement ; branch down = off-diagonal / mechanism
    ax.add_patch(FancyArrowPatch((89, 62.5), (89, 78), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.9, color="black"))
    ax.add_patch(FancyArrowPatch((89, 51.5), (89, 36), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.9, color="black"))
    # diagonal swatch + label (top)
    ax.add_patch(Rectangle((79.5, 84.5), 2.4, 2.4, facecolor=C_DIAG,
                           edgecolor="black", lw=0.5))
    ax.text(83.2, 85.7, "ON the diagonal:\nper-region autocorrelation "
            "changed\n=> MEASUREMENT shift (scanner, TR, sampling)",
            ha="left", va="center", fontsize=5.8)
    # off-diagonal swatch + label (bottom)
    ax.add_patch(Rectangle((79.5, 27.0), 2.4, 2.4, facecolor=C_OFF,
                           edgecolor="black", lw=0.5))
    ax.text(83.2, 28.2, "OFF the diagonal:\ncross-region influence changed\n"
            "=> MECHANISM shift (effective connectivity)",
            ha="left", va="center", fontsize=5.8)

    ax.set_title("A diagnostic for measurement vs mechanism shift across environments",
                 fontsize=8, pad=6)
    save(fig, "fig1_diagnostic_schematic")


# --------------------------------------------------------------------------- #
# condition significance loader (used by the three-panel figure)
# --------------------------------------------------------------------------- #
def _condition_grid():
    df = pd.read_csv(RES / "condition_sms/marginal.csv")
    z = (df.pivot(index="region_i", columns="region_j", values="abs_z")
           .reindex(index=range(50), columns=range(50)).values)
    sig = df[(~df["is_diag"]) & (df["reject_fdr"])]
    n_off_sig = int(((~df["is_diag"]) & (df["reject_fdr"])).sum())
    n_diag_sig = int((df["is_diag"] & (df["reject_fdr"])).sum())
    return df, z, sig, n_off_sig, n_diag_sig


# --------------------------------------------------------------------------- #
# FIGURE 2  -  three panels, ONE binary "unstable" scale (paper's central figure)
# --------------------------------------------------------------------------- #
def _binary_mask(unstable_df):
    """50x50 mask: 1 where a coefficient is unstable, 0 otherwise."""
    m = np.zeros((50, 50))
    for _, r in unstable_df.iterrows():
        m[int(r["region_i"]), int(r["region_j"])] = 1.0
    return m


# Yeo-7 grouping for the display axis (Unlabelled/subcortical last). The network
# label per region is precomputed by scripts/compute_panel_networks.py.
CANON_NET = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont",
             "Default", "Unlabelled"]
NET_ABBR = {"Vis": "Vis", "SomMot": "Som", "DorsAttn": "DAN", "SalVentAttn": "VAN",
            "Limbic": "Lim", "Cont": "FPN", "Default": "DMN", "Unlabelled": "Unl"}


def _network_order(panel):
    """(perm, blocks) for one panel's 50 regions.

    perm: array of original region indices (0-49) reordered so same-network
    regions are contiguous, network order = CANON_NET, atlas_id as stable
    secondary sort. blocks: list of (network, start, end_exclusive) in the new
    display coordinates.
    """
    net = pd.read_csv(RES / "panel_region_networks.csv")
    net = net[net["panel"] == panel].set_index("index").sort_index()
    order_key = net["network"].map(lambda n: CANON_NET.index(n))
    perm = (pd.DataFrame({"k": order_key, "a": net["atlas_id"]})
              .sort_values(["k", "a"]).index.to_numpy())
    nets_sorted = net["network"].to_numpy()[perm]
    blocks, start = [], 0
    for p in range(1, 51):
        if p == 50 or nets_sorted[p] != nets_sorted[start]:
            blocks.append((nets_sorted[start], start, p))
            start = p
    return perm, blocks


def _draw_blocks(ax, blocks, label_y=True):
    """White separators between network blocks (both axes) + block labels."""
    for _net, s, e in blocks[:-1]:
        for line in (e - 0.5,):
            ax.axvline(line, color="white", lw=0.7)
            ax.axhline(line, color="white", lw=0.7)
    centers = [(s + e - 1) / 2 for _n, s, e in blocks]
    labels = [NET_ABBR[n] for n, _s, _e in blocks]
    ax.set_xticks(centers); ax.set_xticklabels(labels, rotation=90, fontsize=4.6)
    if label_y:
        ax.set_yticks(centers); ax.set_yticklabels(labels, fontsize=4.6)
    else:
        ax.set_yticks([])
    ax.tick_params(length=1.5, pad=1)


def fig2_three_panel():
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    # All three panels encode the SAME binary quantity: unstable (1) vs stable (0).
    #   site panels : coefficient rejects in > 20% of that dataset's site pairs
    #   condition   : coefficient significant at FDR 0.05 (a single contrast, so
    #                 ">20% of site pairs" is undefined here)
    ab_mat, ab_un, ab_np = _site_grid(RES / "test_b_marginal.csv")
    ad_mat, ad_un, ad_np = _site_grid(RES / "adhd200/primary_top50_marginal.csv")
    df, _z, sig, n_off_sig, n_diag_sig = _condition_grid()

    ab_bin = _binary_mask(ab_un)
    ad_bin = _binary_mask(ad_un)
    cond_bin = (df.assign(v=df["reject_fdr"].astype(float))
                  .pivot(index="region_i", columns="region_j", values="v")
                  .reindex(index=range(50), columns=range(50)).values)

    ab_d, ab_o = int(ab_un["is_diag"].sum()), int((~ab_un["is_diag"]).sum())
    ad_d, ad_o = int(ad_un["is_diag"].sum()), int((~ad_un["is_diag"]).sum())

    # reorder each panel's rows AND columns by that panel's OWN network grouping
    # (site atlas = CC200, condition atlas = Schaefer-200; region sets differ).
    ab_perm, ab_blk = _network_order("ABIDE")
    ad_perm, ad_blk = _network_order("ADHD200")
    co_perm, co_blk = _network_order("AOMIC")
    ab_bin = ab_bin[np.ix_(ab_perm, ab_perm)]
    ad_bin = ad_bin[np.ix_(ad_perm, ad_perm)]
    cond_bin = cond_bin[np.ix_(co_perm, co_perm)]

    C_STABLE, C_UNSTABLE = "#10233a", "#ffd21e"      # dark navy vs bright gold
    binmap = ListedColormap([C_STABLE, C_UNSTABLE])

    sns.set_theme(style="white", context="paper"); paper_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 3.4))
    panels = [
        (axes[0], ab_bin, ab_blk, f"A  ABIDE-I site:\n{ab_d} diagonal, {ab_o} off-diagonal"),
        (axes[1], ad_bin, ad_blk, f"B  ADHD-200 site:\n{ad_d} diagonal, {ad_o} off-diagonal"),
        (axes[2], cond_bin, co_blk, f"C  AOMIC condition:\n{n_diag_sig} diagonal, {n_off_sig} off-diagonal"),
    ]
    for ax, mat, blk, title in panels:
        ax.imshow(mat, aspect="equal", cmap=binmap, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(title, fontsize=6.8)
        ax.set_xlabel("region (grouped by Yeo-7 network)", fontsize=6.0)
        _draw_blocks(ax, blk, label_y=True)
    axes[0].set_ylabel("region (grouped by Yeo-7 network)", fontsize=6.0)

    # two-entry legend, NOT a continuous colorbar. The network-grouping /
    # criterion explanation lives in the paper's figure caption, not the image.
    handles = [Patch(facecolor=C_UNSTABLE, edgecolor="black", lw=0.5, label="unstable"),
               Patch(facecolor=C_STABLE, edgecolor="black", lw=0.5, label="stable")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.03), handlelength=1.1, columnspacing=1.4)

    fig.subplots_adjust(bottom=0.24, top=0.90, wspace=0.28)
    save(fig, "fig2_three_panel")


# --------------------------------------------------------------------------- #
# FIGURE 3  -  ABIDE marginal effect sizes (unchanged content)
# --------------------------------------------------------------------------- #
def fig3_abide_marginal_effects():
    df = pd.read_csv(RES / "test_b_marginal.csv")
    n_pairs = df.groupby(["site_1", "site_2"]).ngroups
    z = df["abs_z"].values
    rej_per_coef = df.groupby(["region_i", "region_j"])["reject_fdr"].mean()
    pct_above = (z > 1.96).mean() * 100
    sns.set_theme(style="whitegrid", context="paper"); paper_style.apply()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9))
    xmax_left = max(6.0, float(np.percentile(z, 99.5)))
    bins = np.linspace(0, xmax_left, 200)
    above = z > 1.96
    axL.hist(z[~above], bins=bins, color="steelblue", alpha=0.85,
             edgecolor="none", label="|z| ≤ 1.96")
    axL.hist(z[above], bins=bins, color="crimson", alpha=0.85,
             edgecolor="none", label="|z| > 1.96")
    axL.axvline(1.96, ls="--", color="black", lw=1.0)
    axL.set_xlim(0, xmax_left)
    axL.set_xlabel(r"$|d| / \mathrm{SE}$  (per-coefficient Wald z)")
    axL.set_ylabel("count")
    axL.set_title(f"Marginal effect sizes, {len(df):,} tests\n"
                  f"{pct_above:.1f}% above 1.96  ·  "
                  f"FDR reject {df['reject_fdr'].mean()*100:.1f}%")
    axL.legend(frameon=False)
    axR.hist(rej_per_coef.values * 100, bins=np.linspace(0, 100, 26),
             color="slategray", edgecolor="white", alpha=0.9)
    axR.set_xlabel(f"Per-coefficient rejection rate across {n_pairs} site pairs (%)")
    axR.set_ylabel("count of coefficients (of 2,500)")
    axR.set_title("Site-pair rejection rate per coefficient")
    for x in (20, 50, 80):
        axR.axvline(x, ls=":", color="black", lw=0.8, alpha=0.4)
    save(fig, "figS3_abide_marginal_effects")


# --------------------------------------------------------------------------- #
# FIGURE 4  -  condition network enrichment (lighter hatch + SomMot note)
# --------------------------------------------------------------------------- #
def fig4_condition_network_enrichment():
    from test_b import select_top_regions
    from nilearn.datasets import fetch_atlas_schaefer_2018
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
    net50 = np.array([labs[t].split("_")[2] for t in top])

    df = pd.read_csv(RES / "condition_sms/marginal.csv")
    sig = df[(~df["is_diag"]) & (df["reject_fdr"])].copy()
    sig["src"] = net50[sig["region_j"].to_numpy()]
    sig["tgt"] = net50[sig["region_i"].to_numpy()]
    idx = {n: k for k, n in enumerate(NET_ORDER)}
    obs = np.zeros((7, 7))
    for _, r in sig.iterrows():
        obs[idx[r["src"]], idx[r["tgt"]]] += 1
    n = np.array([int((net50 == x).sum()) for x in NET_ORDER], float)
    n_sommot = int(n[NET_ORDER.index("SomMot")])
    avail = np.outer(n, n).copy()
    for k in range(7):
        avail[k, k] = n[k] * (n[k] - 1)
    exp = len(sig) * avail / avail.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        enr = np.where(exp > 0, obs / exp, np.nan)

    cmap = plt.get_cmap("RdBu_r")
    sns.set_theme(style="white", context="paper"); paper_style.apply()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.6))
    ax.imshow(np.nan_to_num(enr, nan=0.0), cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(7)); ax.set_xticklabels(NET_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels(NET_ORDER)
    ax.set_xlabel("target network (predicted)")
    ax.set_ylabel("source network (predictor)")
    for i in range(7):
        for j in range(7):
            if exp[i, j] < 2:                       # unreliable: light hatch + border
                tc = _lum_text(enr[i, j] if not np.isnan(enr[i, j]) else 0, cmap)
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       hatch="//", edgecolor=tc, lw=0.4,
                                       alpha=0.55))
            if obs[i, j] > 0:
                tc = _lum_text(enr[i, j], cmap)
                ax.text(j, i, f"{int(obs[i,j])}\n{enr[i,j]:.1f}x", ha="center",
                        va="center", fontsize=4.2, color=tc)
    ax.set_title("Rest vs WM off-diagonal shifts: obs/exp enrichment\n"
                 "hatched cells (expected < 2) are low-count, unreliable\n"
                 f"SomMot: {n_sommot} of 50 regions selected (variance-based top-50)",
                 fontsize=6.6)
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=2))
    fig.colorbar(sm, ax=ax, fraction=0.046, label="enrichment (obs/exp), 1 = chance")
    save(fig, "figS4_condition_network_enrichment")


# --------------------------------------------------------------------------- #
# SUPPLEMENTARY (re-render at spec, no content change)
# --------------------------------------------------------------------------- #
def figS1_test_a_site_effects():
    df = pd.read_csv(RES / "test_a_results.csv")
    sns.set_theme(style="whitegrid", context="paper"); paper_style.apply()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.3))
    sns.kdeplot(df.eta_squared_raw, ax=ax, label="Raw",
                fill=True, alpha=0.35, clip=(0, None), bw_adjust=0.8)
    sns.kdeplot(df.eta_squared_harmonized, ax=ax, label="ComBat-GAM harmonized",
                fill=True, alpha=0.35, clip=(0, None), bw_adjust=0.8)
    ax.axvline(LARGE_EFFECT, ls="--", color="k", lw=1.0,
               label=r"$\eta^2 = 0.10$ (large effect)")
    ax.set_xlabel(r"Site effect size  $\eta^{2}$  (one-way ANOVA, SITE_ID)")
    ax.set_ylabel("Density over (region, statistic) pairs")
    ax.set_title("Test A: marginal site effects on per-subject BOLD summaries")
    ax.legend(frameon=False)
    save(fig, "figS1_test_a_site_effects")


def figS2_test_b_multivariate():
    df = pd.read_csv(RES / "test_b_results.csv")
    df["pair"] = df["site_1"] + "__" + df["site_2"]
    pair_order = sorted(df["pair"].unique())
    rej_per_region = df.groupby("region")["reject_fdr"].mean()
    region_order = rej_per_region.sort_values(ascending=False).index.tolist()
    atlas_of = df.groupby("region")["region_orig_atlas_idx"].first()
    p_mat = (df.pivot(index="region", columns="pair", values="p_fdr")
               .loc[region_order, pair_order])
    sig_mat = (df.pivot(index="region", columns="pair", values="reject_fdr")
                 .loc[region_order, pair_order]).astype(bool)
    neglog = -np.log10(np.clip(p_mat.values, 1e-300, 1.0))
    vmax = max(float(np.percentile(neglog, 99)), -np.log10(0.05) + 1)
    sns.set_theme(style="white", context="paper"); paper_style.apply()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 4.0))
    im = ax.imshow(neglog, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    sy, sx = np.where(sig_mat.values)
    ax.scatter(sx, sy, marker="x", color="white", s=4, lw=0.3, alpha=0.55)
    ax.set_yticks(range(len(region_order)))
    ax.set_yticklabels([f"r{r} (atlas {int(atlas_of[r])})" for r in region_order],
                       fontsize=3)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, rotation=90, fontsize=2.5)
    ax.set_xlabel("Site pair")
    ax.set_ylabel("Region (top-50, sorted by # rejections desc.)")
    ax.set_title(r"ABIDE-I Test B multivariate: $-\log_{10}(p_{FDR})$;"
                 "  x = reject at FDR 0.05")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label(r"$-\log_{10}(p_{\mathrm{FDR}})$")
    save(fig, "figS2_test_b_multivariate")


def figS3_unstable_by_network():
    out = pd.read_csv(RES / "test_b_unstable_coefficients.csv")
    UNL = "Unlabelled / subcortical"
    src = out["region_j_network"].fillna(UNL).replace({"nan": UNL, "None": UNL})
    tgt = out["region_i_network"].fillna(UNL).replace({"nan": UNL, "None": UNL})
    pair_counts: Counter = Counter()
    for s, t in zip(src, tgt):                     # source j -> target i
        pair_counts[UNL if (s == UNL and t == UNL) else f"{s} -> {t}"] += 1
    pair_df = (pd.DataFrame([{"pair": k, "count": v} for k, v in pair_counts.items()])
                 .sort_values("count", ascending=False).reset_index(drop=True))
    assert not pair_df["pair"].str.contains("nan", case=False).any(), "nan label remains"
    sns.set_theme(style="whitegrid", context="paper"); paper_style.apply()
    fig, ax = plt.subplots(figsize=(SINGLE_COL, max(2.0, 0.3 * len(pair_df) + 1.2)))
    ax.barh(pair_df["pair"], pair_df["count"], color="steelblue", alpha=0.88)
    ax.invert_yaxis()
    ax.set_xlabel("# unstable VAR(1) coefficients")
    ax.set_title("ABIDE-I unstable coefficients by Yeo 7-network\n(source j -> target i)")
    for i, v in enumerate(pair_df["count"]):
        ax.text(v + 0.05, i, str(int(v)), va="center")
    # not cited in the submitted manuscript (superseded by the three-panel Fig 2)
    save(fig, "figS5_unstable_by_network", out=UNUSED)


def main():
    print("Building final paper figures -> figures/paper/")
    # fig1_diagnostic_schematic is hand-built in draw.io; do NOT regenerate it
    # here (would overwrite the maintained version).
    fig2_three_panel()
    fig3_abide_marginal_effects()
    fig4_condition_network_enrichment()
    figS1_test_a_site_effects()
    figS2_test_b_multivariate()
    figS3_unstable_by_network()
    print(f"\n{len(_written)} files written (fig1 excluded, maintained in draw.io).")


if __name__ == "__main__":
    main()
