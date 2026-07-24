"""Network-localize the 129 significant off-diagonal rest-vs-WM condition-shift
coefficients, and run a paired sensitivity check. Works only from saved outputs
(data/condition_sms/ts/*.npy and results/condition_sms/marginal.csv). No re-download
or re-parcellation.

Step 1: recover the Yeo-7 network of each of the top-50 Schaefer regions used in the
        VAR test (recompute top-50 identically; assert shape).
Step 2: 7x7 source->target network matrix of significant off-diagonal edges, with
        OBSERVED/EXPECTED enrichment (expected proportional to available off-diagonal
        region-pairs per cell, since networks differ in size). Rank by enrichment.
Step 3: paired sensitivity (per-subject VAR(1) rest vs WM, paired t per coefficient,
        BH-FDR), off-diagonal reject % vs the unpaired 5.27%.

Coefficient convention (from fit_var1: Ridge.coef_[target, feature]): marginal.csv
region_i = predicted (target), region_j = predictor (source). So an edge is labeled
source-network = net(region_j), target-network = net(region_i).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
from nilearn.datasets import fetch_atlas_schaefer_2018

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from test_b import fit_var1, select_top_regions            # noqa: E402 exact reuse

ROOT = SCRIPTS.parent
TS = ROOT / "data/condition_sms/ts"
RES = ROOT / "results/condition_sms"
FIG = ROOT / "figures/condition_sms"
FIG.mkdir(parents=True, exist_ok=True)

N_TOP = 50
FDR_Q = 0.05
NET_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def load_pairs():
    subs = sorted(p.name[:-9] for p in TS.glob("*_rest.npy")
                  if (TS / f"{p.name[:-9]}_wm.npy").exists())
    rest = np.stack([np.load(TS / f"{s}_rest.npy") for s in subs]).astype(np.float32)
    wm = np.stack([np.load(TS / f"{s}_wm.npy") for s in subs]).astype(np.float32)
    return subs, rest, wm


def region_networks():
    a = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    labs = [x.decode() if isinstance(x, bytes) else str(x) for x in a.labels]
    if labs[0].lower() == "background":
        labs = labs[1:]                       # column k <-> labs[k]
    assert len(labs) == 200, f"expected 200 region labels, got {len(labs)}"
    return np.array([l.split("_")[2] for l in labs])       # (200,) network tokens


def main():
    subs, rest, wm = load_pairs()
    print(f"subjects: {len(subs)}  rest{rest.shape} wm{wm.shape}")

    # ---- Step 1: top-50 selection (identical to test) -> networks ---------- #
    top = select_top_regions(np.concatenate([rest, wm], axis=0), N_TOP)
    assert top.shape == (N_TOP,) and len(set(top.tolist())) == N_TOP
    net200 = region_networks()
    net50 = net200[top]                        # network of each VAR region 0..49
    counts = {n: int((net50 == n).sum()) for n in NET_ORDER}
    print("\n[Step 1] network membership of the 50 VAR regions:")
    print("  " + "  ".join(f"{n}:{counts[n]}" for n in NET_ORDER)
          + f"  (total {sum(counts.values())})")
    assert sum(counts.values()) == N_TOP, "some region network unparsed"

    # ---- Step 2: localize the 129 significant off-diagonal edges ----------- #
    df = pd.read_csv(RES / "marginal.csv")
    sig = df[(~df["is_diag"]) & (df["reject_fdr"])].copy()
    print(f"\n[Step 2] significant off-diagonal edges: {len(sig)} (expected 129)")
    sig["src_net"] = net50[sig["region_j"].to_numpy()]     # predictor = source
    sig["tgt_net"] = net50[sig["region_i"].to_numpy()]     # predicted = target

    idx = {n: k for k, n in enumerate(NET_ORDER)}
    obs = np.zeros((7, 7))
    for _, r in sig.iterrows():
        obs[idx[r["src_net"]], idx[r["tgt_net"]]] += 1

    # available off-diagonal region-pairs per cell (i!=j)
    n = np.array([counts[x] for x in NET_ORDER], float)
    avail = np.outer(n, n).copy()
    for k in range(7):
        avail[k, k] = n[k] * (n[k] - 1)        # within-network excludes i==j
    total_avail = avail.sum()                  # == 2450
    exp = len(sig) * avail / total_avail
    with np.errstate(divide="ignore", invalid="ignore"):
        enr = np.where(exp > 0, obs / exp, np.nan)

    # ranked cells (flag unreliable low-expected cells)
    rows = []
    for i in range(7):
        for j in range(7):
            rows.append((NET_ORDER[i], NET_ORDER[j], int(obs[i, j]),
                         float(exp[i, j]), float(enr[i, j])))
    ranked = sorted([r for r in rows if r[2] > 0],
                    key=lambda r: (-r[4] if not np.isnan(r[4]) else 0))
    print("\n  top source->target cells by enrichment (obs / exp = enrichment):")
    for s, t, o, e, en in ranked[:8]:
        flag = "  [low-exp, noisy]" if e < 2 else ""
        print(f"    {s:11s} -> {t:11s}  obs {o:2d}  exp {e:4.1f}  enr {en:4.2f}{flag}")

    # explicit FrontoParietal (Cont) and Default readout
    def cell(a, b):
        return int(obs[idx[a], idx[b]]), float(exp[idx[a], idx[b]]), float(enr[idx[a], idx[b]])
    print("\n  WM-relevant cells (pre-registered focus):")
    for a, b in [("Cont", "Cont"), ("Cont", "Default"), ("Default", "Cont"),
                 ("Default", "Default")]:
        o, e, en = cell(a, b)
        print(f"    {a} -> {b}: obs {o}, exp {e:.1f}, enrichment {en:.2f}")
    # any-involvement enrichment for Cont and Default
    for name in ["Cont", "Default"]:
        k = idx[name]
        o = obs[k, :].sum() + obs[:, k].sum() - obs[k, k]
        e = exp[k, :].sum() + exp[:, k].sum() - exp[k, k]
        print(f"    {name} involved (as src or tgt): obs {int(o)}, exp {e:.1f}, "
              f"enr {o/e:.2f}")

    # heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.nan_to_num(enr, nan=0.0), cmap="RdBu_r", vmin=0, vmax=2)
    ax.set_xticks(range(7)); ax.set_xticklabels(NET_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(7)); ax.set_yticklabels(NET_ORDER)
    ax.set_xlabel("target network (predicted)")
    ax.set_ylabel("source network (predictor)")
    ax.set_title("Rest vs WM significant off-diagonal shifts\nobserved / expected "
                 "enrichment (1 = chance)")
    for i in range(7):
        for j in range(7):
            if obs[i, j] > 0:
                ax.text(j, i, f"{int(obs[i,j])}\n{enr[i,j]:.1f}x",
                        ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="enrichment (obs/exp)")
    fig.tight_layout()
    fig.savefig(FIG / "network_localization.png", dpi=150)
    print(f"\n  wrote {FIG/'network_localization.png'}")

    # ---- Step 3: paired sensitivity ---------------------------------------- #
    rest_s, wm_s = rest[:, :, top], wm[:, :, top]
    rc = np.stack([fit_var1(rest_s[k:k+1]) for k in range(len(subs))])   # (N,50,50)
    wc = np.stack([fit_var1(wm_s[k:k+1]) for k in range(len(subs))])
    tval, pval = ttest_rel(rc, wc, axis=0)                 # (50,50)
    rej = multipletests(pval.ravel(), alpha=FDR_Q,
                        method="fdr_bh")[0].reshape(N_TOP, N_TOP)
    eye = np.eye(N_TOP, dtype=bool)
    off_rej = int(rej[~eye].sum())
    off_rate = off_rej / (N_TOP * N_TOP - N_TOP)
    diag_rej = int(rej[eye].sum())
    print(f"\n[Step 3] PAIRED test (per-subject VAR, paired t, BH-FDR):")
    print(f"  off-diagonal reject: {off_rej}/2450 = {off_rate*100:.2f}%  "
          f"[unpaired was 129/2450 = 5.27%]")
    print(f"  diagonal reject: {diag_rej}/50")
    direction = ("RAISES" if off_rate > 0.0527 + 0.005 else
                 "LOWERS" if off_rate < 0.0527 - 0.005 else "MATCHES")
    print(f"  paired vs unpaired: {direction}")

    _append_md(counts, obs, exp, enr, ranked, cell, off_rej, off_rate, diag_rej,
               direction, len(sig))


def _append_md(counts, obs, exp, enr, ranked, cell, off_rej, off_rate, diag_rej,
               direction, n_sig):
    L = ["", "## Network localization of the significant off-diagonal shifts", "",
         f"The {n_sig} significant off-diagonal (cross-region) rest-vs-WM shifts, "
         "localized by Yeo-7 network of predictor (source) and predicted (target) "
         "region. Enrichment = observed / expected, where expected is proportional "
         "to the available off-diagonal region-pairs in each network cell (networks "
         "differ in size, so raw counts are biased toward big networks).", "",
         "Region counts among the 50 VAR regions: "
         + ", ".join(f"{k} {counts[k]}" for k in NET_ORDER) + ".", "",
         "Top source->target cells by enrichment (obs, exp, enrichment):"]
    for s, t, o, e, en in ranked[:6]:
        flag = " (low expected, noisy)" if e < 2 else ""
        L.append(f"- {s} -> {t}: obs {o}, exp {e:.1f}, enrichment {en:.2f}{flag}")
    cc = cell("Cont", "Cont"); cd = cell("Cont", "Default")
    dc = cell("Default", "Cont"); dd = cell("Default", "Default")
    L += ["", "WM-relevant (pre-registered focus, Cont = FrontoParietal):",
          f"- Cont -> Cont: obs {cc[0]}, exp {cc[1]:.1f}, enr {cc[2]:.2f}",
          f"- Cont -> Default: obs {cd[0]}, exp {cd[1]:.1f}, enr {cd[2]:.2f}",
          f"- Default -> Cont: obs {dc[0]}, exp {dc[1]:.1f}, enr {dc[2]:.2f}",
          f"- Default -> Default: obs {dd[0]}, exp {dd[1]:.1f}, enr {dd[2]:.2f}",
          "", "See figures/condition_sms/network_localization.png for the full "
          "7x7 enrichment matrix.", "",
          "## Paired sensitivity analysis", "",
          f"Rest and WM are the same 222 subjects. Per-subject VAR(1) coefficients, "
          f"paired t-test per coefficient across subjects, BH-FDR q=0.05.",
          f"- Paired off-diagonal reject: {off_rej}/2450 = {off_rate*100:.2f}% "
          f"(unpaired bootstrap Wald was 129/2450 = 5.27%). Paired {direction} the "
          "estimate.",
          f"- Paired diagonal reject: {diag_rej}/50.",
          "", "The paired result is reported as a robustness check; the primary "
          "number remains the unpaired bootstrap Wald for parallelism with the "
          "site test_b."]
    with open(RES / "results.md", "a") as f:
        f.write("\n".join(L) + "\n")
    print(f"  appended network-localization + paired sections to {RES/'results.md'}")


if __name__ == "__main__":
    main()
