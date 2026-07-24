"""Export the diagnostic-validation "unused" figures into figures/unused/ at paper
spec (300 dpi PNG + vector PDF, Type-42 fonts, tight bbox), using paper_style.

Reads the already-computed simulator/diagnostic_validation/results/all_cells.csv
(no sweep re-run). These are analysis figures NOT cited as standalone display
items in the manuscript; two of them (figS7_sim_localization, figS8_sim_hrf_
localization) are the panel sources for the merged supplementary Figure S5
(scripts/make_figS5_localization.py). The main-text power curve (Figure 4) is
produced by render_power_curve.py, not here.

  figS6_sim_null_fp_rate      C4 null per-coefficient FP rate by N and alpha
  figS7_sim_localization      where detections land (C1/C2/C3)   -> S5 panel A
  figS8_sim_hrf_localization  localisation accuracy, no-HRF vs HRF -> S5 panel B
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style                                          # noqa: E402
from paper_style import SINGLE_COL                          # noqa: E402

CELLS_CSV = HERE / "results" / "all_cells.csv"
OUT = ROOT / "figures" / "unused"
OUT.mkdir(parents=True, exist_ok=True)
NS = [30, 60, 120]
ALPHAS = [0.1, 1.0, 10.0]
F2_MAGS = [0.05, 0.10, 0.20, 0.40]
F2_N = 60
_written = []


def save(fig, name):
    png, pdf = OUT / f"{name}.png", OUT / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    _written.extend([png, pdf])
    print(f"  wrote {name}.png + {name}.pdf")


def main():
    assert CELLS_CSV.exists(), f"missing {CELLS_CSV}; run the sweep first"
    df = pd.read_csv(CELLS_CSV)
    two = df[df.family.isin(["F1", "F2"]) & df.condition.notna()].copy()

    # (The main-text power curve is Figure 4, produced by render_power_curve.py.)

    # ---- figS6: C4 null per-coefficient FP rate by N and alpha ------------- #
    paper_style.apply()
    c4a = two[(two.condition == "C4") & (two.family == "F1")]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.9))
    piv = c4a.groupby(["alpha", "N"])["fpr"].mean().unstack("N")
    x = np.arange(len(piv.index)); w = 0.25
    for k, N_ in enumerate(piv.columns):
        ax.bar(x + (k - 1) * w, piv[N_].values, w, label=f"N={N_}")
    ax.axhline(0.05, ls="--", color="k", lw=0.9, label="q = 0.05")
    ax.set_xticks(x); ax.set_xticklabels([f"α={a}" for a in piv.index])
    ax.set_ylim(0, 0.06)
    ax.set_ylabel("false-positive rate (per coefficient)")
    ax.set_title("C4 null: per-coefficient FP rate")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "figS6_sim_null_fp_rate")

    # ---- figS6: localisation -- where detections land (C1/C2/C3) ----------- #
    paper_style.apply()
    d = two[(two.family == "F1") & (two.alpha == 1.0) & (two.N == 60)
            & two.condition.isin(["C1", "C2", "C3"])]
    conds = ["C1", "C2", "C3"]; x = np.arange(len(conds)); w = 0.38
    det_diag = [d[d.condition == c]["det_diag_injectedDiag"].sum()
                + d[d.condition == c]["det_diag_notInjected"].sum() for c in conds]
    det_off = [d[d.condition == c]["det_off_injectedOff"].sum()
               + d[d.condition == c]["det_off_notInjected"].sum() for c in conds]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.9))
    ax.bar(x - w / 2, det_diag, w, label="detected on diagonal", color="#b5651d")
    ax.bar(x + w / 2, det_off, w, label="detected off-diagonal", color="#3b7dd8")
    ax.set_xticks(x)
    ax.set_xticklabels(["C1\n(inj diag)", "C2\n(inj off)", "C3\n(inj both)"])
    ax.set_ylabel("detected coefficients (Σ seeds, magnitudes)")
    ax.set_title("Localisation: where detections land")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "figS7_sim_localization")

    # ---- figS7: HRF vs no-HRF localisation accuracy ------------------------ #
    paper_style.apply()
    f2 = two[two.family == "F2"]
    f1 = two[(two.family == "F1") & (two.N == F2_N) & (two.alpha == 1.0)
             & two.magnitude.isin(F2_MAGS)]
    conds = ["C1", "C2", "C3"]; x = np.arange(len(conds)); w = 0.38
    acc_raw = [f1[f1.condition == c]["localization_acc"].mean() for c in conds]
    acc_hrf = [f2[f2.condition == c]["localization_acc"].mean() for c in conds]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.9))
    ax.bar(x - w / 2, acc_raw, w, label="no HRF", color="#4c9f70")
    ax.bar(x + w / 2, acc_hrf, w, label="HRF (Glover, TR=2)", color="#d1495b")
    ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylim(0, 1.05)
    ax.set_ylabel("localisation accuracy (mean over magnitudes)")
    ax.set_title("Does localisation survive HRF filtering?")
    ax.legend(fontsize=6, frameon=False)
    save(fig, "figS8_sim_hrf_localization")

    print(f"\n{len(_written)} files written to {OUT}")


if __name__ == "__main__":
    main()
