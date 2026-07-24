"""Re-render ONLY the off-diagonal power curve (figS4 / Figure 6) at print spec.

Rendering fix only: the "0.048 (real effect)" label moved out of the curve
region to clear space near the top; legend anchored in the empty mid-right;
larger fonts and figure size for journal column width. Data is unchanged -- reads
the already-computed results/all_cells.csv, does NOT re-run the sweep, and touches
no other figure or results file.

Writes (300 dpi PNG + vector PDF each):
  figures/paper/fig4_sim_power_curve.{png,pdf}   (v10 main-text Figure 4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style  # noqa: E402  (Type-42 fonts, 300 dpi, tight bbox)

CELLS_CSV = HERE / "results" / "all_cells.csv"
OUT = ROOT / "figures" / "paper"
NS = [30, 60, 120]


def main():
    assert CELLS_CSV.exists(), f"missing {CELLS_CSV}; run the sweep first"
    df = pd.read_csv(CELLS_CSV)
    two = df[df.family.isin(["F1", "F2"]) & df.condition.notna()].copy()
    c2 = two[(two.condition == "C2") & (two.family == "F1") & (two.alpha == 1.0)]
    c4 = two[(two.condition == "C4") & (two.family == "F1") & (two.alpha == 1.0)]

    paper_style.apply()                       # editable PDF fonts, 300 dpi
    # slightly larger fonts for legibility at journal column width
    mpl.rcParams.update({"axes.labelsize": 12, "axes.titlesize": 12,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "legend.fontsize": 10})

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for N_ in NS:                             # same three N series + error bars
        g = c2[c2.N == N_].groupby("magnitude")["tpr"]
        m, s = g.mean(), g.std(ddof=0)
        ax.errorbar(m.index, m.values, yerr=s.values, marker="o", ms=5,
                    capsize=3, lw=1.4, label=f"N={N_}")

    ax.axhline(c4["fpr"].mean(), ls="--", color="k", lw=1.1,
               label=f"C4 FP rate ({c4['fpr'].mean():.3f})")      # dashed at 0
    # The left band is too crowded for an on-line label (all three curves rise
    # steeply through x=0.02-0.10). Label the reference line in the LEGEND, which
    # sits in the empty mid-right, so no text ever crosses a data line.
    ax.axvline(0.048, ls=":", color="crimson", lw=1.1,
               label="0.048 = real off-diagonal effect")

    ax.set_xlabel("injected off-diagonal shift magnitude (VAR-coef units)")
    ax.set_ylabel("power (TPR on injected off-diagonal)")
    ax.set_title("Off-diagonal detection power (C2)")
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlim(0.0, 0.42)
    # legend in the empty mid-right band (all curves saturate at 1.0 on the right)
    ax.legend(loc="center right", frameon=True, framealpha=0.9)
    fig.tight_layout()

    written = []
    for name in ("fig4_sim_power_curve",):     # v10 main-text Figure 4
        png, pdf = OUT / f"{name}.png", OUT / f"{name}.pdf"
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
        written += [png, pdf]
    plt.close(fig)
    print("wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
