"""Test C — does repetition-time (TR) difference between ABIDE-I sites predict
DIAGONAL VAR(1) instability, with the off-diagonal as a control?

Design
------
For each of the 91 site pairs we count how many VAR(1) coefficients differ
significantly between the two sites (FDR-rejected marginal tests, from
results/test_b_marginal.csv). We split those counts into:
    diagonal      (region_i == region_j; 50 self-loop coefficients)
    off-diagonal  (the other 2,450 coefficients)   <-- CONTROL
and ask whether |TR_1 - TR_2| (delta_TR) predicts each count (Spearman).

Prediction: delta_TR predicts DIAGONAL instability specifically, NOT
off-diagonal. If it predicts both roughly equally, delta_TR is just a proxy for
"these two sites differ in general" and does not support a measurement
interpretation. If controlling for min(N) removes the effect, it is a
sample-size (power) artifact.

Per-site TR is measured from NIfTI headers by scripts/step-1 (cached to
results/measured_tr.csv). This script consumes that cache; it does not download.

HONESTY: one test, one answer. Spearman only. No Pearson/Kendall/log/outlier/
threshold hunting. No site pair dropped. Null results reported plainly.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paper_style                                          # noqa: E402
from paper_style import DOUBLE_COL                          # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
OUT = ROOT / "figures/paper"
OUT.mkdir(parents=True, exist_ok=True)

MARGINAL_CSV = RES / "test_b_marginal.csv"
TR_CSV = RES / "measured_tr.csv"


def find_reject_col(df: pd.DataFrame) -> str:
    """Identify the FDR-corrected rejection column and print its name."""
    cands = [c for c in df.columns
             if "reject" in c.lower() and "fdr" in c.lower()]
    if not cands:                       # fall back: any boolean 'reject*'
        cands = [c for c in df.columns if "reject" in c.lower()]
    assert len(cands) == 1, f"ambiguous reject columns: {cands}"
    col = cands[0]
    print(f"[reject-col] using column: '{col}'  (dtype={df[col].dtype})")
    return col


def partial_spearman(x, y, z):
    """Partial Spearman rho of x,y controlling for z: correlate the residuals of
    rank(x)~rank(z) and rank(y)~rank(z). p from a t-test on the partial r with
    df = n - 3."""
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)

    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta

    ex, ey = resid(rx, rz), resid(ry, rz)
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r**2))
    p = 2 * stats.t.sf(abs(t), df=n - 3)
    return r, p


def main() -> None:
    # ---- load per-site TR (measured from headers, cached) ----
    tr_df = pd.read_csv(TR_CSV)
    tr = dict(zip(tr_df.site, tr_df.tr_s))
    print(f"[tr] per-site TR (s): "
          + ", ".join(f"{s}={tr[s]:.3f}" for s in sorted(tr)))

    # ---- load marginal results ----
    df = pd.read_csv(MARGINAL_CSV)
    reject_col = find_reject_col(df)
    df["is_diag"] = df.region_i == df.region_j

    total_reject = int(df[reject_col].sum())
    print(f"[file] total FDR rejections in file: {total_reject}")

    # ---- STEP 2: per-site-pair quantities ----
    sites = sorted(set(df.site_1) | set(df.site_2))
    # canonical pair key (unordered)
    def key(a, b):
        return tuple(sorted((a, b)))

    grp = df.groupby([df.site_1, df.site_2])
    rows = []
    for (s1, s2), g in grp:
        n_diag = int(g.loc[g.is_diag, reject_col].sum())
        n_off = int(g.loc[~g.is_diag, reject_col].sum())
        rows.append(dict(site_1=s1, site_2=s2,
                         n_diag_reject=n_diag, n_offdiag_reject=n_off))
    pairs = pd.DataFrame(rows)
    assert len(pairs) == 91, f"expected 91 pairs, got {len(pairs)}"

    pairs["delta_TR"] = [abs(tr[a] - tr[b]) for a, b in
                         zip(pairs.site_1, pairs.site_2)]
    N_SITE = dict(zip(tr_df.site, tr_df.N))
    pairs["n_min"] = [min(N_SITE[a], N_SITE[b]) for a, b in
                      zip(pairs.site_1, pairs.site_2)]

    # sanity check: rejections summed across pairs == file total
    summed = int(pairs.n_diag_reject.sum() + pairs.n_offdiag_reject.sum())
    print(f"[sanity] sum over pairs (diag+offdiag) = {summed}  "
          f"vs file total = {total_reject}  "
          f"-> {'MATCH' if summed == total_reject else 'MISMATCH'}")
    assert summed == total_reject

    pairs = pairs.sort_values(["site_1", "site_2"]).reset_index(drop=True)
    pairs.to_csv(RES / "test_c_pairs.csv", index=False)
    print(f"[saved] {RES/'test_c_pairs.csv'}  ({len(pairs)} pairs)")

    # ---- STEP 3: the test and its control ----
    rho_d, p_d = stats.spearmanr(pairs.delta_TR, pairs.n_diag_reject)
    rho_o, p_o = stats.spearmanr(pairs.delta_TR, pairs.n_offdiag_reject)
    print("\n=== STEP 3: Spearman across 91 pairs ===")
    print(f"(a) delta_TR vs n_diag_reject    : rho={rho_d:+.4f}  p={p_d:.4g}")
    print(f"(b) delta_TR vs n_offdiag_reject : rho={rho_o:+.4f}  p={p_o:.4g}")

    # ---- STEP 4: control for power ----
    rho_nm, p_nm = stats.spearmanr(pairs.n_min, pairs.n_diag_reject)
    pr, pp = partial_spearman(pairs.delta_TR.values,
                              pairs.n_diag_reject.values,
                              pairs.n_min.values)
    print("\n=== STEP 4: power control ===")
    print(f"    n_min vs n_diag_reject                  : rho={rho_nm:+.4f}  p={p_nm:.4g}")
    print(f"    partial(delta_TR, n_diag_reject | n_min): rho={pr:+.4f}  p={pp:.4g}")

    # ---- STEP 5: figure ----
    paper_style.apply()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL/2.4),
                                   sharex=True)
    axA.scatter(pairs.delta_TR, pairs.n_diag_reject, s=18, alpha=0.75,
                edgecolor="none", color="#c0392b")
    axA.set_xlabel(r"$\Delta$TR between sites (s)")
    axA.set_ylabel("diagonal FDR rejections\n(of 50)")
    axA.set_title(f"A  Diagonal (self-loops)\n"
                  rf"Spearman $\rho$={rho_d:+.2f}, p={p_d:.2g}")

    axB.scatter(pairs.delta_TR, pairs.n_offdiag_reject, s=18, alpha=0.75,
                edgecolor="none", color="#2c3e50")
    axB.set_xlabel(r"$\Delta$TR between sites (s)")
    axB.set_ylabel("off-diagonal FDR rejections\n(of 2450)")
    axB.set_title(f"B  Off-diagonal (control)\n"
                  rf"Spearman $\rho$={rho_o:+.2f}, p={p_o:.2g}")

    for ax in (axA, axB):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = OUT / "fig5_tr_vs_instability.png"
    pdf = OUT / "fig5_tr_vs_instability.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    print(f"\n[saved] {png}")
    print(f"[saved] {pdf}")

    # ---- verdict text ----
    print("\n=== VERDICT ===")
    print(f"diag:    rho={rho_d:+.4f} p={p_d:.4g}")
    print(f"offdiag: rho={rho_o:+.4f} p={p_o:.4g}")
    print(f"partial diag|n_min: rho={pr:+.4f} p={pp:.4g}")


if __name__ == "__main__":
    main()
