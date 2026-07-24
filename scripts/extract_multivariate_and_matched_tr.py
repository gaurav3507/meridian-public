"""Two small extractions from existing results (no new analysis, no re-fitting).

JOB 1 — multivariate (Hotelling T^2) VAR-row rejection rate, for the Discussion.
        results/test_b_results.csv : one row per (region, site pair),
        50 regions x 91 pairs = 4,550 rows.

JOB 2 — matched vs mismatched TR comparison, using measured per-site TRs.
        results/test_b_marginal.csv : 2,500 coeffs x 91 pairs. Split pairs into
        MATCHED (delta_TR == 0) vs MISMATCHED (delta_TR > 0); compare diagonal
        (test) and off-diagonal (control) rejection counts with Mann-Whitney U.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Measured per-site TR (seconds), from the NIfTI headers (already established).
TR = {
    "PITT": 1.500, "LEUVEN_2": 1.652, "KKI": 2.500,
    "UCLA_1": 3.000, "MAX_MUN": 3.000,
    "NYU": 2.000, "UM_1": 2.000, "UM_2": 2.000, "USM": 2.000, "YALE": 2.000,
    "TRINITY": 2.000, "STANFORD": 2.000, "SDSU": 2.000, "CALTECH": 2.000,
}
# Per-site analysis N (sample size).
N_SITE = {"NYU": 169, "UM_1": 81, "USM": 60, "UCLA_1": 54, "YALE": 46,
          "PITT": 44, "TRINITY": 44, "MAX_MUN": 41, "KKI": 39, "STANFORD": 36,
          "CALTECH": 35, "SDSU": 33, "LEUVEN_2": 30, "UM_2": 30}


def job1() -> None:
    print("=" * 60)
    print("JOB 1 — multivariate (Hotelling T^2) rejection rate")
    print("=" * 60)
    df = pd.read_csv(RES / "test_b_results.csv")
    n_total = len(df)
    n_reject = int(df["reject_fdr"].sum())
    rate = 100.0 * n_reject / n_total
    print(f"  rows (region x site-pair combinations tested): {n_total}")
    print(f"  expected 50 x 91 = {50*91}  -> "
          f"{'MATCH' if n_total == 50*91 else 'MISMATCH'}")
    print(f"  reject at FDR q=0.05                          : {n_reject}")
    print(f"  rejection rate                               : {rate:.2f}%")


def job2() -> None:
    print("\n" + "=" * 60)
    print("JOB 2 — matched vs mismatched TR")
    print("=" * 60)
    df = pd.read_csv(RES / "test_b_marginal.csv")
    df["is_diag"] = df.region_i == df.region_j

    rows = []
    for (s1, s2), g in df.groupby(["site_1", "site_2"]):
        rows.append(dict(
            site_1=s1, site_2=s2,
            n_diag_reject=int(g.loc[g.is_diag, "reject_fdr"].sum()),
            n_offdiag_reject=int(g.loc[~g.is_diag, "reject_fdr"].sum()),
            delta_TR=abs(TR[s1] - TR[s2]),
            n_min=min(N_SITE[s1], N_SITE[s2]),
        ))
    pairs = pd.DataFrame(rows)
    assert len(pairs) == 91, f"expected 91 pairs, got {len(pairs)}"

    matched = pairs[pairs.delta_TR == 0]
    mismatched = pairs[pairs.delta_TR > 0]
    print(f"  MATCHED    (delta_TR == 0): {len(matched)} pairs")
    print(f"  MISMATCHED (delta_TR  > 0): {len(mismatched)} pairs")

    metrics = ["n_diag_reject", "n_offdiag_reject", "n_min"]
    print(f"\n  {'metric':17s} {'group':11s} {'median':>8s} {'mean':>8s}")
    for m in metrics:
        for name, grp in [("MATCHED", matched), ("MISMATCHED", mismatched)]:
            print(f"  {m:17s} {name:11s} {grp[m].median():8.2f} "
                  f"{grp[m].mean():8.2f}")

    print("\n  Mann-Whitney U (two-sided), MATCHED vs MISMATCHED:")
    for label, m in [("(a) n_diag_reject   [TEST]   ", "n_diag_reject"),
                     ("(b) n_offdiag_reject [CONTROL]", "n_offdiag_reject")]:
        u, p = stats.mannwhitneyu(matched[m], mismatched[m],
                                  alternative="two-sided")
        print(f"    {label}: U={u:.1f}  p={p:.4g}  "
              f"(median matched={matched[m].median():.1f}, "
              f"mismatched={mismatched[m].median():.1f})")


def main() -> None:
    job1()
    job2()


if __name__ == "__main__":
    main()
