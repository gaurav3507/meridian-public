"""Within-MISMATCHED graded-TR check (no new analysis, no re-fitting).

Among ONLY the 54 pairs that already differ in TR (delta_TR > 0), is the effect
graded in TR magnitude? Spearman(delta_TR, n_diag_reject) with n_offdiag_reject
as the control. A graded diagonal correlation here rules out a binary
matched-vs-mismatched "shared protocol" explanation in favour of a TR-magnitude
effect.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

TR = {"PITT": 1.500, "LEUVEN_2": 1.652, "KKI": 2.500, "UCLA_1": 3.000,
      "MAX_MUN": 3.000, "NYU": 2.000, "UM_1": 2.000, "UM_2": 2.000,
      "USM": 2.000, "YALE": 2.000, "TRINITY": 2.000, "STANFORD": 2.000,
      "SDSU": 2.000, "CALTECH": 2.000}


def main() -> None:
    df = pd.read_csv(RES / "test_b_marginal.csv")
    df["is_diag"] = df.region_i == df.region_j
    rows = []
    for (s1, s2), g in df.groupby(["site_1", "site_2"]):
        rows.append(dict(
            delta_TR=abs(TR[s1] - TR[s2]),
            n_diag_reject=int(g.loc[g.is_diag, "reject_fdr"].sum()),
            n_offdiag_reject=int(g.loc[~g.is_diag, "reject_fdr"].sum()),
        ))
    pairs = pd.DataFrame(rows)
    mis = pairs[pairs.delta_TR > 0]
    print(f"MISMATCHED pairs (delta_TR > 0): {len(mis)}")
    print(f"distinct delta_TR values: {sorted(mis.delta_TR.unique())}")

    rho_d, p_d = stats.spearmanr(mis.delta_TR, mis.n_diag_reject)
    rho_o, p_o = stats.spearmanr(mis.delta_TR, mis.n_offdiag_reject)
    print("\nWithin MISMATCHED, Spearman:")
    print(f"  delta_TR vs n_diag_reject    [TEST]   : rho={rho_d:+.4f}  p={p_d:.4g}")
    print(f"  delta_TR vs n_offdiag_reject [CONTROL]: rho={rho_o:+.4f}  p={p_o:.4g}")


if __name__ == "__main__":
    main()
