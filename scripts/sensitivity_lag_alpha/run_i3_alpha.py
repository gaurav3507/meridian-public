"""I3: ridge-alpha sensitivity of the per-coefficient stability diagnostic.

Refits the EXISTING diagnostic at ridge alpha in {0.1, 1.0, 10.0} on BOTH datasets
at VAR(1) (the primary lag). alpha=1.0 is the primary/reported setting, flanked by
0.1 and 10.0. Everything else identical to the primary analysis.

NOTE: the simulation predicted near-invariance because ridge shrinkage largely
cancels in the standardized Wald z (z = d/se, both scaled together). On REAL data
with subject heterogeneity expect CLOSE but not necessarily identical counts.
Report the actual counts; do not assume invariance.

PRE-REGISTERED EXPECTATION (honest): counts close across alpha, dissociation stable.

Resumable: per-cell atomic JSON with exists-check skip. Writes
results/sensitivity/ridge_alpha_sensitivity.{md,csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_core as D

P = 1
ALPHAS = [0.1, 1.0, 10.0]
PRIMARY_ALPHA = 1.0
OUT = D.ROOT / "results/sensitivity"
CELLS = OUT / "_cells_i3"
CELLS.mkdir(parents=True, exist_ok=True)

LABEL = {"ABIDE": "site (ABIDE-I, CC200)",
         "AOMIC": "condition (AOMIC, Schaefer-200, unpaired bootstrap)"}


def _cell(kind, a):
    return CELLS / f"{kind}__a{a}.json"


def run_cell(kind, a, abide=None, cond=None, n_boot=D.N_BOOT):
    path = _cell(kind, a)
    if path.exists():
        return json.loads(path.read_text())
    if kind == "ABIDE":
        X, sid = abide
        res = D.site_diagnostic(X, sid, P, a, n_boot=n_boot)
    else:
        rest, wm = cond
        res = D.condition_bootstrap_diagnostic(rest, wm, P, a, n_boot=n_boot)
    res.update(kind=kind, alpha=a, p=P)
    D.save_atomic(path, res)
    return res


def run_full():
    need_abide = any(not _cell("ABIDE", a).exists() for a in ALPHAS)
    need_cond = any(not _cell("AOMIC", a).exists() for a in ALPHAS)
    abide = D.load_abide() if need_abide else None
    cond = D.load_condition() if need_cond else None
    print(f"I3: VAR(1), alphas={ALPHAS}")
    for a in ALPHAS:
        run_cell("ABIDE", a, abide=abide); print(f"  ABIDE alpha={a} done")
        run_cell("AOMIC", a, cond=cond); print(f"  AOMIC alpha={a} done")
    aggregate()


def aggregate():
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*__a*.json")]
    df = pd.DataFrame(rows).sort_values(["kind", "alpha"]).reset_index(drop=True)
    df.to_csv(OUT / "ridge_alpha_sensitivity.csv", index=False)

    L = ["# I3: ridge-alpha sensitivity (VAR(1))", "",
         "Per-coefficient two-sample Wald z, BH-FDR q=0.05; site uses the "
         ">20%-of-91-pairs unstable rule, condition uses FDR 0.05 on the single "
         "rest-vs-WM contrast. Same data/harmonization/standardization/top-50 "
         "selection as the primary analyses. Only ridge alpha changes. "
         "alpha=1.0 is the primary/reported setting.", "",
         "Diagonal = 50 self-loops; off-diagonal = 2450 cross terms.", ""]
    for kind in ("ABIDE", "AOMIC"):
        sub = df[df.kind == kind]
        L += [f"## {LABEL[kind]}", "",
              "| ridge alpha | overall reject | diagonal unstable | off-diagonal unstable |",
              "|---|---|---|---|"]
        for _, r in sub.iterrows():
            star = "  **(primary)**" if r["alpha"] == PRIMARY_ALPHA else ""
            L.append(f"| {r['alpha']}{star} | {r['overall_pct']:.2f}% | "
                     f"{r['diag_unstable']}/{r['diag_total']} | "
                     f"{r['off_unstable']}/{r['off_total']} |")
        L.append("")
    L += ["## Stability across alpha", "", _verdict(df)]
    (OUT / "ridge_alpha_sensitivity.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT/'ridge_alpha_sensitivity.md'} and .csv")


def _verdict(df):
    site = df[df.kind == "ABIDE"]
    cond = df[df.kind == "AOMIC"]
    s_off = site["off_unstable"].tolist()
    c_off = cond["off_unstable"].tolist()
    stable = (max(s_off) - min(s_off) <= 2) and all(c > max(s_off) for c in c_off)
    tag = ("STABLE: site off-diagonal stays ~flat and far below condition "
           "off-diagonal across all alpha." if stable else
           "see the rows above -- counts vary more than expected; report as-is.")
    return (f"Site off-diagonal unstable across alpha {ALPHAS}: {s_off}. "
            f"Condition off-diagonal across alpha: {c_off}.\n\n"
            f"The dissociation is {tag}")


def run_smoke():
    print("SMOKE (AOMIC condition, alpha=0.1, VAR(1), 10 bootstraps) ...")
    cond = D.load_condition()
    res = D.condition_bootstrap_diagnostic(*cond, P, 0.1, n_boot=10)
    print(f"  overall {res['overall_pct']:.2f}%  median|z| {res['median_absz']:.3f}  "
          f"diag {res['diag_unstable']}/{res['diag_total']}  "
          f"off {res['off_unstable']}/{res['off_total']}")
    print("SMOKE OK (not saved; full run uses 100 bootstraps).")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        run_smoke()
    elif "--aggregate" in sys.argv:
        aggregate()
    else:
        run_full()
