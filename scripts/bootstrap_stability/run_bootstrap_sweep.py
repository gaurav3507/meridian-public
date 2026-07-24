"""Bootstrap-resample-count (B) stability of the PRIMARY per-coefficient
diagnostics. Re-runs the site (ABIDE-I, CC200) and condition (AOMIC,
Schaefer-200) analyses UNCHANGED except for the bootstrap B in {100, 500, 1000},
to show the reported counts don't depend on the resample count.

Does NOT reimplement anything: it calls scripts/sensitivity_lag_alpha/
diagnostic_core.py (which itself calls scripts/test_b.py fit_var1/bootstrap
machinery and is verified to reproduce the primary Table-3 numbers EXACTLY at
B=100: site 1.06% / 49 diag / 0 off; condition 7.16% / 50 diag / 129 off). The
only swept argument is n_boot; lag p=1 and ridge alpha=1.0 are the primary
settings. Fixed bootstrap seed (20260628) throughout, so each (dataset, B) cell
is deterministic.

PRE-REGISTERED EXPECTATION (report honestly even if it fails): counts
essentially unchanged across B -- site stays 0/2450 off-diagonal, condition stays
~129/2450 off-diagonal (pooled). Small movement in the condition off-diagonal
count is fine (bootstrap Monte-Carlo noise shrinks as B grows); a large drift or
the site off-diagonal leaving 0 would be a real finding to report.

Resumable: per-cell atomic JSON with exists-check skip.
Output: results/bootstrap_stability/bootstrap_B_sweep.{md,csv}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "sensitivity_lag_alpha"))
sys.path.insert(0, str(ROOT / "scripts"))
import diagnostic_core as D                                 # reused, validated

BS = [100, 500, 1000]
P = 1
ALPHA = 1.0
OUT = ROOT / "results/bootstrap_stability"
CELLS = OUT / "_cells"
LABEL = {"site": "site (ABIDE-I, CC200)",
         "condition": "condition (AOMIC, Schaefer-200, pooled bootstrap)"}


def run_cell(dataset, B, abide=None, cond=None):
    tag = f"{dataset}__B{B}.json"
    path = CELLS / tag
    if path.exists():
        return json.loads(path.read_text())
    if dataset == "site":
        X, sid = abide
        res = D.site_diagnostic(X, sid, P, ALPHA, n_boot=B)
    else:
        rest, wm = cond
        res = D.condition_bootstrap_diagnostic(rest, wm, P, ALPHA, n_boot=B)
    res.update(dataset=dataset, B=B, p=P, alpha=ALPHA)
    CELLS.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(path)
    return res


def run_full():
    need_site = any(not (CELLS / f"site__B{B}.json").exists() for B in BS)
    need_cond = any(not (CELLS / f"condition__B{B}.json").exists() for B in BS)
    abide = D.load_abide() if need_site else None
    cond = D.load_condition() if need_cond else None
    print(f"bootstrap-B sweep: B in {BS} (p={P}, alpha={ALPHA})")
    for B in BS:
        run_cell("site", B, abide=abide); print(f"  site B={B} done")
        run_cell("condition", B, cond=cond); print(f"  condition B={B} done")
    aggregate()


def aggregate():
    rows = [json.loads(p.read_text()) for p in sorted(CELLS.glob("*__B*.json"))]
    df = (pd.DataFrame(rows)
          .sort_values(["dataset", "B"]).reset_index(drop=True))
    keep = ["dataset", "B", "overall_pct", "median_absz",
            "diag_unstable", "diag_total", "off_unstable", "off_total"]
    df[keep].to_csv(OUT / "bootstrap_B_sweep.csv", index=False)

    L = ["# Bootstrap resample-count (B) stability of the primary diagnostics", "",
         "Site (ABIDE-I, CC200) and condition (AOMIC, Schaefer-200) per-coefficient "
         "diagnostics re-run UNCHANGED except for the bootstrap B. Ridge VAR(1) "
         "alpha=1.0, per-coefficient two-sample Wald z, BH-FDR q=0.05; site uses "
         "the >20%-of-91-pairs unstable rule, condition uses FDR 0.05 on the "
         "single rest-vs-WM contrast. Fixed seed 20260628. B=100 is the primary "
         "reported setting.", ""]
    for ds in ("site", "condition"):
        sub = df[df.dataset == ds]
        dt = int(sub.diag_total.iloc[0]); ot = int(sub.off_total.iloc[0])
        L += [f"## {LABEL[ds]}", "",
              f"| bootstrap B | overall reject | median |d|/SE | "
              f"diagonal unstable (of {dt}) | off-diagonal unstable (of {ot}) |",
              "|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            star = "  **(primary)**" if int(r.B) == 100 else ""
            L.append(f"| {int(r.B)}{star} | {r.overall_pct:.2f}% | "
                     f"{r.median_absz:.3f} | {int(r.diag_unstable)}/{dt} | "
                     f"{int(r.off_unstable)}/{ot} |")
        L.append("")
    L += ["## Verdict", "", _verdict(df), ""]
    tmp = Path(str(OUT / "bootstrap_B_sweep.md") + ".tmp")
    tmp.write_text("\n".join(L) + "\n")
    tmp.replace(OUT / "bootstrap_B_sweep.md")
    print(f"\nwrote {OUT/'bootstrap_B_sweep.md'} and .csv")


def _verdict(df):
    site = df[df.dataset == "site"]
    cond = df[df.dataset == "condition"]
    s_off = site.off_unstable.tolist()
    c_off = cond.off_unstable.tolist()
    s_diag = site.diag_unstable.tolist()
    c_diag = cond.diag_unstable.tolist()
    site_flat = max(s_off) - min(s_off) <= 1 and max(s_diag) - min(s_diag) <= 1
    cond_flat = (max(c_off) - min(c_off)) <= 8 and max(c_diag) - min(c_diag) <= 1
    stable = site_flat and cond_flat and max(s_off) == 0
    tag = ("STABLE across B: the reported counts do not depend on the bootstrap "
           "resample count." if stable else
           "counts move more than expected across B -- see the tables; report "
           "as-is.")
    return (f"Site off-diagonal across B={BS}: {s_off} (diagonal {s_diag}). "
            f"Condition off-diagonal across B: {c_off} (diagonal {c_diag}).\n\n"
            f"{tag}")


def run_smoke():
    print("SMOKE (condition, B=100 -- one real sweep cell) ...")
    cond = D.load_condition()
    res = run_cell("condition", 100, cond=cond)
    print(f"  overall {res['overall_pct']:.2f}%  median|z| {res['median_absz']:.3f}  "
          f"diag {res['diag_unstable']}/{res['diag_total']}  "
          f"off {res['off_unstable']}/{res['off_total']}   "
          f"[primary: 7.16%, 50/50, 129/2450]")
    print("SMOKE OK. This cell is a real sweep cell (written to _cells/), so the "
          "full tmux run resumes from it.")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        run_smoke()
    elif "--aggregate" in sys.argv:
        aggregate()
    else:
        run_full()
