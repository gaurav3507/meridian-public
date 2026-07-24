"""M1: VAR(p) lag-order sensitivity of the per-coefficient stability diagnostic.

Refits the EXISTING diagnostic (calls diagnostic_core, which calls scripts/test_b)
at VAR(1), VAR(2), VAR(3) on BOTH datasets, ridge alpha fixed at 1.0 (alpha is
swept separately in I3). Generalized partition at higher lags: DIAGONAL =
{A_k[i,i]} across all lags, OFF-DIAGONAL = {A_k[i,j], i!=j} across all lags.

PRE-REGISTERED EXPECTATIONS (interpret honestly, do NOT tune):
  * the diagonal-dominant (site) vs off-diagonal-present (condition) dissociation
    holds at p=2 and p=3;
  * BIC likely selects low order for fMRI at TR~2s;
  * per-subject high-lag fits (condition paired) are noisy -> pooled bootstrap is
    primary at higher lags; paired reported but flagged.

Resumable: per-cell atomic JSON with exists-check skip. Writes
results/sensitivity/var_p_sensitivity.{md,csv}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diagnostic_core as D

ALPHA = 1.0
PS = [1, 2, 3]
OUT = D.ROOT / "results/sensitivity"
CELLS = OUT / "_cells_m1"
CELLS.mkdir(parents=True, exist_ok=True)

PRIMARY = {"ABIDE": "site", "AOMIC_bootstrap": "condition (unpaired bootstrap)"}
FLAGGED = {"AOMIC_paired": "condition (per-subject paired t) -- FLAGGED noisy at p>1"}


def _cell(kind, p):
    return CELLS / f"{kind}__p{p}.json"


def run_diag_cell(kind, p, abide=None, cond=None, n_boot=D.N_BOOT):
    path = _cell(kind, p)
    if path.exists():
        return json.loads(path.read_text())
    if kind == "ABIDE":
        X, sid = abide
        res = D.site_diagnostic(X, sid, p, ALPHA, n_boot=n_boot)
    elif kind == "AOMIC_bootstrap":
        rest, wm = cond
        res = D.condition_bootstrap_diagnostic(rest, wm, p, ALPHA, n_boot=n_boot)
    elif kind == "AOMIC_paired":
        rest, wm = cond
        res = D.condition_paired_diagnostic(rest, wm, p, ALPHA)
    else:
        raise ValueError(kind)
    res.update(kind=kind, p=p, alpha=ALPHA)
    D.save_atomic(path, res)
    return res


def run_aic_bic_cell(dataset, abide=None, cond=None):
    path = CELLS / f"aicbic__{dataset}.json"
    if path.exists():
        return json.loads(path.read_text())
    if dataset == "ABIDE":
        X, sid = abide
        groups = [X[sid == s] for s in sorted(np.unique(sid).tolist())]
    else:
        rest, wm = cond
        groups = [rest, wm]
    aic_sel, bic_sel, aic_tot, bic_tot = D.select_orders(groups, ps=tuple(PS))
    res = dict(dataset=dataset, aic_selected=aic_sel, bic_selected=bic_sel,
               aic_totals=aic_tot, bic_totals=bic_tot)
    D.save_atomic(path, res)
    return res


def run_full():
    need_abide = any(not _cell("ABIDE", p).exists() for p in PS) or \
        not (CELLS / "aicbic__ABIDE.json").exists()
    need_cond = any(not _cell(k, p).exists() for k in
                    ("AOMIC_bootstrap", "AOMIC_paired") for p in PS) or \
        not (CELLS / "aicbic__AOMIC.json").exists()
    abide = D.load_abide() if need_abide else None
    cond = D.load_condition() if need_cond else None
    print(f"M1: alpha={ALPHA}, lags={PS}. Loading "
          f"{'ABIDE ' if need_abide else ''}{'AOMIC' if need_cond else ''}"
          f"{'(all cached)' if not (need_abide or need_cond) else ''}")

    for p in PS:
        run_diag_cell("ABIDE", p, abide=abide); print(f"  ABIDE p={p} done")
        run_diag_cell("AOMIC_bootstrap", p, cond=cond); print(f"  AOMIC bootstrap p={p} done")
        run_diag_cell("AOMIC_paired", p, cond=cond); print(f"  AOMIC paired p={p} done")
    run_aic_bic_cell("ABIDE", abide=abide)
    run_aic_bic_cell("AOMIC", cond=cond)
    aggregate()


def aggregate():
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*__p*.json")]
    df = pd.DataFrame(rows)
    ab = json.loads((CELLS / "aicbic__ABIDE.json").read_text())
    ao = json.loads((CELLS / "aicbic__AOMIC.json").read_text())
    order = {"ABIDE": ab, "AOMIC_bootstrap": ao, "AOMIC_paired": ao}

    df["dataset_label"] = df["kind"].map({**PRIMARY, **FLAGGED})
    df["aic_order"] = df["kind"].map(lambda k: order[k]["aic_selected"])
    df["bic_order"] = df["kind"].map(lambda k: order[k]["bic_selected"])
    df = df.sort_values(["kind", "p"]).reset_index(drop=True)
    df.to_csv(OUT / "var_p_sensitivity.csv", index=False)

    def rowline(r):
        return (f"| {r['dataset_label']} | {int(r['p'])} | {r['overall_pct']:.2f}% | "
                f"{r['diag_unstable']}/{r['diag_total']} | "
                f"{r['off_unstable']}/{r['off_total']} | {int(r['aic_order'])} | "
                f"{int(r['bic_order'])} |")

    L = ["# M1: VAR(p) lag-order sensitivity", "",
         f"Ridge alpha fixed at {ALPHA} (swept in I3). 100-bootstrap subject "
         "resampling, per-coefficient two-sample Wald z, BH-FDR q=0.05; site uses "
         "the >20%-of-91-pairs unstable rule, condition uses FDR 0.05 on the single "
         "rest-vs-WM contrast. Same data/harmonization/standardization/top-50 "
         "selection as the primary analyses (CC200 ABIDE, Schaefer-200 AOMIC).", "",
         "**Generalized partition at higher lags:** with lag matrices A_1..A_p, "
         "DIAGONAL = all self terms {A_k[i,i]} (p x 50 total), OFF-DIAGONAL = all "
         "cross terms {A_k[i,j], i!=j} (p x 50 x 49 total).", "",
         "## Primary (pooled bootstrap)", "",
         "| dataset | lag p | overall reject | diagonal unstable | off-diagonal unstable | AIC order | BIC order |",
         "|---|---|---|---|---|---|---|"]
    for _, r in df[df.kind.isin(list(PRIMARY))].iterrows():
        L.append(rowline(r))
    L += ["", "## Flagged sensitivity: condition per-subject paired t",
          "", "Per-subject VAR(p) fits are underdetermined at p>1 (T=160 vs p x 50 "
          "predictors); ridge alpha=1 regularizes but treat as NOISY. Primary "
          "condition number is the pooled bootstrap above.", "",
          "| dataset | lag p | overall reject | diagonal unstable | off-diagonal unstable | AIC order | BIC order |",
          "|---|---|---|---|---|---|---|"]
    for _, r in df[df.kind.isin(list(FLAGGED))].iterrows():
        L.append(rowline(r))
    L += ["", "## Lag-order selection (pooled OLS VAR, Lutkepohl AIC/BIC)", "",
          f"- **ABIDE**: AIC selects p={ab['aic_selected']}, BIC selects "
          f"p={ab['bic_selected']}  (AIC totals {_fmt(ab['aic_totals'])}; "
          f"BIC totals {_fmt(ab['bic_totals'])})",
          f"- **AOMIC**: AIC selects p={ao['aic_selected']}, BIC selects "
          f"p={ao['bic_selected']}  (AIC totals {_fmt(ao['aic_totals'])}; "
          f"BIC totals {_fmt(ao['bic_totals'])})", "",
          "## Dissociation check", "",
          _dissociation_verdict(df)]
    (OUT / "var_p_sensitivity.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT/'var_p_sensitivity.md'} and .csv")


def _fmt(totals):
    return ", ".join(f"p{p}={v:.1f}" for p, v in sorted(
        ((int(k), val) for k, val in totals.items())))


def _dissociation_verdict(df):
    site = df[df.kind == "ABIDE"].set_index("p")
    cond = df[df.kind == "AOMIC_bootstrap"].set_index("p")
    lines = []
    for p in PS:
        s_off = int(site.loc[p, "off_unstable"]); s_offtot = int(site.loc[p, "off_total"])
        c_off = int(cond.loc[p, "off_unstable"]); c_offtot = int(cond.loc[p, "off_total"])
        lines.append(f"p={p}: site off-diagonal {s_off}/{s_offtot}, "
                     f"condition off-diagonal {c_off}/{c_offtot}")
    holds = all(int(cond.loc[p, "off_unstable"]) > int(site.loc[p, "off_unstable"])
                for p in PS)
    tag = ("HOLDS at every lag (condition off-diagonal > site off-diagonal)."
           if holds else "DOES NOT hold at every lag -- inspect the rows above.")
    return "The site (diagonal-dominant) vs condition (off-diagonal-present) " \
           f"dissociation {tag}\n\n" + "\n".join(f"- {x}" for x in lines)


def run_smoke():
    print("SMOKE (ABIDE site, p=2, alpha=1, 10 bootstraps) ...")
    abide = D.load_abide()
    res = D.site_diagnostic(*abide, 2, ALPHA, n_boot=10)
    print(f"  overall {res['overall_pct']:.2f}%  median|z| {res['median_absz']:.3f}  "
          f"diag {res['diag_unstable']}/{res['diag_total']}  "
          f"off {res['off_unstable']}/{res['off_total']}  ({res['n_pairs']} pairs)")
    print("SMOKE OK (not saved to cells; full run uses 100 bootstraps).")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        run_smoke()
    elif "--aggregate" in sys.argv:
        aggregate()
    else:
        run_full()
