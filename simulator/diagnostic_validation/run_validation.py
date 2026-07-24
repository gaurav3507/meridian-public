"""Simulation validation of the VAR-stability diagnostic (JNM methods paper).

The paper's headline is a NULL: across real ABIDE-I sites, 0 / 2450 off-diagonal
VAR(1) coefficients are unstable. This suite injects KNOWN shifts into synthetic
data and shows the diagnostic recovers WHERE they land, so the null is a real
absence rather than a blind/underpowered off-diagonal channel. It doubles as the
ridge-alpha and sample-size sensitivity analysis.

The diagnostic itself is NOT reimplemented: we CALL scripts/test_b.py
(fit_var1, bootstrap_site, approx_hotelling_t2) and apply the per-coefficient
two-sample Wald z + BH-FDR exactly as scripts/test_b_marginal.py does (the same
math reused by abide_robustness / condition_sms / region_sensitivity).

FAMILY 1 (direct-VAR, clean ground truth): p=50, T=116 (ABIDE short-series case).
  C1 diagonal-only shift; C2 off-diagonal-only (THE power test that defends the
  null); C3 both; C4 null (identical coefficients, only noise variance + N
  imbalance differ -> false-positive test). Plus a 10-level site-like design that
  checks the ">20% of pairs = unstable" rule recovers an injected sparse support.
FAMILY 2 (HRF realism): C1/C2/C3 ground truth passed through the simulator's
  Glover-HRF convolution (TR=2.0) before fitting; asks whether haemodynamic
  filtering creates spurious off-diagonal instability or washes out true
  off-diagonal shifts. Localisation must survive filtering, not exact recovery.

PRE-REGISTERED EXPECTATIONS (interpret honestly; DO NOT tune to hit them):
  * C4 null: FP rate <= ~0.05. If inflated, that is a finding to report.
  * C2: power climbs with magnitude and N and clearly exceeds the FP rate --
    this is what says the off-diagonal channel is not blind.
  * C1/C3: clean diagonal/off localisation in the direct-VAR family.
  * Family 2: report whatever happens under HRF filtering, even if it degrades.

Sweeps: magnitude (curve), N in {30,60,120}, ridge alpha in {0.1,1.0,10.0}, 5
seeds/cell (mean +- sd). Per-cell results saved atomically (.tmp -> rename) with
skip-existing resume. Run --smoke first (one tiny cell) to prove end-to-end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
ROOT = SIM.parent
for pth in (str(HERE), str(SIM), str(ROOT / "scripts")):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from test_b import fit_var1, approx_hotelling_t2               # reused diagnostic
import generate as G

RESULTS = HERE / "results"
CELLS = RESULTS / "cells"
FIGURES = HERE / "figures"
for d in (RESULTS, CELLS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

# ---- fixed experiment constants (NOT swept; not tuned) --------------------- #
P = 50
T = 116
N_BOOT = 100
FDR_Q = 0.05
UNSTABLE_THRESH = 0.20            # ">20% of pairs" rule for the multi-level design
BASE_A_SEED = 7                   # ground-truth base matrix (fixed)
SUPPORT_SEED = 11                 # injected support (fixed)
N_DIAG_INJECT = 8
N_OFF_INJECT = 12
C4_NOISE_A, C4_NOISE_B = 1.0, 1.6   # C4 nuisance: unequal innovation variance
C4_N_IMBALANCE = 0.5                # level B gets half the subjects of level A

# ---- full sweep grid ------------------------------------------------------- #
# Injected shifts are in VAR-coefficient units. The paper's real median
# significant off-diagonal condition shift is 0.048, so the grid STRADDLES it
# (0.02 below, 0.05 at it) to show where detection kicks in rather than saturate.
MAGS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
NS = [30, 60, 120]
ALPHAS = [0.1, 1.0, 10.0]
SEEDS = [0, 1, 2, 3, 4]
F2_MAGS = [0.05, 0.10, 0.20, 0.40]   # HRF comparison; includes the 0.048 anchor
F2_N = 60
N_LEVELS = 10
MAGS_MULTI = [0.02, 0.05, 0.10, 0.15]   # 10-level; smaller (per-level noise stacks)


# --------------------------------------------------------------------------- #
# diagnostic wrappers (call the reused primitives; mirror test_b_marginal)
# --------------------------------------------------------------------------- #
def _bootstrap_alpha(X, n_boot, rng, alpha):
    """Subject-resampling bootstrap of ridge VAR(1), mirroring
    test_b.bootstrap_site but passing the ridge alpha EXPLICITLY. (bootstrap_site
    can't be used for the alpha sweep: it calls fit_var1 with no alpha, so Python
    binds it to the module default 1.0 at definition time.) Reuses the real
    fit_var1; identical to bootstrap_site when alpha=1.0."""
    n, _, p = X.shape
    boots = np.empty((n_boot, p, p), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = fit_var1(X[idx], alpha=alpha)
    return boots


def two_level_diagnostic(XA, XB, alpha, seed):
    """Per-coefficient two-sample Wald z + BH-FDR (exactly test_b_marginal)."""
    rng = np.random.default_rng(seed)
    bA = _bootstrap_alpha(G.zscore_subjects(XA), N_BOOT, rng, alpha)
    bB = _bootstrap_alpha(G.zscore_subjects(XB), N_BOOT, rng, alpha)
    d = bA.mean(0) - bB.mean(0)
    se = np.sqrt(bA.var(0, ddof=1) + bB.var(0, ddof=1))
    z = np.where(se > 0, d / se, 0.0)
    p = 2.0 * norm.sf(np.abs(z))
    reject = multipletests(p.ravel(), alpha=FDR_Q, method="fdr_bh")[0].reshape(P, P)
    return reject, np.abs(z), bA, bB


def multi_level_diagnostic(levels_X, alpha, seed):
    """10-level design: per-pair Wald z, BH-FDR jointly over all pairs, then the
    '>20% of pairs' instability rule. Returns (unstable_mask, rej_frac)."""
    rng = np.random.default_rng(seed)
    boots = [_bootstrap_alpha(G.zscore_subjects(X), N_BOOT, rng, alpha)
             for X in levels_X]
    means = [b.mean(0) for b in boots]
    vars = [b.var(0, ddof=1) for b in boots]
    pairs = list(combinations(range(len(levels_X)), 2))
    Pmat = np.empty((len(pairs), P, P))
    for k, (a, b) in enumerate(pairs):
        d = means[a] - means[b]
        se = np.sqrt(vars[a] + vars[b])
        z = np.where(se > 0, d / se, 0.0)
        Pmat[k] = 2.0 * norm.sf(np.abs(z))
    reject = multipletests(Pmat.ravel(), alpha=FDR_Q,
                           method="fdr_bh")[0].reshape(Pmat.shape)
    rej_frac = reject.mean(0)
    return rej_frac > UNSTABLE_THRESH, rej_frac


def multivariate_reject(bA, bB, alpha):
    """Row-wise Hotelling T^2 (approx_hotelling_t2) per region + BH-FDR, the
    multivariate test from test_b.py. Returns rejection rate over P regions."""
    p = np.empty(P)
    for i in range(P):
        T2 = approx_hotelling_t2(bA[:, i, :], bB[:, i, :])
        p[i] = chi2.sf(T2, df=P)
    reject = multipletests(p, alpha=FDR_Q, method="fdr_bh")[0]
    return float(reject.mean())


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _eye(p=P):
    return np.eye(p, dtype=bool)


def partition_rates(reject):
    e = _eye()
    return dict(overall=float(reject.mean()),
                diag=float(reject[e].mean()),
                off=float(reject[~e].mean()))


def power_fpr(reject, injected):
    """TPR on injected coefficients; FPR on non-injected coefficients."""
    tpr = float(reject[injected].mean()) if injected.any() else float("nan")
    noninj = ~injected
    fpr = float(reject[noninj].mean()) if noninj.any() else float("nan")
    return tpr, fpr


def localization(reject, injected):
    """Confusion of detected coefficients: injected-partition x detected-
    partition, plus localisation accuracy (detected on an injected partition)."""
    e = _eye()
    det = reject
    inj_diag = injected & e
    inj_off = injected & ~e
    conf = dict(
        det_diag_injectedDiag=int((det & inj_diag).sum()),
        det_off_injectedOff=int((det & inj_off).sum()),
        det_diag_notInjected=int((det & e & ~injected).sum()),
        det_off_notInjected=int((det & ~e & ~injected).sum()),
        n_detected=int(det.sum()),
        n_inj_diag=int(inj_diag.sum()), n_inj_off=int(inj_off.sum()))
    # correct partition = the partition(s) where a shift was injected
    correct = np.zeros_like(reject)
    if inj_diag.any():
        correct |= e
    if inj_off.any():
        correct |= ~e
    n_det = det.sum()
    conf["localization_acc"] = float((det & correct).sum() / n_det) if n_det else float("nan")
    return conf


# --------------------------------------------------------------------------- #
# per-cell runners (atomic save + skip-existing resume)
# --------------------------------------------------------------------------- #
def _save_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _cell_path(tag):
    return CELLS / f"{tag}.json"


def run_two_level_cell(base, support, condition, mag, N, alpha, seed, hrf=False):
    tag = (f"{'F2' if hrf else 'F1'}__{condition}__mag{mag:.2f}"
           f"__N{N}__a{alpha}__s{seed}" + ("__hrf" if hrf else ""))
    path = _cell_path(tag)
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(1000 + seed)      # data rng, separate from boot
    if condition == "C4":
        M_A, M_B, injected = base, base, np.zeros((P, P), bool)
        nA, nB = N, max(5, int(round(N * C4_N_IMBALANCE)))
        XA = G.simulate_var(M_A, nA, T, C4_NOISE_A, rng)
        XB = G.simulate_var(M_B, nB, T, C4_NOISE_B, rng)
    else:
        M_A, M_B, injected = G.make_condition_matrix(base, condition, mag, support)
        XA = G.simulate_var(M_A, N, T, 1.0, rng)
        XB = G.simulate_var(M_B, N, T, 1.0, rng)
    if hrf:
        XA, XB = G.apply_hrf(XA), G.apply_hrf(XB)
    reject, absz, bA, bB = two_level_diagnostic(XA, XB, alpha, seed=2000 + seed)
    rates = partition_rates(reject)
    tpr, fpr = power_fpr(reject, injected)
    conf = localization(reject, injected)
    out = dict(tag=tag, family=("F2" if hrf else "F1"), condition=condition,
               magnitude=mag, N=N, alpha=alpha, seed=seed, hrf=hrf,
               median_absz=float(np.median(absz)),
               **{f"reject_{k}": v for k, v in rates.items()},
               tpr=tpr, fpr=fpr, **conf)
    if condition == "C4" and not hrf:
        out["multivariate_reject_rate"] = multivariate_reject(bA, bB, alpha)
    _save_atomic(path, out)
    return out


def run_multilevel_cell(base, support, mag, N, alpha, seed):
    tag = f"F1multi__mag{mag:.2f}__N{N}__a{alpha}__s{seed}"
    path = _cell_path(tag)
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(3000 + seed)
    mats, injected = G.make_multilevel_matrices(base, N_LEVELS, mag, support,
                                                seed=500 + seed)
    levels = [G.simulate_var(M, N, T, 1.0, rng) for M in mats]
    unstable, rej_frac = multi_level_diagnostic(levels, alpha, seed=4000 + seed)
    e = _eye()
    tp = int((unstable & injected).sum())
    fp = int((unstable & ~injected).sum())
    fn = int((~unstable & injected).sum())
    out = dict(tag=tag, family="F1multi", magnitude=mag, N=N, alpha=alpha,
               seed=seed, n_levels=N_LEVELS,
               unstable_overall=int(unstable.sum()),
               unstable_diag=int(unstable[e].sum()),
               unstable_off=int(unstable[~e].sum()),
               injected_total=int(injected.sum()),
               recovered_tp=tp, spurious_fp=fp, missed_fn=fn,
               recall=float(tp / injected.sum()) if injected.sum() else float("nan"),
               precision=float(tp / (tp + fp)) if (tp + fp) else float("nan"))
    _save_atomic(path, out)
    return out


# --------------------------------------------------------------------------- #
# aggregation + figures
# --------------------------------------------------------------------------- #
def aggregate_and_report():
    rows = [json.loads(p.read_text()) for p in sorted(CELLS.glob("*.json"))]
    if not rows:
        print("no cells found"); return
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "all_cells.csv", index=False)

    two = df[df.family.isin(["F1", "F2"]) & df.condition.notna()].copy()
    multi = df[df.family == "F1multi"].copy()

    def msd(g, col):
        return f"{g[col].mean():.3f} ± {g[col].std(ddof=0):.3f}"

    lines = ["# Diagnostic validation results", ""]

    # ---- paper-style per-condition tables (alpha=1.0 default) ----
    lines += ["## Family 1 per-condition rejection rates (paper style, alpha=1.0)",
              "",
              "Rejection rate = fraction of coefficients flagged unstable "
              "(FDR 0.05). Mean ± sd over 5 seeds.", ""]
    d1 = two[(two.family == "F1") & (two.alpha == 1.0)]
    for cond in ["C1", "C2", "C3", "C4"]:
        sub = d1[d1.condition == cond]
        if sub.empty:
            continue
        lines += [f"### {cond}", "",
                  "| N | magnitude | overall | diagonal | off-diagonal | TPR(injected) | FPR(null) |",
                  "|---|---|---|---|---|---|---|"]
        keycols = ["N"] + (["magnitude"] if cond != "C4" else [])
        for key, g in sub.groupby(keycols):
            N_ = key[0] if isinstance(key, tuple) else key
            mag = key[1] if (cond != "C4" and isinstance(key, tuple)) else "-"
            lines.append(f"| {N_} | {mag} | {msd(g,'reject_overall')} | "
                         f"{msd(g,'reject_diag')} | {msd(g,'reject_off')} | "
                         f"{msd(g,'tpr')} | {msd(g,'fpr')} |")
        lines.append("")

    # ---- C4 multivariate contrast ----
    c4 = two[(two.condition == "C4") & (two.family == "F1") & two.multivariate_reject_rate.notna()]
    if not c4.empty:
        lines += ["## C4 null: per-coefficient vs multivariate (Section 5.5 mirror)",
                  "",
                  "| N | alpha | per-coef FPR | multivariate reject rate |",
                  "|---|---|---|---|"]
        for (N_, a), g in c4.groupby(["N", "alpha"]):
            lines.append(f"| {N_} | {a} | {msd(g,'fpr')} | "
                         f"{msd(g,'multivariate_reject_rate')} |")
        lines.append("")

    # ---- ridge-alpha sensitivity (C2 power at a fixed cell) ----
    lines += ["## Ridge-alpha sensitivity (C2 off-diagonal power, N=60)", "",
              "| magnitude | alpha=0.1 | alpha=1.0 | alpha=10.0 |",
              "|---|---|---|---|"]
    c2 = two[(two.condition == "C2") & (two.family == "F1") & (two.N == 60)]
    for mag in sorted(c2.magnitude.unique()):
        cells = [msd(c2[(c2.magnitude == mag) & (c2.alpha == a)], "tpr")
                 if not c2[(c2.magnitude == mag) & (c2.alpha == a)].empty else "-"
                 for a in ALPHAS]
        lines.append(f"| {mag} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- multi-level recovery ----
    if not multi.empty:
        lines += ["## 10-level site-like design: recovery of the injected sparse support",
                  "",
                  "The '>20% of pairs' rule should flag exactly the injected "
                  "support. Mean ± sd over seeds (alpha=1.0).", "",
                  "| magnitude | N | injected | recovered(TP) | spurious(FP) | recall | precision |",
                  "|---|---|---|---|---|---|---|"]
        for (mag, N_), g in multi[multi.alpha == 1.0].groupby(["magnitude", "N"]):
            lines.append(f"| {mag} | {N_} | {int(g.injected_total.iloc[0])} | "
                         f"{g.recovered_tp.mean():.1f} | {g.spurious_fp.mean():.1f} | "
                         f"{msd(g,'recall')} | {msd(g,'precision')} |")
        lines.append("")

    # ---- Family 2 HRF localisation comparison ----
    f2 = two[two.family == "F2"]
    if not f2.empty:
        f1cmp = two[(two.family == "F1") & (two.N == F2_N) & (two.alpha == 1.0)
                    & two.magnitude.isin(F2_MAGS)]
        lines += ["## Family 2: HRF vs no-HRF localisation (N=60, alpha=1.0)", "",
                  "Localisation accuracy = fraction of detected coefficients on the "
                  "injected partition. Also off-diagonal FPR, to see if HRF creates "
                  "spurious cross-region instability.", "",
                  "| condition | magnitude | loc.acc no-HRF | loc.acc HRF | off-FPR no-HRF | off-FPR HRF |",
                  "|---|---|---|---|---|---|"]
        for cond in ["C1", "C2", "C3"]:
            for mag in F2_MAGS:
                a = f1cmp[(f1cmp.condition == cond) & (f1cmp.magnitude == mag)]
                b = f2[(f2.condition == cond) & (f2.magnitude == mag)]
                if a.empty or b.empty:
                    continue
                # off-diagonal FPR: fraction of non-injected off-diagonal rejecting
                lines.append(f"| {cond} | {mag} | {msd(a,'localization_acc')} | "
                             f"{msd(b,'localization_acc')} | {msd(a,'fpr')} | "
                             f"{msd(b,'fpr')} |")
        lines.append("")

    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS/'summary.md'} and {RESULTS/'all_cells.csv'}")
    make_figures(two, multi)


def make_figures(two, multi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) power curve: C2 TPR vs magnitude, by N (alpha=1.0)
    c2 = two[(two.condition == "C2") & (two.family == "F1") & (two.alpha == 1.0)]
    if not c2.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        for N_ in NS:
            g = c2[c2.N == N_].groupby("magnitude")["tpr"]
            if len(g) == 0:
                continue
            m, s = g.mean(), g.std(ddof=0)
            ax.errorbar(m.index, m.values, yerr=s.values, marker="o",
                        capsize=2, label=f"N={N_}")
        # FP reference from C4
        c4 = two[(two.condition == "C4") & (two.family == "F1") & (two.alpha == 1.0)]
        if not c4.empty:
            ax.axhline(c4["fpr"].mean(), ls="--", color="k", lw=1,
                       label=f"C4 FP rate ({c4['fpr'].mean():.3f})")
        ax.axhline(0.05, ls=":", color="gray", lw=1, label="q=0.05")
        ax.set_xlabel("injected off-diagonal shift magnitude")
        ax.set_ylabel("power (TPR on injected off-diagonal)")
        ax.set_title("C2: off-diagonal detection power")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / "fig1_power_curve.png", dpi=200)
        plt.close(fig)

    # (2) FP rate for the null (C4) by N and alpha
    c4 = two[(two.condition == "C4") & (two.family == "F1")]
    if not c4.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        piv = c4.groupby(["alpha", "N"])["fpr"].mean().unstack("N")
        x = np.arange(len(piv.index)); w = 0.25
        for k, N_ in enumerate(piv.columns):
            ax.bar(x + (k - 1) * w, piv[N_].values, w, label=f"N={N_}")
        ax.axhline(0.05, ls="--", color="k", lw=1, label="q=0.05")
        ax.set_xticks(x); ax.set_xticklabels([f"α={a}" for a in piv.index])
        ax.set_ylabel("C4 false-positive rate"); ax.set_title("C4 null: FP rate")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / "fig2_fp_rate.png", dpi=200)
        plt.close(fig)

    # (3) localisation confusion for C1/C2/C3 (alpha=1, N=60, aggregated over seeds/mags)
    d = two[(two.family == "F1") & (two.alpha == 1.0) & (two.N == 60)
            & two.condition.isin(["C1", "C2", "C3"])]
    if not d.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        conds = ["C1", "C2", "C3"]; x = np.arange(len(conds)); w = 0.35
        det_diag = [d[d.condition == c]["det_diag_injectedDiag"].sum()
                    + d[d.condition == c]["det_diag_notInjected"].sum() for c in conds]
        det_off = [d[d.condition == c]["det_off_injectedOff"].sum()
                   + d[d.condition == c]["det_off_notInjected"].sum() for c in conds]
        ax.bar(x - w / 2, det_diag, w, label="detected on diagonal", color="#b5651d")
        ax.bar(x + w / 2, det_off, w, label="detected off-diagonal", color="#3b7dd8")
        ax.set_xticks(x); ax.set_xticklabels(
            ["C1\n(inj diag)", "C2\n(inj off)", "C3\n(inj both)"])
        ax.set_ylabel("detected coefficients (summed over seeds/mags)")
        ax.set_title("Localisation: where detections land"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / "fig3_localization.png", dpi=200)
        plt.close(fig)

    # (4) HRF vs no-HRF localisation accuracy
    f2 = two[two.family == "F2"]
    f1 = two[(two.family == "F1") & (two.N == F2_N) & (two.alpha == 1.0)
             & two.magnitude.isin(F2_MAGS)]
    if not f2.empty and not f1.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        conds = ["C1", "C2", "C3"]; x = np.arange(len(conds)); w = 0.35
        acc_raw = [f1[f1.condition == c]["localization_acc"].mean() for c in conds]
        acc_hrf = [f2[f2.condition == c]["localization_acc"].mean() for c in conds]
        ax.bar(x - w / 2, acc_raw, w, label="no HRF", color="#4c9f70")
        ax.bar(x + w / 2, acc_hrf, w, label="HRF (Glover, TR=2)", color="#d1495b")
        ax.set_xticks(x); ax.set_xticklabels(conds)
        ax.set_ylabel("localisation accuracy"); ax.set_ylim(0, 1.05)
        ax.set_title("Family 2: does localisation survive HRF filtering?")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / "fig4_hrf_localization.png", dpi=200)
        plt.close(fig)
    print(f"wrote figures -> {FIGURES}")


# --------------------------------------------------------------------------- #
def iter_full_cells():
    """Yield callables for every cell in the full sweep (order = cheap first)."""
    base, _, _ = G.make_base_A(P, seed=BASE_A_SEED)
    support = G.pick_support(P, SUPPORT_SEED, N_DIAG_INJECT, N_OFF_INJECT)
    jobs = []
    for cond in ["C1", "C2", "C3"]:
        for mag in MAGS:
            for N in NS:
                for a in ALPHAS:
                    for s in SEEDS:
                        jobs.append(("2l", cond, mag, N, a, s, False))
    for N in NS:                                   # C4 null (no magnitude)
        for a in ALPHAS:
            for s in SEEDS:
                jobs.append(("2l", "C4", 0.0, N, a, s, False))
    for cond in ["C1", "C2", "C3"]:                # Family 2 HRF
        for mag in F2_MAGS:
            for a in [1.0]:
                for s in SEEDS:
                    jobs.append(("2l", cond, mag, F2_N, a, s, True))
    for mag in MAGS_MULTI:                          # 10-level design
        for a in ALPHAS:
            for s in SEEDS:
                jobs.append(("multi", None, mag, 60, a, s, False))
    return base, support, jobs


def run_full():
    base, support, jobs = iter_full_cells()
    done = len(list(CELLS.glob("*.json")))
    print(f"full sweep: {len(jobs)} cells ({done} already done). CPU/NumPy only.")
    for k, job in enumerate(jobs, 1):
        kind, cond, mag, N, a, s, hrf = job
        if kind == "2l":
            run_two_level_cell(base, support, cond, mag, N, a, s, hrf=hrf)
        else:
            run_multilevel_cell(base, support, mag, N, a, s)
        if k % 20 == 0:
            print(f"  {k}/{len(jobs)} cells")
    aggregate_and_report()


def run_smoke():
    """One tiny cell per code path -- proves end-to-end in well under a minute."""
    print("SMOKE TEST (tiny cells) ...")
    base, rho, _ = G.make_base_A(P, seed=BASE_A_SEED)
    support = G.pick_support(P, SUPPORT_SEED, N_DIAG_INJECT, N_OFF_INJECT)
    print(f"  base matrix rho={rho:.3f} (stationary)")
    r_c2 = run_two_level_cell(base, support, "C2", 0.30, 30, 1.0, 0)
    print(f"  C2 mag0.30 N30: off-reject={r_c2['reject_off']:.3f} "
          f"TPR={r_c2['tpr']:.3f} FPR={r_c2['fpr']:.3f} loc={r_c2['localization_acc']:.2f}")
    r_c4 = run_two_level_cell(base, support, "C4", 0.0, 30, 1.0, 0)
    print(f"  C4 null   N30: overall={r_c4['reject_overall']:.3f} "
          f"FPR={r_c4['fpr']:.3f} multivariate={r_c4.get('multivariate_reject_rate'):.3f}")
    r_hrf = run_two_level_cell(base, support, "C2", 0.30, 30, 1.0, 0, hrf=True)
    print(f"  C2 HRF    N30: off-reject={r_hrf['reject_off']:.3f} "
          f"loc={r_hrf['localization_acc']:.2f}")
    r_m = run_multilevel_cell(base, support, 0.30, 30, 1.0, 0)
    print(f"  10-level mag0.30 N30: recovered {r_m['recovered_tp']}/"
          f"{r_m['injected_total']} injected, {r_m['spurious_fp']} spurious "
          f"(recall={r_m['recall']:.2f} precision={r_m['precision']:.2f})")
    print("SMOKE OK -- all four code paths ran end-to-end.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test")
    ap.add_argument("--aggregate", action="store_true", help="rebuild md/csv/figs only")
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
    elif args.aggregate:
        aggregate_and_report()
    else:
        run_full()
