"""Condition-based SMS test: rest vs working-memory VAR(1) coefficient shift.

Mirrors scripts/test_b.py + test_b_marginal.py exactly (top-50 by pooled variance,
ridge VAR(1) alpha=1, 100-boot subject-resampling per group, per-coefficient Wald z,
BH-FDR q=0.05), with CONDITION (rest vs working memory) as the two groups instead of
site. Reports in the SAME format as the site analysis so the contrast is readable
side by side.

Loads data/condition_sms/ts/{sub}_rest.npy and {sub}_wm.npy (each (160, 200),
task-regressed for WM). Writes results/condition_sms/{results.md, marginal.csv}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from test_b import bootstrap_site, select_top_regions      # noqa: E402 exact reuse

ROOT = SCRIPTS.parent
TS = ROOT / "data/condition_sms/ts"
OUT = ROOT / "results/condition_sms"
OUT.mkdir(parents=True, exist_ok=True)

N_TOP_REGIONS = 50
N_BOOT = 100
FDR_Q = 0.05
RNG_SEED = 20260628           # same base seed as test_b


def load_pairs():
    subs = sorted(p.name[:-9] for p in TS.glob("*_rest.npy")
                  if (TS / f"{p.name[:-9]}_wm.npy").exists())
    rest = np.stack([np.load(TS / f"{s}_rest.npy") for s in subs]).astype(np.float32)
    wm = np.stack([np.load(TS / f"{s}_wm.npy") for s in subs]).astype(np.float32)
    return subs, rest, wm


def main():
    subs, rest, wm = load_pairs()
    print(f"subjects with both conditions: {len(subs)}  "
          f"rest{rest.shape} wm{wm.shape}")

    # top-50 by pooled variance across BOTH conditions (subjects x time x region)
    top = select_top_regions(np.concatenate([rest, wm], axis=0), N_TOP_REGIONS)
    R = N_TOP_REGIONS
    rest_s, wm_s = rest[:, :, top], wm[:, :, top]

    rng = np.random.default_rng(RNG_SEED)
    b_rest = bootstrap_site(rest_s, N_BOOT, rng).astype(np.float64)   # (100,50,50)
    b_wm = bootstrap_site(wm_s, N_BOOT, rng).astype(np.float64)
    d = b_rest.mean(0) - b_wm.mean(0)                                 # (50,50)
    se = np.sqrt(b_rest.var(0, ddof=1) + b_wm.var(0, ddof=1))
    z = np.where(se > 0, d / se, 0.0)
    p = 2.0 * norm.sf(np.abs(z))
    reject = multipletests(p.ravel(), alpha=FDR_Q,
                           method="fdr_bh")[0].reshape(R, R)

    eye = np.eye(R, dtype=bool)
    n_tests = R * R
    n_reject = int(reject.sum())
    diag_rej = int(reject[eye].sum())
    off_rej = int(reject[~eye].sum())
    n_off = R * R - R
    off_rate = off_rej / n_off
    diag_rate = diag_rej / R
    med_z = float(np.median(np.abs(z)))

    # per-coefficient table (single contrast)
    ii, jj = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    pd.DataFrame({"region_i": ii.ravel(), "region_j": jj.ravel(),
                  "is_diag": eye.ravel(), "d": d.ravel(), "se": se.ravel(),
                  "abs_z": np.abs(z).ravel(), "p_raw": p.ravel(),
                  "reject_fdr": reject.ravel()}).to_csv(
        OUT / "marginal.csv", index=False)

    if off_rate >= 0.10:
        verdict = ("CLEARLY PRESENT: off-diagonal cross-region reject "
                   f"{off_rate*100:.1f}% (>=10%), a 10x+ contrast with the site "
                   "~1%. Mechanism-shift signal present across conditions.")
    elif off_rate >= 0.03:
        verdict = ("PRESENT BUT MODEST: off-diagonal reject "
                   f"{off_rate*100:.1f}% (3-10%). Substantially more than across "
                   "sites, not dramatic.")
    else:
        verdict = ("WEAK/ABSENT: off-diagonal reject "
                   f"{off_rate*100:.1f}% (<3%), near the site baseline. Surprising; "
                   "weakens the positive-direction claim.")

    lines = [
        "# Condition-based SMS test: rest vs working memory (AOMIC PIOP2)", "",
        f"{len(subs)} subjects, both conditions, Schaefer-200 -> top-50 by pooled "
        "variance, T=160, task-regressed WM. Ridge VAR(1) alpha=1, 100-boot "
        "subject resampling per condition, per-coefficient Wald z, BH-FDR q=0.05 "
        "(same machinery as the site test_b).", "",
        "## Result (same format as the site analysis)", "",
        "| Analysis | grouping | overall reject | median |d|/SE | diagonal reject | "
        "off-diagonal reject |",
        "|---|---|---|---|---|---|",
        "| SITE (ABIDE, test_b) | 14 sites | 1.06% | 0.75 | 49/50 | **0 / 2450 "
        "(0.00%)** |",
        f"| CONDITION (AOMIC) | rest vs WM | {n_reject/n_tests*100:.2f}% | "
        f"{med_z:.2f} | {diag_rej}/{R} ({diag_rate*100:.0f}%) | "
        f"**{off_rej} / {n_off} ({off_rate*100:.2f}%)** |", "",
        "The thesis is the off-diagonal column: all-diagonal / zero off-diagonal "
        "across SITES (measurement), versus cross-region shift across CONDITIONS "
        "(mechanism reconfiguration).", "",
        "## Pre-registered graded reading", "", verdict, "",
        "Honest framing: conceptual contrast, not a matched experiment (different "
        "dataset, atlas, grouping, design). See scripts/condition_sms/README.md.",
    ]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")

    print("\n" + "=" * 70)
    print("CONDITION-BASED SMS: rest vs working memory")
    print("=" * 70)
    print(f"  overall reject @ FDR {FDR_Q} : {n_reject/n_tests*100:.2f}% "
          f"({n_reject}/{n_tests})")
    print(f"  median |d|/SE             : {med_z:.3f}")
    print(f"  DIAGONAL reject           : {diag_rej}/{R} ({diag_rate*100:.0f}%)")
    print(f"  OFF-DIAGONAL reject       : {off_rej}/{n_off} "
          f"({off_rate*100:.2f}%)   [SITE was 0/2450]")
    print(f"\n  {verdict}")
    print(f"\n  wrote {OUT/'results.md'} + marginal.csv")


if __name__ == "__main__":
    main()
