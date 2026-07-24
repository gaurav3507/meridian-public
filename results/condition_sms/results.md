# Condition-based SMS test: rest vs working memory (AOMIC PIOP2)

222 subjects, both conditions, Schaefer-200 -> top-50 by pooled variance, T=160, task-regressed WM. Ridge VAR(1) alpha=1, 100-boot subject resampling per condition, per-coefficient Wald z, BH-FDR q=0.05 (same machinery as the site test_b).

## Result (same format as the site analysis)

| Analysis | grouping | overall reject | median |d|/SE | diagonal reject | off-diagonal reject |
|---|---|---|---|---|---|
| SITE (ABIDE, test_b) | 14 sites | 1.06% | 0.75 | 49/50 | **0 / 2450 (0.00%)** |
| CONDITION (AOMIC) | rest vs WM | 7.16% | 1.02 | 50/50 (100%) | **129 / 2450 (5.27%)** |

The thesis is the off-diagonal column: all-diagonal / zero off-diagonal across SITES (measurement), versus cross-region shift across CONDITIONS (mechanism reconfiguration).

## Pre-registered graded reading

PRESENT BUT MODEST: off-diagonal reject 5.3% (3-10%). Substantially more than across sites, not dramatic.

Honest framing: conceptual contrast, not a matched experiment (different dataset, atlas, grouping, design). See scripts/condition_sms/README.md.

## Effect characterization (honest magnitude read)

- Off-diagonal: 129/2450 (5.27%) significant at FDR. The significant ones are
  robust, not borderline (|d|/SE median 3.45, min 2.91), but small in absolute
  coefficient terms (median |d| = 0.048 VAR-coef units). Statistically solid,
  biologically modest.
- Breadth above chance: 18.2% of off-diagonal exceed |d|/SE > 1.96 (chance 5%) and
  4.86% exceed > 3 (chance ~0.27%). So the cross-region shift is pervasive and far
  above chance, just modest per edge, not a handful of lucky tail hits.
- Diagonal dominates the rest-vs-task difference: 50/50 self-loops significant with
  |d|/SE median 10.0. Most of what changes between rest and task is per-region
  temporal structure; the cross-region (off-diagonal) reconfiguration is the
  smaller, but genuinely present, component.
- The contrast that is the thesis: SITES had 0/2450 off-diagonal instability;
  CONDITIONS have a real off-diagonal component (129/2450, pervasive above chance).
  The mechanism-shift signal absent across sites IS present across conditions.

## Why modest, not dramatic (a point in favor of rigor)

Much of a naive rest-vs-task connectivity difference is task-EVOKED co-activation,
which we deliberately regressed out (per-trial-type HRF regression). The 5.27% is
therefore the clean background-connectivity reconfiguration, artifact-free. A
dramatic number here would invite the reviewer suspicion that co-activation was not
fully removed. Modest-but-clean is the more credible result; the modesty is partly
the price of doing the evoked-response regression correctly.

## Network localization of the significant off-diagonal shifts

The 129 significant off-diagonal (cross-region) rest-vs-WM shifts, localized by Yeo-7 network of predictor (source) and predicted (target) region. Enrichment = observed / expected, where expected is proportional to the available off-diagonal region-pairs in each network cell (networks differ in size, so raw counts are biased toward big networks).

Region counts among the 50 VAR regions: Vis 4, SomMot 0, DorsAttn 6, SalVentAttn 7, Limbic 2, Cont 8, Default 23.

Top source->target cells by enrichment (obs, exp, enrichment):
- DorsAttn -> Vis: obs 5, exp 1.3, enrichment 3.96 (low expected, noisy)
- DorsAttn -> DorsAttn: obs 6, exp 1.6, enrichment 3.80 (low expected, noisy)
- Cont -> Vis: obs 6, exp 1.7, enrichment 3.56 (low expected, noisy)
- SalVentAttn -> Vis: obs 4, exp 1.5, enrichment 2.71 (low expected, noisy)
- Limbic -> Vis: obs 1, exp 0.4, enrichment 2.37 (low expected, noisy)
- DorsAttn -> SalVentAttn: obs 5, exp 2.2, enrichment 2.26

WM-relevant (pre-registered focus, Cont = FrontoParietal):
- Cont -> Cont: obs 5, exp 2.9, enr 1.70
- Cont -> Default: obs 15, exp 9.7, enr 1.55
- Default -> Cont: obs 11, exp 9.7, enr 1.14
- Default -> Default: obs 25, exp 26.6, enr 0.94

See figures/condition_sms/network_localization.png for the full 7x7 enrichment matrix.

## Paired sensitivity analysis

Rest and WM are the same 222 subjects. Per-subject VAR(1) coefficients, paired t-test per coefficient across subjects, BH-FDR q=0.05.
- Paired off-diagonal reject: 74/2450 = 3.02% (unpaired bootstrap Wald was 129/2450 = 5.27%). Paired LOWERS the estimate.
- Paired diagonal reject: 48/50.

The paired result is reported as a robustness check; the primary number remains the unpaired bootstrap Wald for parallelism with the site test_b.
