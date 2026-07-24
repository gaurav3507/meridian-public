# Test C — does TR difference predict diagonal VAR instability?

**Question.** Across the 91 ABIDE-I site pairs, does the repetition-time
difference `delta_TR = |TR_1 - TR_2|` predict how many VAR(1) coefficients differ
significantly between the two sites — and does it do so **specifically** for the
diagonal (self-loop) coefficients, with the off-diagonal as a control?

Backing data: `results/test_b_marginal.csv` (227,500 rows = 2,500 coeffs x 91
pairs). FDR-rejection column identified programmatically: **`reject_fdr`** (bool).
Reproduce with `python scripts/test_c_tr_instability.py`.

## STEP 1 — per-site TR, measured from NIfTI headers

One `func_preproc` volume per site was streamed from the ABIDE S3 mirror
(same path as `build_schaefer_dataset.py`), `pixdim[4]` read, then the volume
deleted. Time unit is unmarked in the header (`xyzt_units = ('mm','unknown')`);
disambiguated by magnitude — **every site landed in 1–4, i.e. already seconds;
no millisecond cases.** Cached to `results/measured_tr.csv`.

| site | N | T | measured TR (s) | published TR (s) | agree |
|------|--:|--:|----------------:|-----------------:|:-----:|
| PITT | 44 | 196 | 1.5000 | 1.500 | yes |
| LEUVEN_2 | 30 | 246 | **1.6519** | 1.667 | ~ (see note) |
| NYU | 169 | 176 | 2.0000 | 2.000 | yes |
| UM_1 | 81 | 296 | 2.0000 | 2.000 | yes |
| UM_2 | 30 | 296 | 2.0000 | 2.000 | yes |
| USM | 60 | 236 | 2.0000 | 2.000 | yes |
| YALE | 46 | 196 | 2.0000 | 2.000 | yes |
| TRINITY | 44 | 146 | 2.0000 | 2.000 | yes |
| STANFORD | 36 | 176 | 2.0000 | 2.000 | yes |
| SDSU | 33 | 176 | 2.0000 | 2.000 | yes |
| KKI | 39 | 152 | 2.5000 | 2.500 | yes |
| UCLA_1 | 54 | 116 | 3.0000 | 3.000 | yes |
| MAX_MUN | 41 | 116 | 3.0000 | 3.000 | yes |
| CALTECH | 35 | 146 | **2.0000** | unknown | n/a |

**Disagreement to flag (not silently accepted):** LEUVEN_2's header reports
**1.6519 s**, the published value is **1.667 s** (= 5/3). The gap is ~0.015 s;
they are not identical. We use the **measured** header value (1.6519 s)
throughout, consistent with the rule of trusting the header. Because delta_TR
enters only through Spearman **ranks**, this 0.015 s difference changes no pair
ordering and has no effect on any result below.

**CALTECH** (TR previously unknown): header says **2.0 s**; that is the value used.

## STEP 2 — per-pair quantities & sanity check

For each of 91 pairs: `n_diag_reject` (of 50), `n_offdiag_reject` (of 2,450),
`delta_TR`, `n_min = min(N_1, N_2)`. Written to `results/test_c_pairs.csv`.

**Sanity check PASSED:** sum of (diag + off-diag) rejections over all 91 pairs =
**2,406**, exactly the total `reject_fdr==True` count in the file (2,406).

## STEP 3 — the test and its control (Spearman, 91 pairs)

| relationship | rho | p |
|---|---:|---:|
| (a) delta_TR vs **n_diag_reject** | **+0.956** | 5.3e-49 |
| (b) delta_TR vs n_offdiag_reject (control) | +0.171 | 0.105 |

**(c) Verdict.** TR difference predicts **diagonal** instability
**specifically**. The diagonal association is near-perfect and overwhelmingly
significant (rho = +0.96); the off-diagonal control is weak and **not**
significant (rho = +0.17, p = 0.11). This is the predicted dissociation: TR is
**not** merely a proxy for "these two sites differ in general" — if it were, the
off-diagonal would track delta_TR just as strongly, and it does not. The result
**supports** a specific measurement (sampling-interval) interpretation of the
diagonal / self-loop concentration.

## STEP 4 — is it just statistical power?

| relationship | rho | p |
|---|---:|---:|
| n_min vs n_diag_reject | +0.210 | 0.045 |
| **partial** delta_TR vs n_diag_reject, controlling n_min | **+0.957** | 4.3e-49 |

Sample size has only a weak, borderline association with diagonal rejection
counts (rho = +0.21). Crucially, partialling n_min **out** leaves the TR effect
completely intact (rho +0.956 -> +0.957). The apparent TR effect is **not** a
sample-size artifact.

## Bottom line (honesty check)

- delta_TR **does** predict diagonal instability — strongly. The Results-section
  claim that TR explains the diagonal concentration is **supported**, not
  contradicted. No softening required on the basis of this test.
- delta_TR does **not** predict off-diagonal instability to a comparable degree
  (rho 0.96 vs 0.17), so the effect is diagonal-specific.
- The effect survives controlling for statistical power.
- Method discipline: one test per question (Spearman), reported as-run. No
  Pearson/Kendall/log/outlier-removal/threshold variants were tried. No site
  pair was dropped (all 91 retained).

Figure: `figures/paper/fig5_tr_vs_instability.{png,pdf}`.
