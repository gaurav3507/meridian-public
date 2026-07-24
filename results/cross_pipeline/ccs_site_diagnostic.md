# M2: cross-pipeline robustness -- ABIDE-I site diagnostic on CCS

Identical machinery, identical retained subjects and covariates; the ONLY change is the input time series (CCS `filt_global/rois_cc200` instead of C-PAC). Ridge VAR(1) alpha=1.0, 100-bootstrap subject resampling, per-coefficient two-sample Wald z, BH-FDR q=0.05, unstable = rejects in >20% of site pairs. ComBat-GAM with the same covariates (batch=SITE, smooth AGE, SEX/DX_GROUP/func_mean_fd). Crop = cohort minimum, exactly as build_dataset.py step 4. No per-region standardization in either pipeline (PCP .1D are raw ROI means).

## Input-convention mismatch between the two PCP products (measured)

C-PAC `filt_global` .1D are **demeaned** (grand mean 0.001; pooled variance 99.9% within-subject temporal). CCS `filt_global` .1D are **not** (grand mean ~8896; pooled variance ~100% between-subject DC offset, temporal signal 0.05% of it). Left uncorrected that is a SECOND difference on top of the pipeline: `select_top_regions` then ranks regions by DC offset rather than temporal signal, and the pooled VAR fit is swamped by subject-specific offsets. The matched column demeans each subject x region over the full series, reproducing C-PAC's own convention so the ONLY remaining difference is the pipeline. Both columns are shown; the as-shipped column is confounded and is reported for transparency, not as the cross-pipeline result.

## C-PAC vs CCS

| metric | C-PAC (primary) | CCS as-shipped (confounded) | CCS matched (valid test) |
|---|---|---|---|
| subjects | 742.0 | 742.0 | 742.0 |
| sites | 14.0 | 14.0 | 14.0 |
| site pairs | 91.0 | 91.0 | 91.0 |
| timepoints T (cohort min crop) | 116.0 | 115.0 | 115.0 |
| overall reject % | 1.06 | 0.23 | 1.2 |
| median |d|/SE | 0.75 | 0.521 | 0.737 |
| diagonal unstable (of 50) | 49.0 | 9.0 | 49.0 |
| off-diagonal unstable (of 2450) | 0.0 | 0.0 | 0.0 |
| top-50 region overlap with C-PAC | 50.0 | 17.0 | 42.0 |

CCS volumes: cohort min T=115 (median 175) vs C-PAC T=116 -- CCS discards one extra volume per subject; all sites/subjects retained. Missing FILE_IDs under CCS: 0.

## Verdict

REPRODUCES: CCS is diagonal-dominant with near-zero off-diagonal site instability, matching C-PAC. The dissociation is not a C-PAC artifact.

