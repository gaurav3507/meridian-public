# Condition-based SMS test (rest vs working memory), AOMIC PIOP2

Companion to the site-based SMS test (scripts/test_b.py / test_b_marginal.py). Same
VAR(1) + bootstrap + Wald + FDR machinery, but the grouping variable is CONDITION
(resting-state vs working-memory task) instead of SITE. The point: across SITES the
residual instability was all DIAGONAL (measurement); across CONDITIONS we expect
substantial OFF-DIAGONAL shift (genuine cross-region mechanism reconfiguration).

## Data
AOMIC PIOP2 (OpenNeuro ds002790), fmriprep MNI152NLin2009cAsym preprocessed BOLD.
Rest (240 vol) and working memory (160 vol), both acq-seq TR 2.0s (TR-matched).
~222 subjects have both runs. Parcellated to Schaefer-200; cropped to 160 volumes.
Downloaded per-subject and deleted after parcellation (never hold >1 subject's BOLD).

## THE CRITICAL STEP: task-design regression (background connectivity)
To capture task-MODULATED connectivity, not task-EVOKED co-activation (which would
be shared-stimulus artifact a reviewer dismisses), the working-memory evoked response
is regressed out before VAR:
  1. Read sub-XXXX_task-workingmemory_acq-seq_events.tsv (onset, duration, trial_type;
     6s blocks; trial types active_change, active_nochange, passive).
  2. Build one regressor per trial_type, convolve with a canonical HRF plus its temporal
     derivative (nilearn make_first_level_design_matrix, drift_model=None since the
     fmriprep cosine columns handle drift).
  3. Regress those task columns out of every region's timeseries TOGETHER with the
     confounds (passed jointly to the parcellation masker), then run VAR(1) on the
     residuals.
Rest gets confounds only (no events, no task regression). Modeling each trial_type
separately removes condition-specific evoked amplitudes thoroughly.

## Confounds
6 motion + 6 motion derivatives + 5 aCompCor (a_comp_cor_00..04) + cosine drift terms,
from the fmriprep confounds TSV (first-row derivative NaNs filled with 0).

## VAR / test (matched to test_b for methodological parallelism)
Top-50 Schaefer regions by pooled variance (across subjects x time x both conditions);
ridge VAR(1) alpha=1.0 fit_intercept; 100-boot subject-resampling per condition;
per-coefficient two-sample Wald z (rest vs WM); BH-FDR q=0.05 across the 2500
coefficients. Report in the SAME format as the site analysis: overall off-diagonal
reject rate AND diagonal-vs-off-diagonal breakdown, so site (all diagonal, ~1%
off-diagonal) and condition sit side by side.

## PRE-REGISTERED GRADED READING (committed before running)
Off-diagonal (cross-region) reject rate between rest and working memory:
  - >= 10%  : signal CLEARLY present. Decisive 10x+ contrast with the site ~1%.
  - 3 - 10% : signal present but MODEST. Real several-fold contrast with sites;
              report as "substantially more than across sites, not dramatic".
  - < 3%    : weak/ABSENT. Surprising; weakens the positive-direction claim. Report
              honestly.
Diagonal self-loops are reported alongside but are not the thesis; the thesis is the
off-diagonal (cross-region) contrast vs the site result.

## HONEST FRAMING (this is a conceptual contrast, not a matched experiment)
Site analysis: between-subject, resting-state only, CC200, ABIDE, 14 sites.
Condition analysis: within-subject, rest-vs-task, Schaefer-200, AOMIC, 2 conditions.
Different dataset, atlas, grouping logic, and design. The claim is "the mechanism-shift
signal is absent across sites and present across conditions, shown with parallel
methodology," NOT a controlled head-to-head. Two further honest notes:
  - We mirror test_b's UNPAIRED per-group bootstrap even though rest and WM are the
    same subjects; a paired test is a possible sensitivity analysis and would only add
    power.
  - Task regression is applied to WM only (rest has no task); this asymmetry is
    intended, since the question is whether intrinsic coupling reconfigures beyond the
    evoked response.
  - Atlas space: Schaefer ships in a slightly different MNI variant than 2009cAsym;
    the masker resamples the atlas to the BOLD grid (sub-voxel difference, standard).
