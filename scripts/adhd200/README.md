# ADHD-200 SMS replication (second-dataset check)

Conceptual replication of the ABIDE-I SMS finding on ADHD-200. This is NOT a
byte-identical replication. Read the honest-differences section before quoting
any number in the paper.

## The finding being replicated (from ABIDE-I)

Cross-region VAR(1) mechanisms are stable across sites; site-related instability
concentrates in per-region diagonal self-loops (a measurement property, not a
cross-region mechanism). ABIDE-I numbers: 227,500 tests, 1.06% rejection at FDR
q=0.05, median per-coefficient |d|/SE = 0.75, and all 49 coefficients unstable
in >20% of site pairs were diagonal self-loops (0 of 2,450 off-diagonal).

## PRE-REGISTERED READING (written before running; do not move the goalposts)

1. Off-diagonal cross-region coefficients stable AND instability concentrated on
   the diagonal self-loops => REPLICATES (the target result).
2. Overall rejection rate HIGHER than ABIDE's 1.06% BUT still diagonal-
   concentrated => REPLICATES and STRENGTHENS. This is the predicted signature of
   flag 2 (see below): ADHD-200 has a wider TR spread across sites than ABIDE,
   and self-loops are per-region lag-1 autocorrelation, which depends on TR. A
   higher rate that lands on the diagonal is confirmation, NOT failure. Do not
   misread it as a miss.
3. Off-diagonal cross-region coefficients ALSO substantially unstable => does
   NOT cleanly replicate. Report honestly. It would mean cross-region mechanisms
   differ across ADHD-200 sites, which is a real (and publishable) negative.

Reporting rule: always report RAW COUNTS alongside percentages. ADHD-200 has far
fewer site pairs than ABIDE (see below), so percentages are noisier.

## The sharp prediction to hold explicitly (flag 2, TR spread)

ABIDE sites are closer in TR; ADHD-200 TR ranges roughly 1.5-2.5 s across sites.
The measurement interpretation predicts that this wider TR spread AMPLIFIES
diagonal self-loop instability while leaving cross-region coefficients stable.
If that is what comes out (higher diagonal rejection, off-diagonal still stable),
it is a stronger confirmation than a flat match to 1.06% would be. This
prediction is registered here in advance.

## Decisions (approved)

- Peking pooled into a single site PEKING (same scanner; gains N). Per-Peking-
  sample counts are logged so pooling is auditable.
- Primary analysis: method-faithful top-50 regions by pooled variance (selected
  on ADHD-200, so the regions differ from ABIDE's 50).
- Secondary robustness check: matched-region variant using ABIDE's exact 50
  atlas indices, to control for "did top-50 coincidentally pick similar regions".

## Honest differences from ABIDE (all go in the paper's methods)

- DATA SOURCE (updated): the C-PAC fcp-indi release was a 162-subject benchmark
  subset (too small, 0 sites >=30). We instead use the classic Neuro Bureau
  ADHD-200 Preprocessed **Athena** release (full N), downloaded as
  ADHD200_CC200_TCs_filtfix.tar. Athena is AFNI+FSL, a DIFFERENT ENGINE from
  C-PAC. Cross-engine replication is accepted (strengthens generalization); a
  non-replication is read cautiously (disorder + engine + GSR triple confound).
- Nuisance model (CONFIRMED from Athena docs): Athena filtered (sfnwmrda) =
  mean WM + mean CSF + 6 motion params + 3rd-order polynomial detrend, band-pass
  0.009-0.08 Hz, and NO global signal regression. ABIDE filt_global HAS global
  signal regression (+ 24 motion params, band-pass 0.01-0.1 Hz). So the Athena
  product matches ABIDE's filt_NOglobal, not filt_global. GSR present in ABIDE,
  absent in Athena, is a documented nuisance-model difference. Cross-region
  coefficients are the ones most plausibly affected by the GSR difference, so a
  cross-region non-replication cannot be cleanly attributed to disorder alone.
- Motion QC (CORRECTED): only KKI ships a motion.csv; 8 of 9 sites have NO motion
  data in the tar and there is no framewise displacement anywhere. So max-motion
  QC is impossible. QC instead uses the ADHD-200 consortium resting-run quality
  flag QC_Rest_1 == pass (available for all sites, bundles motion + registration
  + coverage; a standard ADHD-200 criterion). Deviation from ABIDE's mean-FD<0.2.
- ComBat covariate: no per-subject motion metric exists across sites, so ComBat
  uses age, sex, DX only (ABIDE used func_mean_fd). Minimal-impact deviation:
  ComBat adjusts per-region time MEANS, and a VAR(1) with intercept is invariant
  to per-series constants, so ComBat is near-inert for the actual VAR test.
- WashU dropped: its QC_Rest_1 column is entirely blank, so a consistent QC drops
  the whole site. Six well-powered sites remain (KKI, NYU, PEKING, OHSU,
  Pittsburgh, NeuroIMAGE). Not kept under inconsistent QC (would be a real flaw).
- GSR flip-side (the honest both-directions read): the Athena-vs-ABIDE stack now
  differs on disorder + engine + GSR (Athena filtered = no GSR = filt_noglobal).
  A non-replication is therefore read cautiously (off-diagonal instability could
  be GSR, not mechanism). BUT if the diagonal-self-loop finding REPLICATES
  despite all three differences including GSR, that is a STRONGER generalization
  claim, not a weaker one: surviving disorder + engine + GSR = robust.
- Region count: Athena cc200 files are 3dROIstats output; parcels with no voxels
  are dropped, so subjects have ~190 regions, not 200. Analysis runs on the
  common CC200 labels present across ALL retained subjects (190, uniform).
- Short-protocol site exclusion (pre-specified rule, not hand-named): drop any
  site whose protocol length (site-min T) < 100. This removes only OHSU (T=74).
  Reason: lag-1 autocorrelation from 74 timepoints is biased relative to the
  120-257 point sites, and that bias lands directly on the diagonal self-loops
  the finding is about; keeping OHSU would also crop the whole cohort to 74.
  Dropping it yields T=120 (matches ABIDE's 116 regime, VAR estimates directly
  comparable). Final: 5 sites (PEKING, NYU, KKI, PITTSBURGH, NEUROIMAGE).
- FD definition: func_mean_fd = mean of C-PAC's Power-2012 FD.1D; func_perc_fd =
  percent of frames with FD > 0.2 mm. ABIDE's func_mean_fd came from PCP's QAP
  (also Power-2012) but a different implementation/version. Not bit-identical.
- Phenotype schema: per-site CSVs; SEX recoded from Gender (0/1); DX_GROUP recoded
  from DX (0 control, 1/2/3 ADHD subtypes). DX here is ADHD-vs-control, not ASD.
- TR spread wider than ABIDE (see flag 2).
- Scan length T varies by site; crop-to-shortest-common-T may land low. Report
  the T used and flag if small.
- ComBat harmonization adjusts per-region time MEAN only (broadcast across time);
  a VAR(1) with intercept is invariant to per-series constants, so harmonization
  is near-inert for the tested quantity, in both datasets. Mirrored for fidelity.
- Single scan/session: _scan_rest_1 / _session_1 only, mirroring ABIDE's single
  run. Subjects with only a later scan/session are excluded.

## Held constant vs ABIDE

N_TOP_REGIONS=50, N_BOOT=100, ALPHA_RIDGE=1.0, FDR_Q=0.05, ridge VAR(1) with
intercept, subject-resampling bootstrap, per-coefficient Wald z, BH-FDR jointly,
QC (mean FD < 0.2 AND perc_fd < 30, drop sites N < 30, crop to shortest common T).

## Pipeline order

1. probe_availability.py  - which phenotyped subjects have the filtered CC200 file
2. download_adhd200.py    - download CC200 (filtered+GSR) + FD.1D per subject
3. build_dataset_adhd200.py - compute FD, QC, ComBat-GAM -> adhd200_harmonized.npz
   [GATE: report post-QC site/subject counts, sanity-check N per site before bootstrap]
4. test_b_adhd200.py         - per-site bootstrap VAR(1), Hotelling T^2
5. test_b_marginal_adhd200.py - per-coefficient Wald, FDR
6. unstable_coefs_adhd200.py  - diagonal vs off-diagonal breakdown; primary + matched-region
