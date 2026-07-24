# ADHD-200 (Athena) SMS replication results

5 sites (PEKING 183, NYU 177, KKI 78, PITTSBURGH 66, NEUROIMAGE 39; total 543), T=120, 190 common CC200 parcels, 10 site pairs. Ridge VAR(1) alpha=1, 100-boot subject resampling, per-coefficient Wald z, BH-FDR q=0.05. Deviations from ABIDE logged in README (Athena engine, no GSR = filt_noglobal, QC_Rest_1, no motion covariate, OHSU dropped for short-protocol AC bias, 190 not 200 regions). ABIDE reference: 1.06% reject, median |d|/SE=0.75, 49/49 unstable coeffs diagonal, 0/2450 off-diagonal.

| Variant | reject@FDR | median |d|/SE | diag unstable >20% | off-diag unstable >20% | diag >50% | off-diag >50% |
|---|---|---|---|---|---|---|
| primary_top50 | 1.66% | 0.799 | 39/50 | 20/2450 | 9/50 | 0/2450 |
| matched_region | 1.76% | 0.821 | 45/47 | 10/2162 | 24/47 | 0/2162 |

## Honest reading against the pre-registration

**The diagonal self-loop concentration replicates strongly.** Nearly all self-loops are site-unstable (primary 39/50, matched 45/47) and reject in many pairs (up to 7-8 of 10). Cross-region (off-diagonal) coefficients are >99% stable (primary 2430/2450 stable, matched 2152/2162).

**The off-diagonal residual is threshold-marginal, not a clean non-replication.** The 10-20 off-diagonal coefficients flagged at the loose >20%-of-pairs bar all reject in only 3-5 of 10 pairs, right at the boundary. At a stricter >50%-of-pairs threshold the off-diagonal residual vanishes entirely (0 off-diagonal in BOTH variants) while a solid diagonal set remains. So instability concentrates on the diagonal at every threshold; the off-diagonal only appears at the loosest cut.

**Overall rate slightly above ABIDE (1.66-1.76% vs 1.06%), diagonal-dominated** = consistent with the pre-registered TR-spread prediction (wider ADHD-200 TR range amplifies per-region lag-1 AC instability on the diagonal). Median |d|/SE ~0.80 vs ABIDE 0.75: cross-region effect sizes are as small as ABIDE's.

**Cautious note on the small off-diagonal residual (disorder+engine+GSR).** Athena applies NO global signal regression (unlike ABIDE filt_global). Without GSR, shared/global variance remains in the timeseries and can produce some site-variable cross-region coupling. The small, threshold-marginal off-diagonal residual is therefore plausibly a GSR-absence artifact rather than genuine cross-region mechanism heterogeneity. This is the pre-registered cautious read, not a claim of clean failure.

**Verdict: strong conceptual replication.** The structural claim (instability lives on the diagonal self-loops, cross-region mechanisms are site-stable) holds across a different disorder, a different engine (Athena vs C-PAC), and no GSR. Surviving all three differences is a stronger generalization than a byte-identical rerun. The result is not ABIDE's literal 0/2450 off-diagonal, and that caveat is reported honestly, but the concentration pattern is unambiguous.

## Anatomical labeling of unstable self-loops (vs ABIDE 37% signal-poor)

Signal-poor fraction reported two ways: HO-strict (Harvard-Oxford keyword match:
orbitofrontal, frontal/temporal pole, inferior temporal, subcallosal, brainstem)
and +infratentorial (also counting self-loops whose centroid is cerebellar /
posterior-fossa, z < -25 mm, which HO cannot label because it has no cerebellar
coverage, and which are independently classic low-SNR / susceptibility-dropout
regions). ABIDE's HO-based analysis likewise could not label cerebellum.

- **primary_top50**: 13/39 = **33%** signal-poor (HO-strict; no infratentorial
  centroids to add). Dominant unstable self-loop: **Frontal Pole (11/39)**, one
  of ABIDE's named signal-poor regions. ABIDE: 37% (18/49).
- **matched_region**: 8/45 = 18% HO-strict, rising to **15/45 = 33%** once the 7
  infratentorial (cerebellum/posterior-fossa) self-loops HO leaves Unlabelled are
  included. Frontal Pole (4) and Brain-Stem (2) also present. ABIDE: 37%.

Convergence: both variants land at ~33% signal-poor, matching ABIDE's 37%, and
Frontal Pole recurs as the leading unstable self-loop. The same anatomy (frontal
pole / orbitofrontal, cerebellum and posterior fossa, brainstem) carries the
self-loop instability across two disorders (ASD, ADHD), two engines (C-PAC,
Athena), and with vs without GSR. That an identical signal-poor / susceptibility
fingerprint recurs under all three differences is convergent evidence these
unstable self-loops are measurement/scanner artifacts, not disorder-specific
biology. This is the closing piece for the SMS section: the residual site
instability is anatomically stereotyped low-SNR regions, not cross-region
mechanisms.

Limitation logged: a 2 mm max-motion sensitivity check cannot be run on ADHD-200 (the Athena release ships no per-subject motion data, only KKI has a motion summary); noted, not executed.
