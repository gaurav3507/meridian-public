# ABIDE-I SMS finding: within-dataset robustness

Answers "is the ABIDE-I self-loop finding a sampling fluke or site-driven?" on the
same 742-subject, 14-site harmonized data and the same method that produced the
original result (742 subjects, 91 site pairs, 227,500 tests; the "845" in some
handoff docs is a pre-final-QC count, quote 742). Region set fixed to the original
full-data top-50; ridge VAR(1) alpha=1, 100-boot subject resampling, per-coefficient
Wald, BH-FDR q=0.05, all reused from test_b.py / test_b_marginal.py.

Pre-registered bar (committed before running): ROBUST iff diagonal concentration
holds (off-diagonal unstable count = 0 at the strict >50%-of-pairs threshold) in
ALL 14 leave-one-out subsets AND in >=45/50 split-half resamples (both halves).

## Sanity: full-data reference row reproduces the original

1.06% reject (2406/227,500), median |d|/SE 0.750, 49/50 diagonal self-loops
unstable at >20% of pairs, 0/2450 off-diagonal. Matches the original exactly, so
the helper is faithful and the variants below are trustworthy.

## 1. Leave-one-site-out (drop each site, recompute on remaining 13; 78 pairs)

Holds in ALL 14/14 subsets. No single site drives the finding.
- Rejection rate: 0.84% to 1.17% (full-data reference 1.06%).
- Diagonal unstable >20%: 46 to 49 of 50 (mild dips dropping UCLA_1 = 46, MAX_MUN = 48).
- Off-diagonal unstable: 0 in every one of the 14 subsets, at both >20% and >50%.
- No subset collapses; per-site file in leave_one_out.csv.

## 2. Split-half (50 stratified 50/50 splits, 100 half-datasets ~371 subjects each)

Holds in 100/100 halves; both halves hold in 50/50 splits.
- Rejection rate across halves: mean 0.72%, sd 0.03%, range [0.66, 0.80]%.
- Diagonal unstable >20%: mean 40.7 of 50, range [36, 45].
- Off-diagonal unstable: 0 across all 100 halves (max 0 at >20% and >50%; not a
  single off-diagonal coefficient became unstable in any resample).

Note on the lower rate: halving the subjects per site halves the power, so the
rejection rate falls to ~0.72% and fewer self-loops individually clear the >20%
bar (~41 vs 49). This is a sample-size effect, not a change in the pattern: the
qualitative result (all instability on the diagonal, none off-diagonal) is
invariant across every subsample.

## Verdict: ROBUST (both pre-registered criteria met)

- Leave-one-out: 14/14 subsets hold.
- Split-half: 50/50 splits hold (bar was >=45/50).
- Across all 114 recomputes (14 leave-one-out + 100 halves), off-diagonal unstable
  coefficients = 0, without exception.

The ABIDE-I finding is not a sampling fluke and is not driven by any single site.
The concentration of residual site instability on per-region diagonal self-loops,
with cross-region mechanisms site-stable, survives dropping any site and 50 random
stratified partitions. Combined with the ADHD-200 conceptual replication (different
disorder, engine, and GSR), the SMS section now has both cross-dataset
generalization and within-dataset robustness.
