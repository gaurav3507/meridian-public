# Region-selection sensitivity analysis

Tests whether the site/condition double dissociation survives a principled region
set instead of the near-arbitrary pooled-variance top-50 (which after z-scoring
spans ~1.2% and selected zero somatomotor regions).

Both datasets are Schaefer-200 (identical parcel indexing), so ONE region set is
applied to both. Pipeline is byte-for-byte the primary one: ridge VAR(1)
alpha=1.0 + intercept; 100-bootstrap subject resampling (global seed 20260628);
per-coefficient two-sample Wald z; BH-FDR q=0.05. Site instability = rejects in
>20% of the 91 site pairs; condition = significant at FDR 0.05 on the single
rest-vs-WM contrast. Nothing tuned; one run.

Site file loaded: `data/processed/abide_schaefer_harmonized.npz` (742 subjects,
14 sites, 200 regions; byte-distinct from the CC200 `abide_harmonized.npz`).

## Validation (replication reproduces the stated variance-based baselines exactly)

| analysis | overall reject | median \|d\|/SE | diagonal | off-diagonal |
|---|---|---|---|---|
| site, variance-based (computed here) | 1.02% | 0.727 | 49/50 | **0/2450** |
| site, variance-based (stated) | 1.02% | 0.727 | 49/50 | 0/2450 |
| condition, variance-based (computed here) | 7.16% | 1.018 | 50/50 | **129/2450** |
| condition, variance-based (stated) | 7.16% | 1.02 | 50/50 | 129/2450 |

The machinery reproduces both baselines to the digit, so the new region sets are
trustworthy.

## RUN A: network-balanced 50 regions (seed 20260713)

Allocated proportional to Yeo-7 network size (Hamilton largest-remainder), every
network represented including SomMot:

| network | Vis | SomMot | DorsAttn | SalVentAttn | Limbic | Cont | Default |
|---|---|---|---|---|---|---|---|
| Schaefer-200 size | 29 | 35 | 26 | 22 | 12 | 30 | 46 |
| selected | 7 | 9 | 6 | 5 | 3 | 8 | 12 |

Region IDs (0-based Schaefer): 0,4,5,20,23,26,32,37,38,42,43,44,45,47,55,56,57,
67,71,73,75,86,89,93,98,99,100,104,110,112,115,117,123,128,129,133,141,146,157,
167,170,172,177,178,180,183,190,193,194,196.

| analysis | selection | overall reject | median \|d\|/SE | diagonal | off-diagonal |
|---|---|---|---|---|---|
| SITE | variance-based | 1.02% | 0.727 | 49/50 | **0/2450 (0.00%)** |
| SITE | network-balanced | 1.24% | 0.739 | 50/50 | **0/2450 (0.00%)** |
| CONDITION | variance-based | 7.16% | 1.018 | 50/50 | **129/2450 (5.27%)** |
| CONDITION | network-balanced | 12.20% | 1.112 | 49/50 | **256/2450 (10.45%)** |

Site stays exactly zero off-diagonal; condition off-diagonal is present and
larger than the variance-based estimate.

### Condition enrichment on the balanced set (256 significant off-diagonal shifts)

obs/exp per source (row) -> target (column) cell:

| src \ tgt | Vis | SomMot | DorsAttn | SalVentAttn | Limbic | Cont | Default |
|---|---|---|---|---|---|---|---|
| Vis         | 2.51 | 2.28 | 0.68 | 2.73 | 2.73 | 1.37 | 0.57 |
| SomMot      | 1.37 | 1.99 | 1.24 | 1.91 | 0.00 | 0.40 | 0.89 |
| DorsAttn    | 3.19 | 0.35 | 0.96 | 0.64 | 0.00 | 0.00 | 0.40 |
| SalVentAttn | 2.46 | 0.64 | 0.96 | 1.44 | 0.00 | 1.44 | 1.44 |
| Limbic      | 1.37 | 0.00 | 0.00 | 0.00 | 0.00 | 0.40 | 0.27 |
| Cont        | 1.54 | 0.53 | 1.20 | 0.96 | 0.40 | 1.37 | 0.80 |
| Default     | 0.68 | 0.09 | 0.93 | 0.48 | 0.53 | 1.00 | 1.01 |

Per-network involvement (x baseline density 10.45%): source Vis 1.62, SalVentAttn
1.29, SomMot 1.15, Cont 0.98, DorsAttn 0.78, Default 0.70, Limbic 0.33; target Vis
1.70, SalVentAttn 1.21, DorsAttn 0.94, Cont 0.88, SomMot 0.87, Default 0.81,
Limbic 0.59. (Full table: `runA_condition_enrichment.csv`.)

**Still diffuse.** 41 of 49 network-pair cells are occupied; the largest single
cell holds 15/256 = 5.9%; per-network enrichment stays within 0.33x-1.70x. There
is a mild sensory/attention lean (Visual is the most-enriched target, 1.70x, with
attention->Visual cells at 2.5-3.2x), but it is an emphasis on a diffuse
background, not a localization to one network. Critically, SomMot -- absent from
the variance set -- is now represented (9 regions) and participates at roughly
chance (source 1.15x, target 0.87x): its earlier exclusion did not hide a
somatomotor-localized effect. A balanced set that COULD have detected
network-specific localization did not; the "spatially diffuse" claim survives.

## RUN B: 5 random 50-region subsets (seeds 101, 202, 303, 404, 505)

| seed | site off-diag | site diag | condition off-diag | condition diag |
|---|---|---|---|---|
| 101 | 0/2450 | 50/50 | 211/2450 | 50/50 |
| 202 | 0/2450 | 50/50 | 195/2450 | 50/50 |
| 303 | 0/2450 | 50/50 | 187/2450 | 50/50 |
| 404 | 0/2450 | 50/50 | 176/2450 | 50/50 |
| 505 | 0/2450 | 50/50 | 231/2450 | 50/50 |

| quantity | min | median | max |
|---|---|---|---|
| site off-diagonal unstable (of 2450) | 0 | 0 | 0 |
| condition off-diagonal significant (of 2450) | 176 | 195 | 231 |

## Verdict (against the honesty constraints)

1. **Do the balanced condition shifts concentrate in specific networks?** No. With
   every network represented, they remain diffuse (41/49 blocks, largest 5.9%,
   enrichment 0.33-1.70x), with only a mild Visual/attention lean. The paper's
   "spatially diffuse" claim holds; the previously-excluded SomMot is involved at
   chance, so the diffuse claim is now better supported, not weakened.

2. **Is the site off-diagonal ever non-zero?** No. Site off-diagonal is 0/2450
   under the balanced set AND all 5 random subsets (site diagonal 49-50/50
   throughout). The measurement-only site result is robust to region choice.

3. **Does the condition off-diagonal vary across subsets?** Yes -- the magnitude
   is subset-dependent and the variance-based 5.27% (129/2450) is the LOW end.
   Balanced = 10.45% (256/2450); random subsets = 7.2%-9.4% (176-231/2450). The
   variance-based selection is Default-heavy (23/50 Default) and under-sampled the
   sensory/attention networks where the condition effect is strongest, so it
   under-estimates the effect. Report the condition off-diagonal as a range
   (~5-10%), not a point estimate. The DIRECTION -- present, non-zero, far above
   the site's exact 0 -- is unchanged and robust.

Bottom line: the double dissociation (site = diagonal/measurement only; condition
= off-diagonal/mechanism present) is not an artifact of the variance-based region
set. It holds under a network-balanced set and 5 random sets. The one number that
must change is the condition off-diagonal magnitude: it should be a range, and the
5.27% figure is a conservative low estimate.
