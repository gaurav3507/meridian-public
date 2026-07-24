# I3: ridge-alpha sensitivity (VAR(1))

Per-coefficient two-sample Wald z, BH-FDR q=0.05; site uses the >20%-of-91-pairs unstable rule, condition uses FDR 0.05 on the single rest-vs-WM contrast. Same data/harmonization/standardization/top-50 selection as the primary analyses. Only ridge alpha changes. alpha=1.0 is the primary/reported setting.

Diagonal = 50 self-loops; off-diagonal = 2450 cross terms.

## site (ABIDE-I, CC200)

| ridge alpha | overall reject | diagonal unstable | off-diagonal unstable |
|---|---|---|---|
| 0.1 | 1.06% | 49/50 | 0/2450 |
| 1.0  **(primary)** | 1.06% | 49/50 | 0/2450 |
| 10.0 | 1.06% | 49/50 | 0/2450 |

## condition (AOMIC, Schaefer-200, unpaired bootstrap)

| ridge alpha | overall reject | diagonal unstable | off-diagonal unstable |
|---|---|---|---|
| 0.1 | 7.16% | 50/50 | 129/2450 |
| 1.0  **(primary)** | 7.16% | 50/50 | 129/2450 |
| 10.0 | 7.16% | 50/50 | 129/2450 |

## Stability across alpha

Site off-diagonal unstable across alpha [0.1, 1.0, 10.0]: [0, 0, 0]. Condition off-diagonal across alpha: [129, 129, 129].

The dissociation is STABLE: site off-diagonal stays ~flat and far below condition off-diagonal across all alpha.
