# Bootstrap resample-count (B) stability of the primary diagnostics

Site (ABIDE-I, CC200) and condition (AOMIC, Schaefer-200) per-coefficient diagnostics re-run UNCHANGED except for the bootstrap B. Ridge VAR(1) alpha=1.0, per-coefficient two-sample Wald z, BH-FDR q=0.05; site uses the >20%-of-91-pairs unstable rule, condition uses FDR 0.05 on the single rest-vs-WM contrast. Fixed seed 20260628. B=100 is the primary reported setting.

## site (ABIDE-I, CC200)

| bootstrap B | overall reject | median |d|/SE | diagonal unstable (of 50) | off-diagonal unstable (of 2450) |
|---|---|---|---|---|
| 100  **(primary)** | 1.06% | 0.750 | 49/50 | 0/2450 |
| 500 | 1.04% | 0.749 | 49/50 | 0/2450 |
| 1000 | 1.03% | 0.748 | 49/50 | 0/2450 |

## condition (AOMIC, Schaefer-200, pooled bootstrap)

| bootstrap B | overall reject | median |d|/SE | diagonal unstable (of 50) | off-diagonal unstable (of 2450) |
|---|---|---|---|---|
| 100  **(primary)** | 7.16% | 1.018 | 50/50 | 129/2450 |
| 500 | 6.96% | 1.011 | 50/50 | 124/2450 |
| 1000 | 7.16% | 1.010 | 50/50 | 129/2450 |

## Verdict

Site off-diagonal across B=[100, 500, 1000]: [0, 0, 0] (diagonal [49, 49, 49]). Condition off-diagonal across B: [129, 124, 129] (diagonal [50, 50, 50]).

STABLE across B: the reported counts do not depend on the bootstrap resample count.

