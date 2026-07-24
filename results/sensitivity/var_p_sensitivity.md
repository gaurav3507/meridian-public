# M1: VAR(p) lag-order sensitivity

Ridge alpha fixed at 1.0 (swept in I3). 100-bootstrap subject resampling, per-coefficient two-sample Wald z, BH-FDR q=0.05; site uses the >20%-of-91-pairs unstable rule, condition uses FDR 0.05 on the single rest-vs-WM contrast. Same data/harmonization/standardization/top-50 selection as the primary analyses (CC200 ABIDE, Schaefer-200 AOMIC).

**Generalized partition at higher lags:** with lag matrices A_1..A_p, DIAGONAL = all self terms {A_k[i,i]} (p x 50 total), OFF-DIAGONAL = all cross terms {A_k[i,j], i!=j} (p x 50 x 49 total).

## Primary (pooled bootstrap)

| dataset | lag p | overall reject | diagonal unstable | off-diagonal unstable | AIC order | BIC order |
|---|---|---|---|---|---|---|
| site | 1 | 1.06% | 49/50 | 0/2450 | 3 | 3 |
| site | 2 | 1.12% | 100/100 | 0/4900 | 3 | 3 |
| site | 3 | 1.20% | 150/150 | 0/7350 | 3 | 3 |
| condition (unpaired bootstrap) | 1 | 7.16% | 50/50 | 129/2450 | 2 | 1 |
| condition (unpaired bootstrap) | 2 | 4.58% | 84/100 | 145/4900 | 2 | 1 |
| condition (unpaired bootstrap) | 3 | 2.87% | 85/150 | 130/7350 | 2 | 1 |

## Flagged sensitivity: condition per-subject paired t

Per-subject VAR(p) fits are underdetermined at p>1 (T=160 vs p x 50 predictors); ridge alpha=1 regularizes but treat as NOISY. Primary condition number is the pooled bootstrap above.

| dataset | lag p | overall reject | diagonal unstable | off-diagonal unstable | AIC order | BIC order |
|---|---|---|---|---|---|---|
| condition (per-subject paired t) -- FLAGGED noisy at p>1 | 1 | 4.88% | 48/50 | 74/2450 | 2 | 1 |
| condition (per-subject paired t) -- FLAGGED noisy at p>1 | 2 | 1.50% | 46/100 | 29/4900 | 2 | 1 |
| condition (per-subject paired t) -- FLAGGED noisy at p>1 | 3 | 0.59% | 27/150 | 17/7350 | 2 | 1 |

## Lag-order selection (pooled OLS VAR, Lutkepohl AIC/BIC)

- **ABIDE**: AIC selects p=3, BIC selects p=3  (AIC totals p1=3568.2, p2=2833.4, p3=2229.9; BIC totals p1=3613.6, p2=2924.9, p3=2368.2)
- **AOMIC**: AIC selects p=2, BIC selects p=1  (AIC totals p1=-73.6, p2=-74.4, p3=-74.4; BIC totals p1=-72.4, p2=-72.0, p3=-70.7)

## Dissociation check

The site (diagonal-dominant) vs condition (off-diagonal-present) dissociation HOLDS at every lag (condition off-diagonal > site off-diagonal).

- p=1: site off-diagonal 0/2450, condition off-diagonal 129/2450
- p=2: site off-diagonal 0/4900, condition off-diagonal 145/4900
- p=3: site off-diagonal 0/7350, condition off-diagonal 130/7350
