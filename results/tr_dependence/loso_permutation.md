# TR dependence-robustness: leave-one-site-out + site-level permutation

Per-pair diagonal/off-diagonal rejection counts are the existing diagnostic output (results/test_b_marginal.csv), counted exactly as test_c_tr_instability.py. Start-up check reproduces the published diagonal Spearman rho: **+0.9556** (p=5.33e-49); off-diagonal control **+0.1710** (p=0.105).

## (1) Leave-one-site-out (14 folds, 78 pairs each)

| dropped site | n pairs | rho diagonal | rho off-diagonal |
|---|---|---|---|
| CALTECH | 78 | +0.9604 | +0.2603 |
| KKI | 78 | +0.9394 | +0.0834 |
| LEUVEN_2 | 78 | +0.9450 | +0.1582 |
| MAX_MUN | 78 | +0.9466 | +0.3029 |
| NYU | 78 | +0.9608 | +0.2212 |
| PITT | 78 | +0.9512 | +0.1721 |
| SDSU | 78 | +0.9566 | +0.1218 |
| STANFORD | 78 | +0.9588 | +0.1067 |
| TRINITY | 78 | +0.9618 | +0.1752 |
| UCLA_1 | 78 | +0.9451 | +0.0424 |
| UM_1 | 78 | +0.9603 | +0.1606 |
| UM_2 | 78 | +0.9566 | +0.1777 |
| USM | 78 | +0.9558 | +0.2232 |
| YALE | 78 | +0.9567 | +0.1908 |

**Diagonal rho across folds — min +0.9394, median +0.9566, max +0.9618.**
**Off-diagonal rho across folds — min +0.0424, median +0.1736, max +0.3029.**

## (2) Site-level permutation null (n_perm=10000, TRs permuted across the 14 sites)

| quantity | observed | two-sided permutation p |
|---|---|---|
| diagonal rho | +0.9556 | 9.999e-05 |
| off-diagonal rho (control) | +0.1710 | 0.3459 |
| matched vs mismatched median diag contrast | -39.0 | 0.002 |

Matched-TR pairs (delta_TR=0): n=37, median diagonal rejections=0.0. Mismatched (delta_TR>0): n=54, median=39.0.

## Verdict

ROBUST: the TR->diagonal correlation survives leave-one-site-out (min fold rho +0.939) and is extreme under the site-level permutation null (p=9.999e-05), while the off-diagonal control stays non-significant (permutation p=0.346). The result is not driven by any single site or by pair non-independence.

