# Case 2: Reference-based results

Five layouts × three seeds (0-2). The baseline is SafeDenoiser (SD). `raw_pm` is
the mean oracle posterior mass of the undesired class, and `wass` is the empirical
2-Wasserstein distance to the target GMM; lower is better for both.

## Resampling ON

| Method | `raw_pm` | `wass` |
|--------|--------|------|
| Oracle | 0.012 ± 0.010 | 1.5 ± 0.3 |
| SD (eta=30) | 0.145 ± 0.023 | 2.2 ± 0.3 |
| SD (eta=50) | 0.096 ± 0.026 | 3.6 ± 0.9 |
| **Grad ON** | **0.094 ± 0.020** | **1.9 ± 0.4** |
| **SDProp ON** | **0.105 ± 0.018** | **1.9 ± 0.5** |

Both Grad ON and SDProp ON Pareto-dominate SD(eta=30): lower pm *and* lower wass.
Win rate vs SD(eta=30): 15/15 for both methods.

## Resampling OFF

| Method | `raw_pm` | `w_pm` | `raw_wass` | `w_wass` |
|--------|--------|------|----------|--------|
| SD (eta=50) | 0.096 ± 0.026 | — | 3.6 ± 0.9 | — |
| SD (eta=100) | 0.037 ± 0.018 | — | 6.2 ± 1.3 | — |
| Grad OFF | 0.181 ± 0.045 | **0.011 ± 0.019** | 3.5 ± 0.5 | 4.9 ± 1.9 |
| SDProp OFF | 0.195 ± 0.011 | **0.102 ± 0.026** | 2.7 ± 0.5 | **2.0 ± 0.5** |

- Grad OFF: raw_pm is worse than SD, but **weighted w_pm=0.011** beats
  SD(eta=100)=0.037. Win rate w_pm vs SD(eta=50): 15/15.
- SDProp OFF: **w_wass=2.0** with w_pm=0.102 breaks the SD tradeoff;
  SD(eta=50) has wass=3.6 at similar pm.
