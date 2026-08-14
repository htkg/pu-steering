# Rate: per-batch mean +/- std

Rate is the fraction of a batch that is the avoided target class (single-class) or
the avoided vehicle group (multi-class), evaluated by the ResNet-18 classifier.
Each config is `M=128` per batch x `80` seeds; the numbers below are the leak per
batch, `mean +/- std` over the 80 batches.

The **means equal the tradeoff-plot rates exactly** (verified for all 82 configs,
`max |mean - target_rate| = 0`); only the **std** is new (batch-to-batch spread).
Plotting axes use the mean; the std is not shown there.

## Baselines (SafeDenoiser), single-class [%]

| class | eta=0.0 | eta=0.2 | eta=0.3 | eta=0.5 | eta=0.7 | eta=1.0 |
|---|---|---|---|---|---|---|
| plane | 8.71 +/- 2.49 | 5.93 +/- 2.25 | 4.78 +/- 1.99 | 2.95 +/- 1.77 | 2.24 +/- 1.44 | 1.96 +/- 1.46 |
| cat | 8.26 +/- 2.36 | 7.59 +/- 2.21 | 7.16 +/- 2.47 | 6.10 +/- 2.31 | 4.69 +/- 2.09 | 2.43 +/- 1.58 |
| frog | 9.67 +/- 2.34 | 4.10 +/- 1.70 | 2.55 +/- 1.50 | 0.87 +/- 0.77 | 0.27 +/- 0.46 | 0.13 +/- 0.31 |
| ship | 11.34 +/- 2.90 | 5.62 +/- 2.10 | 3.63 +/- 1.43 | 1.76 +/- 1.03 | 1.04 +/- 0.91 | 0.74 +/- 0.71 |

## Ours, single-class [%]

| class | lam | eta=0.2 | eta=0.3 | eta=0.5 |
|---|---|---|---|---|
| plane | 0.5 | 5.36 +/- 2.37 | 3.88 +/- 2.31 | 2.96 +/- 3.90 |
| plane | 1.0 | 5.70 +/- 7.46 | 3.28 +/- 2.66 | 2.46 +/- 2.99 |
| plane | 1.5 | 4.58 +/- 2.92 | 3.50 +/- 5.27 | 2.18 +/- 2.10 |
| plane | 2.0 | 5.43 +/- 10.84 | 2.54 +/- 3.01 | 2.62 +/- 4.23 |
| cat | 0.5 | 7.51 +/- 2.33 | 6.64 +/- 2.60 | 5.32 +/- 2.64 |
| cat | 1.0 | 7.40 +/- 2.65 | 6.09 +/- 2.98 | 5.12 +/- 3.33 |
| cat | 1.5 | 7.36 +/- 2.85 | 5.66 +/- 4.03 | 3.96 +/- 3.38 |
| cat | 2.0 | 6.89 +/- 2.86 | 5.54 +/- 3.33 | 4.07 +/- 3.63 |
| frog | 0.5 | 3.86 +/- 1.75 | 2.29 +/- 1.49 | 0.87 +/- 0.93 |
| frog | 1.0 | 3.00 +/- 1.98 | 1.65 +/- 1.38 | 0.65 +/- 0.80 |
| frog | 1.5 | 2.30 +/- 2.44 | 1.96 +/- 2.35 | 0.76 +/- 1.10 |
| frog | 2.0 | 1.77 +/- 2.04 | 1.14 +/- 1.96 | 0.73 +/- 1.59 |
| ship | 0.5 | 5.64 +/- 2.46 | 3.50 +/- 1.61 | 1.78 +/- 1.08 |
| ship | 1.0 | 4.91 +/- 3.22 | 2.98 +/- 3.00 | 1.75 +/- 1.22 |
| ship | 1.5 | 4.70 +/- 7.80 | 2.52 +/- 3.95 | 1.41 +/- 1.39 |
| ship | 2.0 | 4.78 +/- 10.58 | 1.46 +/- 1.92 | 1.37 +/- 1.34 |

## Vehicles (multi-class group) [%]

| method | eta=0.0 | eta=0.2 | eta=0.3 | eta=0.5 | eta=0.7 | eta=1.0 |
|---|---|---|---|---|---|---|
| baseline (SafeDenoiser) | 41.37 +/- 4.24 | 31.89 +/- 4.18 | 27.60 +/- 3.98 | 20.70 +/- 3.33 | 15.67 +/- 3.02 | 13.48 +/- 3.50 |

| Ours | lam=0.5 | lam=1.0 | lam=1.5 | lam=2.0 |
|---|---|---|---|---|
| eta=0.2 | 31.70 (std n/a) | 26.91 (std n/a) | 24.24 (std n/a) | 22.30 (std n/a) |
| eta=0.3 | 25.86 +/- 5.27 | 19.87 +/- 6.09 | 18.08 +/- 11.76 | 20.93 +/- 18.06 |
| eta=0.5 | 17.52 (std n/a) | 14.08 (std n/a) | 16.25 (std n/a) | 14.53 (std n/a) |

Vehicles leak is the group rate (per-seed `group_rate`), available only where the
per-seed logs were kept: baselines and Ours `eta=0.3`. For Ours `eta=0.2/0.5` only
the mean survives (`= target_rate`); the std needs a re-run.

## Notes

- Std captures batch-to-batch spread across the 80 seeds; it grows with the
  avoidance strength (`lam` / lower `eta`). Heavy-tailed configs (e.g. plane/ship
  `lam>=1.5` at `eta=0.2`, vehicles `lam=2.0` at `eta=0.3`: `std ~ mean`) have a
  few high-leak batches, consistent with their low ESS.
- Source: `results/{...}/summary_short.json` (`raw_mean/raw_std`, single-class) and
  `summary.json[per_seed].group_rate` (vehicles).
