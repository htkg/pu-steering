# Ours classifier and weight diagnostics — Case 1 (true-score)

**Setup.** The five Ours variants of the main README (OFF / ON / ON_TILT / ON_GRAD 40/40 /
ON_GRAD 48/38; `class_prior = pu_class_prior =` true layout pi, kernel PU ss=0.2 lam=1e-4),
`T=100`, ring GMM (radius 10), 5 layouts × seeds 0-9 = 50 trials, `OMP_NUM_THREADS=4`,
M in {64, 256, 1024, 4096}. Performance vs the matched-M Oracle floor is in
`../ablation/m_merged.txt`; this directory holds the per-step diagnostics.

Proposals: OFF and ON_GRAD use the gradient proposal; ON and ON_TILT the weak-DNG
proposal (lambda0=0.6). The ON_GRAD variants fire exactly 2 ESS-triggered resamples in
all 50 trials at every M.

## f_t classifier diagnostics vs the oracle posterior

Per step, `f_t` is compared to the exact oracle posterior on the same denoised
`x0hat` (snapshot callback; no sampler change): AUC (label = oracle p>0.5),
prob-RMSE (calibration), grad-cos (steering-direction fidelity in x0hat space).
Time bins by `t/T`: early >0.66, mid 0.33-0.66, late ≤0.33. M=1024 shown; the same
pattern holds at M in {64, 256, 4096} (full tables in this directory).

| variant | bin | AUC | prob-RMSE | grad-cos |
|---------|-----|-----|-----------|----------|
| OFF | mid | 0.995 | 0.090 | 0.14±0.10 |
| OFF | late | 0.9995 | 0.074 | 0.22±0.11 |
| ON | mid | 0.985 | 0.141 | 0.09±0.07 |
| ON | late | 0.958 | 0.320 | 0.09±0.07 |
| ON_TILT | mid | 0.986 | 0.148 | 0.09±0.07 |
| ON_TILT | late | 0.947 | 0.371 | 0.07±0.08 |
| ON_GRAD 40/40 | mid | 0.965 | 0.242 | 0.07±0.06 |
| ON_GRAD 40/40 | late | 0.949 | 0.335 | 0.06±0.06 |
| ON_GRAD 48/38 | mid | 0.980 | 0.173 | 0.09±0.07 |
| ON_GRAD 48/38 | late | 0.960 | 0.318 | 0.07±0.05 |

- **Region accuracy is high and M-robust** (AUC 0.94-0.9995 by mid/late at every M and
  variant), while the pointwise posterior is far from accurate (prob-RMSE 0.15-0.37
  overall, grad-cos ≤ 0.25). Low leak therefore comes from coarse region accuracy on a
  forgiving task, not from a precise posterior estimate.
- The resampled variants lose late-time calibration (the path leaves c after
  resampling) with no effect on leak.

## ESS of the classifier-training weights (M=1024)

The PU classifier is fit with w_u (unlabeled term) and w_c (positive term); w_q is the
SMC weight. ESS/M along the path (mean over steps × trials; min = worst step):

| variant | w_u mean | w_u min | w_c mean (late) | w_c min | w_q mean |
|---------|----------|---------|-----------------|---------|----------|
| OFF | 0.98 | 0.90 | 0.19 | 0.11 | 0.87 |
| ON | 0.58 | 0.004 | 0.008 | 0.001 | 0.68 |
| ON_TILT | 0.58 | 0.004 | 0.008 | 0.001 | 0.83 |
| ON_GRAD 40/40 | 0.67 | 0.002 | 0.016 | 0.001 | 0.70 |
| ON_GRAD 48/38 | 0.59 | 0.006 | 0.009 | 0.001 | 0.70 |

- **w_c collapses** as the path leaves c (late ESS/M ≈ 0.01-0.02, worst step ≈ 1
  effective particle in all resampled variants), and w_u also has near-collapsed worst
  steps there.
- **The collapse is benign**: the classifier trained under it still delivers
  AUC ≈ 0.95-0.99 at mid/late, and only w_q enters the estimator (mean 0.68-0.87).

## Reproduction

Run these commands from the `exp_gmm` directory:

```bash
uv run python results/additional_diagnosis/run_ours_M_diag.py \
  --mode OFF --M 1024 --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9
uv run python results/additional_diagnosis/run_weight_ess.py \
  --mode OFF --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9
uv run python results/additional_diagnosis/run_tuned_M_diag.py \
  --variant cd48 --M 1024 --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9
```

Use the loops in `commands.sh` to run all variants and particle counts.
Logs (this directory): `ours_M_diag_{mode}_{M}_log.txt`, `weight_ess_{mode}_log.txt`,
`tuned_M_diag_{variant}_{M}_log.txt` (cd40 = ON_GRAD 40/40, cd48 = ON_GRAD 48/38).
