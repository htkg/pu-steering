# DNG T (diffusion steps) ablation — Case 1 (true-score)

**Setup.** Only `T` varies. DNG fixed to the main-table config
(`lambda0=1, tau=1, pmin=1e-2, lam_max=20`), `M=1024`, ring GMM (radius 10),
5 layouts × seeds 0-9 = 50 trials, `OMP_NUM_THREADS=4`. The Oracle is recomputed at
each `T` as a matched floor. `post_mean` = leak (oracle posterior mass on the
undesired modes); lower is better. `T=100` from the main table (`results/50trials/`).

| T | Oracle pm | DNG post_mean (leak) | DNG lk | DNG Wass | gap to Oracle |
|------|-----------|----------------------|--------|-----------|---------------|
| 100  | 0.0120 | 0.0521 ± 0.0104 | 0.050 | 1.6 ± 0.3 | 0.040 |
| 200  | 0.0118 | 0.0387 ± 0.0091 | 0.035 | 1.6 ± 0.3 | 0.027 |
| 500  | 0.0110 | 0.0285 ± 0.0070 | 0.025 | 1.5 ± 0.3 | 0.018 |
| 1000 | 0.0110 | 0.0227 ± 0.0082 | 0.019 | 1.5 ± 0.3 | 0.012 |
| 2000 | 0.0113 | 0.0196 ± 0.0089 | 0.016 | 1.5 ± 0.3 | 0.008 |

**Findings.**

- Leak decreases monotonically with `T` (0.052 → 0.020) but with **clear diminishing
  returns**: most of the gain is by `T≈500`; 500→2000 only moves leak 0.029 → 0.020.
- The **gap to Oracle shrinks (0.040 → 0.008) but never closes** — DNG keeps a
  structural residual leak (~1.7× the Oracle floor even at `T=2000`, 20× the compute).
- **Wasserstein is flat (1.5-1.6)** across all `T`: more steps lower the leak without
  degrading sample quality.

Run from the `exp_gmm` directory:

```bash
uv run python results/additional_diagnosis/run_dng_T.py \
  --T <T> --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9
```

Logs: `results/additional_diagnosis/dng_T_{200,500,1000,2000}_log.txt`.
