# Quantitative Evaluation on GMMs

This directory evaluates negative guidance on a three-dimensional, twelve-component ring GMM. Three components are treated as the undesired class, and the other nine define the target distribution.

The two settings are:

- **Score-based setting**: The exact scores of the positive and negative GMMs are available.
  - **`ON`**: The **main** setting, using a weak DNG proposal and one scheduled resampling step.
  - **`OFF`**: Resampling is disabled, and the particles are tracked with their weights.
  - **`ON_TILT`**: The importance-ratio factor is set to one, giving the approximation used as a heuristic in the image-scale experiments.
  - **`ON_GRAD40` and `ON_GRAD48`**: Gradient proposals with a cosine guidance schedule and ESS-triggered resampling, using cooldown/stop settings of 40/40 and 48/38, respectively.
  We compare Ours with an exact Oracle sampler, Dynamic Negative Guidance (DNG), and Negative Prompting (NegCFG).
- **Reference-based setting**: The sampler receives reference samples from the undesired class. We compare Ours with gradient and SafeDenoiser proposals, as well as the SafeDenoiser baseline.

All experiments use a cosine VP-DDPM schedule with `T=100` diffusion steps and `M=1024` particles unless stated otherwise. They run on the CPU and require no external data or model checkpoint.

This directory serves as a demonstration that the framework works under several hyperparameter settings. Efficient hyperparameter selection remains future work.

## Directory layout

- **`gmm_ddpm.py`**: Gaussian models, the VP-DDPM schedule, and the Oracle sampler.
- **`gmm_ring.py`**: Ring-GMM construction and evaluation functions.
- **`kernel_model.py`**: Kernel classifier.
- **`sampler_score.py` and `sampler_ref.py`**: Samplers for the two settings.
- **`run_score.py` and `run_ref.py`**: Experiment runners for the two settings.
- **`run_tradeoff_*.py`**: DNG, NegCFG, and SafeDenoiser strength sweeps.
- **`visualize_*.py`**: GMM layouts, comparison figures, and sampling plots.
- **`results/`**: Logs, summaries, ablations, and experiment commands.
- **`plots/`**: Visualization outputs.
- **`pyproject.toml` and `uv.lock`**: Environment specification.

## Setup

Run the following commands from this directory:

```bash
cd exp_gmm
uv sync
export OMP_NUM_THREADS=4
```

The results were produced with the following environment:

| package | version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0+cu130 (CPU only) |
| NumPy | 2.4.2 |
| SciPy | 1.17.0 |
| POT | 0.9.6.post1 |
| Matplotlib | 3.10.8 |

The runs used an x86_64 CPU with `OMP_NUM_THREADS=4`. The complete environment is recorded in `uv.lock`.

## Reproduce the main results

### Run a small set of trials

The following command runs 15 trials for each selected configuration, using five layouts and seeds 0, 1, and 2. It covers both settings and the baseline strength sweeps.

There are three reproduction levels:

- `results/commands.sh`: a quick 15-trial check (5 layouts × 3 seeds).
- `results/50trials/commands.sh`: the main 50-trial score-based results used in the tables below (5 layouts × 10 seeds).
- `results/ablation/commands.sh` and `results/additional_diagnosis/commands.sh`: supplementary ablations and diagnostics.

```bash
bash results/commands.sh
```

To run one configuration directly:

```bash
# five independently generated layouts; three random seeds per layout

uv run python run_score.py \
  --mode ON_TILT \
  --layouts 5 \
  --seeds 0 1 2 \
  --outdir results

uv run python run_ref.py \
  --mode ON \
  --proposal both \
  --layouts 5 \
  --seeds 0 1 2 \
  --outdir results
```

The score-based modes are `OFF`, `ON`, `ON_TILT`, `ON_GRAD40`, and `ON_GRAD48`. The reference-based modes are `OFF` and `ON`, and the proposal choices are `grad`, `sd`, and `both`.

### Score-based setting

The included 50-trial summaries use five layouts and seeds 0 through 9. Each seed is independent.

```bash
bash results/50trials/commands.sh
```

This command writes per-seed logs and merged summaries under `results/50trials/`. It also runs the DNG and NegCFG strength sweeps.

The particle-count and guidance-strength ablations can be run with:

```bash
bash results/ablation/commands.sh
```

The merged ablation tables are written to `results/ablation/m_merged.txt` and `results/ablation/lambda_merged.txt`.

### Reference-based setting

The included reference-based summary uses five layouts and seeds 0, 1, and 2. The corresponding commands are in `results/commands.sh`, and the summary is available in [`results/ref_summary.md`](results/ref_summary.md).

Additional score-based runs are available in `results/additional_diagnosis/commands.sh`. They measure classifier accuracy, classifier-weight ESS across particle counts, and the effect of the number of diffusion steps on DNG.

## Quantitative evaluation

`post_mean` is the mean oracle posterior probability `p(c|x)` of the undesired class on generated samples.

`leak` is the fraction of samples with `p(c|x) > 0.5`. `Wasserstein` is the empirical 2-Wasserstein distance to the target GMM. Lower is better for all three quantities.

`raw` uses the generated particles directly, while `weighted` uses their final weights.

For the headline summaries, `OFF` is evaluated with the final particle weights because resampling is disabled; the resampled `ON` variants are evaluated from the raw particles. Detailed logs may report both views when useful.

### Score-based results

The following values are from the 50-trial score-based setting. Each entry is the mean ± standard deviation over 50 trials.

| Method | `post_mean` | Wasserstein |
|---|---:|---:|
| Oracle | 0.012 ± 0.010 | 1.5 ± 0.3 |
| DNG (`lambda0=1`, `tau=1`) | 0.052 ± 0.010 | 1.6 ± 0.3 |
| **Ours OFF (weighted)** | **0.007 ± 0.005** | **1.7 ± 0.4** |
| **Ours ON (raw)** | **0.017 ± 0.012** | **1.9 ± 0.5** |
| **Ours ON_TILT (raw)** | **0.010 ± 0.007** | **1.7 ± 0.4** |
| **Ours ON_GRAD48 (raw)** | **0.020 ± 0.012** | **1.7 ± 0.5** |
| Ours ON_GRAD40 (raw) | 0.029 ± 0.014 | 1.7 ± 0.5 |

At similar Wasserstein distances, the score-based Ours settings have lower `post_mean` than the DNG sweep. Mode coverage is 1.00 in all 50 trials.

### NegCFG strength sweep

| lambda | `post_mean` | Wasserstein |
|---:|---:|---:|
| 0.0 (unconditional) | 0.253 ± 0.016 | 3.6 ± 0.6 |
| 0.2 | 0.150 ± 0.013 | 3.5 ± 0.7 |
| 0.5 | 0.068 ± 0.008 | 9.7 ± 2.4 |
| 1.0 | 0.018 ± 0.005 | 20.5 ± 6.3 |

### DNG strength sweep

The following tables use 50 trials and show the effect of `lambda0` for `tau=1.0` and `tau=0.2`.

`tau=1.0`:

| `lambda0` | `post_mean` | Wasserstein |
|---:|---:|---:|
| 0.5 | 0.101 ± 0.013 | 2.2 ± 0.4 |
| 1.0 | 0.052 ± 0.010 | 1.6 ± 0.3 |
| 1.2 | 0.040 ± 0.009 | 1.8 ± 0.4 |
| 1.4 | 0.031 ± 0.007 | 2.2 ± 0.5 |
| 1.6 | 0.024 ± 0.007 | 2.6 ± 0.6 |
| 1.8 | 0.019 ± 0.006 | 3.0 ± 0.7 |
| 2.0 | 0.015 ± 0.005 | 3.5 ± 0.7 |

`tau=0.2`:

| `lambda0` | `post_mean` | Wasserstein |
|---:|---:|---:|
| 0.5 | 0.164 ± 0.013 | 2.3 ± 0.4 |
| 1.0 | 0.105 ± 0.011 | 2.1 ± 0.3 |
| 1.2 | 0.086 ± 0.010 | 2.5 ± 0.4 |
| 1.4 | 0.071 ± 0.009 | 3.0 ± 0.5 |
| 1.6 | 0.059 ± 0.008 | 3.5 ± 0.6 |
| 1.8 | 0.049 ± 0.007 | 4.1 ± 0.6 |
| 2.0 | 0.040 ± 0.006 | 4.6 ± 0.7 |

### Particle count

The following results use `ON_TILT`. The Oracle uses the same particle count.

| `M` | `post_mean` | Wasserstein | Oracle `post_mean` | Oracle Wasserstein |
|---:|---:|---:|---:|---:|
| 64 | 0.013 ± 0.015 | 3.63 ± 0.81 | 0.011 ± 0.011 | 3.28 ± 0.68 |
| 256 | 0.015 ± 0.011 | 2.33 ± 0.56 | 0.013 ± 0.013 | 2.40 ± 0.58 |
| 1024 | 0.010 ± 0.007 | 1.68 ± 0.38 | 0.012 ± 0.010 | 1.52 ± 0.34 |
| 4096 | 0.012 ± 0.008 | 1.25 ± 0.30 | 0.012 ± 0.010 | 1.10 ± 0.24 |

The `post_mean` is stable across particle counts, while Wasserstein decreases as the number of particles increases.

### Tilt strength

Each cell reports `post_mean / Wasserstein` at `M=1024`.

| variant | 0.5 | 1 | 2 | 3 | 5 |
|---|---:|---:|---:|---:|---:|
| OFF, weighted | 0.070 / 1.95 | 0.032 / 1.69 | 0.012 / 1.67 | 0.007 / 1.71 | 0.004 / 1.77 |
| ON, raw | 0.057 / 2.09 | 0.032 / 1.94 | 0.016 / 1.85 | 0.012 / 1.82 | 0.009 / 1.88 |
| ON_TILT, raw | 0.029 / 1.69 | 0.019 / 1.64 | 0.012 / 1.68 | 0.010 / 1.68 | 0.008 / 1.75 |
| ON_GRAD40, raw | 0.065 / 1.95 | 0.039 / 1.73 | 0.029 / 1.67 | 0.027 / 1.78 | 0.026 / 1.85 |
| ON_GRAD48, raw | 0.154 / 2.36 | 0.037 / 1.77 | 0.020 / 1.67 | 0.018 / 1.80 | 0.017 / 1.96 |

For the gradient variants, the Wasserstein minimum is near the main setting `lam_end=2`; larger values reduce `post_mean` slightly but increase Wasserstein.

## Visualizations

The following commands write figures to `plots/`:

```bash
uv run python visualize_layouts.py
uv run python visualize_score_paper_plot.py --layout 0 --seed 0  # one score-based example
uv run python visualize_score.py --mode ON_TILT --layout 0 --seed 0  # score-based snapshots
uv run python visualize_ref.py --mode ON --proposal grad --layout 0 --seed 0  # reference-based snapshots
```

`visualize_layouts.py` draws the GMM layouts. `visualize_score_paper_plot.py` creates the four-panel score-based figure used in the paper. `visualize_score.py` and `visualize_ref.py` also show classifier snapshots during sampling.
