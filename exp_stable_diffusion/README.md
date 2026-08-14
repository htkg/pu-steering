# Quantitative Evaluation on Stable Diffusion

This directory evaluates negative guidance for text-to-image generation.
It generates "a cafe terrace at night under a starry sky" while reducing the amount of Van Gogh style without mentioning Van Gogh in the content prompt.

It compares standard negative prompting and several negative-guidance methods with Ours, an SMC sampler with a learned classifier.
The target distribution uses the content prompt without a style modifier, and the avoided distribution appends "in the style of Vincent van Gogh".

The reported experiment uses Stable Diffusion v1.4 and the CSD-ViT-L style embedding model.

## Directory layout

- **`src/`**: Stable Diffusion sampling, proposals, the learned classifier, SMC updates, and metrics.
- **`runs/gen_pool.py`**: Generate the content, Van Gogh, calibration, and independent reference pools.
- **`runs/build_dataset.py`**: Build the reference set and its statistics.
- **`runs/build_floor.py`**: Compute the reference point from two independent reference sets.
- **`runs/gen_method.py`**: Generate one output file for a method.
- **`runs/assemble_frontier.py`**: Build the trade-off table from cached features.
- **`runs/plot_frontier.py`**: Render the trade-off plot.
- **`runs/*_grids.py`**: Render sample and reference grids from cached images.
- **`configs/dataset/`**: Pool-generation and reference-building configurations.
- **`configs/methods/`**: Baseline and Ours configurations for the two output files of each method.
- **`results/dataset/cafe_terrace/`**: Generated caches and summary files.

## Overview

The experiment runs as a fixed pipeline:

```text
gen_pool.py (GPU)
    -> build_dataset.py (GPU; fit the CSD logistic head and select the reference set)
    -> build_floor.py (post-processing; compute the independent reference point)
    -> gen_method.py (GPU; generate two shards for each method)
    -> assemble_frontier.py (post-processing; recompute the metrics from cached features)
    -> plot_frontier.py / *_grids.py (visualization)
```

Run the commands from `exp_stable_diffusion`. The generated data lives below
`results/dataset/cafe_terrace/`:

| stage | required outputs | meaning |
|---|---|---|
| pools | `content_part0..7.pt`, `vangogh_part0..1.pt`, `calib.pt` | content, avoided-style, and classifier-calibration pools |
| reference | `dataset.pt`, `base_info.json` | the lowest-p(c) 25% of the content pool and its statistics |
| floor | `floor_part0..7.pt`, `floor.json` | an independent reference set and the N-matched reference point |
| methods | `{name}_shard0.pt`, `{name}_shard1.pt` | two 1,024-image shards for each method |
| frontier | `frontier.json` | the final metric table used by the plot |

The `_s0` and `_s1` suffixes in method config filenames mean shard 0 and shard 1,
not two different algorithms. Both configurations must be run for every method;
the expected total is 2,048 images per method. The aggregation script can still
read a partial set of shards, so check the `n` field in `frontier.json` before
using the result. A complete method entry should have `n=2048`.

## Setup

### Hardware and environment

The reported results were generated on an aarch64 NVIDIA GH200 120 GB GPU. The code uses CUDA directly.

The base environment is the NVIDIA PyTorch container
`nvcr.io/nvidia/pytorch:25.03-py3`:

| package | version |
|---|---|
| Python | 3.12.3 |
| torch | 2.7.0a0+7c8ec84dab.nv25.03 |
| torchvision | 0.22.0a0 |
| NumPy | 1.26.4 |
| SciPy | 1.15.2 |
| Pillow | 11.1.0 |

Install the following packages in `.deps/` under this directory rather than in the container's system environment:

| package | version |
|---|---|
| diffusers | 0.33.0 |
| transformers | 4.44.0 |
| tokenizers | 0.19.1 |
| huggingface-hub | 0.36.2 |
| safetensors | 0.5.3 |
| accelerate | 1.6.0 |
| pytorch-fid | 0.3.0 |

From the repository root, pull the container and create the local dependency directory:

```sh
cd exp_stable_diffusion
singularity pull sd_nvidia.sif docker://nvcr.io/nvidia/pytorch:25.03-py3
mkdir -p .deps
singularity exec --nv sd_nvidia.sif python -m pip install \
  --target .deps --no-deps \
  diffusers==0.33.0 \
  transformers==4.44.0 \
  tokenizers==0.19.1 \
  huggingface-hub==0.36.2 \
  safetensors==0.5.3 \
  accelerate==1.6.0 \
  pytorch-fid==0.3.0
```

Enter the container with this directory as the working directory:

```sh
singularity shell --nv --pwd "$PWD" sd_nvidia.sif
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
```

Run the commands below inside the container after setting `PYTHONPATH`. The scripts write paths relative to the current working directory.

### Model downloads

On the first run, the scripts download the following public models:

| model | purpose |
|---|---|
| `CompVis/stable-diffusion-v1-4` | image generation |
| `tomg-group-umd/CSD-ViT-L` | style embeddings and the style classifier features |

`pytorch-fid` may also download its InceptionV3 weights on first use. The model files and generated cache files are not included in this repository.

## Build the evaluation dataset

Run the commands below from `exp_stable_diffusion` after entering the container and setting `PYTHONPATH` as described above.

The default tag is `cafe_terrace`.

### Generate the content, style, and calibration pools

The content pool has eight output files (8,192 images total), the Van Gogh pool has two output files (2,048 images total), and the calibration pool is a separate file with Van Gogh positives and Rembrandt/plain negatives.
The content, Van Gogh, and floor files cache images, Inception-2048 features, and CSD-768 features.
`calib.pt` stores images and CSD features for fitting the classifier.

```sh
for part in 0 1 2 3 4 5 6 7; do
  python runs/gen_pool.py "configs/dataset/ds_content${part}.json"
done

for part in 0 1; do
  python runs/gen_pool.py "configs/dataset/ds_vangogh${part}.json"
done

python runs/gen_pool.py configs/dataset/ds_calib0.json
```

The output is written to `results/dataset/cafe_terrace/`.

### Select the reference images

The calibration set fits a logistic head on CSD-768 embeddings. The head is then applied to the 8,192 content images, and the lowest-p(c) 25% - 2,048 images - is kept as the reference set.

```sh
python runs/build_dataset.py configs/dataset/ds_build.json
```

This writes `dataset.pt` and `base_info.json`. The first file is a small index; the large feature arrays remain in the pool files.

## Compute the reference point

The reference point uses fresh seeds, so its images are independent of the main content pool.
Generate the eight floor files and construct a second reference set with the same p(c) threshold:

```sh
for part in 0 1 2 3 4 5 6 7; do
  python runs/gen_pool.py "configs/dataset/ds_floor${part}.json"
done

python runs/build_floor.py configs/dataset/floor_build.json
```

`floor.json` contains the FID, KID, and CSD-Fréchet distance between the two independent 2,048-image reference sets.
These values define the reference point used in the trade-off plot.

## Generate samples

Each method has two configurations, one for each output file.
The following loop generates every configured baseline and Ours output file.
It excludes `assemble_ours.json`, which is an aggregation configuration rather than a generation configuration.

```sh
for config in configs/methods/*_s[01].json; do
  python runs/gen_method.py "$config"
done
```

For a single method, run its two configurations explicitly. For example:

```sh
python runs/gen_method.py configs/methods/ou_m128_e5_l16_s0.json
python runs/gen_method.py configs/methods/ou_m128_e5_l16_s1.json
```

Each output file contains 1,024 images.
The method caches also contain Inception-2048 and CSD-768 features; Ours additionally stores per-particle SMC weights.
A complete regeneration of the method caches requires roughly 100 GB of storage.

## Collect evaluation results and visualize

This step reads the cached features and calculates p(c), FID, KID, CSD-Fréchet distance, and the weighted metrics for Ours. It does not regenerate images or run the feature extractors.

The `unconditional_content` row in `frontier.json` is a fixed random sample of 2,048 images from `floor_part1.pt` through `floor_part7.pt`.
It is an independent reference for the trade-off plot, so its FID is not the same as the content-to-reference value in `base_info.json`.

```sh
python runs/assemble_frontier.py configs/methods/assemble_ours.json
```

The result is written to `results/dataset/cafe_terrace/frontier.json`.

### Render the plot and image grids

The following commands use cached data and do not require a GPU:

```sh
python runs/plot_frontier.py configs/methods/assemble_ours.json
python runs/sample_grids.py
python runs/batch_grids.py
python runs/reference_grids.py
```

They write `frontier.png`, `frontier.pdf`, and image grids below `results/dataset/cafe_terrace/grids/`.

## Quantitative evaluation

All metrics are computed against the 2,048-image reference set. Lower is better.

- `p(c)`: Mean probability assigned to the avoided Van Gogh style by the CSD logistic head.
- `FID`: Fréchet distance in Inception-2048 space.
- `KID`: unbiased cubic-kernel MMD in Inception-2048 space.
- `CSD-FD`: Fréchet distance in the CSD-768 feature space.
- `ESS/M`: mean effective sample size divided by the particle count, reported for the weighted Ours estimates.

For Ours, unweighted metrics describe the generated images with equal weights, while weighted metrics describe the SMC estimate.
Baselines have uniform weights, so their weighted and unweighted metrics are identical.

The included `frontier.json` was produced with 2,048 images per method.

### Reference values

| reference | p(c)↓ | FID↓ | KID↓ | CSD-FD↓ |
|---|---:|---:|---:|---:|
| **reference point** | 0.238 | 18.52 | 0.0002 | 0.009 |
| content (no avoidance) | 0.409 | 60.98 | 0.0552 | 0.213 |
| avoided style (Van Gogh) | 0.645 | 164.32 | 0.1982 | 0.661 |

### Baselines

NegCFG is swept over its scale `s`. DNG uses the content-CFG base and is swept over `lambda0`.

| method | p(c)↓ | FID↓ | KID↓ | CSD-FD↓ |
|---|---:|---:|---:|---:|
| SLD (max) | 0.334 | 43.08 | 0.0294 | 0.114 |
| SLD (xxmax) | 0.317 | 41.82 | 0.0263 | 0.118 |
| SLD (x3max) | 0.296 | 43.70 | 0.0284 | 0.180 |
| NegCFG (s=1.5) | 0.279 | 112.58 | 0.1074 | 0.939 |
| NegCFG (s=2) | 0.246 | 100.44 | 0.0902 | 0.924 |
| NegCFG (s=2.5) | 0.229 | 91.88 | 0.0778 | 0.911 |
| NegCFG (s=3) | 0.222 | 85.56 | 0.0678 | 0.900 |
| NegCFG (s=4) | 0.224 | 82.75 | 0.0608 | 0.897 |
| NegCFG (s=5) | 0.233 | 85.21 | 0.0629 | 0.904 |
| NegCFG (s=6) | 0.254 | 87.86 | 0.0660 | 0.935 |
| NegCFG (s=7.5) | 0.280 | 97.35 | 0.0775 | 0.992 |
| DNG (content, lambda0=100) | 0.382 | 56.87 | 0.0500 | 0.192 |
| DNG (content, lambda0=150) | 0.344 | 49.18 | 0.0370 | 0.155 |
| DNG (content, lambda0=180) | 0.322 | 44.94 | 0.0303 | 0.145 |
| DNG (content, lambda0=200) | 0.303 | 44.88 | 0.0273 | 0.145 |
| DNG (content, lambda0=220) | 0.292 | 47.22 | 0.0269 | 0.163 |
| DNG (content, lambda0=240) | 0.284 | 51.80 | 0.0298 | 0.196 |
| DNG (content, lambda0=260) | 0.277 | 63.84 | 0.0411 | 0.243 |
| DNG (content, lambda0=280) | 0.284 | 77.45 | 0.0556 | 0.304 |
| DNG (content, lambda0=300) | 0.289 | 93.17 | 0.0737 | 0.366 |

### Ours

The first row for each method is unweighted, corresponding to the generated images. The second row is the weighted SMC estimate.

| method | M | weighting | p(c)↓ | FID↓ | KID↓ | CSD-FD↓ | ESS/M | ESS (mean±std) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **ou_m128_e5_l16** | 128 | unweighted | 0.235 | 28.10 | 0.0054 | 0.095 | - | - |
| | | weighted | 0.228 | 38.68 | 0.0047 | 0.110 | 0.47 | 59.9 ± 6.9 |
| ou_m128_e5_l24 | 128 | unweighted | 0.253 | 29.54 | 0.0058 | 0.082 | - | - |
| | | weighted | 0.239 | 39.95 | 0.0060 | 0.111 | 0.50 | 63.9 ± 12.4 |
| ou_m128_e4_l20 | 128 | unweighted | 0.243 | 31.07 | 0.0056 | 0.085 | - | - |
| | | weighted | 0.245 | 43.25 | 0.0072 | 0.092 | 0.51 | 65.0 ± 13.7 |
| ou_m64_e5_l16 | 64 | unweighted | 0.276 | 33.46 | 0.0105 | 0.086 | - | - |
| | | weighted | 0.283 | 43.27 | 0.0115 | 0.096 | 0.48 | 30.9 ± 2.8 |
| ou_e5_l16 | 64 | unweighted | 0.280 | 33.41 | 0.0106 | 0.090 | - | - |
| | | weighted | 0.281 | 43.42 | 0.0116 | 0.094 | 0.49 | 31.5 ± 4.5 |
| ow_nores_l8 | 64 | unweighted | 0.314 | 40.41 | 0.0243 | 0.114 | - | - |
| | | weighted | 0.272 | 39.23 | 0.0104 | 0.103 | 0.50 | 32.0 ± 0.6 |
| ow_nores_l12 | 64 | unweighted | 0.319 | 41.76 | 0.0256 | 0.122 | - | - |
| | | weighted | 0.268 | 38.08 | 0.0090 | 0.106 | 0.51 | 32.5 ± 0.6 |

Ours achieves FID 28-33 with unweighted images, compared with the reference point at 18.5, while reducing p(c) from 0.41 for the content distribution to 0.24-0.32.
NegCFG substantially increases FID across the sweep.
DNG improves over the content distribution until approximately `lambda0=200`, after which its FID increases again.
