# Quantitative Evaluation on CIFAR-10

This directory evaluates negative guidance with an unconditional 32×32 CIFAR-10 diffusion model.
The main result is the trade-off between the rate of images classified as an avoided class and the quality of the remaining distribution.

The single-class experiments avoid Plane, Frog, Cat, or Ship. The multi-class experiment avoids the vehicle group: Plane, Car, Ship, and Truck.

SafeDenoiser is used as the baseline, and Ours combines it with a learned classifier and particle weighting.

## Directory layout

- **`config.py`**: Model loading, data utilities, and class groups.
- **`sampler.py`**: Classifier, proposals, and sampling trials.
- **`generate.py`**: Generate samples for one seed.
- **`evaluate.py`**: Aggregate seeds and compute metrics.
- **`run_sweep.py`**: Single-class parameter sweep.
- **`cifar10_classifier/train_classifier.py`**: Train the evaluation classifier.
- **`plots/plot_multi_eta.py`**: Plot the included multi-scale summaries.
- **`plots/plot_tradeoff.py`**: Plot the 80-seed results stored in the script.
- **`plots/extract_plot_data.py`**: Extract values from locally generated summaries.
- **`results/`**: Per-seed outputs and summary files.
- **`results/diagnostics/`**: Supplementary summaries of per-seed variation.
- **`improved-diffusion/`**: [OpenAI's diffusion implementation](https://github.com/openai/improved-diffusion) under the MIT License.

## Setup

The reported results were generated on an aarch64 NVIDIA GH200 with one GPU, using `nvcr.io/nvidia/pytorch:25.03-py3`.

With Singularity, obtain the container with:

```bash
singularity pull docker://nvcr.io/nvidia/pytorch:25.03-py3
```

| package | version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.7.0a0+7c8ec84dab.nv25.03 |
| CUDA | 12.8 |
| torchvision | 0.22.0a0 |
| NumPy | 1.26.4 |
| SciPy | provided by the container |
| matplotlib | provided by the container |
| Pillow | provided by the container |

The main scripts require an NVIDIA GPU and do not have a CPU fallback. Install only the packages that are added by this experiment into a directory under the repository:

```bash
cd exp_cifar10
mkdir -p packages
python -m pip install --upgrade --no-deps --target packages \
  pytorch-fid==0.3.0 \
  lpips==0.1.4 \
  'blobfile>=1.0.5' \
  adjustText
```

Run the commands below inside the NVIDIA PyTorch container. Add the local packages to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/packages${PYTHONPATH:+:$PYTHONPATH}"
```

For deterministic CUDA operations, set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The reported 80-seed runs used eight seeds per job with NVIDIA MPS, and the jobs were run in parallel across available GPUs.

The following files are required but are not included because of their size:

- `improved-diffusion/cifar10_uncond_50M_500K.pt`: The unconditional CIFAR-10 diffusion checkpoint. Download it from [OpenAI](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt).
- `cifar10_classifier/models/cifar10_classifier.pt`: The ResNet-18 evaluation classifier checkpoint.
- `cifar10_classifier/data/`: The CIFAR-10 train and test sets used for reference images and evaluation features.

Download the CIFAR-10 data with torchvision:

```bash
python - <<'PY'
from torchvision import datasets

root = "cifar10_classifier/data"
datasets.CIFAR10(root=root, train=True, download=True)
datasets.CIFAR10(root=root, train=False, download=True)
PY
```

Train the evaluation classifier if it is not available:

```bash
python cifar10_classifier/train_classifier.py --seed 42
```

The classifier is saved to `cifar10_classifier/models/cifar10_classifier.pt`.

## Reproduce the main results

### Generate one configuration

`generate.py` writes `seed_NNN/samples.pt` and `seed_NNN/log.json` for one seed. It does not compute FID, KL, or MPD.

Single-class example:

```bash
python generate.py \
  --seed 0 \
  --target_class plane \
  --lam_max 1.0 \
  --sd_scale 0.3 \
  --pu_lam 0.001 \
  --out_dir results/plane_ours_lam1.0
```

SafeDenoiser baseline:

```bash
python generate.py \
  --seed 0 \
  --target_class plane \
  --lam_max 0 \
  --sd_scale 0.3 \
  --out_dir results/plane_sd_0.3
```

Set `--group vehicles` for the multi-class experiment.

### Evaluate one configuration

After generating the seeds for one configuration, aggregate them and compute the metrics:

```bash
python evaluate.py \
  --out_dir results/plane_ours_lam1.0 \
  --target_class plane \
  --M 128
```

For the vehicle group, replace `--target_class plane` with `--group vehicles`. The command writes `summary.json`.

## Quantitative evaluation

The metrics are:

- **`Rate`**: The fraction of generated images classified as the avoided class or group by the evaluation classifier. Lower is better.
- **`FID`**: The distance to CIFAR-10 training images excluding the avoided class or group. Lower is better.
- **`KL`**: The KL divergence from the uniform distribution over the non-avoided classes. Lower is better.
- **`MPD`**: The mean pairwise LPIPS distance between generated images. This measures diversity and is reported as an auxiliary metric.

The supplementary per-seed variation of Rate is summarized in `results/diagnostics/LEAK_STD_DIAGNOSTICS.md`.

### Single-class sweep

The reported single-class sweep has 10 configurations: six SafeDenoiser scales (`0.0, 0.2, 0.3, 0.5, 0.7, 1.0`) and four Ours values of `lam_max` (`0.5, 1.0, 1.5, 2.0`). Each configuration uses seeds 0-79 and 128 particles per seed.

The class-specific regularization is:

| avoided class | `pu_lam` |
|---|---:|
| Plane, Cat | 0.001 |
| Frog, Ship | 1.0 |

Run the complete sweep for one class:

```bash
python run_sweep.py --target_class plane --seeds 80 --pu_lam 0.001
```

Use `--target_class frog --pu_lam 1.0`, `--target_class cat --pu_lam 0.001`, or `--target_class ship --pu_lam 1.0` for the other classes. For a shorter run, reduce `--seeds`.

### Vehicle-group sweep

The vehicle experiment avoids Plane, Car, Ship, and Truck. Use `--group vehicles` for all configurations. Ours also uses `--use_denoised_feat --resample_stop 50`.

SafeDenoiser:

```bash
for scale in 0.0 0.2 0.3 0.5 0.7 1.0; do
  out_dir="results/vehicles_sd_${scale}"
  for seed in $(seq 0 79); do
    python generate.py --seed "$seed" --group vehicles \
      --lam_max 0 --sd_scale "$scale" --out_dir "$out_dir"
  done
  python evaluate.py --out_dir "$out_dir" --group vehicles --M 128
done
```

Ours:

```bash
for lam in 0.5 1.0 1.5; do
  out_dir="results/vehicles_ours_lam${lam}"
  for seed in $(seq 0 79); do
    python generate.py --seed "$seed" --group vehicles \
      --lam_max "$lam" --sd_scale 0.3 --pu_lam 1.0 \
      --use_denoised_feat --resample_stop 50 --out_dir "$out_dir"
  done
  python evaluate.py --out_dir "$out_dir" --group vehicles --M 128
done

out_dir="results/vehicles_ours_lam2.0_pl5"
for seed in $(seq 0 79); do
  python generate.py --seed "$seed" --group vehicles \
    --lam_max 2.0 --sd_scale 0.3 --pu_lam 5.0 \
    --use_denoised_feat --resample_stop 50 --out_dir "$out_dir"
done
python evaluate.py --out_dir "$out_dir" --group vehicles --M 128
```

### Multi-scale sweep

The multi-scale sweep repeats the Ours configurations with `sd_scale=0.2` and `sd_scale=0.5`. The output directories use `sd02` and `sd05`:

```bash
for cls in plane frog cat ship; do
  case "$cls" in
    plane|cat) pu_lam=0.001 ;;
    frog|ship) pu_lam=1.0 ;;
  esac

  for pair in "0.2 02" "0.5 05"; do
    read -r scale tag <<< "$pair"
    for lam in 0.5 1.0 1.5 2.0; do
      out_dir="results/${cls}_ours_lam${lam}_sd${tag}"
      for seed in $(seq 0 79); do
        python generate.py --seed "$seed" --target_class "$cls" \
          --lam_max "$lam" --sd_scale "$scale" --pu_lam "$pu_lam" \
          --out_dir "$out_dir"
      done
      python evaluate.py --out_dir "$out_dir" --target_class "$cls" --M 128
    done
  done
done
```

For the vehicle group, use the same two scales with `--group vehicles --use_denoised_feat --resample_stop 50`. Use `pu_lam=5.0` for `lam_max=2.0` and `pu_lam=1.0` for the other values.

## Visualizations

The included `summary_short.json` files are sufficient for the multi-scale plots:

```bash
python plots/plot_multi_eta.py
```

This writes the plots to `plots/plots_multi_eta/`.

`plots/plot_tradeoff.py` uses 80-seed values stored in the script and does not read `results/`. It recreates the single-class and vehicle figures:

```bash
python plots/plot_tradeoff.py --outdir plots
```

Use `plots/extract_plot_data.py` to extract values from locally generated `summary.json` files:

```bash
python plots/extract_plot_data.py --results_dir results
```

To create compact summaries after a local sweep:

```bash
(cd results && python summary_short.py)
```
