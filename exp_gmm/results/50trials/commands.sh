#!/bin/bash
# Commands to reproduce the 50-trial experiments (5 layouts × 10 seeds).
# Each seed is run independently.
# Run this script from the exp_gmm directory.
export OMP_NUM_THREADS=4   # required for reproducibility — changes MKL reduction order

SEEDS="0 1 2 3 4 5 6 7 8 9"
LAYOUTS=5

# Score-based runs without scheduled resampling.
for s in $SEEDS; do
  uv run python run_score.py --mode OFF --layouts $LAYOUTS --seeds $s \
    --outdir results/50trials/score_OFF/seed_${s}
done

# Score-based runs with particle saving for the ON configuration.
for s in $SEEDS; do
  uv run python run_score.py --mode ON --layouts $LAYOUTS --seeds $s \
    --outdir results/50trials/score_ON/seed_${s} \
    --save-particles results/particles
done

# Score-based runs with gradient proposals.
for m in ON_TILT ON_GRAD40 ON_GRAD48; do
  for s in $SEEDS; do
    uv run python run_score.py --mode $m --layouts $LAYOUTS --seeds $s \
      --outdir results/50trials/score_${m}/seed_${s}
  done
done

# NegCFG strength sweep
for s in $SEEDS; do
  uv run python run_tradeoff_negcfg.py --layouts $LAYOUTS --seeds $s \
    --outdir results/50trials/negcfg/seed_${s}
done

# DNG strength sweeps
for s in $SEEDS; do
  uv run python run_tradeoff_dng.py --tau 0.2 --layouts $LAYOUTS --seeds $s \
    --outdir results/50trials/dng/tau0.2/seed_${s}
done

for s in $SEEDS; do
  uv run python run_tradeoff_dng.py --tau 1.0 --layouts $LAYOUTS --seeds $s \
    --outdir results/50trials/dng/tau1.0/seed_${s}
done

uv run python results/50trials/merge.py
