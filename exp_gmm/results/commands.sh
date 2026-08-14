#!/bin/bash
# Commands for the quick 15-trial checks (5 layouts × 3 seeds per command).
# Run this script from the exp_gmm directory.
export OMP_NUM_THREADS=4   # required for reproducibility — changes MKL reduction order

SEEDS="0 1 2"
LAYOUTS=5

# Case 1: true-score
uv run python run_score.py --mode OFF --layouts $LAYOUTS --seeds $SEEDS --outdir results
uv run python run_score.py --mode ON  --layouts $LAYOUTS --seeds $SEEDS --outdir results
uv run python run_score.py --mode ON_TILT --layouts $LAYOUTS --seeds $SEEDS --outdir results

# Case 2: reference-based
uv run python run_ref.py --mode OFF --proposal grad --layouts $LAYOUTS --seeds $SEEDS --outdir results
uv run python run_ref.py --mode OFF --proposal sd   --layouts $LAYOUTS --seeds $SEEDS --outdir results
uv run python run_ref.py --mode ON  --proposal grad --layouts $LAYOUTS --seeds $SEEDS --outdir results
uv run python run_ref.py --mode ON  --proposal sd   --layouts $LAYOUTS --seeds $SEEDS --outdir results

# Baseline strength sweeps
uv run python run_tradeoff_negcfg.py --layouts $LAYOUTS --seeds $SEEDS --outdir results
uv run python run_tradeoff_sd.py     --layouts $LAYOUTS --seeds $SEEDS --outdir results
uv run python run_tradeoff_dng.py    --tau 0.2 --layouts $LAYOUTS --seeds $SEEDS --outdir results
uv run python run_tradeoff_dng.py    --tau 1.0 --layouts $LAYOUTS --seeds $SEEDS --outdir results
