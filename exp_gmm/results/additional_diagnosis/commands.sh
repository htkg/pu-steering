#!/bin/bash
# Additional diagnostics: classifier quality, classifier-weight ESS, and a DNG T sweep.
# Each configuration uses 50 trials (5 layouts × seeds 0-9). Run from the exp_gmm directory.
export OMP_NUM_THREADS=4   # required for reproducibility

# Classifier diagnostics and classifier-weight ESS across M.
for MODE in OFF ON ON_TILT; do
  for M in 64 256 1024 4096; do
    uv run python results/additional_diagnosis/run_ours_M_diag.py --mode $MODE --M $M --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9 --outdir results/additional_diagnosis
  done
  uv run python results/additional_diagnosis/run_weight_ess.py --mode $MODE --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9 --outdir results/additional_diagnosis
done
for V in cd40 cd48; do  # cd40 = ON_GRAD40; cd48 = ON_GRAD48
  for M in 64 256 1024 4096; do
    uv run python results/additional_diagnosis/run_tuned_M_diag.py --variant $V --M $M --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9 --outdir results/additional_diagnosis
  done
done

# DNG diffusion-step sweep
for T in 200 500 1000 2000; do
  uv run python results/additional_diagnosis/run_dng_T.py --T $T --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9 --outdir results/additional_diagnosis
done
