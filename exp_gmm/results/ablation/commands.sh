#!/bin/bash
# Ablations: 50 trials (5 layouts × seeds 0-9) per configuration.
# Run this script from the exp_gmm directory. Configurations run sequentially and append one
# JSON record per trial to dump.txt.
export OMP_NUM_THREADS=4   # required for reproducibility

TUNE="uv run python results/ablation/tune_ours_score.py --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9 --calib-pi --pu-sigma-scale 0.2 --pu-lam 1e-4 --ess-target 0.6 --dump"
DUMP=results/ablation/dump.txt
: > "$DUMP"
OFF="--proposal grad --grad-clip 0.5 --tilt-start 0.6"
ON="--proposal dng --dng-l0 0.6 --dng-tau 1.0 --dng-lam-max 4.0 --tilt-max 0 --resample-at 40"
ONT="--proposal dng --dng-l0 0.6 --dng-tau 1.0 --dng-lam-max 4.0 --wq-tilt-only --resample-at 40"

# Particle-count ablation; the Oracle uses the same M.
for M in 64 256 1024 4096; do
  $TUNE $OFF --tilt-max 3.0  --M $M --oracle --tag m_OFF_M$M     >> $DUMP
  $TUNE $ON  --lam-fixed 2.0 --M $M --oracle --tag m_ON_M$M      >> $DUMP
  $TUNE $ONT --tilt-max 3.0  --M $M --oracle --tag m_ON_TILT_M$M >> $DUMP
done

# Guidance-strength ablation at M=1024.
for L in 0.5 1.0 2.0 3.0 5.0; do
  $TUNE $OFF --tilt-max $L --tag l_OFF_tm$L     >> $DUMP
  $TUNE $ONT --tilt-max $L --tag l_ON_TILT_tm$L >> $DUMP
done
for L in 0.5 1.0 1.5 2.0 3.0 4.0 5.0; do
  $TUNE $ON --lam-fixed $L --tag l_ON_lf$L >> $DUMP
done

# Collect and merge the JSONL records.
uv run python results/ablation/collect.py $DUMP
uv run python results/ablation/merge.py

# Repeat both ablations for the ON_GRAD variants.
ONG40="--grad-clip 0.5 --lam-mode cos --lam-end 2.0 --lam-ramp 0.5 --tilt-max 0 --ess-resample 0.9 --resample-stop 40 --resample-cooldown 40"
ONG48="--grad-clip 0.5 --lam-mode cos --lam-end 2.0 --lam-ramp 0.5 --tilt-max 0 --ess-resample 0.9 --resample-stop 38 --resample-cooldown 48"
for M in 64 256 1024 4096; do
  $TUNE $ONG40 --M $M --oracle --tag m_ON_GRAD40_M$M >> $DUMP
  $TUNE $ONG48 --M $M --oracle --tag m_ON_GRAD48_M$M >> $DUMP
done
for L in 0.5 1.0 2.0 3.0 5.0; do
  $TUNE $ONG40 --lam-end $L --tag l_ON_GRAD40_le$L >> $DUMP
  $TUNE $ONG48 --lam-end $L --tag l_ON_GRAD48_le$L >> $DUMP
done
