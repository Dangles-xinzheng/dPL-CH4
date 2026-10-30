#!/bin/bash
# Stage 1: carbon–water spatial training on bundled example sites.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
ITERATIONS="${ITERATIONS:-50}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-output/stage1/seed_${SEED}}"
mkdir -p "$OUT_DIR"

"$PYTHON" -u spatial_calibration.py \
  --stage 1 \
  --iterations "$ITERATIONS" \
  --batch_size "${BATCH_SIZE:-2}" \
  --learning_rate 5e-4 \
  --train_sites "BW-Gum,ID-Pag" \
  --val_sites "BW-Gum" \
  --test_sites "BW-Nxr" \
  --seed "$SEED" \
  --run_index "$SEED" \
  --output_dir "$OUT_DIR" \
  --drivers_dir "drivers" \
  --site_coords_file "site_coords.csv" \
  --disable_lai \
  --early_stopping_patience 100 \
  --validation_frequency 10 \
  --save_frequency 50 \
  --water_stress_type gpp_acm_uwue

echo "Stage 1 finished: $OUT_DIR"
