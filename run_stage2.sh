#!/bin/bash
# Stage 2: methane module on top of a Stage-1 checkpoint.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
ITERATIONS="${ITERATIONS:-50}"
SEED="${SEED:-42}"
STAGE1_DIR="${STAGE1_DIR:-output/stage1/seed_${SEED}}"
OUT_DIR="${OUT_DIR:-output/stage2/seed_${SEED}}"
mkdir -p "$OUT_DIR"

STAGE1_WEIGHTS=""
for cand in \
  "$STAGE1_DIR/spatial_model_params_run_${SEED}.pickle" \
  "$STAGE1_DIR/spatial_model_params.pickle" \
  "$STAGE1_DIR/spatial_model_params_last_run_${SEED}.pickle"
do
  if [ -f "$cand" ]; then STAGE1_WEIGHTS="$cand"; break; fi
done
if [ -z "$STAGE1_WEIGHTS" ]; then
  echo "ERROR: Stage-1 weights not found under $STAGE1_DIR"
  echo "Run: bash run_stage1.sh"
  exit 1
fi

"$PYTHON" -u spatial_calibration.py \
  --stage 2 \
  --stage1_weights "$STAGE1_WEIGHTS" \
  --enable_methane \
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

echo "Stage 2 finished: $OUT_DIR"
