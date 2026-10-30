#!/bin/bash
# Stage-2 spatial CV using drivers/example_cv_folds.json (requires Stage-1 weights).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
FOLDS_JSON="${FOLDS_JSON:-drivers/example_cv_folds.json}"
CV_ID="${CV_ID:-0}"
SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-50}"
STAGE1_ROOT="${STAGE1_ROOT:-output/stage1_cv}"
OUT_ROOT="${OUT_ROOT:-output/stage2_cv}"
STAGE1_DIR="${STAGE1_ROOT}/cv${CV_ID}/seed_${SEED}"
OUT_DIR="${OUT_ROOT}/cv${CV_ID}/seed_${SEED}"
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
  echo "Run: CV_ID=$CV_ID bash run_stage1_cv.sh"
  exit 1
fi

_SPLIT=$("$PYTHON" - <<PY
import json
with open("$FOLDS_JSON", encoding="utf-8") as f:
    fold = json.load(f)["folds"][int("$CV_ID")]
print(",".join(fold["train_ch4"]))
print(",".join(fold["test_ch4"]))
print(",".join(fold["val_ch4"]))
PY
)
TRAIN_SITES=$(echo "$_SPLIT" | sed -n '1p')
TEST_SITES=$(echo "$_SPLIT" | sed -n '2p')
VAL_SITES=$(echo "$_SPLIT" | sed -n '3p')

cat > "${OUT_DIR}/split.txt" <<EOF
CV_ID=$CV_ID
SEED=$SEED
TRAIN_SITES=$TRAIN_SITES
VAL_SITES=$VAL_SITES
TEST_SITES=$TEST_SITES
STAGE1_WEIGHTS=$STAGE1_WEIGHTS
EOF

"$PYTHON" -u spatial_calibration.py \
  --stage 2 \
  --stage1_weights "$STAGE1_WEIGHTS" \
  --enable_methane \
  --iterations "$ITERATIONS" \
  --batch_size "${BATCH_SIZE:-2}" \
  --learning_rate 5e-4 \
  --train_sites "$TRAIN_SITES" \
  --val_sites "$VAL_SITES" \
  --test_sites "$TEST_SITES" \
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

echo "Stage-2 CV finished: $OUT_DIR"
