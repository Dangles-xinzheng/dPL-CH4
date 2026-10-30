#!/bin/bash
# Stage-1 spatial CV using drivers/example_cv_folds.json.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
FOLDS_JSON="${FOLDS_JSON:-drivers/example_cv_folds.json}"
CV_ID="${CV_ID:-0}"
SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-50}"
OUT_ROOT="${OUT_ROOT:-output/stage1_cv}"
OUT_DIR="${OUT_ROOT}/cv${CV_ID}/seed_${SEED}"
mkdir -p "$OUT_DIR"

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
EOF

"$PYTHON" -u spatial_calibration.py \
  --stage 1 \
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

echo "Stage-1 CV finished: $OUT_DIR"
