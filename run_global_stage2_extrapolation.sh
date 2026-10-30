#!/bin/bash
# Global Stage-2 extrapolation.
# Requires PATHS_ENV, MASK_NPY, and Stage-2 parameter pickles.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
GE_OUTPUT_ROOT="${GE_OUTPUT_ROOT:-output/global_extrap}"
RUN_TAG="${RUN_TAG:-global_stage2}"
STAGE2_ENS_ROOT="${STAGE2_ENS_ROOT:-output/stage2}"
SEEDS="${SEEDS:-42}"
START_YEAR="${START_YEAR:-2001}"
END_YEAR="${END_YEAR:-2001}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RESUME="${RESUME:-0}"

if [ -z "${PATHS_ENV:-}" ] || [ -z "${MASK_NPY:-}" ]; then
  cat <<EOF
Global extrapolation requires grid drivers.

Set at least:
  PATHS_ENV   path to a paths.env listing ERA5 / static / wetland inputs
  MASK_NPY    land mask (.npy)
  STAGE2_ENS_ROOT  directory of Stage-2 ensemble pickles (seed_*/spatial_model_params_run_*.pickle)

Optional: GE_OUTPUT_ROOT RUN_TAG SEEDS START_YEAR END_YEAR BATCH_SIZE RESUME
EOF
  exit 1
fi

# Prefer PATHS_ENV; keep GE_PATHS_ENV as an alias for the Python entrypoint.
export GE_PATHS_ENV="${GE_PATHS_ENV:-$PATHS_ENV}"
export PYTHON GE_OUTPUT_ROOT RUN_TAG STAGE2_ENS_ROOT SEEDS START_YEAR END_YEAR BATCH_SIZE RESUME PATHS_ENV GE_PATHS_ENV MASK_NPY
"$PYTHON" -u tools/global_extrapolation/run_global_stage2_extrapolation.py
