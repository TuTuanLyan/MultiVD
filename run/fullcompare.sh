#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${CONFIG_FILE:-$ROOT/run/config.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

SEED="${1:?usage: bash run/fullcompare.sh SEED}"
TRANSFER_DIR="results/$RUN_NAME/$TRANSFER_NAME/seed_$SEED"
BASELINE_DIR="results/$RUN_NAME/$BASELINE_NAME/seed_$SEED"
COMPARE_DIR="results/$RUN_NAME/$COMPARE_NAME/seed_$SEED"
COMPARE_LOG_DIR="log/$RUN_NAME/$COMPARE_NAME/seed_$SEED"
mkdir -p "$COMPARE_DIR" "$COMPARE_LOG_DIR"

SECONDS=0
echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Full comparison started | Run: $RUN_NAME | Seed: $SEED" | tee -a "$COMPARE_LOG_DIR/pipeline.log"
TRANSFER_STARTED=$SECONDS
bash run/fullpipeline.sh "$SEED"
TRANSFER_SECONDS=$((SECONDS - TRANSFER_STARTED))
BASELINE_STARTED=$SECONDS
bash run/fullbaseline.sh "$SEED"
BASELINE_SECONDS=$((SECONDS - BASELINE_STARTED))

python -u src/compare_results.py \
  --transfer_dir "$TRANSFER_DIR" \
  --baseline_dir "$BASELINE_DIR" \
  --output_dir "$COMPARE_DIR" \
  --seed "$SEED" \
  --transfer_pipeline_seconds "$TRANSFER_SECONDS" \
  --baseline_pipeline_seconds "$BASELINE_SECONDS" \
  2>&1 | tee -a "$COMPARE_LOG_DIR/compare.log"

echo "Comparison: $COMPARE_DIR/comparison.json"
echo "Paired CSV: $COMPARE_DIR/paired_folds.csv"
echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Full comparison finished | Run: $RUN_NAME | Seed: $SEED | Elapsed: ${SECONDS}s" | tee -a "$COMPARE_LOG_DIR/pipeline.log"
