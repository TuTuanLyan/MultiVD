#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${CONFIG_FILE:-$ROOT/run/config.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

SEED="${1:?usage: bash run/fullbaseline.sh SEED}"
LOG_DIR="log/$RUN_NAME/$BASELINE_NAME/seed_$SEED"
MODEL_DIR="model/$RUN_NAME/$BASELINE_NAME/seed_$SEED"
SEED_RESULTS="results/$RUN_NAME/$BASELINE_NAME/seed_$SEED"
mkdir -p "$LOG_DIR" "$MODEL_DIR" "$SEED_RESULTS"

SECONDS=0
echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Baseline pipeline started | Run: $RUN_NAME | Seed: $SEED" | tee -a "$LOG_DIR/pipeline.log"
for fold in 1 2 3 4 5; do
  bash run/train-baseline.sh "$fold" "$SEED"
  bash run/infer-baseline.sh "$fold" "$SEED"
done

python -u src/summarize_results.py \
  --input_dir "$SEED_RESULTS" \
  --output_dir "$SEED_RESULTS" \
  2>&1 | tee -a "$LOG_DIR/summary.log"

echo "Summary: $SEED_RESULTS/summary.json"
echo "Runs CSV: $SEED_RESULTS/all_runs.csv"
echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO - Baseline pipeline finished | Run: $RUN_NAME | Seed: $SEED | Elapsed: ${SECONDS}s" | tee -a "$LOG_DIR/pipeline.log"
