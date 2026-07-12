#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${CONFIG_FILE:-$ROOT/run/config.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

FOLD="${1:?usage: bash run/train-baseline.sh FOLD SEED}"
SEED="${2:?usage: bash run/train-baseline.sh FOLD SEED}"
LOG_DIR="log/$RUN_NAME/$BASELINE_NAME/seed_$SEED"
MODEL_DIR="model/$RUN_NAME/$BASELINE_NAME/seed_$SEED"
RESULT_DIR="results/$RUN_NAME/$BASELINE_NAME/seed_$SEED"
TARGET="$MODEL_DIR/fold$FOLD/best.pt"
mkdir -p "$LOG_DIR" "$(dirname "$TARGET")" "$RESULT_DIR"

python -u src/train_baseline.py \
  --phase train \
  --run_name "$RUN_NAME" \
  --method_name "$BASELINE_NAME" \
  --fold "$FOLD" \
  --seed "$SEED" \
  --checkpoint_path "$TARGET" \
  --epochs "$BASELINE_EPOCHS" \
  --min_epochs "$MIN_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy "$TRUNCATION_STRATEGY" \
  --learning_rate "$BASELINE_LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --patience "$PATIENCE" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --num_workers "$NUM_WORKERS" \
  --model_name "$MODEL_NAME" \
  2>&1 | tee -a "$LOG_DIR/train_fold${FOLD}.log"
