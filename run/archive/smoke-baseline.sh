#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${CONFIG_FILE:-$ROOT/run/config.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

SEED="${1:-42}"
LOG_DIR="log/$RUN_NAME/$BASELINE_NAME/smoke/seed_$SEED"
CHECKPOINT="model/$RUN_NAME/$BASELINE_NAME/smoke/seed_$SEED/fold1/best.pt"
RESULT="results/$RUN_NAME/$BASELINE_NAME/smoke/seed_$SEED/fold1.json"
mkdir -p "$(dirname "$CHECKPOINT")" "$(dirname "$RESULT")" "$LOG_DIR"

COMMON=(--fold 1 --seed "$SEED" --model_name "$MODEL_NAME" \
  --max_length "$SMOKE_MAX_LENGTH" --truncation_strategy "$TRUNCATION_STRATEGY" \
  --eval_batch_size "$EVAL_BATCH_SIZE" --num_workers "$NUM_WORKERS" \
  --max_eval_samples "$SMOKE_EVAL_SAMPLES")

python -u src/train_baseline.py --phase train "${COMMON[@]}" \
  --run_name "$RUN_NAME" --method_name "$BASELINE_NAME" \
  --checkpoint_path "$CHECKPOINT" --epochs 1 --min_epochs 1 --patience 1 \
  --batch_size "$BATCH_SIZE" --learning_rate "$BASELINE_LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$MAX_GRAD_NORM" \
  --max_train_samples "$SMOKE_TRAIN_SAMPLES" \
  2>&1 | tee -a "$LOG_DIR/train.log"

python -u src/train_baseline.py --phase infer "${COMMON[@]}" \
  --run_name "$RUN_NAME" --method_name "$BASELINE_NAME" \
  --checkpoint_path "$CHECKPOINT" --result_path "$RESULT" \
  2>&1 | tee -a "$LOG_DIR/infer.log"

echo "Baseline smoke result: $RESULT"
