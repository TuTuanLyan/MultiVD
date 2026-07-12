#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${CONFIG_FILE:-$ROOT/run/config.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

FOLD="${1:?usage: bash run/train-phase2.sh FOLD SEED}"
SEED="${2:?usage: bash run/train-phase2.sh FOLD SEED}"
LOG_DIR="log/$RUN_NAME/$TRANSFER_NAME/seed_$SEED"
MODEL_DIR="model/$RUN_NAME/$TRANSFER_NAME/seed_$SEED"
RESULT_DIR="results/$RUN_NAME/$TRANSFER_NAME/seed_$SEED"
SOURCE="$MODEL_DIR/source/best.pt"
TARGET="$MODEL_DIR/fold$FOLD/best.pt"
[[ -f "$SOURCE" ]] || { echo "Missing source checkpoint: $SOURCE" >&2; exit 1; }
mkdir -p "$LOG_DIR" "$(dirname "$TARGET")" "$RESULT_DIR"

python -u src/train_transfer.py \
  --phase phase2 \
  --run_name "$RUN_NAME" \
  --method_name "$TRANSFER_NAME" \
  --fold "$FOLD" \
  --seed "$SEED" \
  --source_checkpoint "$SOURCE" \
  --checkpoint_path "$TARGET" \
  --output_dir "results/$RUN_NAME/$TRANSFER_NAME" \
  --epochs "$PHASE2_EPOCHS" \
  --min_epochs "$MIN_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy "$TRUNCATION_STRATEGY" \
  --learning_rate "$PHASE2_LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --patience "$PATIENCE" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --num_workers "$NUM_WORKERS" \
  --model_name "$MODEL_NAME" \
  --anneal_fun "$ANNEAL_FUN" \
  --anneal_k "$ANNEAL_K" \
  --anneal_t0_ratio "$ANNEAL_T0_RATIO" \
  --anneal_w "$ANNEAL_W" \
  --pretrain_cof "$PRETRAIN_COF" \
  2>&1 | tee -a "$LOG_DIR/phase2_fold${FOLD}.log"
