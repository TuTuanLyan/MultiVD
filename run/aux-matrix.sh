#!/usr/bin/env bash
# Compare the four Phase-1 auxiliary modes on one source and one seed.
#
# Every mode shares the same folds, seed, and hyperparameters, so the
# differences between them isolate the auxiliary task alone. aux_mode=cwe is the
# v1 control and aux_mode=none is the lambda=0 ablation that tells us whether
# the auxiliary signal contributes anything at all.
#
# The baseline never sees source data, so it is independent of aux_mode and runs
# once rather than once per mode.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-36}"
RUN_NAME="${RUN_NAME:-auxmatrix_ccppjs}"
DATA="${PHASE1_DATA_PATH:-data/train_ccpp_js.jsonl}"
MODES="${MODES-cwe latent_bottleneck latent_proto none}"
NUM_LATENT="${NUM_LATENT:-8}"
# The baseline is independent of aux_mode, so it can be scheduled separately.
# Run it first when you want a reference point before the modes finish.
RUN_BASELINE="${RUN_BASELINE:-1}"

MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-15}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-30}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LAMBDA_CWE="${LAMBDA_CWE:-0.2}"
PATIENCE="${PATIENCE:-5}"
MIN_EPOCHS="${MIN_EPOCHS:-3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

SHARED=(
  --seed "$SEED" --batch_size "$BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE"
  --max_length "$MAX_LENGTH" --truncation_strategy head_middle_tail
  --weight_decay "$WEIGHT_DECAY" --patience "$PATIENCE" --min_epochs "$MIN_EPOCHS"
  --max_grad_norm "$MAX_GRAD_NORM" --num_workers 0
)

started_all=$SECONDS
for MODE in $MODES; do
  METHOD="transfer_$MODE"
  LOG="log/$RUN_NAME/$METHOD/seed_$SEED"
  MODEL="model/$RUN_NAME/$METHOD/seed_$SEED"
  RES="results/$RUN_NAME/$METHOD/seed_$SEED"
  mkdir -p "$LOG" "$MODEL/source" "$RES"

  echo "=== $(date '+%F %T') | aux_mode=$MODE | phase1 ==="
  $PYTHON -u src/train_transfer.py --phase phase1 \
    --run_name "$RUN_NAME" --method_name "$METHOD" --data_path "$DATA" \
    --aux_mode "$MODE" --num_latent "$NUM_LATENT" \
    --epochs "$PHASE1_EPOCHS" --learning_rate "$LR" --lambda_cwe "$LAMBDA_CWE" \
    --checkpoint_path "$MODEL/source/best.pt" \
    "${SHARED[@]}" >> "$LOG/phase1.log" 2>&1
  if [[ $? -ne 0 ]]; then echo "PHASE1 FAILED for $MODE, skipping"; continue; fi

  for FOLD in 1 2 3 4 5; do
    echo "=== $(date '+%F %T') | aux_mode=$MODE | fold $FOLD ==="
    mkdir -p "$MODEL/fold$FOLD"
    $PYTHON -u src/train_transfer.py --phase phase2 \
      --run_name "$RUN_NAME" --method_name "$METHOD" --fold "$FOLD" \
      --aux_mode "$MODE" --num_latent "$NUM_LATENT" \
      --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
      --source_checkpoint "$MODEL/source/best.pt" \
      --checkpoint_path "$MODEL/fold$FOLD/best.pt" \
      --output_dir "results/$RUN_NAME/$METHOD" \
      "${SHARED[@]}" >> "$LOG/phase2_fold$FOLD.log" 2>&1 || { echo "phase2 fold$FOLD failed"; continue; }

    $PYTHON -u src/train_transfer.py --phase test \
      --run_name "$RUN_NAME" --method_name "$METHOD" --fold "$FOLD" \
      --aux_mode "$MODE" --num_latent "$NUM_LATENT" \
      --checkpoint_path "$MODEL/fold$FOLD/best.pt" \
      --output_dir "results/$RUN_NAME/$METHOD" \
      "${SHARED[@]}" >> "$LOG/test_fold$FOLD.log" 2>&1 || echo "test fold$FOLD failed"
  done

  $PYTHON -u src/summarize_results.py --input_dir "$RES" --output_dir "$RES" \
    >> "$LOG/summary.log" 2>&1 || echo "summarize failed for $MODE"
  echo "=== $(date '+%F %T') | aux_mode=$MODE done | elapsed ${SECONDS}s ==="
done

# Baseline: Python only, no source checkpoint, so one run covers every mode.
if [[ "$RUN_BASELINE" == "1" ]]; then
BLOG="log/$RUN_NAME/baseline/seed_$SEED"
BMODEL="model/$RUN_NAME/baseline/seed_$SEED"
BRES="results/$RUN_NAME/baseline/seed_$SEED"
mkdir -p "$BLOG" "$BMODEL" "$BRES"
for FOLD in 1 2 3 4 5; do
  echo "=== $(date '+%F %T') | baseline | fold $FOLD ==="
  mkdir -p "$BMODEL/fold$FOLD"
  $PYTHON -u src/train_baseline.py --phase train \
    --run_name "$RUN_NAME" --method_name baseline --fold "$FOLD" \
    --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
    --checkpoint_path "$BMODEL/fold$FOLD/best.pt" \
    "${SHARED[@]}" >> "$BLOG/train_fold$FOLD.log" 2>&1 || { echo "baseline fold$FOLD failed"; continue; }
  $PYTHON -u src/train_baseline.py --phase infer \
    --run_name "$RUN_NAME" --method_name baseline --fold "$FOLD" \
    --checkpoint_path "$BMODEL/fold$FOLD/best.pt" \
    --output_dir "results/$RUN_NAME/baseline" \
    "${SHARED[@]}" >> "$BLOG/infer_fold$FOLD.log" 2>&1 || echo "baseline infer fold$FOLD failed"
done
$PYTHON -u src/summarize_results.py --input_dir "$BRES" --output_dir "$BRES" \
  >> "$BLOG/summary.log" 2>&1 || echo "baseline summarize failed"
fi

echo "=== ALL DONE | total $((SECONDS - started_all))s ==="
