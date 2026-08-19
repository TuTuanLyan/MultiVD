#!/usr/bin/env bash
# Fold-major sweep: every config runs on fold 1 before anything runs on fold 2.
#
# The point is early evidence. After one fold you already have a full column
# comparing baseline against every auxiliary mode, so a config that is clearly
# broken shows up in minutes instead of after its five folds finish.
#
# Phase 1 is per-mode, not per-fold, so it is hoisted out of the loop and run
# once per mode up front.
#
# Idempotent: anything whose result JSON already exists is skipped, so this can
# be re-run over a partially completed sweep without repeating work.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-36}"
RUN_NAME="${RUN_NAME:-auxmatrix_ccppjs}"
DATA="${PHASE1_DATA_PATH:-data/train_ccpp_js.jsonl}"
MODES="${MODES-cwe latent_bottleneck latent_proto none}"
FOLDS="${FOLDS:-1 2 3 4 5}"
NUM_LATENT="${NUM_LATENT:-8}"
FREEZE_PROTOTYPES_STEPS="${FREEZE_PROTOTYPES_STEPS:-0}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-0.1}"
SELECTION_METRIC="${SELECTION_METRIC:-macro_f1}"

MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-15}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-30}"
LR="${LR:-2e-5}"
LAMBDA_CWE="${LAMBDA_CWE:-0.2}"

SHARED=(
  --seed "$SEED" --batch_size "$BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE"
  --max_length "$MAX_LENGTH" --truncation_strategy head_middle_tail
  --weight_decay 0.01 --patience 5 --min_epochs 3 --max_grad_norm 1.0 --num_workers 0
)

# train_baseline.py accepts none of these, so they stay out of SHARED and go
# only to train_transfer.py. Passing an unknown flag makes argparse exit, which
# is how an earlier sweep lost every baseline result.
AUX=(
  --num_latent "$NUM_LATENT"
  --freeze_prototypes_steps "$FREEZE_PROTOTYPES_STEPS"
  --latent_temperature "$LATENT_TEMPERATURE"
  --selection_metric "$SELECTION_METRIC"
)

# --- Phase 1 once per mode -------------------------------------------------
for MODE in $MODES; do
  METHOD="transfer_$MODE"
  LOG="log/$RUN_NAME/$METHOD/seed_$SEED"; mkdir -p "$LOG"
  MODEL="model/$RUN_NAME/$METHOD/seed_$SEED"; mkdir -p "$MODEL/source"
  mkdir -p "results/$RUN_NAME/$METHOD/seed_$SEED"
  if [[ -f "$MODEL/source/best.pt" ]]; then
    echo "=== $(date '+%F %T') | phase1 $MODE | already present, skipping ==="
    continue
  fi
  echo "=== $(date '+%F %T') | phase1 $MODE ==="
  $PYTHON -u src/train_transfer.py --phase phase1 \
    --run_name "$RUN_NAME" --method_name "$METHOD" --data_path "$DATA" \
    --aux_mode "${MODE%%_v2}" "${AUX[@]}" \
    --epochs "$PHASE1_EPOCHS" --learning_rate "$LR" --lambda_cwe "$LAMBDA_CWE" \
    --checkpoint_path "$MODEL/source/best.pt" \
    "${SHARED[@]}" >> "$LOG/phase1.log" 2>&1 || echo "  phase1 $MODE FAILED"
done

# --- Fold-major sweep ------------------------------------------------------
for FOLD in $FOLDS; do
  echo ""
  echo "############ FOLD $FOLD ############"

  BLOG="log/$RUN_NAME/baseline/seed_$SEED"; mkdir -p "$BLOG"
  BMODEL="model/$RUN_NAME/baseline/seed_$SEED/fold$FOLD"; mkdir -p "$BMODEL"
  BRES="results/$RUN_NAME/baseline/seed_$SEED"; mkdir -p "$BRES"
  if [[ -f "$BRES/fold$FOLD.json" ]]; then
    echo "=== $(date '+%F %T') | fold $FOLD baseline | already present ==="
  else
    echo "=== $(date '+%F %T') | fold $FOLD | baseline ==="
    $PYTHON -u src/train_baseline.py --phase train \
      --run_name "$RUN_NAME" --method_name baseline --fold "$FOLD" \
      --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
      --checkpoint_path "$BMODEL/best.pt" \
      "${SHARED[@]}" >> "$BLOG/train_fold$FOLD.log" 2>&1 \
      && $PYTHON -u src/train_baseline.py --phase infer \
        --run_name "$RUN_NAME" --method_name baseline --fold "$FOLD" \
        --checkpoint_path "$BMODEL/best.pt" \
          "${SHARED[@]}" >> "$BLOG/infer_fold$FOLD.log" 2>&1 \
      || echo "  baseline fold$FOLD FAILED"
  fi

  for MODE in $MODES; do
    METHOD="transfer_$MODE"
    LOG="log/$RUN_NAME/$METHOD/seed_$SEED"
    MODEL="model/$RUN_NAME/$METHOD/seed_$SEED"
    RES="results/$RUN_NAME/$METHOD/seed_$SEED"
    if [[ -f "$RES/fold$FOLD.json" ]]; then
      echo "=== $(date '+%F %T') | fold $FOLD $MODE | already present ==="
      continue
    fi
    if [[ ! -f "$MODEL/source/best.pt" ]]; then
      echo "  fold $FOLD $MODE skipped, no source checkpoint"; continue
    fi
    echo "=== $(date '+%F %T') | fold $FOLD | $MODE ==="
    mkdir -p "$MODEL/fold$FOLD"
    $PYTHON -u src/train_transfer.py --phase phase2 \
      --run_name "$RUN_NAME" --method_name "$METHOD" --fold "$FOLD" \
      --aux_mode "${MODE%%_v2}" "${AUX[@]}" \
      --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
      --source_checkpoint "$MODEL/source/best.pt" \
      --checkpoint_path "$MODEL/fold$FOLD/best.pt" \
      --output_dir "results/$RUN_NAME/$METHOD" \
      "${SHARED[@]}" >> "$LOG/phase2_fold$FOLD.log" 2>&1 \
      && $PYTHON -u src/train_transfer.py --phase test \
        --run_name "$RUN_NAME" --method_name "$METHOD" --fold "$FOLD" \
        --aux_mode "${MODE%%_v2}" "${AUX[@]}" \
        --checkpoint_path "$MODEL/fold$FOLD/best.pt" \
        --output_dir "results/$RUN_NAME/$METHOD" \
        "${SHARED[@]}" >> "$LOG/test_fold$FOLD.log" 2>&1 \
      || echo "  $MODE fold$FOLD FAILED"
  done

  echo "---- comparison after fold $FOLD ----"
  RUN_NAME="$RUN_NAME" SEED="$SEED" $PYTHON src/report_matrix.py || true
done

echo ""
echo "=== FOLD-MAJOR DONE | total ${SECONDS}s ==="
