#!/usr/bin/env bash
# Two questions, one flag.
#
# 1. Full data: what does RecAdam contribute over plain AdamW? The gap between
#    transfer and baseline has always bundled source exposure, the auxiliary
#    task, and RecAdam together; this separates the last one.
#
# 2. Small target set: at 114 rows the transfer's test probabilities collapse
#    into [0.32, 0.83], std 0.116 against the baseline's 0.431 -- an
#    under-trained model, not a collapsed one. RecAdam scales the target
#    gradient by lambda(t), calibrated to total_steps, and 114 rows give only
#    240 steps with the best checkpoint landing at step 56. If AdamW recovers
#    the delta, the annealing schedule is the culprit, not the transfer.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export SEED=${SEED:-36}
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls

# The AdamW arm changes only Phase 2, so it must start from the SAME Phase-1
# model as the RecAdam arm. Letting it train its own would mix optimizer effect
# with Phase-1 run-to-run variance, which reaches 0.13 val Macro-F1 here.
seed_source() {  # seed_source <run> <from_method> <to_method>
  local src="model/$1/$2/seed_$SEED/source/best.pt"
  local dst="model/$1/$3/seed_$SEED/source"
  if [[ -f "$src" && ! -f "$dst/best.pt" ]]; then
    mkdir -p "$dst" && cp "$src" "$dst/best.pt"
    echo "  seeded $3 from $2"
  elif [[ ! -f "$src" ]]; then
    echo "  WARNING: $src missing, $3 will train its own Phase 1"
  fi
}

echo "=== full data, AdamW instead of RecAdam ==="
seed_source twin_ccppjs transfer_cwe transfer_cwe+adamw
RUN_NAME=twin_ccppjs PHASE2_OPTIMIZER=adamw FOLDS="1 2 3" MODES="cwe+adamw" \
  bash run/fold-major.sh
RUN_NAME=twin_ccppjs SEED=$SEED python src/report_matrix.py

echo "=== 114 rows, AdamW instead of RecAdam ==="
# The 114-row run reuses the same full-data Phase 1: only the target set is cut.
mkdir -p "model/frac114_codebert-base/transfer_cwe+adamw/seed_$SEED/source"
cp "model/twin_ccppjs/transfer_cwe/seed_$SEED/source/best.pt" \
   "model/frac114_codebert-base/transfer_cwe+adamw/seed_$SEED/source/best.pt" 2>/dev/null \
   && echo "  seeded 114-row AdamW arm from the full-data Phase 1"
RUN_NAME=frac114_codebert-base PHASE2_OPTIMIZER=adamw MAX_TRAIN_SAMPLES=114 \
  FOLDS="1 2 3" MODES="cwe+adamw" bash run/fold-major.sh
RUN_NAME=frac114_codebert-base SEED=$SEED python src/report_matrix.py
touch /workspace/OPTABL_DONE
