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

echo "=== full data, AdamW instead of RecAdam ==="
RUN_NAME=twin_ccppjs PHASE2_OPTIMIZER=adamw FOLDS="1 2 3" MODES="cwe+adamw" \
  bash run/fold-major.sh
RUN_NAME=twin_ccppjs SEED=$SEED python src/report_matrix.py

echo "=== 114 rows, AdamW instead of RecAdam ==="
RUN_NAME=frac114_codebert-base PHASE2_OPTIMIZER=adamw MAX_TRAIN_SAMPLES=114 \
  FOLDS="1 2 3" MODES="cwe+adamw" bash run/fold-major.sh
RUN_NAME=frac114_codebert-base SEED=$SEED python src/report_matrix.py
touch /workspace/OPTABL_DONE
