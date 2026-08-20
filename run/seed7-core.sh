#!/usr/bin/env bash
# A third seed, aimed at the one weakness left in the main result.
#
# At n=10 the method wins 10/10 folds on Macro-F1@0.5 with Wilcoxon p=0.0020, and
# four independent Phase-1 draws all agree. What is still not established is the
# magnitude on threshold-independent metrics:
#
#   Macro-F1@0.5   +0.0393  A12 0.86  p 0.0020
#   ROC-AUC        +0.0203  A12 0.65  p 0.0840
#   PR-AUC         +0.0213  A12 0.64  p 0.0840
#
# Both ranking metrics sit just outside 0.05 with sd 0.0286 against an effect of
# 0.020. Five more paired observations take n to 15, which both lowers the
# Wilcoxon floor and tightens the estimate; nothing else queued addresses this.
#
# Only the two arms that carry the argument are run: the transfer, and the
# lambda=0 ablation that says the auxiliary head is why it works. The baseline is
# retrained here at this seed, on this machine, as the fold-major discipline
# requires.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=twin_ccppjs
export SEED=7
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls
export PHASE1_DATA_PATH=data/train_ccpp_js.jsonl

FOLDS="1 2 3" MODES="cwe none" bash run/fold-major.sh
echo "=== seed 7, 3 folds ==="
python src/report_matrix.py
touch /workspace/SEED7_3FOLD

FOLDS="4 5" MODES="cwe none" bash run/fold-major.sh
echo "=== seed 7, all five folds ==="
python src/report_matrix.py

for METRIC in test_macro_f1_at_0.5 test_roc_auc test_pr_auc; do
  echo
  echo "=== pooled over seeds 36, 12, 7 | $METRIC ==="
  python src/paired_stats.py --results_root results/twin_ccppjs --seed 36 12 7 --metric "$METRIC"
done
touch /workspace/SEED7_DONE
