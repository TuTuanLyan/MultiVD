#!/usr/bin/env bash
# A second seed for the finding that survived everything else.
#
# One seed cannot reach significance here: with five folds the smallest
# reachable Wilcoxon p is 0.0625. And four separate signals in this project
# evaporated between n=1 and n=3, so a result standing on one seed is a
# direction, not a claim.
#
# Only the two arms that matter are re-run: the transfer that works, and the
# lambda=0 ablation that says the auxiliary task is why it works. The baseline
# is retrained here too, on this machine, at this seed.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=twin_ccppjs
export SEED=12
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls
export PHASE1_DATA_PATH=data/train_ccpp_js.jsonl

FOLDS="1 2 3" MODES="cwe none" bash run/fold-major.sh
echo "=== SEED 12, 3 FOLDS ==="
python src/report_matrix.py
touch /workspace/SEED12_3FOLD

FOLDS="4 5" MODES="cwe none" bash run/fold-major.sh
echo "=== SEED 12, FINAL ==="
python src/report_matrix.py
touch /workspace/SEED12_DONE
