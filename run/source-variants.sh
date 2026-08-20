#!/usr/bin/env bash
# Does the size of the source taxonomy matter?
#
# full carries 121 CWEs, common 73, and the original ccpp+js file only 4. If the
# latent bottleneck really decouples the auxiliary label space from the target,
# a 73-CWE source should behave like a 121-CWE one, and both should be usable
# where the fixed four-way head could only see four.
#
# The baseline is reused from the CodeBERT twin sweep already measured on this
# machine and these folds; it never sees source data, so it is a valid reference.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export SEED=${SEED:-36}
export CWE_VOCAB=source
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls

export RUN_NAME=twin_pvcommon
mkdir -p "results/$RUN_NAME/baseline/seed_$SEED"
cp results/twin_ccppjs/baseline/seed_$SEED/fold*.json "results/$RUN_NAME/baseline/seed_$SEED/" 2>/dev/null
PHASE1_DATA_PATH=data/ccpp_primevul_paired_common.jsonl \
  FOLDS="1 2 3" MODES="cwe latent_bottleneck" bash run/fold-major.sh
echo "=== COMMON (73 CWE), 3 FOLDS ==="
python src/report_matrix.py
touch /workspace/PVCOMMON_3FOLD
