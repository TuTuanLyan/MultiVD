#!/usr/bin/env bash
# A different source->target pair: C/C++ -> JavaScript, instead of -> Python.
#
# The source MUST be C/C++ only. train_ccpp_js.jsonl contains the JavaScript
# rows that are now the target, so using it here would put the target corpus
# inside the source and invalidate the whole comparison.
#
# Everything else matches the Python setup: pair-preserving folds, fold-major,
# one seed, its own baseline trained on this machine.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/js_twin
export RUN_NAME=jstarget_pvcommon
export SEED=${SEED:-36}
export CWE_VOCAB=source
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls
export PHASE1_DATA_PATH=data/ccpp_primevul_paired_common.jsonl

FOLDS="1 2 3" MODES="cwe latent_bottleneck none" bash run/fold-major.sh
echo "=== C/C++ -> JS, 3 FOLDS ==="
python src/report_matrix.py
touch /workspace/JSTARGET_3FOLD
