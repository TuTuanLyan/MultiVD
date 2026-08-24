#!/usr/bin/env bash
# Can the method use a source whose CWE taxonomy does not match the target?
#
# PrimeVul carries 121 CWEs; the Python target has four. The original four-way
# map sent every other CWE to -100, so those rows contributed nothing to the
# auxiliary task. CWE_VOCAB=source builds the auxiliary label space from the
# data instead, letting the whole taxonomy participate.
#
# The baseline never sees source data, so the one already measured on these
# folds and this machine is reused rather than retrained.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python

export DATA_ROOT=data/sven_python_twin
export RUN_NAME=twin_primevul
export SEED=36
export CWE_VOCAB=source
export PHASE1_DATA_PATH=data/ccpp_primevul_paired_full.jsonl

FOLDS="1 2 3" MODES="cwe latent_bottleneck" bash run/fold-major.sh
echo "=== TREND AFTER 3 FOLDS ==="
python src/report_matrix.py
touch /workspace/PV_3FOLD

FOLDS="4 5" MODES="cwe latent_bottleneck" bash run/fold-major.sh
echo "=== FINAL ==="
python src/report_matrix.py
touch /workspace/PV_DONE
