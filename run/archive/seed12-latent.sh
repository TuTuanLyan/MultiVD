#!/usr/bin/env bash
# A second seed for the two latent heads, so they get the evidence cwe has.
#
# The point of the latent heads was to stop the method depending on exactly four
# CWEs: latent_bottleneck routes through K units before the C-way output, and
# latent_proto needs no CWE labels at all. They are the generalisation, and they
# are the arms still stuck at one seed:
#
#   transfer_cwe                n=10  +0.0393  Wilcoxon p 0.0020   10/10 folds
#   transfer_latent_bottleneck  n=5   +0.0268  Wilcoxon p 0.0625   floor, unreachable
#   transfer_latent_proto       n=5   +0.0254  Wilcoxon p 0.1250
#
# At n=5 the Wilcoxon floor is 0.0625, so neither can reach 0.05 whatever the
# data. A second seed takes the floor to 0.0020 and makes the question answerable.
#
# latent_bottleneck is also the arm that behaved better in the common-vs-full
# comparison -- same direction as cwe but without leaning on a single fold -- so
# strengthening it is worth more than another cwe seed.
#
# The seed-12 baseline is already on this machine from run/seed12-core.sh and is
# reused; it never sees source data, so it is identical across auxiliary modes.
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

FOLDS="1 2 3" MODES="latent_bottleneck latent_proto" bash run/fold-major.sh
echo "=== seed 12 latent heads, 3 folds ==="
python src/report_matrix.py
touch /workspace/SEED12LATENT_3FOLD

FOLDS="4 5" MODES="latent_bottleneck latent_proto" bash run/fold-major.sh
echo "=== seed 12 latent heads, all five folds ==="
python src/report_matrix.py
echo
echo "=== pooled over both seeds ==="
python src/paired_stats.py --results_root results/twin_ccppjs --seed 36 12
touch /workspace/SEED12LATENT_DONE
