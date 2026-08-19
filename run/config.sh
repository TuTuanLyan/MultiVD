#!/usr/bin/env bash
# Edit this file to configure all training scripts. Environment variables still override these defaults.

: "${CONDA_ENV:=vdenv}"
: "${MODEL_NAME:=microsoft/codebert-base}"

# Change RUN_NAME for every experiment you want to preserve. Transfer,
# baseline, logs, models, results, and comparisons share this parent folder.
: "${RUN_NAME:=seed12_ccppjs_py_compare_v1}"
: "${TRANSFER_NAME:=transfer}"
: "${BASELINE_NAME:=baseline}"
: "${COMPARE_NAME:=compare}"

# Shared model/data settings.
: "${MAX_LENGTH:=512}"
: "${TRUNCATION_STRATEGY:=head_middle_tail}"
: "${BATCH_SIZE:=16}"
: "${EVAL_BATCH_SIZE:=16}"
: "${NUM_WORKERS:=0}"
: "${WEIGHT_DECAY:=0.01}"
: "${MAX_GRAD_NORM:=1.0}"
: "${PATIENCE:=5}"
: "${MIN_EPOCHS:=3}"

# Phase 1: any JSONL with code, label, CWE fields, and lang/language.
: "${PHASE1_DATA_PATH:=data/train_ccpp_js.jsonl}"
: "${PHASE1_EPOCHS:=15}"
: "${PHASE1_LEARNING_RATE:=2e-5}"
: "${LAMBDA_CWE:=0.2}"

# Phase 2: Python RecAdam transfer.
: "${PHASE2_EPOCHS:=30}"
: "${PHASE2_LEARNING_RATE:=2e-5}"
: "${ANNEAL_FUN:=sigmoid}"
: "${ANNEAL_K:=0.05}"
: "${ANNEAL_T0_RATIO:=0.05}"
: "${ANNEAL_W:=1.0}"
: "${PRETRAIN_COF:=5000.0}"

# Plain CodeBERT baseline: Python only, no source checkpoint and no CWE head.
: "${BASELINE_EPOCHS:=$PHASE2_EPOCHS}"
: "${BASELINE_LEARNING_RATE:=$PHASE2_LEARNING_RATE}"

# Smoke test only.
: "${SMOKE_MAX_LENGTH:=128}"
: "${SMOKE_TRAIN_SAMPLES:=32}"
: "${SMOKE_EVAL_SAMPLES:=32}"
