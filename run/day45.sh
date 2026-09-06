#!/usr/bin/env bash
# NHIEM VU CHINH: lambda=0.02, ASAM rho=0.1 o Phase 2, RecAdam, 3 nguon x 3 backbone.
#   bash run/day45.sh <nguon> <backbone-spec>
#
# lambda DOI PHASE 1 nen phai huan luyen Phase 1 moi — TRU nhanh `none`:
# src/model.py:320 tra ve aux_loss=None khi aux_mode="none", nen lambda khong bao
# gio vao ham loss. Checkpoint `none` cua lambda=0.05 dung lai duoc y nguyen
# (da symlink sang ten _l02), tiet kiem 9 job Phase 1.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?4cwe|com|full}"; BB="${2:?backbone-spec}"
case "$SRC" in
  full) DATA=data/phase1_full.jsonl;   VOCAB=precomputed; M="none latent_bottleneck latent_proto" ;;
  com)  DATA=data/phase1_common.jsonl; VOCAB=precomputed; M="none latent_bottleneck latent_proto" ;;
  4cwe) DATA=data/phase1_4cwe.jsonl;   VOCAB=fixed4;      M="none cwe latent_bottleneck latent_proto" ;;
  *) echo "nguon sai"; exit 2 ;;
esac
RUN_NAME="${RUN_NAME:-s42}" SEED=42 FOLDS="${FOLDS:-1 2 3 4 5}" \
BACKBONES="$BB" MODES="${MODES:-$M}" OPTIMIZERS=recadam \
PHASE1_DATA_PATH="$DATA" CWE_VOCAB="$VOCAB" \
ARM_TAG="_${SRC}_l02" PHASE1_TAG="_${SRC}_l02" \
LAMBDA_CWE=0.02 PHASE1_EPOCHS=15 \
PHASE1_EXTRA="--sam_rho 0" \
PHASE2_EXTRA="--sam_rho ${ASAM_RHO:-0.1} --sam_variant asam --asam_eta 0.01" \
DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
bash run/matrix.sh
