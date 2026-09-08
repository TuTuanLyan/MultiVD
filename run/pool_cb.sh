#!/usr/bin/env bash
# POOL_CB — lap lai LUOI TINH KHIET tren backbone THU HAI (codebert).
#
# Phat hien tinh khiet la ung vien dong gop manh nhat hien nay: thu tu `pur75 > pur100 >
# pur50` lap lai tren HAI MAY doc lap (161 va ntat2), va pur50 am tren ca hai chi so
# (dAUC 0/5 fold). Nhung ca hai deu la t5p. Neu no giu tren codebert thi day la ket qua
# lap lai qua BACKBONE, thu manh hon nhieu so voi lap lai qua may.
#
# PHASE1_MIN_VAL=0 (keo theo PHASE1_MIN_EPOCH=0): nguon pha loang PHAI duoc chay ke ca khi
# Pha 1 suy bien — do la ket qua can do, khong phai ly do bo o.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/ntat/envs/vdenv/bin/python}"
NEED="${NEED:-9000}"; RETRY="${RETRY:-3}"
exec 8>/tmp/mvd_poolcb.lock || exit 1
flock -n 8 || { echo "DA CO pool_cb dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(ts) | cho VRAM ${a}MiB"
  done; }
echo "########## POOL_CB bat dau $(ts) | $(hostname) ##########"
for FOLD in ${FOLD_LIST:-1 2 3}; do
  for SRC in ${POOLS:-pur100_n930 pur75_n930 pur50_n930 pur25_n930 pur12_n930}; do
    D="data/pool/${SRC}.jsonl"; [[ -f "$D" ]] || { echo "  !! thieu $D"; continue; }
    RES="results/poolcb_codebert/transfer_latent_bottleneck_${SRC}_adamw/seed_42/fold${FOLD}.json"
    for try in $(seq 1 "$RETRY"); do
      [[ -f "$RES" ]] && break
      wait_vram; (( try > 1 )) && echo "$(ts) | THU LAI lan $try"
      echo "===== $(ts) | fold $FOLD | $SRC ====="
      RUN_NAME=poolcb SEED=42 FOLDS="$FOLD" \
      BACKBONES="codebert=microsoft/codebert-base:cls" \
      MODES="latent_bottleneck" OPTIMIZERS="adamw" \
      PHASE1_DATA_PATH="$D" CWE_VOCAB=precomputed \
      ARM_TAG="_${SRC}" PHASE1_TAG="_${SRC}" PHASE1_STORE="model/poolcb/phase1" \
      LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
      PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
      DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
      PYTHON="$PY" bash run/matrix.sh 8>&-
    done
  done
done
echo "########## POOL_CB xong $(ts) ##########"
