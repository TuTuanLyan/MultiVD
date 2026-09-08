#!/usr/bin/env bash
# ASAM_CB — lap lai phat hien ASAM/AUC tren BACKBONE THU HAI, trong MOT PHIEN.
#
# Bang chung hien tai cho ASAM la 190 o ghep cap gom tu nhieu khoi cu — manh ve so luong
# nhung yeu ve thiet ke: cac fold dung lai qua nhieu khoi nen khong doc lap. Khoi nay chay
# codebert tu dau trong mot phien, ro=0 vs ro=0.1, de co mot lan lap lai SACH.
#
# Chi so CHINH khai bao truoc: ROC-AUC. Phu: PR-AUC, F1@0.5, F1@nguong-val.
# Ky vong tu du lieu cu: codebert dAUC +0.0027 (24/40). Neu khoi nay cho cung dau thi
# phat bieu "ASAM nang thu hang" co mot lan lap lai doc lap; neu nguoc dau thi phai rut.
#
# Dung kho Pha 1 s42 (ten khong co hau to lambda) nen P1LTAG rong.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/ntat/envs/vdenv/bin/python}"
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_asamcb.lock || exit 1
flock -n 8 || { echo "DA CO asam_cb dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
w=0
while true; do
  free=0; flock -n "$LOCK" -c true 2>/dev/null && free=1
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
  use=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  avail=$(( ${tot:-0} - ${use:-0} ))
  (( free == 1 && avail >= 9000 )) && { echo "$(ts) | GPU ranh (${avail}MiB)"; break; }
  sleep 60; w=$((w+60)); (( w % 600 == 0 )) && echo "$(ts) | cho... lock=$free VRAM=${avail}MiB"
done
echo "########## ASAM_CB bat dau $(ts) | $(hostname) ##########"
for FOLD in ${FOLD_LIST:-1 2 3 4 5}; do
  for SRC in ${SOURCES_LIST:-4cwe com}; do
    echo "===== $(ts) | fold $FOLD | nguon $SRC ====="
    RUN=asamcb SEEDS=42 SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 \
    BB="codebert=microsoft/codebert-base:cls" P1STORE="model/s42/phase1" P1LTAG="" \
    PYTHON="$PY" \
    CONFIGS="r0|recadam|--sam_rho 0
r0p1|recadam|--sam_rho 0.1 --sam_variant asam" \
    bash run/opt1.sh 8>&-
  done
done
echo "########## ASAM_CB xong $(ts) ##########"
