#!/usr/bin/env bash
# VIEC 3 cho 158: chay dung CHE DO CUA BAI GOC — 60 epoch, patience 10.
# Dieu kien da ghi truoc trong CURRENT_RUN muc "viec cho sang 07/09" #4: chi chay neu
# ME10 duong. ME10 DA DUONG (+0.0689, 6/6) nen dieu kien thoa.
# Ly do tai lieu: RecAdam goc chay 50-100 epoch tren RTE/MRPC va KHONG early-stop
# (RESEARCH §8.1). Ta dang chay toi da 30 epoch, patience 5.
set -uo pipefail
cd /data/ntat/MultiVD
exec 8>/tmp/mvd_queue158b.lock || exit 1
flock -n 8 || { echo "DA CO queue158b"; exit 3; }
ts(){ date -u '+%F %T'; }
while ! flock -n /tmp/multivd_opt1.lock -c true 2>/dev/null; do sleep 60; done
echo "########## QUEUE158B bat dau $(ts) | 60 epoch, patience 10 ##########"
MIN_EP=10 PHASE2_EPOCHS=60 PATIENCE=10 \
SOURCES="4cwe com" FOLD_LIST="3" PYTHON=/data/ntat/envs/vdenv/bin/python \
CONFIGS="c5000_t0p05_e60|recadam|--sam_rho 0
plain_e60|adamw|--sam_rho 0
c5_t0p05_e60|recadam|--sam_rho 0 --pretrain_cof 5
c5000_t0p2k02_e60|recadam|--sam_rho 0 --anneal_t0_ratio 0.2 --anneal_k 0.02
c5000_t0p5k005_e60|recadam|--sam_rho 0 --anneal_t0_ratio 0.5 --anneal_k 0.005" \
bash run/opt1.sh 8>&-
echo "########## QUEUE158B xong $(ts) ##########"
