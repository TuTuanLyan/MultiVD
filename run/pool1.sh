#!/usr/bin/env bash
# POOL1 — DO TINH KHIET NHAN hay CO DU LIEU moi quyet dinh transfer?
#
# Ba nguon hien co lam hai bien dinh chat nhau (tinh khiet va co tuong quan NGHICH hoan hao):
#   4cwe 930 hang / 100% tinh khiet -> +0.0352
#   com  3744     /  24.8%          -> +0.0549
#   full 7598     /  12.2%          -> -0.2908, 0/3 fold (SUP duoi muc ngau nhien)
# Khong the noi cai nao gay ra cai gi. Luoi nay tach chung:
#   truc TINH KHIET (co CO DINH 930): pur100, pur75, pur50, pur25, pur12
#   truc CO (tinh khiet CO DINH 100%): n232, n465, n930
#
# DU DOAN CO THE SAI — day la diem chinh cua khoi:
#   `pur12_n930` co DUNG do tinh khiet cua `full` nhung chi 1/8 co du lieu.
#   Neu no cung SUP  -> TINH KHIET la nguyen nhan, co du lieu khong phai. Co quy tac chon nguon.
#   Neu no KHONG sup -> nguyen nhan la co du lieu (hoac tuong tac), gia thuyet bi bac.
#
# Bac 1: 3 fold, seed 42, t5p. Doc CA BON chi so bang tools/report2.py.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( [ -x /venv/main/bin/python ] && echo /venv/main/bin/python \
  || { [ -x /data/ntat/envs/vdenv/bin/python ] && echo /data/ntat/envs/vdenv/bin/python \
  || echo /home/ntat/miniconda3/envs/vdenv/bin/python; } )}"
export PYTHON
POOLS="${POOLS:-pur100_n930 pur75_n930 pur50_n930 pur25_n930 pur12_n930 pur100_n465 pur100_n232}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
SEED="${SEED:-42}"
# GPU dung chung: cong VRAM TRUOC TUNG O + THU LAI. OOM la loi ha tang, khong phai ket qua —
# 08/09 mat 26 o vi cong chi kiem mot lan luc khoi dong roi nguoi dung khac gianh bo nho.
NEED="${NEED:-13000}"
RETRY="${RETRY:-3}"
ts(){ date -u "+%F %T"; }
wait_vram(){
  local w=0 tot use avail
  while true; do
    tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    use=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    avail=$(( ${tot:-0} - ${use:-0} ))
    (( avail >= NEED )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(ts) | cho VRAM ${avail}MiB < $NEED ... ${w}s"
  done
}
echo "########## POOL1 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  nguon: $POOLS | fold: $FOLD_LIST"
for FOLD in $FOLD_LIST; do
  for SRC in $POOLS; do
    D="data/pool/${SRC}.jsonl"
    [[ -f "$D" ]] || { echo "  !! thieu $D"; continue; }
    echo "===== $(date -u '+%F %T') | fold $FOLD | nguon $SRC ====="
    RES="results/pool1_t5p/transfer_latent_bottleneck_${SRC}_adamw/seed_42/fold${FOLD}.json"
    for try in $(seq 1 "$RETRY"); do
      [[ -f "$RES" ]] && break
      wait_vram
      (( try > 1 )) && echo "$(ts) | THU LAI lan $try"
    # PHASE1_MIN_VAL=0: nguon tinh khiet thap DUOC PHEP lam Pha 1 sap — do chinh la ket qua
    # can do, khong phai ly do bo o (CLAUDE.md muc 3).
    RUN_NAME=pool1 SEED="$SEED" FOLDS="$FOLD" \
    BACKBONES="t5p=Salesforce/codet5p-220m-bimodal:mean" \
    MODES="latent_bottleneck" OPTIMIZERS="adamw" \
    PHASE1_DATA_PATH="$D" CWE_VOCAB=precomputed \
    ARM_TAG="_${SRC}" PHASE1_TAG="_${SRC}" PHASE1_STORE="model/pool1/phase1" \
    LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 PHASE1_MIN_EPOCH=0 MIN_EPOCHS=3 \
    PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    bash run/matrix.sh 8>&-
    done
  done
done
echo "########## POOL1 xong $(date -u '+%F %T') ##########"
