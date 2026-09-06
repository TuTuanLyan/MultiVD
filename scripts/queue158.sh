#!/usr/bin/env bash
# queue158.sh — hang doi cho may 158, chay TUAN TU khong can nguoi trong.
#
# Chay TREN 158:  setsid nohup bash scripts/queue158.sh > log/queue158.log 2>&1 < /dev/null &
#
# Ba viec, dung thu tu, moi viec cho viec truoc nha lock `/tmp/multivd_opt1.lock`:
#
#   0. (dang chay khi phong) OPT1 nguon `full`, fold 3        <- cho no xong
#   1. ME10 — tra loi cau hoi bi early stopping chan          <- ~1,4 h
#   2. CODEBERT — truc gamma tren backbone thu hai            <- ~5 h
#
# Vi sao viec 1: do duoc 06/09 tu val_history, moi o co lambda@epoch5 ~ 0 deu dung o
# DUNG epoch 7 (patience 5, min_epochs 3) va sap duoi F1 0.6, con o co lambda@epoch5
# = 0.99 chay 13-18 epoch va khong sap. Nghia la early stopping cat nhanh neo ben
# TRUOC khi lambda kip len, nen "neo ben co tot khong" chua duoc tra loi. Khoi nay
# nang min_epochs len 10 cho CA nhanh chinh LAN doi chung, nen phep so doi dung mot bien.
#
# Vi sao viec 2: truc gamma tren t5p cho neo YEU thang neo mac dinh (gamma 5 -> +0.0295
# so voi AdamW thuan, 3/4 fold). Cau hoi ke tiep la dac tinh do co CHUYEN duoc sang
# backbone khac khong, hay chi la cua t5p. codebert co san checkpoint Pha 1 lambda=0.05
# o kho s42 (4cwe val 0.6532, com val 0.5598) nen chi ton Pha 2.
#
# BAC 1 — KIEM CHUNG, seed 42 (CLAUDE.md muc 1). Khong phai ket qua cuoi.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/ntat/envs/vdenv/bin/python}"
LOCK=/tmp/multivd_opt1.lock
# Lock RIENG cua hang doi (khac lock cua driver): de watchdog biet hang doi con song
# khong ma phong lai. Con chau ke thua fd nen phong driver phai dong bang 8>&- .
exec 8>/tmp/mvd_queue158.lock || exit 1
flock -n 8 || { echo "DA CO queue158 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }

wait_free(){ # cho toi khi khong con driver opt1 nao giu lock
  local waited=0
  while ! flock -n "$LOCK" -c true 2>/dev/null; do
    sleep 120; waited=$((waited+120))
    (( waited % 1800 == 0 )) && echo "$(ts) | cho lock... ${waited}s"
  done
  # them mot nhip cho tien trinh con dong het fd
  sleep 10
}

echo "########## QUEUE158 bat dau $(ts) ##########"

# ---- viec 1: ME10 ----
wait_free
echo "########## $(ts) | VIEC 1: ME10 (min_epochs=10) ##########"
MIN_EP=10 \
SOURCES="4cwe com" FOLD_LIST="3" PYTHON="$PY" \
CONFIGS="c5000_t0p05_me10|recadam|--sam_rho 0
c5000_t0p2k02_me10|recadam|--sam_rho 0 --anneal_t0_ratio 0.2 --anneal_k 0.02
c50_t0p2k02_me10|recadam|--sam_rho 0 --pretrain_cof 50 --anneal_t0_ratio 0.2 --anneal_k 0.02
c5000_t0p5_me10|recadam|--sam_rho 0 --anneal_t0_ratio 0.5" \
bash run/opt1.sh 8>&-
echo "########## $(ts) | VIEC 1 ket thuc | thieu: $(MISSING_LIST=0 bash scripts/opt1_missing.sh '4cwe com' '3' 42) o cua CONFIGS mac dinh ##########"

# ---- viec 2: CODEBERT ----
wait_free
echo "########## $(ts) | VIEC 2: CODEBERT truc gamma ##########"
BB="codebert=microsoft/codebert-base:cls" \
P1STORE="model/s42/phase1" P1LTAG="" \
SOURCES="4cwe com" FOLD_LIST="1 2 3" PYTHON="$PY" \
CONFIGS="c5000_t0p05|recadam|--sam_rho 0
plain|adamw|--sam_rho 0
c500_t0p05|recadam|--sam_rho 0 --pretrain_cof 500
c50_t0p05|recadam|--sam_rho 0 --pretrain_cof 50
c5_t0p05|recadam|--sam_rho 0 --pretrain_cof 5
c0p5_t0p05|recadam|--sam_rho 0 --pretrain_cof 0.5
warm|adamw|--sam_rho 0 --adamw_anneal_lr" \
bash run/opt1.sh 8>&-
echo "########## QUEUE158 xong $(ts) ##########"
