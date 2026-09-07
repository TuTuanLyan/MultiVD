#!/usr/bin/env bash
# SPD1 — thay LICH THOI GIAN cua RecAdam bang DIEU KIEN GRADIENT (SPD, NeurIPS 2024).
#
# VI SAO (RESEARCH §10): tren 100 o ghep cap, ba backbone, ba nguon, RecAdam - AdamW =
# -0.0041 (44/100). Quet gamma khong cuu duoc (nut phang trong dai 0.5..50), va he qua
# gamma ~ N cua chinh bai RecAdam da kiem va KHONG dung. Nen thu doi CO CHE.
#
#   RecAdam:  neo manh theo lambda(t) — mot ham cua THOI GIAN, khong biet luc nay neo
#             dang giup hay dang can.
#   SPD:      chi phat khi  c_t = (-g_t)·(theta_{t-1} - theta_0) < 0, tinh RIENG tung lop,
#             tuc khi gradient da quay dau ma quan tinh van day ra xa neo. Cuong do = r_t,
#             phan do lech TANG THEM trong chinh buoc do. Khong co tham so thoi gian nao.
#
# DIEU PHAI BIET TRUOC KHI DOC KET QUA (do trong tests/test_spd.py muc 3): tren mot quy dao
# tien bo nhat quan, SPD KHONG BAO GIO phat va no DUNG BANG AdamW. Nen o nao co
# `ti le kich hoat = 0` phai doc la "AdamW", khong phai "SPD kem". Ti le do duoc ghi vao log.
#
# Ba muc so sanh, CUNG MAY CUNG PHIEN (CLAUDE.md muc 4):
#   plain (adamw)   fine-tune hai lan, khong neo
#   c50_t0p05       RecAdam o gamma tot nhat do duoc — doi thu that su
#   spdl1 (spd)     SPD lambda = 1, dung gia tri bai goc khuyen
#
# Kem theo MIEN PHI: moi o deu cham lai tren tap val Pha 1 (`--source_eval_data`), nen khoi
# nay tra loi luon "SPD co giu duoc nguon khong" ma khong ton them o nao. Nguoi dung neu
# 07/09: ca RecAdam lan SPD deu la co che chong quen, chua tung thu cham lai tren nguon.
#
# BAC 1 (kiem chung): 3 fold, seed 42. Tot moi len n=5.
#   FOLD_LIST="1 2 3" bash run/spd1.sh
#
# Ket qua: results/spd1_t5p/ (cay RIENG de baseline va ca hai doi chung deu chay lai cung phien)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCES_LIST="${SOURCES_LIST:-4cwe com}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
SEED="${SEED:-42}"
ALL_SRC="data/phase1_4cwe.jsonl data/phase1_common.jsonl data/phase1_full.jsonl"

data_of(){ case "$1" in
  4cwe) echo "data/phase1_4cwe.jsonl" ;;
  com)  echo "data/phase1_common.jsonl" ;;
  full) echo "data/phase1_full.jsonl" ;;
  *) echo "" ;; esac; }

echo "########## SPD1 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  nguon: $SOURCES_LIST | fold: $FOLD_LIST | seed: $SEED"

for FOLD in $FOLD_LIST; do
  for SRC in $SOURCES_LIST; do
    DATA="$(data_of "$SRC")"
    [[ -n "$DATA" ]] || { echo "  !! nguon la: $SRC"; continue; }
    echo "===== $(date -u '+%F %T') | fold $FOLD | nguon $SRC ====="
    RUN=spd1 SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 \
    SOURCE_EVAL_DATA="$DATA" BASELINE_SOURCE_EVAL="$ALL_SRC" \
    CONFIGS="plain|adamw|--sam_rho 0
c50_t0p05|recadam|--sam_rho 0 --pretrain_cof 50
spdl1|spd|--sam_rho 0 --spd_lambda 1.0" \
    bash run/opt1.sh
  done
done
echo "########## SPD1 xong $(date -u '+%F %T') ##########"
