#!/usr/bin/env bash
# ASAM1 — ASAM co nang ROC-AUC khong? Chi so chinh KHAI BAO TRUOC: ROC-AUC.
#
# VI SAO (do 08/09 tren 190 o ghep cap cu, doc rho TU FILE chu khong doan tu ten):
#   macro-F1@0.5 : +0.0015, 100/190 duong, p=0.51   -> khong co gi
#   ROC-AUC      : +0.0037, 119/190 duong, p=0.0006 -> CUNG CHIEU o ca ba backbone
#   PR-AUC       : +0.0045, 111/190 duong, p=0.024
# Ket luan cu "ASAM null" (+0.0020, 70/130, p=0.43) tinh tren macro-F1 VA CHI macro-F1.
# Khong ai nhin AUC. Nguoi dung phat hien 08/09.
#
# Tach theo backbone (kiem lap lai doc lap), tat ca CUNG DAU tren AUC:
#   codebert +0.0027 (24/40) | t5p +0.0025 (59/100) | unixcoder +0.0069 (36/50, p=0.0026)
# Tach theo nhanh: latent_bottleneck — dung nhanh da chot — +0.0040, 62/95, p=0.0038.
# Ngoai le: nguon `com` hoi am (-0.0017). Nen dua CA `com` vao de kiem lai cho do.
#
# Doi chung cua khoi nay la chinh nhanh rho=0, ghep cap tung fold — do moi la phep so
# tra loi cau hoi. matrix.sh van chay baseline mot lan moi fold (no luon lam vay); giu lai
# vi no cho them cot "D so voi baseline" mien phi trong cung cay, khong phai vi phep so can.
#
#   FOLD_LIST="1 2 4" bash run/asam1.sh      # ntat
#   FOLD_LIST="3 5"   bash run/asam1.sh      # ntat2
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( [ -x /venv/main/bin/python ] && echo /venv/main/bin/python \
  || { [ -x /data/ntat/envs/vdenv/bin/python ] && echo /data/ntat/envs/vdenv/bin/python \
  || echo /home/ntat/miniconda3/envs/vdenv/bin/python; } )}"
export PYTHON
SOURCES_LIST="${SOURCES_LIST:-4cwe com}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
SEED="${SEED:-42}"

echo "########## ASAM1 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  chi so CHINH: ROC-AUC (khai bao truoc). Phu: PR-AUC, macro-F1@0.5"
echo "  nguon: $SOURCES_LIST | fold: $FOLD_LIST | seed: $SEED"

for FOLD in $FOLD_LIST; do
  for SRC in $SOURCES_LIST; do
    case "$SRC" in 4cwe) DATA=data/phase1_4cwe.jsonl;; com) DATA=data/phase1_common.jsonl;;
      full) DATA=data/phase1_full.jsonl;; *) echo "  !! nguon la: $SRC"; continue;; esac
    echo "===== $(date -u '+%F %T') | fold $FOLD | nguon $SRC ====="
    RUN=asam1 SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 \
    CONFIGS="r0|recadam|--sam_rho 0
r0p1|recadam|--sam_rho 0.1 --sam_variant asam" \
    bash run/opt1.sh
  done
done
echo "########## ASAM1 xong $(date -u '+%F %T') ##########"
