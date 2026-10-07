#!/usr/bin/env bash
# Hai muc lr cao cho khoi assemble t5p, CHAY LAN LUOT (mot GPU mot chuoi).
# Bac 1 — kiem chung, n=3 fold, seed 42. Moi muc lr mot CAY RIENG de ghep cap trong cay.
set -u
cd /drive1/cuongtm/ntat/MultiVD || exit 1
export PYTHON=/home/ntat/miniconda3/envs/vdenv/bin/python
echo "===== LR CHAIN bat dau $(date -u '+%F %T') ====="
RUN_NAME=mw_asm_t5p_lr5e5 LR=5e-5 FOLDS="1 2 3" ARMS="mwK8 mwTR" bash run/mw_t5p_lr.sh
echo "===== xong muc 5e-5 $(date -u '+%F %T') ====="
RUN_NAME=mw_asm_t5p_lr1e4 LR=1e-4 FOLDS="1 2 3" ARMS="mwK8 mwTR" bash run/mw_t5p_lr.sh
echo "===== LR CHAIN xong $(date -u '+%F %T') ====="
