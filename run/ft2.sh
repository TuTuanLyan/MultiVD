#!/usr/bin/env bash
# ft2.sh — dien not o CON THIEU cua doi chung "finetune hai lan thuan":
#   Pha 1 KHONG head (aux_mode=none), Pha 2 AdamW, SAM=0 ca hai pha.
#
# Nguoi dung 08/09 yeu cau doi chung nay tren t5p va codebert. Da co san seed 42 du 5 fold:
#   t5p      x {4cwe, com, full}
#   codebert x {4cwe, com}
# Thieu DUNG MOT o: codebert x full. Checkpoint Pha 1 cua no la file .rejected 0 BYTE
# (di chung cua su co day dia thang 8), nen phai huan luyen lai Pha 1.
#
# PHASE1_MIN_VAL=0 la CO Y: CLAUDE.md muc 3 — khong bao gio bo mot o vi Pha 1 cua no yeu.
# `codebert x full` chinh la o Pha 1 sap ve ~0.34, va o "hong" do la bang chung tot nhat
# cho negative transfer. Chay va ghi kem val Pha 1 chu khong de trong.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/ntat/miniconda3/envs/vdenv/bin/python}"
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_ft2.lock || exit 1
flock -n 8 || { echo "DA CO ft2 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
# GPU dung chung: cho ca lock lan VRAM (bai hoc 08/09)
w=0
while true; do
  free=0; flock -n "$LOCK" -c true 2>/dev/null && free=1
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  avail=$(( 16376 - ${used:-16376} ))
  (( free == 1 && avail >= 11500 )) && { echo "$(ts) | GPU ranh (${avail}MiB)"; break; }
  sleep 60; w=$((w+60)); (( w % 600 == 0 )) && echo "$(ts) | cho... lock=$free VRAM=${avail}MiB"
done
echo "########## FT2 bat dau $(ts) | $(hostname) ##########"
# Ghi vao CAY RIENG de khong dung vao ket qua cu; baseline cua chinh cay nay se duoc chay.
RUN_NAME=ft2 SEED=42 FOLDS="1 2 3 4 5" \
BACKBONES="codebert=microsoft/codebert-base:cls" \
MODES="none" OPTIMIZERS="adamw" \
PHASE1_DATA_PATH=data/phase1_full.jsonl CWE_VOCAB=precomputed \
ARM_TAG="_full" PHASE1_TAG="_full" PHASE1_STORE="model/ft2/phase1" \
LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
PYTHON="$PY" bash run/matrix.sh 8>&-
echo "########## FT2 xong $(ts) ##########"
