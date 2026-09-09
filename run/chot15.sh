#!/usr/bin/env bash
# CHOT15 — bo sung seed 7 va 1234 cho khoi `chot` de len bac 3 (n=15).
#
# Nguoi dung 10/09: "chay them 2 seed nua cho du n=15 cho ca 3 source va setting.
# chay them n=10 la duoc cai chay roi khong can chay lai dau."
#
# Seed 42 DA XONG (70 o, FACTS §34) — khong dung toi. File nay chi chay seed 7 va 1234.
#
# PHA 1 HUAN LUYEN LAI CHO TUNG SEED, khong dung lai ban seed 42. Ly do: phat bieu cua ca
# khoi la "loi ich den tu Pha 1 + head". Neu giu nguyen mot Pha 1 va chi doi seed Pha 2 thi
# khong the biet loi ich la tinh chat cua PHUONG PHAP hay cua DUNG MOT CHECKPOINT do. Seed
# con quyet dinh ca cach chia train/val cua Pha 1 (train_transfer.py:334), nen doi seed ma
# giu Pha 1 la doi nua bien.
#
#   161 -> codebert (rho 0.1)      158 -> t5p (rho 2.0)
# Mot backbone tron mot may (CLAUDE.md muc 4). Cay ket qua mang TEN MAY de khong bao gio
# ghep cap nham qua may.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python /venv/main/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch,numpy,sklearn" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch+numpy+sklearn"; exit 2; }
export PYTHON

RUN="${RUN:?can RUN, vd chot161}"
BB="${BB:?can BB, vd 'codebert=microsoft/codebert-base:cls'}"
CFG_A="${CFG_A:?can CFG_A}"
CFG_B="${CFG_B:-plain|adamw|--sam_rho 0}"
SEED_LIST="${SEED_LIST:-7 1234}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
SOURCES_LIST="${SOURCES_LIST:-4cwe com full}"
STORE="${STORE:-model/n48/phase1}"
# NEED_VRAM phai theo BACKBONE, khong dat chung mot muc. Do that: codebert dinh 6674 MiB,
# t5p dinh ~13468 MiB. Dat chung 13000 cho ca hai la qua chat cho codebert — 00:17 VN 10/09
# `cuongtm` chiem 2738 MiB tren 161 (con trong 12798) va job codebert LE RA van chay thoai mai
# nhung se ngoi cho vo ich, tham chi kich hoat thue vast khong can thiet.
case "${BB%%=*}" in
  codebert|unixcoder) NEED_VRAM="${NEED_VRAM:-8500}" ;;
  *)                  NEED_VRAM="${NEED_VRAM:-13000}" ;;
esac
LOCK="${CHOT15_LOCK:-/tmp/mvd_chot15.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO chot15 dang chay tren may nay — dung"; exit 3; }

data_of(){ case "$1" in
  4cwe) echo "data/phase1_4cwe.jsonl fixed4" ;;
  com)  echo "data/phase1_common.jsonl precomputed" ;;
  full) echo "data/phase1_full.jsonl precomputed" ;;
esac; }
wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED_VRAM )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(date -u '+%F %T') | cho VRAM ${a}MiB"
  done; }

L="${BB%%=*}"
echo "########## CHOT15 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  RUN=$RUN | backbone=$L | seed: $SEED_LIST | fold: $FOLD_LIST | nguon: $SOURCES_LIST"
echo "  A = ${CFG_A%%|*}   B = ${CFG_B%%|*}   | python=$PYTHON"

for SEED in $SEED_LIST; do
  # --- Pha 1 cho seed nay ---
  for SRC in $SOURCES_LIST; do
    read -r DATA VOCAB <<< "$(data_of "$SRC")"
    T="$STORE/${L}__latent_bottleneck_${SRC}_l0p05/seed_${SEED}/best.pt"
    if [[ -f "$T" ]]; then echo "=== Pha 1 $L/$SRC seed $SEED | da co ==="; continue; fi
    echo "===== $(date -u '+%F %T') | Pha 1 | $L | $SRC | seed $SEED ====="
    wait_vram
    RUN_NAME=p1_${RUN} SEED="$SEED" FOLDS="" \
    BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS="adamw" \
    PHASE1_DATA_PATH="$DATA" CWE_VOCAB="$VOCAB" \
    PHASE1_TAG="_${SRC}_l0p05" ARM_TAG="_${SRC}_l0p05" PHASE1_STORE="$STORE" \
    LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 PHASE1_EXTRA="--sam_rho 0" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PYTHON" bash run/matrix.sh
    [[ -f "$T" ]] && echo "  => DA TAO: $T" || echo "  !! VAN THIEU: $T — o cua $L/$SRC/seed$SEED se TRONG"
  done
  # --- Pha 2: FOLD vong ngoai, nguon vong trong (CLAUDE.md muc 1) ---
  for FOLD in $FOLD_LIST; do
    for SRC in $SOURCES_LIST; do
      echo "===== $(date -u '+%F %T') | $L | seed $SEED | fold $FOLD | $SRC ====="
      wait_vram
      RUN="$RUN" SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 BB="$BB" \
      P1STORE="$STORE" CONFIGS="$CFG_A
$CFG_B" bash run/opt1.sh
    done
  done
  echo "########## seed $SEED xong $(date -u '+%F %T') | $(find results/${RUN}_${L} -name 'fold*.json' 2>/dev/null | wc -l) o ##########"
done
echo "########## CHOT15 xong $(date -u '+%F %T') | $(find results/${RUN}_${L} -name 'fold*.json' 2>/dev/null | wc -l) o ##########"
