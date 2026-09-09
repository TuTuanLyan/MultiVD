#!/usr/bin/env bash
# CHOT2BB — cau hinh CHOT tren HAI backbone, bat/tat toan bo optimizer. Nguoi dung 09/09:
#
#   "chot gan nhu la latent_bottleneck + asam 2.0 + recadam 0.05 tren t5p nhung codebert
#    thi toi chua ro. Hyper param co the dieu chinh theo tung backbone duoc."
#
# Dung vay: bang tong hop cho thay codebert CHUA HE duoc chay voi ASAM rho=2.0 — muc tot
# nhat cua no hien la r0p1 (+0.0125 ROC, 10/10) va plain (+0.0112, 11/16), deu o n nho.
#
# THIET KE: 2 backbone x 2 dieu kien x 5 fold = 20 o Pha 2 + 10 baseline.
#   A  r2p0   RecAdam + ASAM rho=2.0   — DAY DU optimizer cua phuong phap
#   B  plain  AdamW, khong SAM/ASAM    — TAT ca hai cung luc
# Doi chung `baseline` (khong Pha 1) chay cung may cung fold, dung chung cho ca hai dieu kien.
#
# lambda = 0.05, KHONG phai 0.01. Dong lambda=0.01 ma nguoi dung thay (r2p0, ROC +0.0553)
# chi co o n=3 — bac 1, noi p=0.250 la SAN nen khong phan biet duoc voi may man. Ban
# lambda=0.05 co n=18 va ROC +0.0399 (17/18). Ngoai ra lambda nam o Pha 1 (train.py:53)
# nen doi lambda la phai huan luyen lai Pha 1 ca hai backbone — xem CLAUDE.md muc 5.
#
# Nguon: 4cwe. Tinh khiet nhat (100% hang thuoc 4 CWE dich), Pha 1 re nhat (930 dong), va
# la nguon co ket qua r2p0 tot nhat tren t5p khi tach rieng (+0.0170 ROC, 3/3).
#
# Pha 1: t5p da co san o model/n48/phase1. codebert THIEU (checkpoint cu nam tren ntat2
# da huy) nen script tu huan luyen vao DUNG kho ma opt1.sh doc.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
RUN="${RUN:-chot}"
SRC="${SRC:-4cwe}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
SEED="${SEED:-42}"
STORE="${STORE:-model/n48/phase1}"
BBS="${BBS:-t5p=Salesforce/codet5p-220m-bimodal:mean codebert=microsoft/codebert-base:cls}"
NEED_VRAM="${NEED_VRAM:-13000}"

LOCK="${MVD_LOCK:-/tmp/multivd_opt1.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver giu GPU tren may nay — dung"; exit 3; }

data_of(){ case "$1" in
  4cwe) echo "data/phase1_4cwe.jsonl fixed4" ;;
  com)  echo "data/phase1_common.jsonl precomputed" ;;
  full) echo "data/phase1_full.jsonl precomputed" ;;
esac; }
read -r DATA VOCAB <<< "$(data_of "$SRC")"

wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED_VRAM )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(date -u '+%F %T') | cho VRAM ${a}MiB"
  done; }

echo "########## CHOT2BB bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  nguon $SRC | fold $FOLD_LIST | seed $SEED | lambda 0.05"
echo "  A = r2p0  (RecAdam + ASAM rho=2.0)   B = plain (AdamW, khong SAM)"

# --- 1) bu Pha 1 cho backbone nao con thieu ---
for BB in $BBS; do
  L="${BB%%=*}"
  T="$STORE/${L}__latent_bottleneck_${SRC}_l0p05/seed_${SEED}/best.pt"
  if [[ -f "$T" ]]; then echo "=== Pha 1 $L | da co: $T ==="; continue; fi
  echo "===== $(date -u '+%F %T') | Pha 1 | $L | $SRC -> $T ====="
  wait_vram
  RUN_NAME=p1fill SEED="$SEED" FOLDS="" \
  BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS="adamw" \
  PHASE1_DATA_PATH="$DATA" CWE_VOCAB="$VOCAB" \
  PHASE1_TAG="_${SRC}_l0p05" ARM_TAG="_${SRC}_l0p05" PHASE1_STORE="$STORE" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 PHASE1_EXTRA="--sam_rho 0" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PYTHON" bash run/matrix.sh 9>&-
  [[ -f "$T" ]] && echo "  => DA TAO: $T" || echo "  !! VAN THIEU: $T — cac o cua $L se TRONG"
done

# --- 2) FOLD la vong ngoai (CLAUDE.md muc 1): xong mot fold la co mot lat cat so duoc ---
for FOLD in $FOLD_LIST; do
  for BB in $BBS; do
    L="${BB%%=*}"
    echo "===== $(date -u '+%F %T') | fold $FOLD | $L ====="
    wait_vram
    RUN="$RUN" SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 BB="$BB" \
    P1STORE="$STORE" \
    CONFIGS="r2p0|recadam|--sam_rho 2.0 --sam_variant asam
plain|adamw|--sam_rho 0" \
    bash run/opt1.sh 9>&-
  done
done
echo "########## CHOT2BB xong $(date -u '+%F %T') | $(find results -path "*${RUN}_*" -name 'fold*.json' 2>/dev/null | wc -l) o ##########"
