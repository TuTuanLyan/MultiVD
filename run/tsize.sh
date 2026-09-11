#!/usr/bin/env bash
# TSIZE — loi ich cua transfer co phai la hieu ung DU LIEU IT khong? (bac 1, n=3 fold, seed 42)
#
# Du doan khai bao TRUOC khi chay o records/prediction_2026-09-11_duong_cong_co_dich.md.
# Tom tat: so hang train theo tung CWE (~40 / ~47 / ~124 / ~244) xep dung thu tu voi loi ich
# do duoc o n=15 (+0.37 / +0.35 / +0.02 / -0.01). Va §36 do duoc dac trung Pha 1 cai thien o
# CA BON lop tren codebert — manh nhat lai la CWE-078, dung lop null end-to-end. Nen gia
# thuyet: mo hinh chi dua vao dac trung nguon khi dich KHONG du vi du cho lop do.
#
# Phep kiem: cat NGAU NHIEN tap train Python xuong N = 228 / 152 / 76, giu NGUYEN val va test.
# N=456 da co san o khoi bridge3.
#
# HAI DIEU PHAI DUNG, neu khong phep so doi hai bien:
#   1. CA HAI nhanh (chuyen giao VA baseline) phai bi cat CUNG mot N. Vi the can BASELINE_EXTRA
#      trong matrix.sh — truoc do doi chung khong nhan duoc co nao.
#   2. Cung seed => `limit_records` tra CUNG mot tap con cho ca hai nhanh (lay mau ngau nhien
#      co seed, khong thay the). Da kiem: src/train_transfer.py va src/train_baseline.py deu
#      goi limit_records(records, args.max_train_samples, args.seed).
#
# Chi cat TRAIN. val dung --max_eval_samples (khong dat), test giu nguyen 152 hang — cat test
# thi Δ doi ca thuoc do chu khong con do cung mot thu.
#
# Moi N mot cay ket qua rieng (`results/sz<N>_<bb>`) vi baseline cua matrix.sh ghi vao duong dan
# co dinh theo RUN_NAME; chung cay thi cac N se de len nhau.
#
#   BB=codebert=microsoft/codebert-base:cls        bash run/tsize.sh   # 161
#   BB=t5p=Salesforce/codet5p-220m-bimodal:mean    bash run/tsize.sh   # 158
#
# Doc: python3 tools/tsize_report.py
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
: "${BB:?can BB=<label>=<hf-name>:<pooling>}"
SIZES="${SIZES:-228 152 76}"
SRC="${SRC:-4cwe}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
SEED="${SEED:-42}"
STORE="${STORE:-model/n48/phase1}"
L="${BB%%=*}"
case "$L" in
  codebert|unixcoder) NEED_VRAM="${NEED_VRAM:-8500}" ;;
  *)                  NEED_VRAM="${NEED_VRAM:-13000}" ;;
esac

LOCK="${TSIZE_LOCK:-/tmp/mvd_tsize.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO tsize dang chay tren may nay — dung"; exit 3; }

wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED_VRAM )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(date -u '+%F %T') | cho VRAM ${a}MiB"
  done; }

CKPT="$STORE/${L}__latent_bottleneck_${SRC}_l0p05/seed_${SEED}/best.pt"
NF=$(echo $FOLD_LIST | wc -w); NS=$(echo $SIZES | wc -w)
echo "########## TSIZE bat dau $(date -u '+%F %T') | $(hostname) | $L ##########"
echo "  N: $SIZES (N=456 da co o bridge3) | fold $FOLD_LIST | seed $SEED | nguon $SRC"
echo "  ky vong $(( NS*NF )) o chuyen giao + $(( NS*NF )) baseline = $(( 2*NS*NF )) o MOI"
echo "  Pha 1: $CKPT $( [[ -f "$CKPT" ]] && echo "(co)" || echo "!! THIEU — dung" )"
[[ -f "$CKPT" ]] || exit 4

for N in $SIZES; do
  echo "===== $(date -u '+%F %T') | $L | N=$N hang train ====="
  # Truyen cho CA HAI nhanh. BASELINE_EXTRA duoc matrix.sh doc tu moi truong.
  export BASELINE_EXTRA="--max_train_samples $N"
  for FOLD in $FOLD_LIST; do
    wait_vram
    RUN="sz$N" SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 BB="$BB" \
    P1STORE="$STORE" MVD_LOCK=/tmp/mvd_tsize_opt1.lock \
    CONFIGS="plain|adamw|--sam_rho 0 --max_train_samples $N" \
    bash run/opt1.sh 9>&-
  done
  unset BASELINE_EXTRA
  T=$(ls results/sz${N}_${L}/transfer_*/seed_${SEED}/fold*.json 2>/dev/null | wc -l)
  B=$(ls results/sz${N}_${L}/baseline/seed_${SEED}/fold*.json 2>/dev/null | wc -l)
  echo "########## TSIZE $L N=$N xong $(date -u '+%F %T') | $T/$NF chuyen giao + $B/$NF baseline ##########"
done
echo "########## TSIZE xong $(date -u '+%F %T') | $(hostname) | $L | $(find results/sz*_${L} -name 'fold*.json' 2>/dev/null | wc -l) o ##########"
