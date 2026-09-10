#!/usr/bin/env bash
# LPFT3 — nhanh doi chung BAT BUOC ma phan tra cuu 11/09 chi ra (RESEARCH_2026-09-10 §5.2a).
#
# Kumar, Raghunathan, Jones, Ma, Liang — ICLR 2022 Oral (arXiv:2202.10054) ke don NGUOC voi
# nhanh `rh`: ho do duoc head NGAU NHIEN lam MEO dac trung tot khi fine-tune toan bo, va thuoc
# chua la fit head TRUOC voi backbone dong bang (LP-FT): +1% ID / +10% OOD so voi FT tren 10 bo
# dich chuyen phan phoi. Reviewer nao biet LP-FT se hoi ngay "sao khong fit head truoc?" —
# phai tra loi bang SO, khong bang lap luan.
#
# Hai nhanh, va chung KHAC NHAU o mot diem quan trong:
#   lp3     giu head cua Pha 1 roi PROBE no (backbone dong bang, 3 epoch) — tinh chinh mot ham
#           quyet dinh da co san nhung do duoc la gan ngau nhien (§36)
#   rhlp3   KHOI TAO LAI head roi probe — day moi la LP-FT SACH GIAO KHOA (head ngau nhien)
# `plain` va `baseline` dung lai cua khoi bridge3 cung may cung fold cung ngay.
#
#   BB=codebert=microsoft/codebert-base:cls        bash run/lpft3.sh   # 161
#   BB=t5p=Salesforce/codet5p-220m-bimodal:mean    bash run/lpft3.sh   # 158
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
: "${BB:?can BB=<label>=<hf-name>:<pooling>}"
RUN="${RUN:-bridge3}"; SRC="${SRC:-4cwe}"; FOLD_LIST="${FOLD_LIST:-1 2 3}"; SEED="${SEED:-42}"
STORE="${STORE:-model/n48/phase1}"; L="${BB%%=*}"
case "$L" in codebert|unixcoder) NEED_VRAM="${NEED_VRAM:-8500}" ;; *) NEED_VRAM="${NEED_VRAM:-13000}" ;; esac
CONFIGS="${CONFIGS:-\
lp3|adamw|--sam_rho 0 --lp_epochs 3
rhlp3|adamw|--sam_rho 0 --lp_epochs 3 --phase2_reinit_head}"
LOCK="${LPFT_LOCK:-/tmp/mvd_lpft3.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO lpft3 dang chay tren may nay — dung"; exit 3; }
wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED_VRAM )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(date -u '+%F %T') | cho VRAM ${a}MiB"
  done; }
CKPT="$STORE/${L}__latent_bottleneck_${SRC}_l0p05/seed_${SEED}/best.pt"
NC=$(printf '%s\n' "$CONFIGS" | grep -c '|'); NF=$(echo $FOLD_LIST | wc -w)
echo "########## LPFT3 bat dau $(date -u '+%F %T') | $(hostname) | $L ##########"
echo "  fold $FOLD_LIST | seed $SEED | $NC nhanh | ky vong $(( NC*NF )) o MOI | cay results/${RUN}_${L}"
[[ -f "$CKPT" ]] || { echo "!! THIEU Pha 1 $CKPT"; exit 4; }
printf '%s\n' "$CONFIGS" | sed 's/^/  nhanh: /'
for FOLD in $FOLD_LIST; do
  echo "===== $(date -u '+%F %T') | $L | fold $FOLD | $SRC ====="
  wait_vram
  RUN="$RUN" SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 BB="$BB" \
  P1STORE="$STORE" MVD_LOCK=/tmp/mvd_lpft3_opt1.lock CONFIGS="$CONFIGS" \
  bash run/opt1.sh 9>&-
  echo "########## LPFT3 $L fold $FOLD xong $(date -u '+%F %T') ##########"
done
echo "########## LPFT3 xong $(date -u '+%F %T') | $(hostname) | $L | $(find results/${RUN}_${L} -name 'fold*.json' | wc -l) o trong cay ##########"
