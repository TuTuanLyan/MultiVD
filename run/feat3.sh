#!/usr/bin/env bash
# FEAT3 — kiem chung bac 1 (n=3 fold, seed 42) cho can thiep suy TRUC TIEP tu FACTS §36.
#
# §36 do duoc, tren chinh checkpoint Pha 1 nay:
#   * DAC TRUNG chuyen giao   — linear probe dong bang tren dac trung Pha 1 hon pretrained
#                               +0.1125 ROC / +0.0849 F1, CA NAM fold (codebert)
#   * HAM QUYET DINH thi KHONG — `vul_head` cua chinh no cham thang tren Python: 0.537 F1 /
#                               0.645 ROC (codebert), 0.503 / 0.545 (t5p) ~ ngau nhien
#
# Hai can thiep duy nhat ma cap so do TRUC TIEP goi ra, va ca hai deu CHUA TUNG chay:
#   fd   NEO KHONG GIAN DAC TRUNG: cong beta*(1 - cos(f_theta(x), f_theta*(x))) tren input DICH.
#        Khac han neo TRONG SO (RecAdam/L2-SP/SPD/Fisher — bon khoi da bac): rang buoc nay
#        BIET du lieu dich, va no giu dung dai luong da do la co gia tri. Thay = chinh mo hinh
#        luc khoi tao Pha 2, dac trung tinh san mot lan => 0 VRAM them, 0 giay them moi buoc.
#   rh   KHOI TAO LAI vul_head: duong chay mac dinh nap nguyen `vul_head` cua Pha 1 (strict=True)
#        roi bat Pha 2 go mot ham quyet dinh SAI mot cach TU TIN. Bo no di co hon khong?
#
# THIET KE: 4 nhanh x 3 fold. `plain` va `baseline` DUNG LAI cua khoi bridge3 cung ngay,
# cung may, cung fold, cung ma (matrix.sh bo qua o da co JSON) — dung CLAUDE.md muc 4, va
# tiet kiem 6 o GPU. Vi the RUN mac dinh la `bridge3`, KHONG phai ten rieng.
#   fd1     beta 1     rang buoc nhe   (d(f,f*) do duoc ~0.05 => dong gop ~7% loss)
#   fd10    beta 10    rang buoc that  (dong gop ~0.5, ngang cross-entropy)
#   rh      chi khoi tao lai head
#   fd10rh  ca hai
#
#   BB=codebert=microsoft/codebert-base:cls        bash run/feat3.sh   # 161
#   BB=t5p=Salesforce/codet5p-220m-bimodal:mean    bash run/feat3.sh   # 158
#
# Doc: python3 tools/bridge_report.py --b plain --a fd1,fd10,rh,fd10rh results/bridge3_<bb>
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
: "${BB:?can BB=<label>=<hf-name>:<pooling>}"
RUN="${RUN:-bridge3}"
SRC="${SRC:-4cwe}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
SEED="${SEED:-42}"
STORE="${STORE:-model/n48/phase1}"
L="${BB%%=*}"
case "$L" in
  codebert|unixcoder) NEED_VRAM="${NEED_VRAM:-8500}" ;;
  *)                  NEED_VRAM="${NEED_VRAM:-13000}" ;;
esac
CONFIGS="${CONFIGS:-\
fd1|adamw|--sam_rho 0 --feat_distill_beta 1.0
fd10|adamw|--sam_rho 0 --feat_distill_beta 10.0
rh|adamw|--sam_rho 0 --phase2_reinit_head
fd10rh|adamw|--sam_rho 0 --feat_distill_beta 10.0 --phase2_reinit_head}"

LOCK="${FEAT_LOCK:-/tmp/mvd_feat3.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO feat3 dang chay tren may nay — dung"; exit 3; }

wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED_VRAM )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(date -u '+%F %T') | cho VRAM ${a}MiB"
  done; }

CKPT="$STORE/${L}__latent_bottleneck_${SRC}_l0p05/seed_${SEED}/best.pt"
NC=$(printf '%s\n' "$CONFIGS" | grep -c '|'); NF=$(echo $FOLD_LIST | wc -w)
echo "########## FEAT3 bat dau $(date -u '+%F %T') | $(hostname) | $L ##########"
echo "  nguon $SRC | fold $FOLD_LIST | seed $SEED | $NC nhanh | ky vong $(( NC*NF )) o MOI"
echo "  ghi vao cay results/${RUN}_${L} — plain/baseline cua bridge3 duoc DUNG LAI, khong chay lai"
echo "  Pha 1: $CKPT $( [[ -f "$CKPT" ]] && echo "(co, $(stat -c %s "$CKPT") B)" || echo "!! THIEU — dung" )"
[[ -f "$CKPT" ]] || exit 4
printf '%s\n' "$CONFIGS" | sed 's/^/  nhanh: /'

for FOLD in $FOLD_LIST; do
  echo "===== $(date -u '+%F %T') | $L | fold $FOLD | $SRC ====="
  wait_vram
  RUN="$RUN" SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 BB="$BB" \
  P1STORE="$STORE" MVD_LOCK=/tmp/mvd_feat3_opt1.lock CONFIGS="$CONFIGS" \
  bash run/opt1.sh 9>&-
  N=$(ls results/${RUN}_${L}/transfer_latent_bottleneck_${SRC}_l0p05_*/seed_${SEED}/fold${FOLD}.json 2>/dev/null | wc -l)
  echo "########## FEAT3 $L fold $FOLD xong $(date -u '+%F %T') | $N o tong cong trong cay ##########"
done
TOT=$(find results/${RUN}_${L} -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## FEAT3 xong $(date -u '+%F %T') | $(hostname) | $L | $TOT o trong cay ##########"
