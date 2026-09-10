#!/usr/bin/env bash
# BRIDGE3 — kiem chung bac 1 (n=3 fold, seed 42) cho "cau CWE": dua du lieu NGUON vao Pha 2.
#
# Vi sao (RESEARCH_2026-09-06 §11–12, FACTS §35): bon khoi OPT1/RET1/SPD1/INT1 cho thay tri
# thuc nguon vao dich CHI qua diem khoi tao; neo trong so (RecAdam/SPD/Fisher/WiSE-FT) khong
# them gi vi neo la RANG BUOC, khong phai KENH TRUYEN. Duong con lai duy nhat la cho gradient
# cua du lieu nguon nan truc tiep nghiem dich — replay nguon trong Pha 2 (Đ5), kem cau CWE:
# head phu 4 lop cua Pha 1 hoc tiep tren CA HAI ngon ngu (cung bang CWE_MAPPING).
#
# THIET KE 2x2, cung fold cung may cung phien voi doi chung (CLAUDE.md muc 4):
#   plain   AdamW thuan                          <- doi chung (fine-tune hai lan)
#   cwe05   + lambda_cwe 0.05 tren DICH          <- chi nua "dich" cua cau
#   rp50    + replay 4cwe mu0=0.5 giam ve 0 sau 6 epoch, can tang   <- chi replay
#   rpc     + ca hai: replay + lambda_cwe 0.05 tren nguon VA dich    <- cau CWE day du
# `baseline` (khong Pha 1) chay cung fold do matrix.sh.
#
# Nguon: 4cwe (100%% hang thuoc 4 CWE dich, cung khong gian nhan). Pha 1 DUNG LAI, khong
# huan luyen lai (model/n48/phase1/<bb>__latent_bottleneck_4cwe_l0p05/seed_42/best.pt).
# Fold la vong ngoai: het fold 1 la co mot lat cat 4 nhanh + baseline so duoc ngay.
#
#   BB=codebert=microsoft/codebert-base:cls        bash run/bridge3.sh   # 161
#   BB=t5p=Salesforce/codet5p-220m-bimodal:mean    bash run/bridge3.sh   # 158
#
# Ket qua: results/bridge3_<bb>/{baseline,transfer_latent_bottleneck_4cwe_l0p05_<tag>_adamw}/seed_42/fold<k>.json
# Doc: python3 tools/report2.py --a <nhanh> --b <doi chung> results/bridge3_<bb>
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
: "${BB:?can BB=<label>=<hf-name>:<pooling>, vd BB=codebert=microsoft/codebert-base:cls}"
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
RPL="--replay_data data/phase1_4cwe.jsonl --replay_cwe_vocab fixed4 --replay_mu ${MU0:-0.5} --replay_epochs ${RP_EPOCHS:-6} --replay_stratify"
CONFIGS="${CONFIGS:-\
plain|adamw|--sam_rho 0
cwe05|adamw|--sam_rho 0 --phase2_lambda_cwe 0.05
rp50|adamw|--sam_rho 0 $RPL
rpc|adamw|--sam_rho 0 $RPL --replay_lambda_cwe 0.05 --phase2_lambda_cwe 0.05}"

LOCK="${BRIDGE_LOCK:-/tmp/mvd_bridge3.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO bridge3 dang chay tren may nay — dung"; exit 3; }

wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED_VRAM )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(date -u '+%F %T') | cho VRAM ${a}MiB"
  done; }

CKPT="$STORE/${L}__latent_bottleneck_${SRC}_l0p05/seed_${SEED}/best.pt"
NC=$(printf '%s\n' "$CONFIGS" | grep -c '|'); NF=$(echo $FOLD_LIST | wc -w)
echo "########## BRIDGE3 bat dau $(date -u '+%F %T') | $(hostname) | $L ##########"
echo "  nguon $SRC | fold $FOLD_LIST | seed $SEED | $NC nhanh | ky vong $(( NC*NF )) o + $NF baseline"
echo "  Pha 1: $CKPT $( [[ -f "$CKPT" ]] && echo "(co, $(stat -c %s "$CKPT") B)" || echo "!! THIEU — dung" )"
[[ -f "$CKPT" ]] || exit 4
[[ -f data/phase1_4cwe.jsonl ]] || { echo "!! thieu data/phase1_4cwe.jsonl"; exit 4; }
printf '%s\n' "$CONFIGS" | sed 's/^/  nhanh: /'

for FOLD in $FOLD_LIST; do
  echo "===== $(date -u '+%F %T') | $L | fold $FOLD | $SRC ====="
  wait_vram
  RUN="$RUN" SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 BB="$BB" \
  P1STORE="$STORE" MVD_LOCK=/tmp/mvd_bridge3_opt1.lock CONFIGS="$CONFIGS" \
  bash run/opt1.sh 9>&-
  N=$(ls results/${RUN}_${L}/transfer_latent_bottleneck_${SRC}_l0p05_*/seed_${SEED}/fold${FOLD}.json 2>/dev/null | wc -l)
  B=$(ls results/${RUN}_${L}/baseline/seed_${SEED}/fold${FOLD}.json 2>/dev/null | wc -l)
  echo "########## BRIDGE3 $L fold $FOLD xong $(date -u '+%F %T') | $N/$NC o + $B/1 baseline ##########"
done
TOT=$(find results/${RUN}_${L} -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## BRIDGE3 xong $(date -u '+%F %T') | $(hostname) | $L | $TOT/$(( NC*NF + NF )) o ##########"
