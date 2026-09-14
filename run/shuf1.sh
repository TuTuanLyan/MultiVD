#!/usr/bin/env bash
# SHUF1 — doi chung XAO NHAN: loi ich Pha 1 la TRI THUC LO HONG hay chi PHOI NHIEM MIEN?
#
# Ghi du doan truoc khi do: records/prediction_2026-09-14_xao_nhan_phoi_nhiem_hay_tri_thuc.md
# Boi canh: FACTS §51 — tac vu nguon chi hoc duoc toi 0.57-0.59, CleanVul tu cong bo
# bo du lieu lo hong mang 40-75% nhieu nhan.
#
# Ba nhanh Pha 1, deu `aux_mode=none` (co lap dung nhan nhi phan, khong lan head phu),
# CUNG seed, CUNG split (phep chia Pha 1 chia theo NHOM va khong doc nhan), chi khac cot label:
#   real      nhan that
#   shufall   hoan vi toan cuc      -> pha ca lien he code-nhan LAN can bang trong cap
#   shufpair  doi cho trong cap     -> pha DUY NHAT: ban nao la ban truoc khi va
#
# KHOP SO BUOC GRADIENT: ca ba dung --save_last_epoch, chay dung PHASE1_EPOCHS epoch,
# tat dung som, lay checkpoint CUOI. Khong lam vay thi nhanh xao dung rat som va ta
# doi HAI bien (nhan + so epoch) chu khong mot.
#
# Bac 1 — kiem chung, n=3 fold. Khong con so nao o day duoc viet vao bai.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
RN="${RUN_NAME:-shuf1}"
E="${PHASE1_EPOCHS:-12}"
FOLDS_LIST="${FOLDS:-1 2 3}"
BB_LIST="${BB_LIST:-codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean}"

exec 4>/tmp/mvd_shuf1.lock || exit 1
flock -n 4 || { echo "DA CO shuf1 dang chay"; exit 3; }
# Ghi PID cua CHINH driver nay, mang ten cua chinh no. Dung lai file PID cua khoi khac
# thi mot PID da chet va mot PID CHUA BAO GIO DUNG cho ra cung mot tin hieu "da xong".
PIDFILE="${PIDFILE:-/workspace/shuf1.pid}"
echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

# Ba nhanh: <ten>|<file du lieu>
ARMS=(
  "real|data/phase1_common.jsonl"
  "shufall|data/phase1_common_shufall.jsonl"
  "shufpair|data/phase1_common_shufpair.jsonl"
)

# Kiem MOI duong dan chuoi se cham vao TRUOC khi chay (bai hoc: file tao sau lan rsync).
for a in "${ARMS[@]}"; do
  f="${a#*|}"
  [ -f "$f" ] || { echo "!! THIEU $f — DUNG"; exit 5; }
done
for f in run/matrix.sh src/train_transfer.py src/train_baseline.py src/report_fold.py; do
  [ -f "$f" ] || { echo "!! THIEU $f — DUNG"; exit 5; }
done
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__,transformers.__version__)" || exit 5
grep -q "save_last_epoch" src/train_transfer.py || { echo "!! src/train_transfer.py chua co --save_last_epoch"; exit 5; }

echo "########## SHUF1 bat dau $(ts) | $(hostname) | E=$E epoch | fold: $FOLDS_LIST ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for BB in $BB_LIST; do
  LABEL="${BB%%=*}"
  echo "==================== BACKBONE $LABEL ===================="
  for FOLD in $FOLDS_LIST; do
    first=1
    for a in "${ARMS[@]}"; do
      ARM="${a%%|*}"; DATA="${a#*|}"
      echo "===== $(ts) | $LABEL | FOLD $FOLD | nhanh $ARM ====="
      SKIP_BASELINE=$([ "$first" = 1 ] && echo 0 || echo 1) \
      RUN_NAME="$RN" SEED=42 FOLDS="$FOLD" \
      BACKBONES="$BB" MODES=none OPTIMIZERS=adamw \
      PHASE1_DATA_PATH="$DATA" CWE_VOCAB=precomputed \
      ARM_TAG="_com_${ARM}" PHASE1_TAG="_com_${ARM}" PHASE1_STORE="model/${RN}/phase1" \
      LAMBDA_CWE=0.05 PHASE1_EPOCHS="$E" PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
      PHASE1_EXTRA="--sam_rho 0 --save_last_epoch" PHASE2_EXTRA="--sam_rho 0" \
      DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
      PYTHON="$PY" bash run/matrix.sh 4>&-
      first=0
    done
    echo "----- $(ts) | het $LABEL fold $FOLD | o: $(find results/${RN}_${LABEL} -name 'fold*.json' 2>/dev/null | wc -l) -----"
  done
done

TOT=0
for BB in $BB_LIST; do
  L="${BB%%=*}"; n=$(find "results/${RN}_${L}" -name 'fold*.json' 2>/dev/null | wc -l)
  echo "  $L: $n o"; TOT=$((TOT+n))
done
echo "########## SHUF1 xong $(ts) | $TOT/24 o ##########"
