#!/usr/bin/env bash
# GATE2 — phep so con thieu cua De xuat 1: ghep voi "y kien thu hai CUNG TRUONG".
#
# gate1 da cho: ghep(transfer, baseline_seed7) chi hon ghep(transfer, baseline_seed42) rat it.
# Cau con lai: loi ich cua ghep la do HAI MODEL BIET THU KHAC NHAU, hay chi do CO HAI MODEL?
#   X = ghep(transfer_42, baseline_42)   -> hai model biet thu KHAC nhau
#   Y = ghep(transfer_42, transfer_7)    -> hai model biet thu GIONG nhau, khac moi hat giong
#   X > Y  => tinh bu tru la THAT
#   X ~ Y  => tat ca chi la giam phuong sai
#
# Khoi nay chi chay MOT nhanh moi: transfer o SEED 7, dung LAI dung checkpoint Pha 1 cua
# seed 42 (symlink). Tuc hai model transfer co CUNG tri thuc nguon, chi khac ngau nhien
# o Pha 2 — dung phep doi ung voi "baseline 42 vs baseline 7".
#
# Ghi vao CHINH cay gate1 de ghep cap duoc theo fold (cung may, cung phien du lieu).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
FOLDS_LIST="${FOLDS:-1 2 3}"
BB_LIST="${BB_LIST:-codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean}"
P1STORE="${PHASE1_STORE:-model/shuf1/phase1}"
NEWSEED="${NEWSEED:-7}"

exec 4>/tmp/mvd_gate2.lock || exit 1
flock -n 4 || { echo "DA CO gate2 dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/data/ntat/gate2.pid}"; echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

# Symlink Pha 1: seed moi TRO VE checkpoint seed 42. Lam tuong minh va in ra, vi neu khong
# thi matrix.sh se thay thieu file va HUAN LUYEN MOT PHA 1 MOI (~30 phut/backbone) — luc do
# phep so doi HAI bien (tri thuc nguon + hat giong) chu khong con mot.
for BB in $BB_LIST; do
  L="${BB%%=*}"; D="$P1STORE/${L}__none_com_real"
  [ -f "$D/seed_42/best.pt" ] || { echo "!! THIEU $D/seed_42/best.pt"; exit 5; }
  # Ba tinh huong, KHONG gop lam mot (bai hoc §3: gop "hong" voi "moi truong hong" da
  # tung XOA MAT hai checkpoint tot):
  #   khong co        -> tao symlink
  #   la symlink      -> giu, kiem lai o duoi
  #   la thu muc THAT -> rong thi xoa roi symlink; CO NOI DUNG thi DUNG HAN, khong tu xoa
  if [ -L "$D/seed_$NEWSEED" ]; then
    :
  elif [ -d "$D/seed_$NEWSEED" ]; then
    if [ -z "$(ls -A "$D/seed_$NEWSEED" 2>/dev/null)" ]; then
      rmdir "$D/seed_$NEWSEED" && ln -s seed_42 "$D/seed_$NEWSEED"
      echo "  ($L) seed_$NEWSEED la thu muc RONG -> da thay bang symlink"
    else
      echo "!! $D/seed_$NEWSEED la thu muc THAT va CO NOI DUNG."
      echo "   Do co the la mot Pha 1 seed $NEWSEED da huan luyen that. KHONG tu xoa."
      echo "   Xoa tay neu chac, hoac doi NEWSEED."
      exit 5
    fi
  else
    ln -s seed_42 "$D/seed_$NEWSEED"
  fi
  T=$(readlink -f "$D/seed_$NEWSEED/best.pt")
  S=$(readlink -f "$D/seed_42/best.pt")
  [ "$T" = "$S" ] || { echo "!! symlink seed_$NEWSEED KHONG tro ve seed_42 ($T)"; exit 5; }
  echo "  Pha 1 $L: seed_$NEWSEED -> seed_42 ($(stat -c%s "$S") byte)"
done
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__)" || exit 5

echo "########## GATE2 bat dau $(ts) | $(hostname) | seed $NEWSEED | fold: $FOLDS_LIST ##########"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
for FOLD in $FOLDS_LIST; do
  for BB in $BB_LIST; do
    L="${BB%%=*}"; GC=""; [ "$L" = "t5p" ] && GC=" --grad_checkpointing"
    echo "===== $(ts) | $L | FOLD $FOLD | transfer seed $NEWSEED ====="
    SKIP_BASELINE=1 RUN_NAME=gate1 SEED="$NEWSEED" FOLDS="$FOLD" BACKBONES="$BB" \
    MODES=none OPTIMIZERS=adamw CWE_VOCAB=precomputed \
    ARM_TAG="_com_real" PHASE1_TAG="_com_real" PHASE1_STORE="$P1STORE" \
    PHASE1_DATA_PATH=data/phase1_common.jsonl LAMBDA_CWE=0.05 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
    PHASE2_EXTRA="--sam_rho 0$GC" DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PY" bash run/matrix.sh 4>&-
  done
  echo "----- $(ts) | het fold $FOLD -----"
done
echo "########## GATE2 xong $(ts) | $(find results/gate1_*/transfer_none_com_real_adamw/seed_$NEWSEED -name 'fold*.json' 2>/dev/null | wc -l)/6 o ##########"
