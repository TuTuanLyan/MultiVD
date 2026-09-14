#!/usr/bin/env bash
# FUSNONE — bo head `latent_bottleneck` khoi Pha 1, GIU NGUYEN adapter + fusion.
#
# Cau hoi: phan loi ich cua fusion co CAN head phu o Pha 1 khong, hay Pha 1 nhi phan thuan
# (`none`) + adapter cung cho ket qua tuong duong? §46 da do head phu chi dang +0.0005 tren
# 132 o BINH THUONG — nhung chua ai do dieu do TRONG cau hinh fusion.
#
# Hai nhanh khac DUNG MOT bien (aux_mode), moi thu khac giong het:
#   A  fusft    aux_mode=latent_bottleneck  (Pha 1 DUNG LAI model/fus2/phase1/...l0p05_ad48)
#   B  nonefus  aux_mode=none               (Pha 1 huan luyen moi, cung giao thuc)
#
# Giao thuc Pha 1 doc THANG tu training_args cua checkpoint nhanh A, khong doan:
#   epochs 15 · patience 10 · min_epochs 3 · lr 2e-5 · adapter_dim 48 · adapter_lr 1e-4
#   sam_rho 0 · batch 16 · max_length 512 · seed 42 · data phase1_common.jsonl
#
# CAY RIENG `results/fusnone_codebert` voi BASELINE cua chinh no — ca ba nhanh cung may
# cung phien, nen Delta ghep cap sach (muc 4). Khong muon cay fus5060 vi no chay cach day
# vai gio; sau §52.1 thi khong con tin "cung may khac gio" la du nua.
#
# Bac 2 — xac nhan, n=5 fold, seed 42, codebert. 15 o + 1 lan Pha 1.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
RN="${RUN_NAME:-fusnone}"
BB="${BB:-codebert=microsoft/codebert-base:cls}"
FOLDS_LIST="${FOLDS:-1 2 3 4 5}"

exec 4>/tmp/mvd_fusnone.lock || exit 1
flock -n 4 || { echo "DA CO fusnone dang chay"; exit 3; }
# Ghi PID cua CHINH driver nay, mang ten cua chinh no. Dung lai file PID cua khoi khac
# thi mot PID da chet va mot PID CHUA BAO GIO DUNG cho ra cung mot tin hieu "da xong".
PIDFILE="${PIDFILE:-/workspace/fusnone.pid}"
echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

P1_HEAD="model/fus2/phase1/codebert__latent_bottleneck_com_l0p05_ad48/seed_42/best.pt"
[ -f "$P1_HEAD" ] || { echo "!! THIEU Pha 1 cua nhanh CO head: $P1_HEAD"; exit 5; }
for f in run/matrix.sh src/train_transfer.py src/train_baseline.py src/adapters.py data/phase1_common.jsonl; do
  [ -f "$f" ] || { echo "!! THIEU $f"; exit 5; }
done
"$PY" -c "import torch,transformers,sklearn" || exit 5
echo "Pha 1 nhanh CO head: da co, se dung lai (khong huan luyen lai)"

echo "########## FUSNONE bat dau $(ts) | $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for FOLD in $FOLDS_LIST; do
  echo "===== $(ts) | FOLD $FOLD | A: fusft (CO head latent_bottleneck) + baseline ====="
  SKIP_BASELINE=0 RUN_NAME="$RN" SEED=42 FOLDS="$FOLD" \
  BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05_ad48_fusft" PHASE1_TAG="_com_l0p05_ad48" PHASE1_STORE="model/fus2/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_PATIENCE=10 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 --adapter_dim 48 --adapter_lr 1e-4" \
  PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr 1e-4" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-

  echo "===== $(ts) | FOLD $FOLD | B: nonefus (KHONG head) ====="
  SKIP_BASELINE=1 RUN_NAME="$RN" SEED=42 FOLDS="$FOLD" \
  BACKBONES="$BB" MODES=none OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_ad48_fusft" PHASE1_TAG="_com_ad48" PHASE1_STORE="model/${RN}/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_PATIENCE=10 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 --adapter_dim 48 --adapter_lr 1e-4" \
  PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr 1e-4" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-

  echo "----- $(ts) | het FOLD $FOLD | o: $(find results/${RN}_codebert -name 'fold*.json' 2>/dev/null | wc -l)/15 -----"
done
N=$(find results/${RN}_codebert -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## FUSNONE xong $(ts) | $N/15 o ##########"
