#!/usr/bin/env bash
# FUSION_5060 — khoi fus3 chay tron ven tren MOT may vast RTX 5060 Ti (nhan `ntat`).
#
# Vi sao day len vast: GPU local (A4000) bi nguoi dung khac chiem lien tuc, driver
# `run/fusion_base.sh` cho 9h43 ma chua chay duoc o nao. Muc 4 CLAUDE.md doi nhanh
# doi chung phai CUNG MAY CUNG PHIEN voi nhanh no doi chung, nen ca ba nhanh chay o day.
#
#   baseline            khong Pha 1
#   latent_bottleneck   phuong phap chot (Pha 1 + head phu, KHONG adapter)
#   fusft               adapter nguon + adapter dich + fusion, fine-tune ca backbone
#
# Cay ket qua rieng: `results/fus5060_codebert` — KHONG trung ten voi cay local
# `results/fus3_codebert`. Hai cay khac may thi khong duoc gop (muc 2b).
#
# Pha 1 dung lai, khong huan luyen lai:
#   khong adapter: model/n48/phase1/codebert__latent_bottleneck_com_l0p05
#   co adapter   : model/fus2/phase1/codebert__latent_bottleneck_com_l0p05_ad48
#
# GPU o day la CUA RIENG — khong can cong cho VRAM nhu ban local.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
RN="${RUN_NAME:-fus5060}"

# Mot driver mot lock (memory: one-driver-one-lock-and-count-artifacts).
exec 4>/tmp/mvd_fusion_5060.lock || exit 1
flock -n 4 || { echo "DA CO fusion_5060 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }

echo "########## FUSION_5060 bat dau $(ts) | $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import torch,transformers,sklearn; print('torch',torch.__version__,'transformers',transformers.__version__,'sklearn',sklearn.__version__)"

for FOLD in ${FOLDS:-1 2 3 4 5}; do
  echo "===== $(ts) | FOLD $FOLD | baseline + latent_bottleneck ====="
  SKIP_BASELINE=0 RUN_NAME="$RN" SEED=42 FOLDS="$FOLD" \
  BACKBONES="codebert=microsoft/codebert-base:cls" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05" PHASE1_TAG="_com_l0p05" PHASE1_STORE="model/n48/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-

  echo "===== $(ts) | FOLD $FOLD | fusft ====="
  SKIP_BASELINE=1 RUN_NAME="$RN" SEED=42 FOLDS="$FOLD" \
  BACKBONES="codebert=microsoft/codebert-base:cls" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05_ad48_fusft" PHASE1_TAG="_com_l0p05_ad48" PHASE1_STORE="model/fus2/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 --adapter_dim 48 --adapter_lr 1e-4" \
  PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr 1e-4" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-

  echo "----- $(ts) | het FOLD $FOLD | o hien co: $(find results/${RN}_codebert -name 'fold*.json' 2>/dev/null | wc -l)/15 -----"
done
N=$(find results/${RN}_codebert -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## FUSION_5060 xong $(ts) | $N/15 o ##########"
