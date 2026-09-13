#!/usr/bin/env bash
# FUSION_W5 — do trong so lop fusion tren CA 5 FOLD, khong chi fold 1.
#
# §48.1 hien la n=1: ban DA HOC cho do tan giua cac lop 0.1850 con ban NGAU NHIEN 0.0379
# (gap 4,9 lan). Ket luan "co che CO hoat dong" dang dua tren MOT fold — ma trong chinh
# ngay hom nay fold 1 da hai lan cho mot buc tranh khong giu duoc o n=5. Nen phai lam lai
# cho du 5 fold truoc khi tin.
#
# Chi so cuoi cua 10 o nay DA BIET (§48); chay lai chi de GIU CHECKPOINT (KEEP_CKPT=1), roi
# do trong so xong thi xoa ngay de khong day dia.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
exec 5>/tmp/mvd_fusion_w5.lock || exit 1
flock -n 5 || { echo "DA CO fusion_w5 dang chay"; exit 3; }
FOLDS="${FOLDS:-1 2 3 4 5}"
ts(){ date -u '+%F %T'; }
mkdir -p log/fusw
echo "########## FUSION_W5 bat dau $(ts) | fold: $FOLDS ##########"
for FOLD in $FOLDS; do
  for V in "fusft:" "fusftrnd:--fusion_src_random"; do
    TAG="${V%%:*}"; EXTRA="${V#*:}"
    CK="model/fus2w_codebert/transfer_latent_bottleneck_com_l0p05_ad48_${TAG}_adamw/seed_42/fold${FOLD}/best.pt"
    OUT="log/fusw/w_${TAG}_fold${FOLD}.json"
    if [[ -f "$OUT" ]]; then echo "=== fold $FOLD $TAG | da do roi ==="; continue; fi
    echo "===== $(ts) | fold $FOLD | $TAG ====="
    KEEP_CKPT=1 SKIP_BASELINE=1 RUN_NAME=fus2w SEED=42 FOLDS="$FOLD" \
    BACKBONES="codebert=microsoft/codebert-base:cls" MODES=latent_bottleneck OPTIMIZERS=adamw \
    PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
    ARM_TAG="_com_l0p05_ad48_${TAG}" PHASE1_TAG="_com_l0p05_ad48" PHASE1_STORE="model/fus2/phase1" \
    LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
    PHASE1_EXTRA="--sam_rho 0 --adapter_dim 48 --adapter_lr 1e-4" \
    PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr 1e-4 $EXTRA" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PY" bash run/matrix.sh 5>&-
    if [[ -f "$CK" ]]; then
      "$PY" tools/fusion_weights.py --checkpoint "$CK" --model_name microsoft/codebert-base \
        --pooling cls --data "data/sven_python_folds_norm/fold${FOLD}/test.jsonl" \
        --batch_size 8 --json_out "$OUT" > "log/fusw/w_${TAG}_fold${FOLD}.log" 2>&1 \
        && echo "  da do -> $OUT"
      rm -rf "$(dirname "$CK")"        # xoa ngay, khong de day dia
    else
      echo "  !! khong co checkpoint $CK"
    fi
  done
done
echo "########## FUSION_W5 xong $(ts) | $(ls log/fusw/*.json 2>/dev/null | wc -l)/10 phep do ##########"
