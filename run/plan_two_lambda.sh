#!/usr/bin/env bash
# Kế hoạch một máy: chạy ma trận ở λ=0.2 rồi λ=0.05, dùng chung baseline và `none`.
#
# Khối 2 chỉ chạy ba nhánh có head phụ. `none` và baseline KHÔNG chạy lại vì cả
# hai độc lập với λ: baseline không đọc dữ liệu source, còn với aux_mode=none thì
# src/train.py cho aux_loss = None nên λ không xuất hiện trong hàm loss. Chúng nằm
# chung thư mục run và bị bỏ qua vì đã tồn tại — không sao chép file giữa các run.
#
# Chi phí khối 2 vì thế chỉ là Phase 1 mới + 3 nhánh × 2 optimizer × 5 fold, chứ
# không phải gấp đôi khối 1.
#
# CẢNH BÁO: đừng sửa file này (hay run/matrix.sh) trong lúc nó đang chạy. Bash đọc
# script theo offset byte; ghi đè một file đang chạy làm tiến trình nhảy vào giữa
# câu lệnh. Muốn đổi thì dừng, sửa, chạy lại — matrix.sh idempotent nên chỉ mất
# đúng job đang dở.
#
# Cách gọi:
#   BACKBONES="..." RUN_NAME=m1 bash run/plan_two_lambda.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_NAME="${RUN_NAME:-m1}"
SEED="${SEED:-42}"
FOLDS="${FOLDS:-1 2 3 4 5}"
BACKBONES="${BACKBONES:?can BACKBONES}"
export PYTHON="${PYTHON:-/venv/main/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

common() {
  env RUN_NAME="$RUN_NAME" SEED="$SEED" FOLDS="$FOLDS" BACKBONES="$BACKBONES" \
      OPTIMIZERS="recadam adamw" CWE_VOCAB=fixed4 \
      DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
      PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
      PYTHON="$PYTHON" HF_HOME="$HF_HOME" "$@"
}

echo "########## KHOI 1 — lambda 0.2, du 4 nhanh ##########"
common MODES="none cwe latent_bottleneck latent_proto" LAMBDA_CWE=0.2 ARM_TAG="" \
  bash run/matrix.sh

echo ""
echo "########## KHOI 2 — lambda 0.05, ba nhanh co head phu ##########"
echo "  baseline va none dung lai cua khoi 1 (doc lap voi lambda)"
common MODES="cwe latent_bottleneck latent_proto" LAMBDA_CWE=0.05 ARM_TAG="_l05" \
  bash run/matrix.sh

echo ""
echo "########## CA HAI KHOI XONG ##########"
$PYTHON src/report_fold.py --prefix "${RUN_NAME}_" --seed "$SEED" || true
