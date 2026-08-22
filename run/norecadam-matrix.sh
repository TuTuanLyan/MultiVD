#!/usr/bin/env bash
# Y HỆT ma trận họ backbone, chỉ BỎ RecAdam ở Phase 2.
#
# Câu hỏi: RecAdam đóng góp bao nhiêu vào phương pháp? Cả dự án tới giờ luôn dùng
# nó ở Phase 2, nên chưa bao giờ tách được "pretrain đa nhiệm có tác dụng" khỏi
# "RecAdam có tác dụng".
#
# §40.3 đã tính tay và cho một dự đoán cụ thể: hệ số kéo mỗi bước là
# lr × pretrain_cof = 2e-5 × 5000 = 0.1, nên sau MỘT epoch chỉ còn 9.1% khoảng
# cách tới điểm neo và sau ba epoch còn 1.2%, trên tổng 30 epoch. Tức RecAdam ở
# pipeline này hoạt động như một lịch warmup chứ không phải bộ chính quy hoá.
#
# NẾU tính toán đó đúng thì bỏ RecAdam phải làm thay đổi RẤT ÍT. Đây là phép thử
# trực tiếp của một dự đoán đã ghi trước, không phải thử cho biết.
#
# Chỉ đổi ĐÚNG MỘT biến: --phase2_optimizer adamw. Phase 1 giữ nguyên hoàn toàn,
# nên checkpoint nguồn dùng lại được từ ma trận cũ và không tốn GPU chạy lại.
#
# Baseline cũng dùng lại: nó không đọc dữ liệu source VÀ không dùng RecAdam
# (train_baseline.py không có Phase 2), nên nó y hệt ở cả hai ma trận.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"; export PYTHON

MAY="${MAY:?can MAY=A hoac B}"
SEED="${SEED:-42}"; export SEED
LAM="${LAM:-0.2}"

case "$MAY" in
  A) BACKBONES="codet5=Salesforce/codet5-base:mean codebert=microsoft/codebert-base:cls" ;;
  B) BACKBONES="t5p=Salesforce/codet5p-220m:mean unixcoder=microsoft/unixcoder-base:cls" ;;
  *) echo "MAY phai la A hoac B"; exit 1 ;;
esac

RUN_NAME=nora

echo "################################################################"
echo "  MA TRAN KHONG RECADAM — may $MAY, lambda $LAM, seed $SEED"
echo "  Phase 2 dung AdamW thuong, KHONG co luc keo ve diem neo"
echo "################################################################"

# Sao chép baseline và checkpoint nguồn từ ma trận cũ: cả hai đều không phụ thuộc
# vào lựa chọn optimizer của Phase 2, nên chạy lại chỉ tốn GPU mà ra đúng số cũ.
for BB in $BACKBONES; do
  L="${BB%%=*}"
  mkdir -p "results/${RUN_NAME}_${L}/baseline/seed_$SEED"
  cp results/fam1_${L}/baseline/seed_$SEED/fold*.json \
     "results/${RUN_NAME}_${L}/baseline/seed_$SEED/" 2>/dev/null \
     && echo "  sao chep baseline cua $L"
  for MODE in none cwe latent_bottleneck latent_proto; do
    SRC="model/fam1_${L}/transfer_${MODE}/seed_$SEED/source/best.pt"
    DST="model/${RUN_NAME}_${L}/transfer_${MODE}/seed_$SEED/source"
    if [[ -f "$SRC" ]]; then
      mkdir -p "$DST" && cp "$SRC" "$DST/best.pt" && echo "  dung lai Phase 1 cua $L/$MODE"
    else
      echo "  !! thieu $SRC — $L/$MODE se bi bo qua"
    fi
  done
done

RUN_NAME="$RUN_NAME" BACKBONES="$BACKBONES" \
  MODES="none cwe latent_bottleneck latent_proto" \
  LAMBDA_CWE="$LAM" CWE_VOCAB=fixed4 \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
  PHASE2_EXTRA="--phase2_optimizer adamw" \
  GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

touch "/workspace/NORA_${MAY}_DONE"
