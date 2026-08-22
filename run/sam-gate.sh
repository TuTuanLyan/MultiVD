#!/usr/bin/env bash
# SAM ở Phase 2, đặt cạnh đúng cấu hình đã chạy không-SAM để so trực tiếp.
#
# Phép đo độ nhọn (§45) đã bác MỘT cơ chế cụ thể: CodeT5+ là backbone PHẲNG NHẤT
# chứ không nhọn nhất, nên "Phase 1 nhọn khiến Phase 2 phá nhiều" không đứng.
#
# Nhưng nó KHÔNG chứng minh SAM vô dụng, và tôi đã kết luận quá mạnh khi nói bỏ.
# SAM cải thiện tổng quát hoá nói chung; bài báo báo cáo lợi ích trên chính các
# tác vụ FINE-TUNING, và Phase 2 ở đây học trên 456 dòng với hơn 100M tham số —
# đúng vùng quá khớp mà SAM sinh ra để chữa. Đó là một lý do độc lập với §40.
#
# So sánh sạch: dùng lại Phase 1 và baseline của ma trận fam1, chỉ đổi ĐÚNG MỘT
# biến là --sam_rho. Nhánh không-SAM tương ứng đã có sẵn trong fam1_*.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"; export PYTHON

MAY="${MAY:?can MAY=A hoac B}"
SEED="${SEED:-42}"; export SEED
RHO="${RHO:-0.05}"
MODES="${MODES:-none cwe}"

case "$MAY" in
  A) BACKBONES="codebert=microsoft/codebert-base:cls" ;;
  B) BACKBONES="t5p=Salesforce/codet5p-220m:mean" ;;
  *) echo "MAY phai la A hoac B"; exit 1 ;;
esac

RUN_NAME="sam"

echo "################################################################"
echo "  SAM Phase 2 — may $MAY, rho $RHO, seed $SEED"
echo "  doi chieu voi fam1_* (cung Phase 1, cung baseline, chi khac SAM)"
echo "  moi buoc 2 luot forward-backward -> ~2x thoi gian"
echo "################################################################"

for BB in $BACKBONES; do
  L="${BB%%=*}"
  mkdir -p "results/${RUN_NAME}_${L}/baseline/seed_$SEED"
  cp results/fam1_${L}/baseline/seed_$SEED/fold*.json \
     "results/${RUN_NAME}_${L}/baseline/seed_$SEED/" 2>/dev/null && echo "  sao chep baseline $L"
  for MODE in $MODES; do
    SRC="model/fam1_${L}/transfer_${MODE}/seed_$SEED/source/best.pt"
    DST="model/${RUN_NAME}_${L}/transfer_${MODE}/seed_$SEED/source"
    if [[ -f "$SRC" ]]; then
      mkdir -p "$DST" && cp "$SRC" "$DST/best.pt" && echo "  dung lai Phase 1 $L/$MODE"
    else
      echo "  !! thieu $SRC"
    fi
  done
done

RUN_NAME="$RUN_NAME" BACKBONES="$BACKBONES" MODES="$MODES" \
  LAMBDA_CWE=0.2 CWE_VOCAB=fixed4 \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
  PHASE2_EXTRA="--sam_rho $RHO" \
  GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

touch "/workspace/SAM_${MAY}_DONE"
