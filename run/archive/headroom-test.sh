#!/usr/bin/env bash
# Kiểm thẳng cơ chế §34: có phải head phụ tắt tác dụng vì HẾT DƯ ĐỊA không?
#
# §34 đo được: trên target Python, CodeT5+ đã đạt 0.6586 và 0.6756 ở hai lớp CWE
# hiếm — đúng hai lớp mà toàn bộ lợi ích của phương pháp nằm ở đó — trong khi
# CodeBERT chỉ đạt 0.4876 và 0.5793. Giải thích: dư địa đã bị backbone mạnh ăn mất
# nên head phụ không còn gì để thêm.
#
# Nếu giải thích đó ĐÚNG thì nó dự đoán được: đưa CodeT5+ sang một target mà nó
# CÒN dư địa, head phụ phải có tác dụng trở lại. Target JS là phép thử sẵn có —
# baseline ở đó chỉ 0.5432 so với 0.88 trên Python, tức dư địa gấp nhiều lần.
#
#   head phụ có tác dụng trên JS  -> cơ chế dư địa đúng, và "không phụ thuộc
#                                    pretrained" là bài toán CHỌN TARGET chứ không
#                                    phải bài toán sửa phương pháp
#   vẫn bằng không trên JS        -> cơ chế dư địa SAI, và §34 phải viết lại
#
# Đây là phép thử của một GIẢI THÍCH, không phải của một ý tưởng mới. Nó có giá
# trị dù ra kết quả nào, khác với việc thử tiếp một tín hiệu phụ khác mà không
# biết vì sao tín hiệu trước hỏng.
#
# Source phải là C/C++ THUẦN. Kiểm ở §24: phần JS của train_ccpp_js trùng
# 1138/1138 hàm với target js_twin, nên dùng nó sẽ là rò rỉ toàn phần. Bộ
# ccpp_primevul_paired_common trùng 0/1138.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"
export PYTHON

SEED="${SEED:-42}"
export SEED
RUN_NAME=headroom_js \
  BACKBONES="t5p=Salesforce/codet5p-220m:cls" \
  MODES="none cwe" \
  DATA_ROOT=data/js_twin TARGET_LANG=js \
  PHASE1_DATA_PATH=data/ccpp_primevul_paired_common.jsonl \
  CWE_VOCAB=source \
  GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

echo ""
echo "=== doi chieu: cung backbone, cung nhanh, nhung target Python ==="
echo "    (Python: baseline ~0.88, it du dia -> head phu cong them ~0)"
RUN_PREFIX=edit_ref SEED="$SEED" $PYTHON src/report_gate.py 2>/dev/null | sed -n '1,9p' || true
touch /workspace/HEADROOM_DONE
