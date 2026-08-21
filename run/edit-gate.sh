#!/usr/bin/env bash
# Cổng 1 cho tín hiệu phụ thay thế: kích thước sửa đổi so với nhãn CWE.
#
# Câu hỏi: §34.3 đo được giá trị gia tăng của head phụ, Δ(cwe) − Δ(none), là
# +0.076 và +0.080 trên hai lớp hiếm của CodeBERT nhưng chỉ −0.014 và +0.012 trên
# CodeT5+. Tín hiệu CWE tắt tác dụng khi backbone đã tự biểu diễn được lớp đó.
# Tín hiệu kích thước sửa đổi có NMI 0.029 với CWE và 0.000 với nhãn nhị phân
# (§35.5) — mới thật, và không rò rỉ task chính. Nó có bù được không?
#
# Vì sao phải chạy lại nhánh CWE ở đây: `gate1` chạy trên máy đã hủy, GPU khác
# (4070 Ti SUPER so với 5060 Ti hiện tại), và chênh lệch phần cứng đo được là
# 0.028 Macro-F1 — lớn hơn hiệu ứng. Lấy số cũ so với số mới là so nhầm biến.
#
# Ba nhánh dùng CHUNG một baseline vì baseline không hề thấy dữ liệu source:
#   none  -> pretrain trần, không head phụ      (mốc dưới)
#   cwe   -> head phụ học nhãn CWE              (bản hiện tại)
#   edit  -> head phụ học nhóm kích thước sửa   (bản đề xuất)
#
# `edit` chạy được mà không sửa code model: build_edit_labels.py chỉ ghi đè trường
# `cwe_class`, nên AUX_MODE=cwe trên file đó là học tín hiệu mới. Nhờ vậy so sánh
# chỉ có đúng một biến thay đổi.
#
# Số cần nhìn KHÔNG phải Δ tuyệt đối mà là Δ(edit) − Δ(none) trên CodeT5+. Nếu nó
# cũng gần 0 như tín hiệu CWE thì hướng này bị bác ngay ở fold 1-3.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"
export PYTHON

SEED="${SEED:-42}"
BACKBONES="${BACKBONES:-codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m:cls}"
GATE_FOLDS="${GATE_FOLDS:-1 2 3}"
REST_FOLDS="${REST_FOLDS:-4 5}"

# Bước 1: baseline + none + cwe, dùng source gốc.
echo "################ BUOC 1: baseline, none, cwe (nhan CWE) ################"
RUN_NAME=edit_ref SEED="$SEED" BACKBONES="$BACKBONES" MODES="none cwe" \
  PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
  GATE_FOLDS="$GATE_FOLDS" REST_FOLDS="" STOP_AT_GATE=1 \
  bash run/gated.sh

# Bước 2: nhánh edit, dùng source đã thay nhãn. Baseline sao chép từ bước 1 —
# cùng máy, cùng seed, cùng fold, cùng backbone nên nó y hệt; huấn luyện lại chỉ
# tốn GPU mà không đổi con số nào.
echo ""
echo "################ BUOC 2: nhanh edit (kich thuoc sua doi) ################"
for BB in $BACKBONES; do
  LABEL="${BB%%=*}"
  mkdir -p "results/edit_new_${LABEL}/baseline/seed_$SEED"
  cp results/edit_ref_${LABEL}/baseline/seed_$SEED/fold*.json \
     "results/edit_new_${LABEL}/baseline/seed_$SEED/" 2>/dev/null
done
RUN_NAME=edit_new SEED="$SEED" BACKBONES="$BACKBONES" MODES="cwe" \
  PHASE1_DATA_PATH=data/train_ccpp_js_editsize.jsonl \
  GATE_FOLDS="$GATE_FOLDS" REST_FOLDS="" STOP_AT_GATE=1 \
  bash run/gated.sh

echo ""
echo "################ CONG CHAN: so sanh ba nhanh ################"
$PYTHON src/report_edit_gate.py --seed "$SEED"
touch /workspace/EDITGATE_3FOLD

if [[ "${STOP_AFTER_GATE:-0}" == "1" ]]; then
  echo "STOP_AFTER_GATE=1 — dung de nguoi quyet dinh."; exit 0
fi

echo ""
echo "################ CHAY NOT FOLD ${REST_FOLDS// /,} ################"
RUN_NAME=edit_ref SEED="$SEED" BACKBONES="$BACKBONES" MODES="none cwe" \
  PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
  GATE_FOLDS="$REST_FOLDS" REST_FOLDS="" STOP_AT_GATE=1 bash run/gated.sh
for BB in $BACKBONES; do
  LABEL="${BB%%=*}"
  cp results/edit_ref_${LABEL}/baseline/seed_$SEED/fold*.json \
     "results/edit_new_${LABEL}/baseline/seed_$SEED/" 2>/dev/null
done
RUN_NAME=edit_new SEED="$SEED" BACKBONES="$BACKBONES" MODES="cwe" \
  PHASE1_DATA_PATH=data/train_ccpp_js_editsize.jsonl \
  GATE_FOLDS="$REST_FOLDS" REST_FOLDS="" STOP_AT_GATE=1 bash run/gated.sh

echo ""
echo "################ DU 5 FOLD ################"
$PYTHON src/report_edit_gate.py --seed "$SEED"
touch /workspace/EDITGATE_DONE
