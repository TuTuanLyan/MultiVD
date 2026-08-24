#!/usr/bin/env bash
# Sàng lọc một seed, đủ 5 fold: head latent trên backbone T5.
#
# Khoảng trống thật trong bằng chứng, không phải chạy lại: **mọi** lần chạy trên
# họ T5 từ trước tới nay đều chỉ dùng nhánh `cwe`. Nhánh `latent_bottleneck` —
# thứ gỡ ràng buộc taxonomy và là phần tổng quát hóa của phương pháp — chưa từng
# được chạy trên backbone nào khác CodeBERT.
#
# Điều này đáng kiểm vì hai nhánh **không** hành xử giống nhau: trên CodeBERT,
# `cwe` cho Δ F1 lớn hơn (+0.0410 so với +0.0264) còn `latent_bottleneck` cho p
# tốt hơn trên ROC-AUC. Không có gì bảo đảm thứ tự đó giữ nguyên khi đổi backbone,
# và câu "phương pháp không phụ thuộc pretrained" hiện chỉ được kiểm cho `cwe`.
#
# Theo quy trình: **một seed (42), chạy đủ 5 fold để thấy xu hướng.** Chỉ khi
# seed này cho tín hiệu thì mới mở rộng sang seed khác để loại trừ may rủi.
# Chạy nhiều seed ngay từ đầu là lãng phí khi chưa biết có gì để xác nhận.
#
# `none` được giữ lại vì nó là ablation duy nhất tách được "head phụ có tác dụng"
# khỏi "pretrain rồi RecAdam có tác dụng" — thiếu nó thì Δ dương không diễn giải được.
#
# Baseline huấn luyện lại tại seed 42 trên chính máy này, theo đúng kỷ luật
# fold-major: không tái dùng baseline từ seed khác hay máy khác.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=t5cls_s42
export SEED=42
export MODEL_NAME=Salesforce/codet5p-220m
export POOLING=cls
export PHASE1_DATA_PATH=data/train_ccpp_js.jsonl

FOLDS="1 2 3" MODES="cwe latent_bottleneck none" bash run/fold-major.sh
echo "=== CodeT5+ cls, seed 42, 3 fold ==="
python src/report_matrix.py
touch /workspace/T5S42_3FOLD

FOLDS="4 5" MODES="cwe latent_bottleneck none" bash run/fold-major.sh
echo "=== CodeT5+ cls, seed 42, du 5 fold ==="
python src/report_matrix.py
echo
echo "=== doi chieu: CodeBERT cls (gop 3 seed) va CodeT5+ cls seed 36 ==="
RUN_NAME=twin_ccppjs python src/paired_stats.py --results_root results/twin_ccppjs --seed 36 12 7 2>/dev/null || true
RUN_NAME=t5p_twin_cls SEED=36 python src/report_matrix.py
touch /workspace/T5S42_DONE
