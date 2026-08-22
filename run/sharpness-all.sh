#!/usr/bin/env bash
# Đo độ nhọn cực tiểu Phase 1 cho hai backbone của một máy.
#
# Đây là phép thử QUYẾT ĐỊNH có đáng chạy SAM hay không, và nó rẻ: chỉ nạp
# checkpoint rồi chạy forward/backward, không huấn luyện gì. Vài phút GPU.
#
# SAM cực tiểu giá trị lớn nhất của loss trong lân cận bán kính ρ, tức đi tìm cực
# tiểu PHẲNG. Nó chỉ có mục tiêu nếu cực tiểu hiện tại thật sự nhọn — và nhọn hơn
# ở đúng backbone đang hỏng.
#
# Hai chẩn đoán đang cạnh tranh:
#   §34  tín hiệu CWE thừa vì backbone mạnh đã biểu diễn được lớp đó
#        -> SAM KHÔNG chữa được, vì nó không đổi tín hiệu
#   §40  Phase 1 đẩy backbone mạnh ra xa (CodeT5+ gấp 1.86× CodeBERT), nghiệm nhọn
#        nên Phase 2 phá nhiều
#        -> SAM CÓ THỂ chữa
#
# Nếu backbone T5 không nhọn hơn rõ rệt thì §34 là chẩn đoán đúng và bỏ SAM luôn,
# tiết kiệm được nhiều giờ GPU vì SAM tốn 2× thời gian huấn luyện.
#
# Dùng nhánh `cwe` của mỗi backbone vì mọi backbone đều có nó và nó là cấu hình
# gốc của phương pháp.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"

MAY="${1:?can MAY=A hoac B}"
SEED="${SEED:-42}"

case "$MAY" in
  A) PAIRS="codet5:Salesforce/codet5-base:mean codebert:microsoft/codebert-base:cls" ;;
  B) PAIRS="t5p:Salesforce/codet5p-220m:mean unixcoder:microsoft/unixcoder-base:cls" ;;
  *) echo "MAY phai la A hoac B"; exit 1 ;;
esac

# Chờ GPU thực sự trống trước khi nạp model thứ hai. Bản đầu chạy thẳng và OOM:
# job huấn luyện chiếm 12.5 GB trên card 15.5 GB, không còn chỗ. Job đo chết còn
# job huấn luyện sống, nhưng đó là may chứ không phải thiết kế.
echo "cho GPU trong truoc khi do..."
for _ in $(seq 1 120); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${USED:-99999}" -lt 2000 ] && break
  sleep 15
done
echo "GPU dang dung ${USED} MiB — bat dau"

OUT="/workspace/sharpness_${MAY}.txt"
: > "$OUT"
for P in $PAIRS; do
  LABEL="${P%%:*}"; REST="${P#*:}"; MODEL="${REST%:*}"; POOL="${REST##*:}"
  CKPT="model/fam1_${LABEL}/transfer_cwe/seed_${SEED}/source/best.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "bo qua $LABEL: chua co $CKPT" | tee -a "$OUT"
    continue
  fi
  echo "=== $(date -u '+%F %T UTC') | do do nhon: $LABEL ($MODEL, $POOL) ===" | tee -a "$OUT"
  $PYTHON src/measure_sharpness.py \
    --checkpoint "$CKPT" --model_name "$MODEL" --pooling "$POOL" \
    --aux_mode cwe --seed "$SEED" --batch_size 16 \
    --rho_mode relative --rhos 0.005 0.01 0.02 0.05 --n_random 3 \
    2>&1 | tee -a "$OUT"
done

echo "" | tee -a "$OUT"
echo "So hai backbone o CUNG rho. Delta loss lon hon = cuc tieu NHON hon." | tee -a "$OUT"
touch "/workspace/SHARPNESS_${MAY}_DONE"
