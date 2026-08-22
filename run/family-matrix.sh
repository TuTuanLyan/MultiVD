#!/usr/bin/env bash
# Ma trận bằng chứng: phương pháp có phụ thuộc HỌ backbone không?
#
# Câu hỏi không phải "ô này bao nhiêu" mà "quy luật RoBERTa-hợp / T5-không có giữ
# qua nhiều điều kiện không". Đó là một phép thử DẤU trên nhiều ô, và phép đó đọc
# được ở một seed kể cả khi từng ô riêng lẻ nằm trong nhiễu — điều quan trọng, vì
# sd giữa seed của họ T5 (0.0131–0.0206) LỚN HƠN hiệu ứng cần đo (0.010–0.027).
#
# Cố định mọi thứ trừ backbone, nhánh, và λ:
#   source  train_ccpp_js.jsonl  (1284 dòng, đúng 4 CWE — bắt buộc cho head 4-explicit;
#                                 hai source PrimeVul chỉ có 6% và 2% dòng thuộc 4 lớp đó)
#   target  Python, bộ fold gốc sven_python_folds_norm, 5 fold
#
# LƯỢT=1  λ=0.2   nhánh: none cwe latent_bottleneck latent_proto
# LƯỢT=2  λ=0.05  nhánh: cwe latent_bottleneck latent_proto
#                 -> `none` KHÔNG chạy lại: src/train.py:66 cho thấy khi aux_mode=none
#                    thì aux_loss là None nên λ không xuất hiện trong hàm loss. Một lần
#                    chạy dùng chung cả hai lượt. Baseline cũng vậy, nó không đọc source.
#
# Lượt 1 và 2 dùng CÙNG seed 42. Đổi seed giữa hai lượt sẽ làm λ và seed lẫn vào
# nhau, và khi đó thấy chênh lệch cũng không biết do cái nào.
#
# Đợt này CHỈ chạy seed 42. Đa seed là bước sau, và chỉ chạy khi đợt này cho thấy
# quy luật ổn định — đúng quy trình đã dùng ở đợt trước, nơi CodeT5-base λ=0.05
# dương ở seed 42 rồi âm 0/5 fold ở seed 7 (§42).
#
# MAY=A  CodeT5-base(mean) + CodeBERT(cls)
# MAY=B  CodeT5+(mean)     + UniXcoder(cls)
# Mỗi máy một RoBERTa và một T5, nên so sánh liên-họ nằm trong cùng phần cứng —
# đúng cái so sánh quyết định. Chênh lệch phần cứng đo được là 0.028 Macro-F1,
# lớn hơn hiệu ứng, nhưng Δ tính trong nội bộ máy nên nó phần lớn tự triệt tiêu.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"; export PYTHON

MAY="${MAY:?can MAY=A hoac MAY=B}"
LUOT="${LUOT:-1}"
SEED="${SEED:-42}"; export SEED

case "$MAY" in
  A) BACKBONES="codet5=Salesforce/codet5-base:mean codebert=microsoft/codebert-base:cls" ;;
  B) BACKBONES="t5p=Salesforce/codet5p-220m:mean unixcoder=microsoft/unixcoder-base:cls" ;;
  *) echo "MAY phai la A hoac B"; exit 1 ;;
esac

case "$LUOT" in
  1) LAM=0.2  ; MODES="none cwe latent_bottleneck latent_proto" ;;
  2) LAM=0.05 ; MODES="cwe latent_bottleneck latent_proto" ;;
  *) echo "LUOT phai la 1 hoac 2"; exit 1 ;;
esac

RUN_NAME="fam${LUOT}"

echo "################################################################"
echo "  MA TRAN HO BACKBONE — may $MAY, luot $LUOT, lambda $LAM, seed $SEED"
echo "  backbone : $BACKBONES"
echo "  nhanh    : $MODES"
echo "  source   : data/train_ccpp_js.jsonl (4 CWE)"
echo "################################################################"

# Lượt 2 dùng lại baseline và none của lượt 1: cùng máy, cùng seed, cùng fold,
# cùng backbone. Huấn luyện lại chỉ tốn GPU mà ra đúng con số cũ.
if [[ "$LUOT" == "2" ]]; then
  for BB in $BACKBONES; do
    L="${BB%%=*}"
    for ARM in baseline transfer_none; do
      SRC="results/fam1_${L}/${ARM}/seed_$SEED"
      DST="results/fam2_${L}/${ARM}/seed_$SEED"
      if [[ -d "$SRC" ]]; then
        mkdir -p "$DST" && cp "$SRC"/fold*.json "$DST/" 2>/dev/null \
          && echo "  sao chep $ARM cua $L tu luot 1"
      else
        echo "  !! chua co $SRC — luot 1 cua $L chua chay xong"
      fi
    done
  done
fi

RUN_NAME="$RUN_NAME" BACKBONES="$BACKBONES" MODES="$MODES" \
  LAMBDA_CWE="$LAM" CWE_VOCAB=fixed4 \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
  GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

echo ""
echo "################ CHECK CHAT LUONG PHASE 1 ################"
$PYTHON src/check_phase1.py --run_prefix "$RUN_NAME" --seed "$SEED" || true
touch "/workspace/FAM${LUOT}_${MAY}_DONE"
