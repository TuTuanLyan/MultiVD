#!/usr/bin/env bash
# THÍ NGHIỆM RIÊNG, chạy một lần: codet5p-110m-embedding.
#
# Tách khỏi run/family-matrix.sh có chủ đích — đây là thí nghiệm dùng một lần,
# không nên làm phình cái script đang gánh ma trận chính.
#
# Vì sao checkpoint này đáng thử: encoder của nó GIỐNG HỆT encoder của
# codet5p-220m đến từng tham số — 84,954,240 tham số ngoài embedding ở cả hai,
# cùng 768 chiều, cùng 12 lớp, cùng RobertaTokenizerFast, cùng <s> ở vị trí 0.
# Khác biệt duy nhất là nó có THÊM một giai đoạn pretrain (contrastive/embedding).
#
# Nên đặt cạnh codet5p-220m thì nó tách được đúng một biến: **pretrain**. Mọi
# thí nghiệm trước đây đổi backbone là đổi luôn cả kiến trúc lẫn cỡ lẫn vocab,
# nên chưa lần nào tách được biến đó.
#
# Ràng buộc kỹ thuật: AutoModel của checkpoint này trả về vector 256 chiều đã
# chuẩn hoá, không phải chuỗi hidden states. build_backbone có một nhánh khoá
# theo tên model để lấy `.encoder` của nó; nhánh đó không đụng tới backbone nào
# khác.
#
# Hai lượt, một seed, đúng quy tắc của ma trận chính:
#   LUOT=1  λ=0.2   none cwe latent_bottleneck latent_proto
#   LUOT=2  λ=0.05  cwe latent_bottleneck latent_proto  (dùng lại baseline+none)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"; export PYTHON

LUOT="${LUOT:-1}"
SEED="${SEED:-42}"; export SEED
BB="emb=Salesforce/codet5p-110m-embedding:mean"

case "$LUOT" in
  1) LAM=0.2  ; MODES="none cwe latent_bottleneck latent_proto" ;;
  2) LAM=0.05 ; MODES="cwe latent_bottleneck latent_proto" ;;
  *) echo "LUOT phai la 1 hoac 2"; exit 1 ;;
esac

RUN_NAME="emb${LUOT}"

echo "################################################################"
echo "  codet5p-110m-embedding — luot $LUOT, lambda $LAM, seed $SEED"
echo "  doi chieu truc tiep voi fam*_t5p (codet5p-220m, cung may)"
echo "################################################################"

# `none` và baseline không phụ thuộc λ, nên lượt 2 dùng lại của lượt 1.
if [[ "$LUOT" == "2" ]]; then
  for ARM in baseline transfer_none; do
    SRC="results/emb1_emb/${ARM}/seed_$SEED"
    DST="results/emb2_emb/${ARM}/seed_$SEED"
    [[ -d "$SRC" ]] && mkdir -p "$DST" && cp "$SRC"/fold*.json "$DST/" 2>/dev/null \
      && echo "  sao chep $ARM tu luot 1"
  done
fi

RUN_NAME="$RUN_NAME" BACKBONES="$BB" MODES="$MODES" \
  LAMBDA_CWE="$LAM" CWE_VOCAB=fixed4 \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
  GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

touch "/workspace/EMB${LUOT}_DONE"
