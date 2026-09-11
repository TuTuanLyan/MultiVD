#!/usr/bin/env bash
# GATE3 — ĐƯỜNG QUYẾT ĐỊNH KÉP CÓ CỔNG (bậc 1: n=3 fold, seed 42). Mảnh 2.
#
#     logit = (1 − g) · W₇₆₈ · f   +   g · W₈ · P(f)
#
# `P` = `latent_proj` của Pha 1 (768→8), ĐÓNG BĂNG. Nhánh thứ hai chỉ 8·C + C tham số.
# `g` là cổng học được VÀ là một số đọc được: phần quyết định đi qua không gian 8 chiều.
#
# VÌ SAO cơ chế này chứ không phải cơ chế khác — FACTS §40/§40.2: lợi ích transfer TĂNG ĐƠN
# ĐIỆU khi tập đích co lại (ROC-AUC +0.011 → +0.194 codebert, +0.017 → +0.100 t5p, đơn điệu
# chặt ở n=5 trên CẢ HAI backbone). Mô hình hiện KHÔNG CÓ cách nào biểu đạt lựa chọn đó — nó
# chỉ có một đường quyết định. Đây là chỗ biểu đạt nó ra.
#
# DỰ ĐOÁN kiểm được VÀ CÓ THỂ SAI: **N nhỏ ⇒ g lớn**. `g` không đổi theo cỡ tập đích ⇒ cơ chế
# bị BÁC, dù điểm số có tốt lên. Vì thế khối này chạy HAI đầu mút N = 456 và N = 76, không
# phải một mức.
#
# BA nhánh, và nhánh thứ ba là ĐỐI CHỨNG BẮT BUỘC:
#   plain  không cổng                              <- cấu hình chốt hiện tại
#   gl     cổng, `latent_proj` ĐÃ HỌC             <- cơ chế
#   gr     cổng, `latent_proj` thay bằng ma trận NGẪU NHIÊN đóng băng   <- ĐỐI CHỨNG
#
# Vì sao `gr` bắt buộc: `tools/latent_probe.py` đo được trên codebert rằng ảnh 8 chiều đã học
# KHÔNG hơn một phép chiếu Gauss (+0.0083 ROC, 3/5 fold) và THUA PCA-8 (−0.0259, 0/5). Nếu
# `gl` và `gr` chạy ngang nhau thì cơ chế là "một nhánh ÍT THAM SỐ", không phải "neo vào bảng
# phân loại của nguồn". Hai phát biểu khác hẳn về độ mới, và chỉ phát biểu thứ hai cần
# `latent_proj`. Không có đối chứng này thì không phân biệt được, và bài sẽ nói quá.
#
#   BB=codebert=microsoft/codebert-base:cls        bash run/gate3.sh   # 161
#   BB=t5p=Salesforce/codet5p-220m-bimodal:mean    bash run/gate3.sh   # 158
#
# Đọc: python3 tools/gate_report.py
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
: "${BB:?can BB=<label>=<hf-name>:<pooling>}"

SIZES="${SIZES:-456 76}"        # HAI dau mut, du de doc XU HUONG cua g
SRC="${SRC:-4cwe}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
SEED="${SEED:-42}"
STORE="${STORE:-model/n48/phase1}"      # Pha 1 DA CHOT (§39). Doi bang bien neu dung ban can bang.
P1TAG="${P1TAG:-_${SRC}_l0p05}"
ALPHA="${ALPHA:-0.3}"
L="${BB%%=*}"
case "$L" in
  codebert|unixcoder) NEED_VRAM="${NEED_VRAM:-8500}" ;;
  *)                  NEED_VRAM="${NEED_VRAM:-13000}" ;;
esac

LOCK="${GATE_LOCK:-/tmp/mvd_gate3.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO gate3 dang chay tren may nay — dung"; exit 3; }

wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED_VRAM )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(date -u '+%F %T') | cho VRAM ${a}MiB"
  done; }

CKPT="$STORE/${L}__latent_bottleneck${P1TAG}/seed_${SEED}/best.pt"
NF=$(echo "$FOLD_LIST" | wc -w); NS=$(echo "$SIZES" | wc -w)
echo "########## GATE3 bat dau $(date -u '+%F %T') | $(hostname) | $L ##########"
echo "  N: $SIZES | fold $FOLD_LIST | seed $SEED | nguon $SRC | alpha $ALPHA"
echo "  ky vong $(( 3*NS*NF )) o chuyen giao + $(( NS*NF )) baseline = $(( 4*NS*NF )) o"
echo "  Pha 1: $CKPT $( [[ -f "$CKPT" ]] && echo "(co)" || echo "!! THIEU — dung" )"
[[ -f "$CKPT" ]] || exit 4

for N in $SIZES; do
  echo "===== $(date -u '+%F %T') | $L | N=$N hang train ====="
  export BASELINE_EXTRA="--max_train_samples $N"
  for FOLD in $FOLD_LIST; do
    wait_vram
    # Ba nhanh chay CANH NHAU trong cung fold (CLAUDE.md muc 1): xong mot fold la co ngay
    # mot lat cat so duoc, va Δ ghep cap theo fold sach.
    RUN="gt$N" SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 BB="$BB" \
    P1STORE="$STORE" MVD_LOCK=/tmp/mvd_gate3_opt1.lock \
    CONFIGS="plain|adamw|--sam_rho 0 --max_train_samples $N
gl|adamw|--sam_rho 0 --max_train_samples $N --phase2_gate scalar --phase2_gate_alpha $ALPHA --phase2_gate_init 0.0
gr|adamw|--sam_rho 0 --max_train_samples $N --phase2_gate scalar --phase2_gate_alpha $ALPHA --phase2_gate_init 0.0 --phase2_gate_proj random" \
    bash run/opt1.sh 9>&-
  done
  unset BASELINE_EXTRA
  T=$(ls results/gt${N}_${L}/transfer_*/seed_${SEED}/fold*.json 2>/dev/null | wc -l)
  B=$(ls results/gt${N}_${L}/baseline/seed_${SEED}/fold*.json 2>/dev/null | wc -l)
  echo "########## GATE3 $L N=$N xong $(date -u '+%F %T') | $T/$(( 3*NF )) chuyen giao + $B/$NF baseline ##########"
done
echo "########## GATE3 xong $(date -u '+%F %T') | $(hostname) | $L | $(find results/gt*_${L} -name 'fold*.json' 2>/dev/null | wc -l) o ##########"
