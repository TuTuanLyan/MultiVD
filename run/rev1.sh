#!/usr/bin/env bash
# REV1 — ĐẢO CHIỀU chuyển giao: nguồn = PYTHON, đích = JAVASCRIPT. (bậc 1, 1 fold, seed 42)
#
# Người dùng 11/09: "thử chạy thêm cho tôi 1 cái đảo nguồn và đích ... dùng py transfer sang js
# only, dùng 1 fold là được. chia lại js có train val test còn py thì gộp thêm để chia train val
# thôi. Dùng adamw trước và thêm 1 cái có asam và recadam theo setting backbone tốt nhất."
#
# Vì sao đáng chạy: mọi số của dự án tới giờ đều đo MỘT chiều (C/C++ + JS -> Python). Nếu hiệu
# ứng là tính chất của PHƯƠNG PHÁP thì đảo chiều vẫn phải thấy; nếu nó là tính chất của riêng
# cặp (nguồn này, đích này) thì đảo chiều sẽ tắt. Chưa phép đo nào của dự án phân biệt được hai
# khả năng đó.
#
# NGUỒN Pha 1: data/phase1_python.jsonl — 760 dòng SVEN gộp lại, Pha 1 tự chia train/val bằng
#   `split_source_records` nên chỉ cần một file phẳng (đúng yêu cầu "py thì gộp thêm để chia
#   train val thôi").
# ĐÍCH Pha 2: data/js_4cwe_folds/fold1 — 812 dòng JS của `phase1_4cwe`, chia 60/20/20 = 486/162/164,
#   PHÂN TẦNG theo (nhãn, CWE), chia THEO DÒNG đúng quy ước của bộ `norm` bên Python để biến duy
#   nhất đổi là CHIỀU. Rò rỉ cặp đo được 227/406 (56%), cùng dạng với `norm` (~40%).
#
# BẤT ĐỐI XỨNG PHẢI NÊU KHI ĐỌC: đích Python bị CWE-089 áp đảo (54% hàng), còn đích JS bị
# CWE-079 áp đảo (82% hàng test, 134/164). CWE-022 và 089 ở JS chỉ có 8 hàng test mỗi lớp —
# per-CWE bên JS gần như không đọc được, chỉ số tổng mới có nghĩa.
#
# Hai nhánh, đúng yêu cầu:
#   plain  AdamW trần, không SAM         <- cấu hình chốt hiện tại
#   A      RecAdam + ASAM ở rho TỐT NHẤT CỦA CHÍNH BACKBONE (codebert 0.1, t5p 2.0)
# Đối chứng `baseline` (không Pha 1) do matrix.sh tự chạy cùng fold.
#
#   BB=codebert=microsoft/codebert-base:cls        bash run/rev1.sh   # 161
#   BB=t5p=Salesforce/codet5p-220m-bimodal:mean    bash run/rev1.sh   # 158
#
# Đọc: python3 tools/bridge_report.py --b plain --a r0p1,r2p0 results/rev1_<bb>
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
: "${BB:?can BB=<label>=<hf-name>:<pooling>}"
RUN="${RUN:-rev1}"; SEED="${SEED:-42}"; FOLDS="${FOLDS:-1}"
SRC_DATA="${SRC_DATA:-data/phase1_python.jsonl}"
TGT_ROOT="${TGT_ROOT:-data/js_4cwe_folds}"
STORE="${STORE:-model/rev1/phase1}"
L="${BB%%=*}"
case "$L" in
  codebert|unixcoder) NEED_VRAM="${NEED_VRAM:-8500}"; RHO="${RHO:-0.1}";  RTAG="${RTAG:-r0p1}" ;;
  *)                  NEED_VRAM="${NEED_VRAM:-13000}"; RHO="${RHO:-2.0}"; RTAG="${RTAG:-r2p0}" ;;
esac
# "tag|optimizer|co Pha 2" — doi chung AdamW dung DAU, dung yeu cau "dung adamw truoc".
CONFIGS="${CONFIGS:-\
plain|adamw|--sam_rho 0
$RTAG|recadam|--sam_rho $RHO --sam_variant asam}"

LOCK="${REV_LOCK:-/tmp/mvd_rev1.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO rev1 dang chay tren may nay — dung"; exit 3; }

wait_vram(){ local w=0 t u a
  while true; do
    t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    u=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    a=$(( ${t:-0} - ${u:-0} )); (( a >= NEED_VRAM )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(date -u '+%F %T') | cho VRAM ${a}MiB"
  done; }

for f in "$SRC_DATA" "$TGT_ROOT/fold1/train.jsonl" "$TGT_ROOT/fold1/val.jsonl" "$TGT_ROOT/fold1/test.jsonl"; do
  [[ -f "$f" ]] || { echo "!! THIEU du lieu: $f"; exit 4; }
done
NC=$(printf '%s\n' "$CONFIGS" | grep -c '|')
echo "########## REV1 bat dau $(date -u '+%F %T') | $(hostname) | $L ##########"
echo "  DAO CHIEU: nguon PYTHON ($(wc -l < "$SRC_DATA") dong) -> dich JAVASCRIPT"
echo "  dich: $TGT_ROOT/fold1 = $(wc -l < "$TGT_ROOT/fold1/train.jsonl")/$(wc -l < "$TGT_ROOT/fold1/val.jsonl")/$(wc -l < "$TGT_ROOT/fold1/test.jsonl") (train/val/test)"
echo "  fold $FOLDS | seed $SEED | rho cua $L = $RHO | $NC nhanh + baseline"
printf '%s\n' "$CONFIGS" | sed 's/^/  nhanh: /'

while IFS='|' read -r TAG OPT EXTRA; do
  [[ -z "$TAG" ]] && continue
  echo "===== $(date -u '+%F %T') | $L | $TAG ($OPT) | $EXTRA ====="
  wait_vram
  # PHASE1_TAG GIONG NHAU giua hai nhanh => Pha 1 huan luyen DUNG MOT LAN roi dung lai.
  # Tach PHASE1_TAG khoi ARM_TAG la bat buoc (CLAUDE.md muc 5), neu khong nhanh thu hai se
  # tu huan luyen mot Pha 1 khac va phep so doi hai bien.
  RUN_NAME="$RUN" SEED="$SEED" FOLDS="$FOLDS" \
  BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS="$OPT" \
  PHASE1_DATA_PATH="$SRC_DATA" CWE_VOCAB=fixed4 \
  ARM_TAG="_py_l0p05_${TAG}" PHASE1_TAG="_py_l0p05" PHASE1_STORE="$STORE" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="$EXTRA" \
  DATA_ROOT="$TGT_ROOT" TARGET_LANG=js \
  bash run/matrix.sh 9>&-
done <<< "$CONFIGS"

echo "########## REV1 xong $(date -u '+%F %T') | $(hostname) | $L | $(find results/${RUN}_${L} -name 'fold*.json' 2>/dev/null | wc -l) o ##########"
