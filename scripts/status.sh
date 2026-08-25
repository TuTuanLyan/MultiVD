#!/usr/bin/env bash
# Trạng thái ba máy đang chạy ma trận, trong một lần gọi.
#
# Mỗi backbone nằm TRỌN trên một máy — cả 5 fold, cả hai λ, cả hai optimizer và
# baseline của nó. Nhờ vậy mọi Δ được tính trong cùng phần cứng, và chênh lệch
# giữa máy (đo được 0.028 Macro-F1, lớn hơn hiệu ứng) không lọt vào phép so nào.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSHOPT="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"

# Địa chỉ giải theo NHÃN tại thời điểm gọi — IP công khai của instance vast đổi
# khi nó được dời máy chủ, và một script hardcode IP sẽ timeout trong khi máy vẫn
# đang chạy bình thường.
source "$(dirname "${BASH_SOURCE[0]}")/endpoints.sh"

# nhãn|thư mục|file log|backbone
MACHINES=(
  "ntat2|/workspace/MultiVD|/workspace/overnight_m1.log|t5 t5p t5pe"
  "ntat|/workspace/MultiVD|/workspace/overnight_n1.log|codebert unixcoder"
)
echo "=============== TRANG THAI $(date -u '+%F %T UTC') ==============="

# Local CO chay, nhung la may server dung chung: dung DUNG MOT process, nice 15,
# OMP_NUM_THREADS=2, va khong bao gio co watchdog — hang doi local luon huu han
# (FOLDS=1) de no tu dung thay vi chiem GPU vo thoi han.
#
# Chu may luon duoc nhuong: mot job cua ta da OOM ngay khi ho quay lai. Nen dong
# "GPU dang dung boi" duoi day phai doc ky truoc khi phong them.
printf "\n--- local (RTX A4000) · run l1, 1 process, nice 15 ---\n"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while read -r pid mem; do
  pid=${pid%,}
  printf "  GPU dang dung boi pid=%s user=%s (%s)\n" "$pid" "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" "$mem"
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU: /'

for M in "${MACHINES[@]}"; do
  IFS='|' read -r NAME DIR LOG BB <<< "$M"
  printf "\n--- %s · backbone: %s ---\n" "$NAME" "$BB"
  if ! read -r HOST PORT <<< "$(vast_endpoint "$NAME")" || [[ -z "${HOST:-}" ]]; then
    echo "  KHONG GIAI DUOC DIA CHI — trang thai instance: $(vast_state "$NAME")"
    continue
  fi
  SSH="ssh $SSHOPT -p $PORT"
  echo "  ssh $HOST:$PORT"
  # shellcheck disable=SC2059
  OUT=$($SSH -p "$PORT" "root@$HOST" "
    ps -eo pid,args --no-headers | awk '\$2==\"bash\" && \$3 ~ /overnight/ {c++} END{print (c+0>0) ? \"  dang chay hang doi\" : \"  KHONG CO TIEN TRINH\"}'
    echo \"  phase1=\$(ls $DIR/model/*/phase1/*/seed_*/best.pt 2>/dev/null | wc -l) ket_qua=\$(find $DIR/results -name 'fold*.json' 2>/dev/null | wc -l)\"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU: /'
    grep '^=== ' $LOG 2>/dev/null | tail -1 | sed 's/^/  /'
    # Chi dem tu lan khoi dong hang doi GAN NHAT. Tong tich luy trong file log
    # gom ca cac dot da bi bo di, va doc nham no thanh 'dang hong' la sai.
    awk '/HANG DOI QUA DEM/{n=0} /THAT BAI/{n++} END{print \"  job hong tu lan khoi dong gan nhat: \" n+0}' $LOG 2>/dev/null
  " 2>/dev/null | grep -v "AI agents:\|Welcome to vast\|Have fun")
  echo "${OUT:-  khong ket noi duoc}"
done

printf "\n--- hang doi local (l1) ---\n"
if ps -eo pid,args --no-headers | awk '$2=="bash" && $3 ~ /overnight/ {f=1} END{exit !f}'; then
  echo "  dang chay"
else
  echo "  KHONG CO TIEN TRINH$([ -f /home/ntat/.multivd_state/overnight_l1/ALL_DONE ] && echo ' (da ALL_DONE)')"
fi
# Lay log MOI NHAT chu khong hardcode mot ten: hang doi local duoc phong lai
# nhieu lan trong ngay voi STATE khac nhau (overnight_l1, overnight_l1_f23, ...),
# va tro vao ten cu se hien dong cuoi cua mot hang doi DA KET THUC — o day no
# hien mot loi cu phat 5 tieng truoc trong khi hang doi that dang chay binh thuong.
LLOG=$(ls -t /home/ntat/.multivd_state/overnight_l1*.log 2>/dev/null | head -1)
[[ -n "$LLOG" ]] && { echo "  ($(basename "$LLOG"))"; tail -1 "$LLOG" | sed 's/^/  /'; }

cat <<'TONG'

Phan cong hien tai (moi backbone tron ven tren MOT may, khong phep so nao vat qua):
  ntat2  t5 t5p t5pe      07 SAM Phase 2  ->  05 SAM Phase 1
  ntat   codebert unixcoder  01 lam=0.2 -> 03 lam=0.05 -> 05 SAM Phase 1 -> 07 SAM Phase 2
  local  t5pe               01 lam=0.2 -> 05 SAM Phase 1, CHI fold 1 (pilot)
TONG
