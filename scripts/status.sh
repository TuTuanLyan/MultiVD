#!/usr/bin/env bash
# Trạng thái ba máy đang chạy ma trận, trong một lần gọi.
#
# Mỗi backbone nằm TRỌN trên một máy — cả 5 fold, cả hai λ, cả hai optimizer và
# baseline của nó. Nhờ vậy mọi Δ được tính trong cùng phần cứng, và chênh lệch
# giữa máy (đo được 0.028 Macro-F1, lớn hơn hiệu ứng) không lọt vào phép so nào.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes"

# nhãn|host|port|thư mục|file log|backbone
MACHINES=(
  "ntat2|1.54.247.106|30634|/workspace/MultiVD|/workspace/overnight_m1.log|t5 t5p t5pe"
  "dung |115.73.216.179|53259|/workspace/ntat_MultiVD|/workspace/ntat_MultiVD/overnight_r1.log|codebert unixcoder"
)
echo "=============== TRANG THAI $(date -u '+%F %T UTC') ==============="

# Local KHONG nam trong ma tran. GPU local dung chung, va cuongtm/tranmanhcuong
# la chu may nen luon duoc nhuong; mot job cua ta da OOM ngay khi ho quay lai.
# Chi in trang thai de biet khi nao GPU trong tro lai.
printf "\n--- local (RTX A4000) · KHONG chay ma tran ---\n"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while read -r pid mem; do
  pid=${pid%,}
  printf "  GPU dang dung boi pid=%s user=%s (%s)\n" "$pid" "$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" "$mem"
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU: /'

for M in "${MACHINES[@]}"; do
  IFS='|' read -r NAME HOST PORT DIR LOG BB <<< "$M"
  printf "\n--- %s · backbone: %s ---\n" "$(echo "$NAME" | tr -d ' ')" "$BB"
  # shellcheck disable=SC2059
  OUT=$($SSH -p "$PORT" "root@$HOST" "
    pgrep -f plan_two_lambda.sh >/dev/null && echo '  dang chay' || echo '  KHONG CO TIEN TRINH'
    echo \"  phase1=\$(ls $DIR/model/*/phase1/*/seed_*/best.pt 2>/dev/null | wc -l) ket_qua=\$(find $DIR/results -name 'fold*.json' 2>/dev/null | wc -l)\"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU: /'
    grep '^=== ' $LOG 2>/dev/null | tail -1 | sed 's/^/  /'
    # Chi dem tu lan khoi dong hang doi GAN NHAT. Tong tich luy trong file log
    # gom ca cac dot da bi bo di, va doc nham no thanh 'dang hong' la sai.
    awk '/HANG DOI QUA DEM/{n=0} /THAT BAI/{n++} END{print \"  job hong tu lan khoi dong gan nhat: \" n+0}' $LOG 2>/dev/null
  " 2>/dev/null | grep -v "AI agents:\|Welcome to vast\|Have fun")
  echo "${OUT:-  khong ket noi duoc}"
done

cat <<'TONG'

Tong ket qua khi xong: 5 backbone x 5 fold x 9 (lambda 0.2) + 5 x 5 x 6 (lambda 0.05) = 375
  ntat2: t5 t5p t5pe = 225   |   dung: codebert unixcoder = 150
TONG
