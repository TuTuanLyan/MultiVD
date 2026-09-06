#!/usr/bin/env bash
# Bao MOI KHI xong mot fold cua khoi ASAM@Phase2, kem so sanh ngay voi nhanh
# khong-ASAM tuong ung. Thoat de lop tren duoc goi len.
#   bash scripts/watch43.sh [so_moc_da_thay]
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEN="${1:-0}"; MAX="${MAX:-288}"
source scripts/endpoints.sh
remote_drivers() {
  read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
  [[ -z "${H:-}" ]] && { echo 0; return; }
  timeout 20 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$P" root@"$H" \
    'ps -eo pid,args --no-headers | awk "\$3 ~ /day43_machine|day43_ctl/" | wc -l' 2>/dev/null | tail -1 || echo 0
}
for ((i=0;i<MAX;i++)); do
  # Dem moc XONG FOLD o CA HAI may. Tren ntat phai keo log ve truoc.
  bash scripts/pull43.sh >/dev/null 2>&1
  N=$(grep -h "@@@@@ XONG FOLD" log/day43_*.log log/vast_ntat/day43_*.log 2>/dev/null | wc -l)
  REMOTE_D=$(remote_drivers); REMOTE_D=${REMOTE_D:-0}
  D=$(( $(ps -eo pid,args --no-headers | awk '$3 ~ /day43_machine|day43_ctl/' | wc -l) + REMOTE_D ))
  if (( N > SEEN )); then
    echo "=== moc moi ==="; grep -h "@@@@@ XONG FOLD" log/day43_*.log log/vast_ntat/day43_*.log 2>/dev/null | tail -$((N-SEEN))
    bash scripts/report43.sh
    exit 0
  fi
  if (( D == 0 )); then echo "=== DRIVER DA DUNG (xong hoac hong) ==="; bash scripts/report43.sh; exit 0; fi
  sleep 120
done
echo "het $MAX vong"
