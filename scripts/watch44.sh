#!/usr/bin/env bash
# Bao khi tap dich SACH du de doc: >= 15 o cua o thang, hoac xong mot fold,
# hoac local mat driver.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX="${MAX:-200}"; TARGET="${1:-30}"
SEEN=$(grep -hc "@@@@@ XONG FOLD" log/day44_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
for ((i=0;i<MAX;i++)); do
  N=$(ls results/s42tw_*/*/seed_42/fold*.json 2>/dev/null | wc -l)
  D=$(ps -eo pid,args --no-headers | awk '$3 ~ /day44_machine/' | wc -l)
  F=$(grep -hc "@@@@@ XONG FOLD" log/day44_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
  EV=""
  (( N >= TARGET )) && EV+="da co $N o tren fold sach. "
  (( F > SEEN )) && { EV+="xong fold moi. "; SEEN=$F; }
  [[ "$D" == "0" ]] && EV+="local MAT DRIVER. "
  if [[ -n "$EV" ]]; then echo "$(date -u '+%F %T') >>> $EV"; bash scripts/report44.sh | tail -25; exit 0; fi
  sleep 180
done
