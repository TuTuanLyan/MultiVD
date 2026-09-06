#!/usr/bin/env bash
# Bao khi KHOI ASAM du ca 3 backbone (t5p dat 100/100 sau khi gop hai may),
# hoac khi mot may mat driver.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
MAX="${MAX:-240}"
for ((i=0;i<MAX;i++)); do
  bash scripts/pull43.sh >/dev/null 2>&1
  N=$(( $(ls results/s42_t5p/transfer_*_asam/seed_42/fold*.json 2>/dev/null|wc -l) \
      + $(ls results/s42_t5p/transfer_*_ctl/seed_42/fold*.json 2>/dev/null|wc -l) ))
  LD=$(ps -eo pid,args --no-headers | awk '$3 ~ /day43_machine/' | wc -l)
  read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
  RD=0
  [[ -n "${H:-}" && "$P" != "None" ]] && RD=$(timeout 20 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$P" root@"$H" \
      'ps -eo pid,args --no-headers | awk "\$3 ~ /day43_machine|day44_machine/" | wc -l' 2>/dev/null | tail -1)
  EV=""
  (( N >= 100 )) && EV+="t5p ASAM DU 100/100. "
  (( LD == 0 && RD == 0 )) && EV+="CA HAI MAY het driver (t5p=$N/100). "
  if [[ -n "$EV" ]]; then
    echo "$(date -u '+%F %T') >>> $EV"; bash scripts/report43.sh | tail -12; exit 0
  fi
  echo "$(date -u '+%F %T')  t5p=$N/100  local_dr=$LD ntat_dr=$RD" >> log/watch_asam.log
  sleep 240
done
