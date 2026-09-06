#!/usr/bin/env bash
# Watcher nen chay dai cho ca hai may. Ghi tien do vao log/watch_all.log moi vong,
# THOAT (de duoc goi len) khi:
#   - t5p Phase 1 du 10/10
#   - mot may mat driver
#   - local xong mot fold (moc @@@@@ XONG FOLD moi)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
OUT=log/watch_all.log; mkdir -p log
MAX="${MAX:-200}"; SEEN_FOLD=$(grep -hc "@@@@@ XONG FOLD" log/day43_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
for ((i=0;i<MAX;i++)); do
  TS=$(date -u '+%F %T'); EV=""
  read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
  if [[ -n "${H:-}" && "$P" != "None" ]]; then
    V=$(timeout 25 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$P" root@"$H" '
      cd /workspace/MultiVD 2>/dev/null || exit
      echo "$(find model/s42/phase1 -name best.pt -path "*t5p*" 2>/dev/null|wc -l)/$(ps -eo pid,args --no-headers | awk "\$3 ~ /phase1_fill|day43_machine/"|wc -l)"' 2>/dev/null | tail -1)
    T5=${V%%/*}; DR=${V##*/}
    [[ -z "$V" ]] && EV+="[ntat] mat lien lac. "
    [[ "${T5:-0}" -ge 10 ]] && EV+="[ntat] t5p Phase 1 DU 10/10. "
    [[ "${DR:-1}" == "0" && "${T5:-0}" -lt 10 ]] && EV+="[ntat] MAT DRIVER khi moi co ${T5}/10. "
  else EV+="[ntat] khong giai duoc dia chi. "; fi
  LD=$(ps -eo pid,args --no-headers | awk '$3 ~ /day43_ctl|day43_machine/' | wc -l)
  LC=$(ls results/s42_codebert/transfer_*_ctl/seed_42/fold*.json 2>/dev/null|wc -l)
  LA=$(ls results/s42_codebert/transfer_*_asam/seed_42/fold*.json 2>/dev/null|wc -l)
  NF=$(grep -hc "@@@@@ XONG FOLD" log/day43_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)
  [[ "$LD" == "0" ]] && EV+="[local] MAT DRIVER. "
  (( NF > SEEN_FOLD )) && { EV+="[local] xong fold moi. "; SEEN_FOLD=$NF; }
  echo "$TS  ntat t5p=${T5:-?}/10 dr=${DR:-?} | local dr=$LD ctl=$LC asam=$LA fold=$NF" >> "$OUT"
  [[ -n "$EV" ]] && { echo "$TS >>> $EV" | tee -a "$OUT"; exit 0; }
  sleep 180
done
echo "het $MAX vong" | tee -a "$OUT"
