#!/usr/bin/env bash
# In MOT dong khi co su kien dang chu y, im lang khi moi thu binh thuong.
# Su kien: driver chet, GPU rong khi dang le phai ban, checkpoint bi tu choi,
#          so checkpoint Phase 1 tang, job hong tang.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
STATE="${MVD_WATCH_STATE:-/tmp/mvd_watch42.state}"

snap_remote='cd /workspace/MultiVD 2>/dev/null || { echo "0 0 0 0 NOREPO"; exit; }
d=$(ps -eo pid,args --no-headers | awk "\$3 ~ /day42_machine/" | wc -l)
g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
c=$(ls model/s42/phase1/*/seed_42/best.pt 2>/dev/null | wc -l)
r=$(ls model/s42/phase1/*/seed_42/best.pt.rejected 2>/dev/null | wc -l)
f=$(ls results/s42_*/*/seed_42/fold*.json 2>/dev/null | wc -l)
echo "$d $g $c $r $f"'

now=""
for L in ntat ntat2; do
  read -r H P <<< "$(vast_endpoint "$L")"
  v=$(timeout 25 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$P" root@"$H" "$snap_remote" 2>/dev/null | grep -E '^[0-9]' | tail -1)
  now+="$L=${v:-UNREACHABLE};"
done
d=$(ps -u ntat -o pid,args --no-headers | awk '$3 ~ /day42_machine/' | wc -l)
g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
c=$(ls model/s42/phase1/*/seed_42/best.pt 2>/dev/null | wc -l)
r=$(ls model/s42/phase1/*/seed_42/best.pt.rejected 2>/dev/null | wc -l)
f=$(ls results/s42_*/*/seed_42/fold*.json 2>/dev/null | wc -l)
now+="local=$d $g $c $r $f;"

prev=$(cat "$STATE" 2>/dev/null || echo "")
echo "$now" > "$STATE"

alert=""
for part in ${now//;/ }; do :; done
IFS=';' read -ra ARR <<< "$now"
for e in "${ARR[@]}"; do
  [[ -z "$e" ]] && continue
  lab="${e%%=*}"; rest="${e#*=}"
  if [[ "$rest" == "UNREACHABLE" || "$rest" == *NOREPO* ]]; then alert+="[$lab] KHONG LIEN LAC DUOC. "; continue; fi
  read -r dd gg cc rr ff <<< "$rest"
  (( dd != 1 ))            && alert+="[$lab] driver=$dd (phai la 1). "
  (( gg < 500 ))           && alert+="[$lab] GPU chi $gg MiB — co the da dung han. "
  (( rr > 0 ))             && alert+="[$lab] $rr checkpoint Phase 1 BI TU CHOI (SAM rho=0.05?). "
done
if [[ -n "$alert" ]]; then echo "$(date -u '+%F %T') CANH BAO: $alert"; fi
if [[ "$now" != "$prev" && -z "$alert" ]]; then echo "$(date -u '+%F %T') tien do: $now"; fi
