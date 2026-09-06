#!/usr/bin/env bash
# Cho vast 49839302 provision xong roi bao. Chi doi, khong lam gi khac.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
ID=49839302
for i in $(seq 1 180); do   # toi da 60 phut
  R=$(vastai show instance $ID --raw 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
if isinstance(d,list): d=d[0] if d else {}
p=(d.get('ports') or {}).get('22/tcp') or [{}]
print('%s|%s|%s'%(d.get('actual_status'), d.get('public_ipaddr'), p[0].get('HostPort')))" 2>/dev/null)
  ST=${R%%|*}; PORT=${R##*|}
  echo "$(date '+%F %H:%M:%S')  $R" >> log/wait_vast47.log
  if [[ "$ST" == "running" && "$PORT" != "None" && -n "$PORT" ]]; then
    echo "$(date '+%F %H:%M:%S')  SAN SANG: $R" >> log/wait_vast47.log
    exit 0
  fi
  sleep 20
done
echo "$(date '+%F %H:%M:%S')  het gio cho" >> log/wait_vast47.log
exit 1
