#!/usr/bin/env bash
# Cho 10 checkpoint t5p ve du va KHOP TUNG BYTE roi moi xoa tren ntat.
# Ngay 27/08 toi xoa truoc khi xac minh va mat 6 checkpoint — khong lam lai.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
read -r H P <<< "$(vast_endpoint ntat)"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
for ((i=0;i<80;i++)); do
  [[ "$(find model/s42/phase1 -name best.pt -path '*t5p*' | wc -l)" -ge 10 ]] && break
  sleep 30
done
R=$($SSH "root@$H" 'cd /workspace/MultiVD/model/s42/phase1 && find . -name best.pt -path "*t5p*" -printf "%p %s\n" | LC_ALL=C sort' 2>/dev/null)
L=$(cd model/s42/phase1 && find . -name best.pt -path '*t5p*' -printf '%p %s\n' | LC_ALL=C sort)
NR=$(echo "$R"|grep -c .); MISS=$(LC_ALL=C comm -23 <(echo "$R") <(echo "$L")|grep -c .)
echo "tren ntat $NR | lech/thieu o local $MISS"
if [[ "$NR" -ge 10 && "$MISS" == "0" ]]; then
  $SSH "root@$H" 'cd /workspace/MultiVD/model/s42/phase1 && rm -rf t5p__* && df -h / | tail -1' 2>/dev/null | tail -1
  echo "-> da xoa t5p tren ntat (da xac minh khop tung byte o local)"
else
  echo "-> KHONG xoa: chua khop"
fi
