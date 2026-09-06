#!/usr/bin/env bash
# Cho 4 checkpoint t5p da co len den ntat roi phong bu Phase 1 cho 6 o con thieu.
# Phong som hon se lam matrix.sh huan luyen LAI ca 4 cai da co.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
read -r H P <<< "$(vast_endpoint ntat)"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
for ((i=0;i<120;i++)); do
  N=$($SSH "root@$H" 'find /workspace/MultiVD/model/s42/phase1 -name best.pt -path "*t5p*" 2>/dev/null | wc -l' 2>/dev/null | tail -1)
  [[ "${N:-0}" -ge 4 ]] && break
  sleep 20
done
echo "t5p checkpoint tren ntat: ${N:-0}/4"
[[ "${N:-0}" -ge 4 ]] || { echo "chua du — van phong, nhung se huan luyen lai vai o da co"; }
rsync -az -e "$SSH" run/ "root@$H:/workspace/MultiVD/run/" 2>/dev/null
rsync -az -e "$SSH" scripts/ "root@$H:/workspace/MultiVD/scripts/" 2>/dev/null
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
  "cd /workspace/MultiVD && SAM_RHO=0 PYTHON=python3 setsid bash run/phase1_fill.sh 't5p=Salesforce/codet5p-220m-bimodal:mean' </dev/null >/dev/null 2>&1"
sleep 8
echo "driver bu Phase 1: $($SSH "root@$H" 'ps -eo pid,args --no-headers | awk "\$3 ~ /phase1_fill/" | wc -l' 2>/dev/null | tail -1)"
