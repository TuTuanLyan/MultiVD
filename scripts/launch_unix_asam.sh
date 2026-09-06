#!/usr/bin/env bash
# Cho checkpoint unixcoder len du 10/10 roi phong khoi ASAM day du tren ntat.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
read -r H P <<< "$(vast_endpoint ntat)"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
for ((i=0;i<120;i++)); do
  N=$($SSH "root@$H" 'find /workspace/MultiVD/model/s42/phase1 -name best.pt 2>/dev/null | wc -l' 2>/dev/null | tail -1)
  [[ "${N:-0}" == "10" ]] && break
  sleep 30
done
[[ "${N:-0}" == "10" ]] || { echo "chi co ${N:-0}/10 checkpoint sau 60 phut — khong phong"; exit 3; }
echo "du 10/10 checkpoint, phong khoi unixcoder ASAM rho=0.1 + doi chung"
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
  "cd /workspace/MultiVD && ASAM_RHO=0.1 PYTHON=python3 setsid bash run/day43_machine.sh 'unixcoder=microsoft/unixcoder-base:cls' </dev/null >/dev/null 2>&1"
sleep 8
echo "driver tren ntat: $($SSH "root@$H" 'ps -eo pid,args --no-headers | awk "\$3 ~ /day43_machine/" | wc -l' 2>/dev/null | tail -1)"
