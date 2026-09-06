#!/usr/bin/env bash
# Ngu 30 phut roi in bao cao — dung lam nhip bao cho nguoi dung.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sleep "${SLEEP:-1800}"
source scripts/endpoints.sh
read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
if [[ -n "${H:-}" && "$P" != "None" ]]; then
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $P" \
    --include='s42_*/' --include='s42_*/**' --exclude='*' \
    "root@$H:/workspace/MultiVD/results/" results/ 2>/dev/null
  RD=$(timeout 20 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$P" root@"$H" \
    'echo "$(ps -eo pid,args --no-headers | awk "\$3 ~ /day45_machine/"|wc -l)|$(df -h / | tail -1 | awk "{print \$4}")|$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"' 2>/dev/null | tail -1)
else RD="DA HUY"; fi
LD=$(ps -eo pid,args --no-headers | awk '$3 ~ /day45_machine/'|wc -l)
echo "===== BAO CAO $(date -u '+%F %H:%M') UTC ====="
echo "  ntat : ${RD}   (driver|dia|GPU)"
echo "  local: driver=$LD  GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)"
echo "  watchdog: $(ps -eo pid,args --no-headers | awk '$3 ~ /overnight45/'|wc -l) tien trinh"
for bb in codebert unixcoder t5p; do
  printf "  %-10s " "$bb"
  for s in 4cwe com full; do
    n=$(ls results/s42_${bb}/transfer_*_${s}_l02/seed_42/fold*.json 2>/dev/null|wc -l)
    need=20; [ "$s" != "4cwe" ] && need=15
    printf "%s=%s/%s  " "$s" "$n" "$need"
  done; echo
done
echo "  TONG: $(ls results/s42_*/transfer_*_l02/seed_42/fold*.json 2>/dev/null|wc -l)/150"
tail -2 log/overnight45.log 2>/dev/null | sed 's/^/  wd: /'
