#!/usr/bin/env bash
# Trang thai run s42 tren ca 3 may (ntat=codebert, ntat2=t5p, local=unixcoder).
#
# LOC TIEN TRINH THEO CO T, KHONG DUNG pkill/pgrep -f:
#   `ps -eo pid,args` -> $1=pid $2=bash $3=<script>
#   `ps -eo args`     -> $1=bash $2=<script>      <-- lech mot cot
# Dung nham chi so cot lam lenh kill khong giet gi, va lan phong lai tao driver
# THU HAI cung tranh mot GPU -> OOM. Da xay ra that sang 27/08.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

probe() {  # $1=nhan hien thi  $2=lenh chay
  echo "───────── $1 ─────────"
  eval "$2" 2>/dev/null | grep -v "Welcome\|AI agents\|Have fun\|Warning" | sed 's/^/  /'
}

REMOTE='cd /workspace/MultiVD 2>/dev/null || exit
echo -n "driver: "; ps -eo pid,args --no-headers | awk "\$3 ~ /day42_machine|day42_fold/" | wc -l
echo -n "job dang chay: "; ps -eo pid,args --no-headers | awk "\$0 ~ /train_transfer|train_baseline/" | wc -l
echo -n "GPU: "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo -n "checkpoint Phase 1: "; ls model/s42/phase1/*/seed_42/best.pt 2>/dev/null | wc -l
echo -n "  bi tu choi: "; ls model/s42/phase1/*/seed_42/best.pt.rejected 2>/dev/null | wc -l
echo -n "ket qua fold: "; ls results/s42_*/*/seed_42/fold*.json 2>/dev/null | wc -l
echo -n "job hong: "; grep -c "THAT BAI\|KHONG DAT" log/day42_*.log 2>/dev/null | tail -1
echo "moc gan nhat:"; grep -E "^=== |^===== |XONG" log/day42_*.log 2>/dev/null | tail -3'

for L in ntat ntat2; do
  read -r H P <<< "$(vast_endpoint "$L")"
  probe "$L  ($H:$P)" "timeout 25 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $P root@$H '$REMOTE'"
done
probe "local (unixcoder)" "bash -c '
echo -n \"driver: \"; ps -u ntat -o pid,args --no-headers | awk \"\\\$3 ~ /day42_machine|day42_fold/\" | wc -l
echo -n \"job dang chay: \"; ps -u ntat -o pid,args --no-headers | awk \"\\\$0 ~ /train_transfer|train_baseline/\" | wc -l
echo -n \"GPU: \"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo -n \"checkpoint Phase 1: \"; ls model/s42/phase1/*/seed_42/best.pt 2>/dev/null | wc -l
echo -n \"ket qua fold: \"; ls results/s42_*/*/seed_42/fold*.json 2>/dev/null | wc -l
echo \"moc gan nhat:\"; grep -E \"^=== |^===== |XONG\" log/day42_*.log 2>/dev/null | tail -3'"
