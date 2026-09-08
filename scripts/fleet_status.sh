#!/usr/bin/env bash
# fleet_status.sh — mot man hinh trang thai cho ca ba may. Doc nhanh, khong sua gi.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
R=/workspace/MultiVD
echo "===== $(date -u '+%F %T UTC') / $(date '+%H:%M gio VN') ====="
for L in ntat ntat2; do
  VAST_CACHE_TTL=1 read -r H P <<< "$(vast_endpoint "$L" 2>/dev/null)" || true
  if [[ -z "${H:-}" || "${P:-None}" == "None" ]]; then echo "$L: KHONG giai duoc dia chi"; continue; fi
  out=$(timeout 60 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p "$P" root@"$H" \
    "cd $R 2>/dev/null || exit 1
     echo \"  job=\$(ps -eo args --no-headers|grep -c '[s]rc/train_[a-z]*\.py') worklist=\$(ps -eo args --no-headers|grep -c '[v]ast_worklist.sh') vram=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB\"
     echo \"  muc xong: \$(grep -c . log/worklist.done 2>/dev/null)/\$(grep -vc '^\s*#\|^\s*\$' log/worklist.txt 2>/dev/null)\"
     echo \"  o: int1=\$(ls results/int1_t5p/*/seed_*/fold*.json 2>/dev/null|wc -l) asam1=\$(ls results/asam1_t5p/*/seed_*/fold*.json 2>/dev/null|wc -l)\"
     echo \"  dang chay: \$(grep 'CHAY:' log/worklist.log 2>/dev/null | tail -1 | cut -c1-84)\"
     echo \"  con lai:\"; grep -v '^\s*#\|^\s*\$' log/worklist.txt 2>/dev/null | while read -r l; do
        grep -Fqx \"\$l\" log/worklist.done 2>/dev/null || echo \"    - \$l\"; done" 2>/dev/null) || true
  echo "$L  ($H:$P)"; [[ -n "$out" ]] && echo "$out" || echo "  KHONG SSH DUOC"
done
echo "161 (local)"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1)
echo "  job=$(ps -eo args --no-headers|grep -c '[s]rc/train_[a-z]*\.py') ft2=$(ps -eo args --no-headers|grep -c '[r]un/ft2.sh') vram=${used}MiB"
tail -1 log/ft2.log 2>/dev/null | sed 's/^/  ft2: /'
echo "158"
ssh -o BatchMode=yes -o ConnectTimeout=8 tranmanhcuong@112.137.129.158 \
  'echo "  job=$(ps -eo args --no-headers|grep -c "[s]rc/train_[a-z]*\.py") vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB"' 2>/dev/null || echo "  khong ssh duoc"
