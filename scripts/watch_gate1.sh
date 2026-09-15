#!/usr/bin/env bash
# Theo doi gate1 tren may vast: driver con song khong, bao nhieu o, GPU the nao.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/ntat/.local/bin:$PATH"
source scripts/endpoints.sh
LABEL="${1:-ntat}"; R=/workspace/MultiVD
read -r H P <<< "$(vast_endpoint "$LABEL" 2>/dev/null)"
[[ -n "${H:-}" && "${P:-None}" != "None" ]] || { echo "may '$LABEL' khong giai duoc dia chi"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $P"
$SSH root@"$H" "cd $R 2>/dev/null || exit 9
  echo \"o: \$(find results/gate1_* -name 'fold*.json' 2>/dev/null | wc -l)\"
  # CON SONG = lock van bi giu. Khong dem tien trinh bang ps|grep (tu khop chinh minh).
  if flock -n /tmp/mvd_gate1.lock -c true 2>/dev/null; then echo 'driver: DA DUNG'; else echo 'driver: DANG CHAY'; fi
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  df -h /workspace | tail -1
  echo '--- 6 dong cuoi ---'; tail -6 log/gate1.log 2>/dev/null"
