#!/usr/bin/env bash
set -uo pipefail
H=tranmanhcuong@112.137.129.158; R=/data/ntat/MultiVD
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $H "cd $R 2>/dev/null || exit 9
  echo \"o: \$(find results/gate1_* -name 'fold*.json' 2>/dev/null | wc -l)\"
  if flock -n /tmp/mvd_gate1.lock -c true 2>/dev/null; then echo 'driver: DA DUNG'; else echo 'driver: DANG CHAY'; fi
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader
  echo '--- 6 dong cuoi ---'; tail -6 log/gate1.log 2>/dev/null"
