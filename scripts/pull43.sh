#!/usr/bin/env bash
# Keo ket qua + log cua khoi ASAM tu ntat ve local.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
[[ -z "${H:-}" || "$P" == "None" ]] && exit 0
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $P"
mkdir -p log/vast_ntat
rsync -az -e "$SSH" --include='s42_*/' --include='s42_*/**' --exclude='*' \
  "root@$H:/workspace/MultiVD/results/" results/ 2>/dev/null
rsync -az -e "$SSH" --include='day43_*.log' --include='asam_sweep_*.log' --exclude='*' \
  "root@$H:/workspace/MultiVD/log/" log/vast_ntat/ 2>/dev/null
exit 0
