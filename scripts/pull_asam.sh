#!/usr/bin/env bash
# Keo results tu 2 vast ve cay RIENG cho tung may.
# Vi sao rieng: Delta ghep cap phai trong cung may (CLAUDE.md muc 4). Gop chung
# mot cay thi baseline cua may nay se ghep voi nhanh cua may kia.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
for NAME in ntat ntat2; do
  LDIR="results_asam1_$NAME"
  mkdir -p "$LDIR"
  if ! read -r HOST PORT <<< "$(vast_endpoint "$NAME")" || [[ -z "${HOST:-}" ]]; then
    echo "$NAME: khong giai duoc dia chi ($(vast_state "$NAME"))"; continue
  fi
  SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -p $PORT"
  B=$(find "$LDIR" -name 'fold*.json' 2>/dev/null | wc -l)
  rsync -az -e "$SSH" "root@$HOST:/workspace/MultiVD/results/" "$LDIR/" 2>/dev/null
  rsync -az -e "$SSH" "root@$HOST:/workspace/MultiVD/log/" "log_run_$NAME/" 2>/dev/null || true
  A=$(find "$LDIR" -name 'fold*.json' 2>/dev/null | wc -l)
  echo "$NAME: $A o (+$((A-B)))"
done
