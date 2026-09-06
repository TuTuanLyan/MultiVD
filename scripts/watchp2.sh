#!/usr/bin/env bash
# Watcher giai doan Phase 2: thoat khi tong so ket qua fold dat NGUONG,
# hoac khi mot may mat driver / xuat hien checkpoint bi tu choi moi.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
TARGET="${1:-20}"; MAX="${2:-144}"
OUT=log/watch42.log; mkdir -p log
declare -A BASE_REJ
for ((i=0;i<MAX;i++)); do
  TS=$(date -u '+%F %T'); TOT=0; LINE="$TS"; EVENT=""
  for L in ntat ntat2; do
    read -r H P <<< "$(vast_endpoint "$L")"
    V=$(timeout 25 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$P" root@"$H" '
      cd /workspace/MultiVD 2>/dev/null || exit
      d=$(ps -eo pid,args --no-headers | awk "\$3 ~ /day42_machine/" | wc -l)
      r=$(ls model/s42/phase1/*/seed_42/best.pt.rejected 2>/dev/null | wc -l)
      f=$(ls results/s42_*/*/seed_42/fold*.json 2>/dev/null | wc -l)
      echo "$d/$r/$f"' 2>/dev/null | tail -1)
    LINE+="  $L=${V:-UNREACH}"
    [[ -z "$V" ]] && { EVENT+="[$L] mat lien lac. "; continue; }
    IFS=/ read -r d r f <<< "$V"; TOT=$((TOT+f))
    [[ "$d" == "0" ]] && EVENT+="[$L] KHONG CON DRIVER. "
    if [[ -z "${BASE_REJ[$L]:-}" ]]; then BASE_REJ[$L]=$r
    elif (( r > BASE_REJ[$L] )); then EVENT+="[$L] THEM $((r-BASE_REJ[$L])) checkpoint bi tu choi. "; BASE_REJ[$L]=$r; fi
  done
  d=$(ps -u "$(id -un)" -o pid,args --no-headers | awk '$3 ~ /day42_machine/' | wc -l)
  r=$(ls model/s42/phase1/*/seed_42/best.pt.rejected 2>/dev/null | wc -l)
  # CHI dem ket qua CUA unixcoder. `pull42.sh` rsync ket qua cua hai may vast
  # VAO chinh results/ nay, nen dem `s42_*` se cong ca chung vao va bao local co
  # 24 fold trong khi that ra no van dang o Phase 1. Da bi lua dung mot lan.
  f=$(ls results/s42_unixcoder/*/seed_42/fold*.json 2>/dev/null | wc -l)
  TOT=$((TOT+f)); LINE+="  local=$d/$r/$f  TONG_FOLD=$TOT"
  [[ "$d" == "0" ]] && EVENT+="[local] KHONG CON DRIVER. "
  if [[ -z "${BASE_REJ[local]:-}" ]]; then BASE_REJ[local]=$r
  elif (( r > BASE_REJ[local] )); then EVENT+="[local] THEM $((r-BASE_REJ[local])) checkpoint bi tu choi. "; BASE_REJ[local]=$r; fi
  (( TOT >= TARGET )) && EVENT+="da co $TOT ket qua fold (nguong $TARGET). "
  echo "$LINE" >> "$OUT"
  if [[ -n "$EVENT" ]]; then echo "$TS  >>> $EVENT" | tee -a "$OUT"; exit 0; fi
  sleep 150
done
echo "$(date -u '+%F %T') het $MAX vong" | tee -a "$OUT"
