#!/usr/bin/env bash
# Watcher nen chay dai. Ghi tien do vao log/watch42.log moi vong, va THOAT
# (de lop tren duoc bao) khi co su kien dang chu y:
#   - t5p hoac unixcoder xong none/full  -> tra loi cau hoi "full co sap voi moi backbone khong"
#   - xuat hien checkpoint .rejected moi
#   - mot may mat driver
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
OUT=log/watch42.log; mkdir -p log
MAX="${1:-144}"        # 144 vong x 150s = 6 gio
declare -A BASE_REJ   # so checkpoint .rejected o VONG DAU, chi bao khi TANG them

for ((i=0;i<MAX;i++)); do
  TS=$(date -u '+%F %T')
  LINE="$TS"; EVENT=""
  for L in ntat ntat2; do
    read -r H P <<< "$(vast_endpoint "$L")"
    V=$(timeout 25 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$P" root@"$H" '
      cd /workspace/MultiVD 2>/dev/null || exit
      d=$(ps -eo pid,args --no-headers | awk "\$3 ~ /day42_machine/" | wc -l)
      c=$(ls model/s42/phase1/*/seed_42/best.pt 2>/dev/null | wc -l)
      r=$(ls model/s42/phase1/*/seed_42/best.pt.rejected 2>/dev/null | wc -l)
      f=$(ls results/s42_*/*/seed_42/fold*.json 2>/dev/null | wc -l)
      j=$(ls log/s42/*.log 2>/dev/null | tail -1 | xargs -r basename)
      echo "$d/$c/$r/$f/$j"' 2>/dev/null | tail -1)
    LINE+="  $L=${V:-UNREACH}"
    [[ "$V" == UNREACH || -z "$V" ]] && EVENT+="[$L] mat lien lac. "
    [[ "${V%%/*}" == "0" ]] && EVENT+="[$L] KHONG CON DRIVER. "
    R=$(echo "$V" | cut -d/ -f3); R=${R:-0}
    if [[ -z "${BASE_REJ[$L]:-}" ]]; then BASE_REJ[$L]=$R
    elif (( R > BASE_REJ[$L] )); then
      EVENT+="[$L] co THEM $(( R - BASE_REJ[$L] )) checkpoint bi tu choi (tong $R). "; BASE_REJ[$L]=$R
    fi
  done
  d=$(ps -u "$(id -un)" -o pid,args --no-headers | awk '$3 ~ /day42_machine/' | wc -l)
  c=$(ls model/s42/phase1/*/seed_42/best.pt 2>/dev/null | wc -l)
  r=$(ls model/s42/phase1/*/seed_42/best.pt.rejected 2>/dev/null | wc -l)
  f=$(ls results/s42_*/*/seed_42/fold*.json 2>/dev/null | wc -l)
  LINE+="  local=$d/$c/$r/$f"
  [[ "$d" == "0" ]] && EVENT+="[local] KHONG CON DRIVER. "
  if [[ -z "${BASE_REJ[local]:-}" ]]; then BASE_REJ[local]=$r
  elif (( r > BASE_REJ[local] )); then
    EVENT+="[local] co THEM $(( r - BASE_REJ[local] )) checkpoint bi tu choi (tong $r). "; BASE_REJ[local]=$r
  fi
  echo "$LINE" >> "$OUT"

  # Su kien theo doi hien tai: xong het Phase 1 cua mot may (bat dau co ket qua
  # fold), hoac codebert xong ba nhanh con lai cua nguon `full`.
  read -r H P <<< "$(vast_endpoint ntat)"
  CB=$(timeout 20 ssh -o StrictHostKeyChecking=no -p "$P" root@"$H" \
      'ls /workspace/MultiVD/log/s42/phase1_codebert_*_full.log 2>/dev/null | wc -l' 2>/dev/null)
  [[ "${CB:-0}" -ge 4 ]] && EVENT+="codebert xong ca 4 nhanh cua nguon full. "
  for L in ntat ntat2; do
    FF=$(echo "$LINE" | grep -o "$L=[0-9]*/[0-9]*/[0-9]*/[0-9]*" | cut -d/ -f4)
    [[ "${FF:-0}" -gt 0 ]] && EVENT+="[$L] da co ket qua fold dau tien. "
  done
  [[ "$f" -gt 0 ]] && EVENT+="[local] da co ket qua fold dau tien. "

  if [[ -n "$EVENT" ]]; then echo "$TS  >>> $EVENT" | tee -a "$OUT"; exit 0; fi
  sleep 150
done
echo "$(date -u '+%F %T') het $MAX vong, khong co su kien" | tee -a "$OUT"
