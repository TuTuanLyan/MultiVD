#!/usr/bin/env bash
# Monitor nen cho khoi lambda=0.02 + ASAM rho=0.1 + RecAdam.
# Ghi tien do moi vong vao log/monitor45.log; THOAT (de duoc goi len) khi:
#   - mot may mat driver ma chua xong
#   - xong them mot nguon tron ven cua mot backbone
#   - ca hai may xong het
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
OUT=log/monitor45.log; mkdir -p log
MAX="${MAX:-240}"
# Khoi tao PREV bang trang thai HIEN TAI, khong phai rong: neu de rong thi vong
# dau tien coi moi nguon da xong tu truoc la "su kien moi" va monitor thoat ngay.
PREV=$(for bb in codebert unixcoder t5p; do for s in 4cwe com full; do
  n=$(ls results/s42_${bb}/transfer_*_${s}_l02/seed_42/fold*.json 2>/dev/null|wc -l)
  [ "$n" -ge 20 ] && echo -n "$bb/$s "; done; done)
pull() {
  read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
  [[ -z "${H:-}" || "$P" == "None" ]] && return 0
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $P" \
    --include='s42_*/' --include='s42_*/**' --exclude='*' \
    "root@$H:/workspace/MultiVD/results/" results/ 2>/dev/null
  return 0
}
for ((i=0;i<MAX;i++)); do
  pull
  LINE=""; EV=""
  for bb in codebert unixcoder t5p; do
    part=""
    for s in 4cwe com full; do
      n=$(ls results/s42_${bb}/transfer_*_${s}_l02/seed_42/fold*.json 2>/dev/null|wc -l)
      part+="$s=$n "
    done
    LINE+="$bb[$part] "
  done
  LD=$(ps -eo pid,args --no-headers | awk '$3 ~ /day45_machine/' | wc -l)
  read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
  RD=0
  if [[ -n "${H:-}" && "$P" != "None" ]]; then
    RD=$(timeout 20 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$P" root@"$H" \
      'ps -eo pid,args --no-headers | awk "\$3 ~ /day45_machine/" | wc -l' 2>/dev/null | tail -1)
  else EV+="[ntat] mat lien lac. "; fi
  TOT=$(ls results/s42_*/transfer_*_l02/seed_42/fold*.json 2>/dev/null|wc -l)
  echo "$(date -u '+%F %T') local_dr=$LD ntat_dr=${RD:-?} tong=$TOT/150 | $LINE" >> "$OUT"
  # su kien: mot nguon tron ven moi (20/20) xuat hien
  DONE=$(for bb in codebert unixcoder t5p; do for s in 4cwe com full; do
      n=$(ls results/s42_${bb}/transfer_*_${s}_l02/seed_42/fold*.json 2>/dev/null|wc -l)
      [ "$n" -ge 20 ] && echo -n "$bb/$s "; done; done)
  [[ -n "$DONE" && "$DONE" != "$PREV" ]] && { EV+="xong tron ven: $DONE. "; PREV="$DONE"; }
  (( LD == 0 )) && (( ${RD:-0} == 0 )) && EV+="CA HAI MAY het driver (tong $TOT/150). "
  if [[ -n "$EV" ]]; then echo "$(date -u '+%F %T') >>> $EV" | tee -a "$OUT"; exit 0; fi
  sleep 300
done
