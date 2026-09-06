#!/usr/bin/env bash
# Watchdog qua dem cho hai may vast. Chay nen, tu lo tu dau den cuoi:
#
#   moi vong (mac dinh 5 phut):
#     - may nao CON driver  -> ghi tien do, khong dong gi
#     - may nao HET driver  -> goi finish42.sh cho rieng nhan do:
#                                keo ket qua + checkpoint ve, doi chieu tung file
#                                ke ca kich thuoc byte; KHOP thi moi huy/dung.
#     - bat thuong (mat lien lac, driver chet giua chung, dia gan day)
#                           -> ghi CANH BAO va THOAT de duoc goi len
#
#   bash scripts/overnight42.sh                    # keo ve + xac minh, KHONG huy
#   ARM=destroy bash scripts/overnight42.sh        # xac minh xong thi HUY
#   ARM=stop    bash scripts/overnight42.sh        # xac minh xong thi DUNG (van tinh tien dia)
#
# Vi sao co cong xac minh chat den vay: huy may la KHONG HOI LAI DUOC. Mot lan
# rsync dut giua chung de lai file ngan hon ma dem dong van thay "du" — nen phai
# doi chieu ca KICH THUOC BYTE, khong chi dem file.
#
# CHI dong toi nhan ho ntat. Tai khoan con instance khac (`dung`, `cuongtm*`)
# khong lien quan va khong duoc cham vao.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

LABELS="${LABELS:-ntat ntat2}"
ARM="${ARM:-none}"                 # none | stop | destroy
EVERY="${EVERY:-300}"
MAX="${MAX:-288}"                  # 288 x 5 phut = 24 gio
LOG=log/overnight42.log; mkdir -p log
declare -A DONE

say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
say "== bat dau, ARM=$ARM, nhan: $LABELS =="

for ((i=0; i<MAX; i++)); do
  ALIVE=0
  for L in $LABELS; do
    [[ -n "${DONE[$L]:-}" ]] && continue
    read -r H P <<< "$(vast_endpoint "$L")"
    if [[ -z "${H:-}" ]]; then
      say "CANH BAO [$L] khong giai duoc dia chi — co the da bi huy ben ngoai."
      DONE[$L]=unreachable; continue
    fi
    SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
    ST=$($SSH "root@$H" '
      cd /workspace/MultiVD 2>/dev/null || { echo "NOREPO"; exit; }
      d=$(ps -eo pid,args --no-headers | awk "\$3 ~ /day42_machine|day42_fold/" | wc -l)
      f=$(ls results/s42_*/*/seed_42/fold*.json 2>/dev/null | wc -l)
      g=$(df -P / | tail -1 | awk "{print \$4}")
      echo "$d $f $g"' 2>/dev/null | tail -1)
    if [[ -z "$ST" || "$ST" == NOREPO ]]; then
      say "CANH BAO [$L] mat lien lac hoac mat repo."; continue
    fi
    read -r D F G <<< "$ST"
    (( G < 1048576 )) && say "CANH BAO [$L] dia con $((G/1024)) MB — sap day."
    if [[ "$D" == "0" ]]; then
      say "[$L] HET driver, $F ket qua fold."
      # NAP THEM TRUOC, HUY SAU. May da xong viec cua no van con GPU tot; giao
      # cho no mot fold cua backbone dang cham (unixcoder) truoc khi tra may.
      # Chuyen TRON fold, khong chia theo nhanh — xem run/day42_fold.sh.
      if [[ "${TOPUP:-1}" == "1" ]]; then
        TU=$(bash scripts/topup_unix.sh "$L" 2>&1); RC=$?
        echo "$TU" >> "$LOG"
        if (( RC == 0 )); then
          say "[$L] $(echo "$TU" | tail -1)"
          ALIVE=$((ALIVE+1)); continue
        fi
        say "[$L] khong nap them duoc: $(echo "$TU" | tail -1)"
      fi
      say "[$L] Keo ve va doi chieu..."
      OUT=$(LABELS="$L" ACTION="$ARM" bash scripts/finish42.sh 2>&1)
      echo "$OUT" >> "$LOG"
      if echo "$OUT" | grep -q "doi chieu KHONG khop"; then
        say "[$L] doi chieu KHONG khop — GIU MAY, se thu lai vong sau."
        ALIVE=$((ALIVE+1))
      else
        say "[$L] doi chieu KHOP. $(echo "$OUT" | grep -o '\-> .*' | tail -1)"
        DONE[$L]=ok
      fi
    else
      ALIVE=$((ALIVE+1))
      say "[$L] dang chay ($D driver, $F fold, dia $((G/1024)) MB)"
    fi
  done
  if (( ALIVE == 0 )); then
    say "== moi may vast da xong va da xu ly. Ket thuc watchdog. =="
    say "tong ket qua fold o local: $(find results -name 'fold*.json' 2>/dev/null | wc -l)"
    exit 0
  fi
  sleep "$EVERY"
done
say "== het $MAX vong, van con may dang chay =="
