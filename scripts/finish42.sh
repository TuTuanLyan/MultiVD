#!/usr/bin/env bash
# Khi mot may vast xong hang doi: KEO VE -> XAC MINH -> roi moi (tuy chon) HUY.
#
#   bash scripts/finish42.sh              # chi kiem tra + keo ve, KHONG huy (mac dinh)
#   ACTION=destroy bash scripts/finish42.sh
#   ACTION=stop    bash scripts/finish42.sh
#   LABELS="ntat2" ACTION=destroy bash scripts/finish42.sh
#
# CONG XAC MINH — khong qua thi TUYET DOI khong huy:
#   1. tren may khong con driver day42 nao (hang doi da xong that)
#   2. moi fold*.json tren may deu co ban o local VA DUNG KICH THUOC BYTE
#   3. moi best.pt cua Phase 1 tren may deu co ban o local
# Chi so 1 va 2 thoi la khong du: mot lan rsync dut giua chung van de lai file
# ngan hon ma `wc -l` dem la "co du".
#
# CHI dung nhan ho ntat. Tai khoan con instance `dung`/`cuongtm*` KHONG phai cua
# viec nay va khong duoc dong toi trong bat ky truong hop nao.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

LABELS="${LABELS:-ntat ntat2}"
ACTION="${ACTION:-none}"          # none | stop | destroy
MODEL_DIR="${MODEL_DIR:-model_s42}"
mkdir -p results "$MODEL_DIR"

for LABEL in $LABELS; do
  case "$LABEL" in
    ntat|ntat[0-9]*) ;;
    *) echo "[$LABEL] KHONG thuoc ho ntat — bo qua."; continue ;;
  esac
  read -r H P <<< "$(vast_endpoint "$LABEL")"
  [[ -z "${H:-}" ]] && { echo "[$LABEL] khong giai duoc dia chi"; continue; }
  SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
  echo "───────── $LABEL ($H:$P) ─────────"

  DRV=$($SSH "root@$H" 'ps -eo pid,args --no-headers | awk "\$3 ~ /day42_machine|day42_fold/" | wc -l' 2>/dev/null | tail -1)
  if [[ "${DRV:-1}" != "0" ]]; then
    echo "  van con $DRV driver — CHUA xong, khong lam gi."
    continue
  fi
  echo "  hang doi da xong (0 driver). Keo ve..."

  rsync -az -e "$SSH" --include='s42_*/' --include='s42_*/**' --exclude='*' \
        "root@$H:/workspace/MultiVD/results/" results/ 2>/dev/null
  rsync -az -e "$SSH" --include='*/' --include='phase1/**/best.pt' --exclude='*' \
        "root@$H:/workspace/MultiVD/model/" "$MODEL_DIR/" 2>/dev/null
  rsync -az -e "$SSH" "root@$H:/workspace/MultiVD/log/" "log/vast_$LABEL/" 2>/dev/null

  # --- cong 2: doi chieu TUNG FILE ke ca kich thuoc byte ---
  R=$($SSH "root@$H" 'cd /workspace/MultiVD/results 2>/dev/null && find . -name "fold*.json" -printf "%p %s\n" | LC_ALL=C sort' 2>/dev/null)
  L=$(cd results && find . -name 'fold*.json' -printf '%p %s\n' 2>/dev/null | LC_ALL=C sort)
    # LC_ALL=C cho CA `comm`, khong chi `sort`. Input da sap bang LC_ALL=C ma
  # `comm` chay duoi locale khac se bao "file 2 is not in sorted order" va tra ve
  # RAC — sang 30/08 no bao lech 100/105 trong khi moi file deu co o local dung
  # tung byte, lam watchdog kẹt va de may vast nam khong. Da tung dinh dung bug
  # nay voi `sort` (memory: pin sort locale).
  MISS=$(LC_ALL=C comm -23 <(echo "$R") <(echo "$L") | grep -c . || true)
  NR=$(echo "$R" | grep -c . || true)

  # --- cong 3: checkpoint Phase 1 ---
  RC=$($SSH "root@$H" 'cd /workspace/MultiVD/model 2>/dev/null && find . -path "*phase1*" -name "best.pt" -printf "%p\n" | LC_ALL=C sort' 2>/dev/null)
  LC=$(cd "$MODEL_DIR" && find . -path '*phase1*' -name 'best.pt' -printf '%p\n' 2>/dev/null | LC_ALL=C sort)
  MISSC=$(LC_ALL=C comm -23 <(echo "$RC") <(echo "$LC") | grep -c . || true)
  NRC=$(echo "$RC" | grep -c . || true)

  echo "  ket qua fold: $NR tren may, thieu/lech o local: $MISS"
  echo "  checkpoint P1: $NRC tren may, thieu o local: $MISSC"

  if (( MISS > 0 || MISSC > 0 )); then
    echo "  -> KHONG $ACTION: doi chieu KHONG khop. Chay lai lenh nay sau."
    LC_ALL=C comm -23 <(echo "$R") <(echo "$L") | head -5 | sed 's/^/       thieu: /'
    continue
  fi

  case "$ACTION" in
    none)    echo "  -> doi chieu KHOP. ACTION=none nen giu may (dat ACTION=destroy de huy)." ;;
    # PHAI co -y: `vastai destroy instance` hoi xac nhan tuong tac
    # ("Are you sure ... [y/N]"). Watchdog chay khong co stdin nen lenh bi
    # "Aborted." va may VAN CHAY. Do duoc 27/08: ntat2 nam khong 38 phut sau khi
    # nhat ky da ghi "HUY may" — neu de qua dem thi la ~10 tieng tinh tien.
    stop)    echo "  -> doi chieu KHOP. DUNG may."; vastai stop instance "$(vast_id "$LABEL")" ;;
    destroy) echo "  -> doi chieu KHOP. HUY may."
             OUT=$(vastai destroy instance "$(vast_id "$LABEL")" -y 2>&1)
             echo "$OUT" | sed 's/^/     /'
             if echo "$OUT" | grep -qi "abort\|error\|fail"; then
               echo "     !! LENH HUY KHONG THANH CONG — may van chay, se thu lai vong sau."
               return 1 2>/dev/null || DESTROY_FAILED=1
             fi ;;
  esac
done
echo
echo "tong ket qua fold o local: $(find results -name 'fold*.json' 2>/dev/null | wc -l)"
echo "tong checkpoint P1 o local: $(find "$MODEL_DIR" model/s42 -path '*phase1*' -name best.pt 2>/dev/null | wc -l)"
