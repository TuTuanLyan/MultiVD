#!/usr/bin/env bash
# Watchdog qua dem cho khoi ASAM. Chay khong nguoi truc ~10 gio.
#
#   ARM=destroy bash scripts/overnight43.sh
#
# Chuoi viec tren ntat, tu dong:
#   1. unixcoder ASAM+doi chung dang chay  -> chi ghi tien do
#   2. het driver, unixcoder xong          -> keo ket qua ve + DOI CHIEU TUNG BYTE
#                                             -> xoa checkpoint unixcoder tren may
#                                                (chi khi da xac minh co o local)
#                                             -> day checkpoint t5p len
#                                             -> chay t5p ASAM+doi chung
#   3. het driver, t5p cung xong           -> keo ve + doi chieu -> HUY MAY
#
# Local chay codebert doc lap, khong dung den watchdog nay.
#
# BA CHO DE HONG DA GAP THAT, deu da co cong chan:
#   * `vastai destroy instance` HOI XAC NHAN tuong tac -> phai co -y, va phai
#     KIEM TRA output vi no in "Aborted." roi thoat 0.
#   * Xoa checkpoint truoc khi xac minh da co o local -> mat 6 checkpoint t5p
#     hom 27/08. Luon doi chieu ca KICH THUOC BYTE, khong chi dem file.
#   * Dem driver phai tinh CA `day43_machine` LAN cac driver khac, neu khong se
#     tuong may ranh va huy may dang chay.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

ARM="${ARM:-none}"          # none | destroy
EVERY="${EVERY:-300}"
MAX="${MAX:-160}"           # 160 x 5 phut = ~13 gio
LOG=log/overnight43.log; mkdir -p log
STATE=log/overnight43.state; touch "$STATE"

say() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
mark() { grep -qx "$1" "$STATE" || echo "$1" >> "$STATE"; }
done_() { grep -qx "$1" "$STATE"; }

ssh_ntat() {
  read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
  [[ -z "${H:-}" || "$P" == "None" ]] && return 1
  timeout 30 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$P" root@"$H" "$1" 2>/dev/null
}

# Keo ket qua ve va doi chieu tung file + KICH THUOC BYTE cho mot backbone.
verify_pull() {   # $1 = ten backbone
  local bb="$1" H P
  read -r H P <<< "$(vast_endpoint ntat)"
  local SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
  rsync -az -e "$SSH" --include='s42_*/' --include='s42_*/**' --exclude='*' \
        "root@$H:/workspace/MultiVD/results/" results/ 2>/dev/null
  rsync -az -e "$SSH" --include='day43_*.log' --exclude='*' \
        "root@$H:/workspace/MultiVD/log/" log/vast_ntat/ 2>/dev/null
  local R L MISS
  R=$($SSH "root@$H" "cd /workspace/MultiVD/results 2>/dev/null && find . -path '*s42_${bb}*' -name 'fold*.json' -printf '%p %s\n' | LC_ALL=C sort" 2>/dev/null)
  L=$(cd results && find . -path "*s42_${bb}*" -name 'fold*.json' -printf '%p %s\n' 2>/dev/null | LC_ALL=C sort)
    # LC_ALL=C cho CA `comm`, khong chi `sort`. Input da sap bang LC_ALL=C ma
  # `comm` chay duoi locale khac se bao "file 2 is not in sorted order" va tra ve
  # RAC — sang 30/08 no bao lech 100/105 trong khi moi file deu co o local dung
  # tung byte, lam watchdog kẹt va de may vast nam khong. Da tung dinh dung bug
  # nay voi `sort` (memory: pin sort locale).
  MISS=$(LC_ALL=C comm -23 <(echo "$R") <(echo "$L") | grep -c . || true)
  echo "$(echo "$R" | grep -c .) $MISS"
}

say "== overnight43 bat dau, ARM=$ARM =="
for ((i=0;i<MAX;i++)); do
  read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
  if [[ -z "${H:-}" || "$P" == "None" ]]; then
    say "[ntat] khong giai duoc dia chi — co the da bi huy. Ket thuc."; exit 0
  fi
  D=$(ssh_ntat 'ps -eo pid,args --no-headers | awk "\$3 ~ /day43_machine|day44_machine|day43_ctl|phase1_fill/" | wc -l' | tail -1)
  NU=$(ssh_ntat 'ls /workspace/MultiVD/results/s42_unixcoder/*/seed_42/fold*.json 2>/dev/null | wc -l' | tail -1)
  NT=$(ssh_ntat 'ls /workspace/MultiVD/results/s42_t5p/transfer_*_asam/seed_42/fold*.json /workspace/MultiVD/results/s42_t5p/transfer_*_ctl/seed_42/fold*.json 2>/dev/null | wc -l' | tail -1)
  G=$(ssh_ntat 'df -P / | tail -1 | awk "{print \$4}"' | tail -1)
  NL=$(( $(ls results/s42_codebert/transfer_*_asam/seed_42/fold*.json 2>/dev/null|wc -l) + $(ls results/s42_codebert/transfer_*_ctl/seed_42/fold*.json 2>/dev/null|wc -l) ))
  LD=$(ps -eo pid,args --no-headers | awk '$3 ~ /day43_machine/' | wc -l)
  say "[ntat] dr=${D:-?} unix=${NU:-?}/100 t5p=${NT:-?}/100 dia=$(( ${G:-0}/1024 ))MB | [local] dr=$LD codebert=$NL/80"

  if [[ "${D:-1}" == "0" ]]; then
    if ! done_ UNIX_DONE; then
      read -r NR MISS <<< "$(verify_pull unixcoder)"
      say "[ntat] unixcoder xong: $NR ket qua tren may, lech o local $MISS"
      if [[ "${MISS:-1}" != "0" || "${NR:-0}" -lt 90 ]]; then
        say "[ntat] doi chieu CHUA khop — thu lai vong sau."; sleep "$EVERY"; continue
      fi
      mark UNIX_DONE
      # unixcoder Phase-1 checkpoint da co ban goc o local -> an toan de xoa tren may
      NLOC=$(find model/s42/phase1 -name best.pt -path '*unixcoder*' | wc -l)
      if [[ "$NLOC" -ge 10 ]]; then
        ssh_ntat 'cd /workspace/MultiVD/model/s42/phase1 && rm -rf unixcoder__*' >/dev/null
        say "[ntat] da xoa checkpoint unixcoder tren may (co $NLOC ban o local), dia con $(ssh_ntat 'df -h / | tail -1 | awk "{print \$4}"' | tail -1)"
      else
        say "[ntat] KHONG xoa unixcoder: local chi co $NLOC/10"
      fi
      say "[ntat] day checkpoint t5p len..."
      rsync -az -e "ssh -o StrictHostKeyChecking=no -p $P" \
        --include='t5p__*/' --include='t5p__*/seed_42/' --include='t5p__*/seed_42/best.pt' --exclude='*' \
        model/s42/phase1/ "root@$H:/workspace/MultiVD/model/s42/phase1/" 2>/dev/null
      NUP=$(ssh_ntat 'find /workspace/MultiVD/model/s42/phase1 -name best.pt -path "*t5p*" | wc -l' | tail -1)
      say "[ntat] t5p checkpoint tren may: ${NUP:-0}/10"
      if [[ "${NUP:-0}" -ge 10 ]]; then
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
          "cd /workspace/MultiVD && ASAM_RHO=0.1 PYTHON=python3 setsid bash run/day43_machine.sh 't5p=Salesforce/codet5p-220m-bimodal:mean' </dev/null >/dev/null 2>&1"
        sleep 15
        say "[ntat] da phong t5p ASAM (driver=$(ssh_ntat 'ps -eo pid,args --no-headers | awk "\$3 ~ /day43_machine/" | wc -l' | tail -1))"
      else
        say "[ntat] chua du checkpoint t5p — se thu lai vong sau."
      fi
    else
      # unixcoder da xong. Tu day tro di: xac minh + keo ve, roi LAY JOB KE TIEP
      # tu hang doi. Chi huy may khi hang doi RONG.
      CUR=$(cat log/overnight43.job 2>/dev/null || echo "t5p_asam")
      read -r NR MISS <<< "$(verify_pull t5p)"
      say "[ntat] job '$CUR': $NR ket qua tren may, lech o local ${MISS}"

      # Job coi la xong khi khong con driver VA da keo ve khop. Neu chua khop thi
      # cho vong sau — khong bao gio huy hay chuyen job khi du lieu chua an toan.
      if [[ "${MISS:-1}" != "0" ]]; then
        say "[ntat] doi chieu chua khop — cho vong sau."; sleep "$EVERY"; continue
      fi
      bash scripts/queue43.sh mark "$CUR" 2>/dev/null
      NEXT=$(bash scripts/queue43.sh next 2>/dev/null)
      if [[ -n "$NEXT" ]]; then
        JN="${NEXT%%|*}"; REST="${NEXT#*|}"; JBB="${REST%%|*}"; REST2="${REST#*|}"
        JENV="${REST2%%|*}"; JSCRIPT="${REST2#*|}"
        [[ "$JSCRIPT" == "$JENV" || -z "$JSCRIPT" ]] && JSCRIPT="run/day43_machine.sh"
        say "[ntat] job ke tiep: $JN  ($JENV)"
        NUP=$(ssh_ntat 'find /workspace/MultiVD/model/s42/phase1 -name best.pt | wc -l' | tail -1)
        if [[ "${NUP:-0}" -lt 4 ]]; then
          say "[ntat] tren may chi con ${NUP:-0} checkpoint — day lai truoc khi chay."
          BBN="${JBB%%=*}"
          rsync -az -e "ssh -o StrictHostKeyChecking=no -p $P" \
            --include="${BBN}__*/" --include="${BBN}__*/seed_42/" --include="${BBN}__*/seed_42/best.pt" \
            --exclude='*' model/s42/phase1/ "root@$H:/workspace/MultiVD/model/s42/phase1/" 2>/dev/null
        fi
        echo "$JN" > log/overnight43.job
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
          "cd /workspace/MultiVD && $JENV PYTHON=python3 setsid bash $JSCRIPT '$JBB' </dev/null >/dev/null 2>&1"
        sleep 15
        say "[ntat] da phong $JN qua $JSCRIPT (driver=$(ssh_ntat 'ps -eo pid,args --no-headers | awk "\$3 ~ /day43_machine|day44_machine/" | wc -l' | tail -1))"
      else
        verify_pull unixcoder >/dev/null
        say "[ntat] HANG DOI RONG va moi thu da ve khop."
        if [[ "$ARM" == "destroy" ]]; then
          ID=$(vast_id ntat)
          OUT=$(vastai destroy instance "$ID" -y 2>&1)
          say "[ntat] huy may $ID: $OUT"
          if echo "$OUT" | grep -qi "abort\|error\|fail"; then
            say "[ntat] !! LENH HUY THAT BAI — thu lai vong sau."
          else
            say "== xong het, da huy ntat. Tong ket qua o local: $(find results/s42_* -name 'fold*.json' | wc -l) =="
            exit 0
          fi
        else
          say "[ntat] ARM=none nen giu may."; exit 0
        fi
      fi
    fi
  fi
  sleep "$EVERY"
done
say "== het $MAX vong =="
