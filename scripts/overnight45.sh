#!/usr/bin/env bash
# Watchdog qua dem cho khoi lambda=0.02 + ASAM rho=0.1 + RecAdam.
#
#   ARM=destroy bash scripts/overnight45.sh
#
# QUY TAC COT LOI: driver chet KHONG co nghia la xong viec.
#
# Ban truoc huy may ngay khi dem driver = 0. Chieu 30/08 driver t5p chet vi DIA
# DAY 100%, va neu watchdog do dang chay thi no da huy may trong khi t5p con
# nguyen 0/50 o. Nen bay gio:
#
#   driver chet + CON viec  -> don dia (co xac minh) roi PHONG LAI driver
#   driver chet + HET viec  -> doi chieu tung byte -> huy may
#
# Viec cua ntat = t5p: 4cwe 20 o + com 15 + full 15 = 50 (nhanh `cwe` chi co o
# nguon 4cwe nen com/full chi 3 nhanh x 5 fold).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

ARM="${ARM:-none}"; EVERY="${EVERY:-600}"; MAX="${MAX:-200}"
NEED_T5P="${NEED_T5P:-50}"
BB_T5P='t5p=Salesforce/codet5p-220m-bimodal:mean'
LOG=log/overnight45.log; mkdir -p log
say(){ echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
FAILS=0

sshx(){ timeout 30 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$2" root@"$1" "$3" 2>/dev/null; }

say "== overnight45 bat dau, ARM=$ARM, chu ky ${EVERY}s, can t5p=$NEED_T5P o =="
for ((i=0;i<MAX;i++)); do
  read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
  if [[ -z "${H:-}" || "$P" == "None" ]]; then
    say "[ntat] khong giai duoc dia chi — co the da huy. Ket thuc."; exit 0
  fi
  SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"

  rsync -az -e "$SSH" --include='s42_*/' --include='s42_*/**' --exclude='*' \
        "root@$H:/workspace/MultiVD/results/" results/ 2>/dev/null
  rsync -az -e "$SSH" --include='day45_*.log' --exclude='*' \
        "root@$H:/workspace/MultiVD/log/" log/vast_ntat/ 2>/dev/null

  D=$(sshx "$H" "$P" 'ps -eo pid,args --no-headers | awk "\$3 ~ /day45_machine/ || \$0 ~ /src\/train_transfer|src\/train_baseline/" | wc -l' | tail -1)
  G=$(sshx "$H" "$P" 'df -P / | tail -1 | awk "{print \$4}"' | tail -1)
  NT=$(ls results/s42_t5p/transfer_*_l02/seed_42/fold*.json 2>/dev/null | wc -l)
  LD=$(ps -eo pid,args --no-headers | awk '$3 ~ /day45_machine/ || $0 ~ /src\/train_transfer|src\/train_baseline/' | wc -l)
  TOT=$(ls results/s42_*/transfer_*_l02/seed_42/fold*.json 2>/dev/null | wc -l)
  say "[ntat dr=${D:-?} dia=$(( ${G:-0}/1024 ))MB t5p=$NT/$NEED_T5P] [local dr=$LD] tong=$TOT/150"

  # --- canh bao dia som, TRUOC khi no giet driver ---
  if (( ${G:-999999} < 2097152 )); then
    say "[ntat] CANH BAO dia con $(( ${G:-0}/1024 ))MB — don checkpoint cua backbone DA XONG."
    for BB in unixcoder; do
      NLOC=$(ls -d model/s42/phase1/${BB}__*_l02/seed_42/best.pt 2>/dev/null | wc -l)
      NREM=$(sshx "$H" "$P" "ls -d /workspace/MultiVD/model/s42/phase1/${BB}__*_l02/seed_42/best.pt 2>/dev/null | wc -l" | tail -1)
      if (( NLOC >= NREM && NREM > 0 )); then
        sshx "$H" "$P" "cd /workspace/MultiVD/model/s42/phase1 && rm -rf ${BB}__*_l02" >/dev/null
        say "[ntat] da xoa checkpoint $BB (co $NLOC ban o local, may co $NREM), dia con $(sshx "$H" "$P" 'df -h / | tail -1 | awk "{print \$4}"' | tail -1)"
      else
        say "[ntat] KHONG xoa $BB: local $NLOC < may $NREM"
      fi
    done
  fi

  if [[ "${D:-1}" == "0" ]]; then
    if (( NT < NEED_T5P )); then
      # --- CHUA XONG: khoi phuc, khong huy ---
      FAILS=$((FAILS+1))
      say "[ntat] driver CHET nhung t5p moi $NT/$NEED_T5P — KHOI PHUC (lan $FAILS)."
      if (( FAILS > 6 )); then say "[ntat] khoi phuc that bai $FAILS lan — dung watchdog de nguoi xem."; exit 1; fi
      sshx "$H" "$P" 'rm -f /tmp/multivd_day45.lock*' >/dev/null
      ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
        "cd /workspace/MultiVD && ASAM_RHO=0.1 PYTHON=python3 setsid bash run/day45_machine.sh '$BB_T5P' </dev/null >/dev/null 2>&1"
      sleep 20
      say "[ntat] sau khoi phuc: driver=$(sshx "$H" "$P" 'ps -eo pid,args --no-headers | awk "\$3 ~ /day45_machine/" | wc -l' | tail -1)"
    else
      # --- XONG THAT: doi chieu roi huy ---
      say "[ntat] t5p da du $NT/$NEED_T5P. Doi chieu truoc khi huy..."
      R=$($SSH "root@$H" "cd /workspace/MultiVD/results 2>/dev/null && find . -name 'fold*.json' -printf '%p %s\n' | LC_ALL=C sort" 2>/dev/null)
      L=$(cd results && find . -name 'fold*.json' -printf '%p %s\n' 2>/dev/null | LC_ALL=C sort)
      NR=$(echo "$R" | grep -c . || true)
      MISS=$(LC_ALL=C comm -23 <(echo "$R") <(echo "$L") | grep -c . || true)
      say "[ntat] $NR file tren may, lech o local: $MISS"
      if (( MISS > 0 )); then
        say "[ntat] CHUA khop — giu may, thu lai vong sau."
        LC_ALL=C comm -23 <(echo "$R") <(echo "$L") | head -3 | sed 's/^/     thieu: /' | tee -a "$LOG"
      else
        # checkpoint _l02 cung phai ve truoc khi huy
        rsync -a -e "$SSH" --include='*_l02/' --include='*_l02/seed_42/' --include='*_l02/seed_42/best.pt' \
              --exclude='*' "root@$H:/workspace/MultiVD/model/s42/phase1/" model/s42/phase1/ 2>/dev/null
        say "[ntat] doi chieu KHOP ($NR file). Checkpoint _l02 o local: $(ls -d model/s42/phase1/*_l02/seed_42/best.pt 2>/dev/null|wc -l)"
        if [[ "$ARM" == "destroy" ]]; then
          ID=$(vast_id ntat)
          OUT=$(vastai destroy instance "$ID" -y 2>&1)
          say "[ntat] huy $ID: $OUT"
          if echo "$OUT" | grep -qi "abort\|error\|fail"; then
            say "[ntat] !! HUY THAT BAI — thu lai vong sau."
          else
            say "== da huy ntat. Tong: $(ls results/s42_*/transfer_*_l02/seed_42/fold*.json 2>/dev/null|wc -l)/150 =="
            exit 0
          fi
        else say "[ntat] ARM=none nen giu may."; exit 0; fi
      fi
    fi
  fi
  sleep "$EVERY"
done
say "== het $MAX vong =="
