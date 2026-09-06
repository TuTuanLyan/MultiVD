#!/usr/bin/env bash
# Watchdog cho khoi lambda=0.02 + ASAM rho=0.1 + RecAdam.  ARM=destroy bash scripts/overnight47.sh
#
# THAY DOI SO VOI overnight45: DIEU KIEN "XONG" KHONG CON LA DEM O.
#
# Ban 45 huy may khi t5p du 50 o. Nhung 50 la khong the dat: 3 nhanh Phase 1 cua
# t5p bi cong chat luong tu choi (best_epoch<=1 hoac val<0.55) nen 15 o se KHONG
# BAO GIO sinh ra. Watchdog dem o se cho mai mai va may cu tinh tien.
#
# Dieu kien dung bay gio la tin hieu do CHINH driver phat ra khi no chay het:
#     day45_machine.sh in "########## L02 t5p xong <gio> ##########"
# Driver chet + CO dong do  -> xong that -> doi chieu tung byte -> huy
# Driver chet + KHONG co    -> chet giua chung -> phong lai (toi da 6 lan)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

ARM="${ARM:-none}"; EVERY="${EVERY:-600}"; MAX="${MAX:-300}"
BB_T5P='t5p=Salesforce/codet5p-220m-bimodal:mean'
DONE_RE='L02 t5p xong'
LOG=log/overnight47.log; mkdir -p log log/vast_ntat
say(){ echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
FAILS=0

sshx(){ timeout 30 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$2" root@"$1" "$3" 2>/dev/null; }

say "== overnight47 bat dau, ARM=$ARM, chu ky ${EVERY}s, dung khi driver in '$DONE_RE' =="
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
  FIN=$(grep -c "$DONE_RE" log/vast_ntat/day45_t5p.log 2>/dev/null); FIN="${FIN:-0}"
  NT=$(ls results/s42_t5p/transfer_*_l02/seed_42/fold*.json 2>/dev/null | wc -l)
  LD=$(ps -eo pid,args --no-headers | awk '$3 ~ /day45_machine/ || $0 ~ /src\/train_transfer|src\/train_baseline/' | wc -l)
  TOT=$(ls results/s42_*/transfer_*_l02/seed_42/fold*.json 2>/dev/null | wc -l)
  say "[ntat dr=${D:-?} dia=$(( ${G:-0}/1024 ))MB t5p=$NT xong=$FIN] [local dr=$LD] tong=$TOT/150"

  # canh bao dia — KHONG tu xoa nua. Xoa checkpoint da lam hong symlink cua nhanh
  # `none` mot lan roi (t5p__none_*_l02/seed_42 -> t5p__none_*/seed_42 da bi xoa),
  # lam mat 15 o. Neu thieu dia thi bao de nguoi quyet dinh.
  (( ${G:-999999} < 1572864 )) && say "[ntat] !! DIA CON $(( ${G:-0}/1024 ))MB — can nguoi xu ly, watchdog KHONG tu xoa."

  if [[ "${D:-1}" == "0" ]]; then
    if (( FIN == 0 )); then
      FAILS=$((FAILS+1))
      say "[ntat] driver CHET nhung chua thay dong ket thuc — KHOI PHUC (lan $FAILS)."
      if (( FAILS > 6 )); then say "[ntat] khoi phuc that bai $FAILS lan — dung watchdog de nguoi xem."; exit 1; fi
      sshx "$H" "$P" 'rm -f /tmp/multivd_day45.lock*' >/dev/null
      ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
        "cd /workspace/MultiVD && ASAM_RHO=0.1 PYTHON=python3 setsid bash run/day45_machine.sh '$BB_T5P' </dev/null >/dev/null 2>&1"
      sleep 20
      say "[ntat] sau khoi phuc: driver=$(sshx "$H" "$P" 'ps -eo pid,args --no-headers | awk "\$3 ~ /day45_machine/" | wc -l' | tail -1)"
    else
      say "[ntat] driver da in dong ket thuc va da thoat. Doi chieu truoc khi huy..."
      R=$($SSH "root@$H" "cd /workspace/MultiVD/results 2>/dev/null && find . -name 'fold*.json' -printf '%p %s\n' | LC_ALL=C sort" 2>/dev/null)
      L=$(cd results && find . -name 'fold*.json' -printf '%p %s\n' 2>/dev/null | LC_ALL=C sort)
      NR=$(echo "$R" | grep -c . || true)
      MISS=$(LC_ALL=C comm -23 <(echo "$R") <(echo "$L") | grep -c . || true)
      say "[ntat] $NR file tren may, lech o local: $MISS"
      if (( MISS > 0 )); then
        say "[ntat] CHUA khop — giu may, thu lai vong sau."
        LC_ALL=C comm -23 <(echo "$R") <(echo "$L") | head -3 | sed 's/^/     thieu: /' | tee -a "$LOG"
      else
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
            say "== da huy ntat. Local van chay tiep. Tong: $(ls results/s42_*/transfer_*_l02/seed_42/fold*.json 2>/dev/null|wc -l)/150 =="
            exit 0
          fi
        else say "[ntat] ARM=none nen giu may."; exit 0; fi
      fi
    fi
  fi
  sleep "$EVERY"
done
say "== het $MAX vong =="
