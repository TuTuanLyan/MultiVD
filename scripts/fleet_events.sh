#!/usr/bin/env bash
# fleet_events.sh — nguon SU KIEN cho Monitor. Moi dong stdout = mot thong bao.
#
# CHI in khi TRANG THAI DOI, khong in moi vong — neu khong thi mot vong 2 phut se thanh
# spam va monitor bi tu dong tat.
#
# PHU KIN CA HAI CHIEU (yeu cau cua Monitor: im lang khong phai la thanh cong):
#   IDLE   — may khong co job va khong co worklist  -> tien GPU dang chay khong
#   DEAD   — worklist chet ma con muc chua chay
#   DONE   — so o cua may tang qua moc, hoac worklist bao xong het
#   SSHERR — ba vong lien khong ssh duoc  (khong ket luan may chet, chi bao)
#   FAIL   — log co dong THAT BAI moi
set -uo pipefail
cd /drive1/cuongtm/ntat/MultiVD || exit 1
source scripts/endpoints.sh
R=/workspace/MultiVD
declare -A PREV ERRN IDLEN
tick=0
while true; do
  tick=$((tick+1))
  # ---- hai may vast ----
  for L in ntat ntat2; do
    VAST_CACHE_TTL=1 read -r H P <<< "$(vast_endpoint "$L" 2>/dev/null)" || true
    if [[ -z "${H:-}" || "${P:-None}" == "None" ]]; then
      ERRN[$L]=$(( ${ERRN[$L]:-0} + 1 ))
      (( ${ERRN[$L]} == 3 )) && echo "SSHERR $L | khong giai duoc dia chi 3 vong lien"
      continue
    fi
    out=$(timeout 60 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p "$P" root@"$H" \
      "cd $R 2>/dev/null || exit 1
       echo j=\$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
       echo w=\$(ps -eo args --no-headers | grep -c '[v]ast_worklist.sh')
       echo d=\$(grep -c . log/worklist.done 2>/dev/null)
       echo t=\$(grep -vc '^\s*#\|^\s*\$' log/worklist.txt 2>/dev/null)
       echo n=\$(ls results/*_t5p/*/seed_*/fold*.json 2>/dev/null | wc -l)
       echo f=\$(grep -c 'THAT BAI' log/*.log 2>/dev/null | awk -F: '{s+=\$2} END{print s+0}')" 2>/dev/null) || true
    if [[ -z "$out" ]]; then
      ERRN[$L]=$(( ${ERRN[$L]:-0} + 1 ))
      (( ${ERRN[$L]} == 3 )) && echo "SSHERR $L | khong ssh duoc 3 vong lien ($H:$P)"
      continue
    fi
    ERRN[$L]=0
    j=$(sed -n 's/^j=//p' <<<"$out"|head -1); w=$(sed -n 's/^w=//p' <<<"$out"|head -1)
    d=$(sed -n 's/^d=//p' <<<"$out"|head -1); t=$(sed -n 's/^t=//p' <<<"$out"|head -1)
    n=$(sed -n 's/^n=//p' <<<"$out"|head -1); f=$(sed -n 's/^f=//p' <<<"$out"|head -1)
    st="j${j:-?}w${w:-?}d${d:-?}"
    # BAO DONG GIA da gap 08/09: giua hai o, job=0 trong vai giay khi Pha 1 cua nguon ke
    # dang nap. Phai thay IDLE HAI VONG LIEN TIEP (4 phut) moi bao.
    if (( ${j:-0} == 0 && ${w:-0} == 0 )); then
      IDLEN[$L]=$(( ${IDLEN[$L]:-0} + 1 ))
      (( ${IDLEN[$L]} >= 2 )) && st="IDLE" || st="j0w0-cho-xac-nhan"
    else IDLEN[$L]=0; fi
    if [[ "${PREV[${L}_st]:-}" != "$st" ]]; then
      case "$st" in
        IDLE) echo "IDLE $L | het job VA het worklist — ${d:-?}/${t:-?} muc xong, $n o. CAN VIEC MOI" ;;
        *) [[ "${PREV[${L}_d]:-}" != "${d:-}" && -n "${PREV[${L}_d]:-}" ]] && \
             echo "DONE $L | xong muc ${d}/${t} | tong $n o" ;;
      esac
      PREV[${L}_st]="$st"
    fi
    if [[ -n "${PREV[${L}_f]:-}" && "${f:-0}" -gt "${PREV[${L}_f]}" ]]; then
      echo "FAIL $L | so dong THAT BAI tang ${PREV[${L}_f]} -> ${f}"
    fi
    PREV[${L}_d]="${d:-}"; PREV[${L}_f]="${f:-0}"
  done
  # ---- may local 161 ----
  lj=$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
  lft=$(ps -eo args --no-headers | grep -cE '[r]un/ft2.sh|[r]un/pool1.sh|[s]cripts/queue_pool1.sh')
  # 161 cung vay: phai IDLE hai vong lien tiep
  lst="j${lj}f${lft}"
  if (( lj == 0 && lft == 0 )); then
    IDLEN[161]=$(( ${IDLEN[161]:-0} + 1 ))
    (( ${IDLEN[161]} >= 2 )) && lst="IDLE" || lst="j0-cho-xac-nhan"
  else IDLEN[161]=0; fi
  if [[ "${PREV[161_st]:-}" != "$lst" ]]; then
    [[ "$lst" == "IDLE" ]] && echo "IDLE 161 | khong con job train nao — CAN VIEC MOI"
    [[ "${PREV[161_st]:-}" == "IDLE" && "$lst" != "IDLE" ]] && echo "DONE 161 | da co job chay lai"
    PREV[161_st]="$lst"
  fi
  sleep 120
done
