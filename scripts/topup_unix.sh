#!/usr/bin/env bash
# Nap them viec cua unixcoder cho mot may vast da xong hang doi cua no.
#
#   bash scripts/topup_unix.sh ntat2
#
# Chon fold tu CAO xuong (5,4,3...) trong khi may local di tu THAP len (1,2,3...)
# — hai ben gap nhau o giua, khong dam nhau. Fold nao da co ket qua o local hoac
# da bi mot may khac nhan thi bo qua.
#
# Chuyen TRON mot fold (baseline + moi nhanh) — xem giai thich trong
# run/day42_fold.sh ve ly do khong duoc chia theo nhanh.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

LABEL="${1:?ntat | ntat2}"
BB="${BB:-unixcoder=microsoft/unixcoder-base:cls}"
NAME="${BB%%=*}"
CLAIMS=log/unix_claims.txt; mkdir -p log; touch "$CLAIMS"

case "$LABEL" in ntat|ntat[0-9]*) ;; *) echo "[$LABEL] khong thuoc ho ntat"; exit 2 ;; esac
read -r H P <<< "$(vast_endpoint "$LABEL")"
[[ -z "${H:-}" ]] && { echo "[$LABEL] khong giai duoc dia chi"; exit 2; }
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"

# --- chon fold ---
PICK=""
for F in 5 4 3 2 1; do
  grep -qx "$F" "$CLAIMS" && continue
  n=$(ls results/s42_${NAME}/*/seed_42/fold${F}.json 2>/dev/null | wc -l)
  (( n > 0 )) && continue          # local (hoac lan truoc) da lam fold nay
  PICK="$F"; break
done
[[ -z "$PICK" ]] && { echo "[$LABEL] khong con fold nao de giao — co the huy may."; exit 10; }

# GHI CLAIM NGAY, khong doi rsync xong. Buoc day checkpoint mat vai phut; neu
# chi claim sau do thi watchdog goi topup o vong ke tiep se chon TRUNG fold nay
# va tai thua 4.8 GB, roi mot trong hai ben bi flock chan. Claim som + go claim
# tren moi duong thoat that bai la cach re nhat de tranh.
echo "$PICK" >> "$CLAIMS"
unclaim() { sed -i "/^${PICK}\$/d" "$CLAIMS"; }
trap 'rc=$?; (( rc != 0 )) && unclaim; exit $rc' EXIT

# --- dia con du khong? can ~5 GB cho 10 checkpoint ---
FREE=$($SSH "root@$H" 'df -P / | tail -1 | awk "{print \$4}"' 2>/dev/null)
if [[ -z "$FREE" ]]; then echo "[$LABEL] khong doc duoc dia"; exit 2; fi
if (( FREE < 6000000 )); then
  echo "[$LABEL] dia con $((FREE/1024)) MB — thu don checkpoint cua backbone KHAC."
  # QUY TAC: KHONG BAO GIO xoa mot checkpoint chua duoc XAC MINH la da co o local.
  # Ban dau ham nay lam `rm -rf model/s42/phase1` khi dia chat. Luc 13:27 ngay
  # 27/08 no da xoa 10 checkpoint Phase 1 cua t5p trong khi mot rsync cuu ho dang
  # chay dang do — mat 6 cai, phai huan luyen lai ~2 gio neu con can. Ket qua
  # Phase 2 thi khong mat vi da ve truoc do, nhung do la may man chu khong phai
  # thiet ke.
  for D in $($SSH "root@$H" 'ls -d /workspace/MultiVD/model/s42/phase1/*/ 2>/dev/null | xargs -r -n1 basename' 2>/dev/null); do
    case "$D" in ${NAME}__*) continue ;; esac
    RSZ=$($SSH "root@$H" "stat -c %s /workspace/MultiVD/model/s42/phase1/$D/seed_42/best.pt 2>/dev/null" 2>/dev/null)
    [[ -z "$RSZ" ]] && continue
    LSZ=""
    for CAND in "model_s42/phase1/$D/seed_42/best.pt" "model/s42/phase1/$D/seed_42/best.pt"; do
      [[ -f "$CAND" ]] && LSZ=$(stat -c %s "$CAND") && break
    done
    if [[ "$LSZ" == "$RSZ" ]]; then
      $SSH "root@$H" "rm -rf /workspace/MultiVD/model/s42/phase1/$D" 2>/dev/null
      echo "    xoa $D (da xac minh co o local, $RSZ byte)"
    else
      echo "    GIU $D — chua co ban khop o local (may $RSZ, local ${LSZ:-khong co})"
    fi
  done
  FREE=$($SSH "root@$H" 'df -P / | tail -1 | awk "{print \$4}"' 2>/dev/null)
  (( FREE < 5500000 )) && { echo "[$LABEL] van khong du dia ($((FREE/1024)) MB) — bo qua."; exit 3; }
fi

# --- day checkpoint Phase 1 cua unixcoder len (chi nhung cai con thieu) ---
echo "[$LABEL] day checkpoint Phase 1 cua $NAME len..."
rsync -az --info=stats1 -e "$SSH" \
  --include='*/' --include="${NAME}__*/**/best.pt" --exclude='*' \
  model/s42/phase1/ "root@$H:/workspace/MultiVD/model/s42/phase1/" 2>&1 | tail -2
NCK=$($SSH "root@$H" "ls -d /workspace/MultiVD/model/s42/phase1/${NAME}__*/seed_42/best.pt 2>/dev/null | wc -l" 2>/dev/null)
echo "[$LABEL] co $NCK checkpoint $NAME tren may"
(( NCK < 1 )) && { echo "[$LABEL] chua co checkpoint nao — bo qua."; exit 4; }

# --- dong bo ma nguon + du lieu roi phong ---
rsync -az -e "$SSH" run/ "root@$H:/workspace/MultiVD/run/" 2>/dev/null
rsync -az -e "$SSH" scripts/ "root@$H:/workspace/MultiVD/scripts/" 2>/dev/null
rsync -az -e "$SSH" data/phase1_4cwe.jsonl data/phase1_full.jsonl data/phase1_common.jsonl \
      "root@$H:/workspace/MultiVD/data/" 2>/dev/null

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
  "cd /workspace/MultiVD && SAM_RHO=0 PYTHON=python3 setsid bash run/day42_fold.sh '$BB' $PICK </dev/null >/dev/null 2>&1"
sleep 6
D=$($SSH "root@$H" 'ps -eo pid,args --no-headers | awk "\$3 ~ /day42_fold/" | wc -l' 2>/dev/null | tail -1)
echo "[$LABEL] da giao fold $PICK cho $NAME (driver=$D)"
[[ "${D:-0}" == "0" ]] && { echo "[$LABEL] PHONG HONG — go claim."; exit 5; }
exit 0
