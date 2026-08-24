#!/usr/bin/env bash
# Phát một dòng sự kiện mỗi khi có thứ ĐÁNG BÁO. Dùng làm nguồn cho Monitor.
#
# Bản trước phát ở mức "xong một BƯỚC", mà một bước là trọn một khối λ — vài giờ.
# Hệ quả đo được: chạy 1,5 giờ, phát đúng 0 sự kiện, trong khi ba fold đã hoàn tất.
# Im lặng khi đó trông y hệt "chưa có gì xảy ra", và đó là kiểu im lặng tệ nhất.
#
# Đơn vị đúng là FOLD: một fold xong nghĩa là mọi backbone và mọi nhánh trên máy
# đó đều có kết quả cho fold ấy — tức một lát cắt đọc được, so sánh được.
#
# Phủ cả nhánh hỏng, không chỉ nhánh tốt: máy không với tới được, hàng đợi biến
# mất, bước hỏng, và ALL_DONE. Một bộ lọc chỉ bắt tin vui thì im lặng suốt lúc
# treo, và im lặng đó không phân biệt được với đang chạy.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

INTERVAL="${INTERVAL:-600}"
# nhãn|thư mục dự án|thư mục trạng thái
MACHINES=(
  "ntat2|/workspace/MultiVD|/workspace/overnight_m1"
)

declare -A PREV
while true; do
  for M in "${MACHINES[@]}"; do
    IFS='|' read -r NAME DIR STATE <<< "$M"

    if ! read -r HOST PORT <<< "$(vast_endpoint "$NAME")" || [[ -z "${HOST:-}" ]]; then
      if [[ "${PREV[$NAME.reach]:-1}" != "0" ]]; then
        echo "[$NAME] KHONG GIAI DUOC DIA CHI — instance $(vast_state "$NAME") luc $(date -u +%H:%M) UTC"
      fi
      PREV[$NAME.reach]=0
      continue
    fi

    OUT=$(timeout 45 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes \
          -p "$PORT" "root@$HOST" "
      for f in 1 2 3 4 5; do
        printf 'f%s=%s ' \"\$f\" \"\$(find $DIR/results -name \"fold\$f.json\" 2>/dev/null | wc -l)\"
      done
      printf 'steps=%s ' \"\$(ls $STATE/*.done 2>/dev/null | wc -l)\"
      printf 'fail=%s ' \"\$(wc -l < $STATE/failed.txt 2>/dev/null || echo 0)\"
      printf 'alldone=%s ' \"\$(test -f $STATE/ALL_DONE && echo 1 || echo 0)\"
      printf 'queue=%s ' \"\$(ps -eo pid,args --no-headers | awk '\\\$2==\"bash\" && \\\$3 ~ /overnight/ {c++} END{print c+0}')\"
      printf 'jobfail=%s\\n' \"\$(awk '/HANG DOI QUA DEM/{n=0} /THAT BAI/{n++} END{print n+0}' $STATE/../overnight_*.log 2>/dev/null | tail -1)\"
    " 2>/dev/null)

    if [[ -z "$OUT" ]]; then
      if [[ "${PREV[$NAME.reach]:-1}" != "0" ]]; then
        echo "[$NAME] KHONG SSH DUOC luc $(date -u +%H:%M) UTC (instance $(vast_state "$NAME")) — hang doi tren may van chay doc lap"
      fi
      PREV[$NAME.reach]=0
      continue
    fi
    [[ "${PREV[$NAME.reach]:-1}" == "0" ]] && echo "[$NAME] ket noi lai duoc ($HOST:$PORT)"
    PREV[$NAME.reach]=1

    val() { sed -n "s/.*$1=\([0-9]*\).*/\1/p" <<< "$OUT"; }
    STEPS=$(val steps); FAIL=$(val fail); ALL=$(val alldone); Q=$(val queue); JF=$(val jobfail)

    # Số nhánh mong đợi mỗi fold = số lớn nhất từng thấy ở một fold bất kỳ. Nó hội
    # tụ về đúng giá trị sau fold đầu tiên và không cần biết trước cấu hình máy.
    EXPECT=0; COMPLETE=0; DETAIL=""
    for f in 1 2 3 4 5; do
      c=$(sed -n "s/.*f$f=\([0-9]*\).*/\1/p" <<< "$OUT")
      (( c > EXPECT )) && EXPECT=$c
    done
    for f in 1 2 3 4 5; do
      c=$(sed -n "s/.*f$f=\([0-9]*\).*/\1/p" <<< "$OUT")
      (( EXPECT > 0 && c == EXPECT )) && COMPLETE=$((COMPLETE + 1))
      DETAIL="$DETAIL f$f=$c"
    done

    # "So nhanh mong doi" duoc suy la lon nhat qua cac fold, va no BO LEN DAN khi
    # mot khoi moi dang lap day fold 1. Nen moi lan no doi, hai cach doc deu sai:
    # goi la "xong fold" thi bon fold cu bong trong nhu thut lui, con goi la "khoi
    # moi bat dau" thi lan thu hai tro di lai sai. Khi nguong con dang doi thi chi
    # bao TIEN DO tho, khong ket luan gi — cho no on dinh roi hay noi "xong fold".
    # Nguong con dang bo len => IM LANG. Khong bao gi ca.
    #
    # Truoc do no bao moi lan nhich, va da bao sai hai lan lien: mot khoi moi lap
    # day fold 1 dan dan nen nguong nhich lien tuc, moi buoc nhich lai la mot su
    # kien vo nghia. Cadence nguoi dung can la ~2 fold mot lan, khong phai moi lan
    # them mot job. Trang thai chi tiet luon lay duoc bang scripts/status.sh.
    if [[ "${PREV[$NAME.expect]:--1}" != "$EXPECT" ]]; then
      :
    elif [[ "${PREV[$NAME.folds]:--1}" != "$COMPLETE" && "${PREV[$NAME.folds]:--1}" != "-1" ]]; then
      echo "[$NAME] XONG FOLD — da du $COMPLETE/5 fold (moi fold $EXPECT nhanh:$DETAIL). Keo ve: bash scripts/pull_results.sh"
    fi
    # CHI bao khi TANG. Ba bo dem nay deu co the GIAM, va giam luon co nghia la
    # "vua khoi dong lai" chu khong phai su co: xoa mot co XONG sai lam so buoc
    # tut xuong, va bo dem job hong reset ve 0 moi lan hang doi chay lai. Bao ca
    # chieu giam thi moi lan sua chua deu sinh ra ba bao dong gia.
    if (( STEPS > ${PREV[$NAME.steps]:-0} )) && [[ -n "${PREV[$NAME.steps]:-}" ]]; then
      echo "[$NAME] XONG MOT KHOI (buoc $STEPS cua hang doi)"
    fi
    if (( FAIL > ${PREV[$NAME.fail]:-0} )); then
      echo "[$NAME] BUOC HONG — tong $FAIL. Xem $STATE/failed.txt"
    fi
    if (( JF > ${PREV[$NAME.jobfail]:-0} )); then
      echo "[$NAME] JOB HONG — $JF job khong sinh ra ket qua tu lan khoi dong gan nhat"
    fi
    if [[ "$Q" == "0" && "$ALL" == "0" ]]; then
      echo "[$NAME] KHONG CO HANG DOI dang chay va chua ALL_DONE — watchdog se bat lai trong 5 phut; lap lai thi la su co"
    fi
    if [[ "$ALL" == "1" && "${PREV[$NAME.all]:-0}" != "1" ]]; then
      echo "[$NAME] ALL_DONE — het hang doi. Tai ve: bash scripts/pull_results.sh --with-phase1"
    fi

    PREV[$NAME.folds]=$COMPLETE; PREV[$NAME.steps]=$STEPS; PREV[$NAME.expect]=$EXPECT
    PREV[$NAME.fail]=$FAIL; PREV[$NAME.jobfail]=$JF; PREV[$NAME.all]=$ALL
  done
  sleep "$INTERVAL"
done
