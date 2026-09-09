#!/usr/bin/env bash
# CHOT_CHAIN — hai giai doan, chay noi tiep trong MOT driver.
# Nguoi dung 09/09: "chay 2 nguon cwe va common, sau khi xong TOAN BO khoi thi chay voi
# full tu phase 1, vi toi muon thay du ket qua cwe va common truoc."
#
#   GD1  4cwe + com   — Pha 1 co san (t5p) hoac re (codebert 930 dong)
#   GD2  full         — Pha 1 phai huan luyen tu dau CA HAI backbone (7 598 dong, ~40 phut)
#
# Trong moi giai doan: BACKBONE vong ngoai (codebert truoc) -> fold -> nguon.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec 7>/tmp/mvd_chot_chain.lock || exit 1
flock -n 7 || { echo "DA CO chot_chain dang chay"; exit 3; }
echo "########## CHOT_CHAIN bat dau $(date -u '+%F %T') ##########"
echo "===== GIAI DOAN 1: 4cwe + com ====="
SOURCES_LIST="4cwe com" bash run/chot2bb.sh 7>&-
n1=$(find results/chot_codebert results/chot_t5p -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## GD1 xong $(date -u '+%F %T') | $n1 o ##########"
if (( n1 < 50 )); then
  echo "!! GD1 moi co $n1/50 o — KHONG sang GD2, de nguoi xem."
  echo "   (watchdog se phong lai chuoi nay; chot2bb bo qua o da co nen no chi chay bu)"
  exit 1
fi
echo "===== GIAI DOAN 2: full (Pha 1 tu dau ca hai backbone) ====="
SOURCES_LIST="full" bash run/chot2bb.sh 7>&-
n2=$(find results/chot_codebert results/chot_t5p -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## CHOT_CHAIN xong $(date -u '+%F %T') | $n2 o (ky vong 70) ##########"
