#!/usr/bin/env bash
# queue_bac2.sh — BAC 2 (xac nhan, 5 fold) cho khoi OPT1. Chay tren CA HAI may,
# moi may mot fold moi, dat bang bien FOLD_NEW.
#
#   161:  FOLD_NEW=4 setsid nohup bash scripts/queue_bac2.sh > log/queue_bac2.log 2>&1 &
#   158:  FOLD_NEW=5 ...
#
# Fold TRON VEN tren mot may (CLAUDE.md muc 4): fold 4 hoan toan o 161, fold 5 hoan
# toan o 158, baseline cua chinh fold do cung chay tren may do.
#
# CAU HOI CUA BAC 2: cau chuyen "neo nang SAN" o bac 1 dua vao DUNG MOT fold kho
# (fold 3, baseline 0.7036). Trong hai fold moi co fold nao kho khong, va neu co thi
# neo co lai nang san o dung cho do khong? Day la diem yeu chi mang cua ket luan hien
# tai — trong du an nay mot fold ngoai le da tung ganh ca mot ket luan roi bi rut lai.
#
# BAY GIU NGUYEN BIEN: min_epochs phai KHOP voi luc chay bac 1, neu khong thi fold 4,5
# khac fold 1,2,3 va phep gop khong hop le. Nen chay HAI luot:
#   luot A: 6 cau hinh o min_epochs=3   (dung nhu bac 1)
#   luot B: nhanh me10 o min_epochs=10  (dung nhu bac 1)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLD_NEW="${FOLD_NEW:?phai dat FOLD_NEW=4 hoac 5}"
PY="${PYTHON:-$( [ -x /data/ntat/envs/vdenv/bin/python ] && echo /data/ntat/envs/vdenv/bin/python || echo /home/ntat/miniconda3/envs/vdenv/bin/python )}"
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_queue_bac2.lock || exit 1
flock -n 8 || { echo "DA CO queue_bac2 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
IS161=0; [ -d /home/ntat/miniconda3 ] && IS161=1

wait_free(){
  local w=0
  while true; do
    if flock -n "$LOCK" -c true 2>/dev/null; then
      if [ "$IS161" = 1 ]; then
        local used free
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1)
        free=$(( 16376 - ${used:-16376} ))
        if (( free >= 13000 )); then sleep 10; return 0; fi
        (( w % 1800 == 0 )) && echo "$(ts) | lock trong nhung VRAM ${free}MiB — NHUONG"
      else sleep 10; return 0; fi
    fi
    sleep 120; w=$((w+120)); (( w % 1800 == 0 )) && echo "$(ts) | cho... ${w}s"
  done
}

echo "########## BAC2 bat dau $(ts) | fold $FOLD_NEW | $(hostname) ##########"

wait_free
echo "########## $(ts) | luot A: 6 cau hinh, min_epochs=3 ##########"
MIN_EP=3 SOURCES="4cwe com full" FOLD_LIST="$FOLD_NEW" PYTHON="$PY" \
CONFIGS="c5000_t0p05|recadam|--sam_rho 0
plain|adamw|--sam_rho 0
c50_t0p05|recadam|--sam_rho 0 --pretrain_cof 50
c5_t0p05|recadam|--sam_rho 0 --pretrain_cof 5
c0p5_t0p05|recadam|--sam_rho 0 --pretrain_cof 0.5
warm|adamw|--sam_rho 0 --adamw_anneal_lr" \
bash run/opt1.sh 8>&-

wait_free
echo "########## $(ts) | luot B: nhanh neo ben, min_epochs=10 ##########"
MIN_EP=10 SOURCES="4cwe com full" FOLD_LIST="$FOLD_NEW" PYTHON="$PY" \
CONFIGS="c5000_t0p2k02_me10|recadam|--sam_rho 0 --anneal_t0_ratio 0.2 --anneal_k 0.02" \
bash run/opt1.sh 8>&-

echo "########## BAC2 xong $(ts) | fold $FOLD_NEW ##########"
