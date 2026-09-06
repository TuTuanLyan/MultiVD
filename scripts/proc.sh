#!/usr/bin/env bash
# Ham dung chung de TIM va DUNG tien trinh cua run nay. Source file nay, dung goi truc tiep.
#
# Vi sao ton tai file nay — hai cai bay da lam mat thoi gian that:
#
# 1. `pkill -f <mau>` / `pgrep -f <mau>`: dong lenh SSH cua chinh minh cung chua
#    <mau>, nen no tu giet phien dang chay. Lenh dung sau do khong bao gio chay,
#    va ta tuong da don sach.
# 2. Chi so cot cua awk KHAC NHAU giua hai dang ps:
#       ps -eo pid,args  ->  $1=pid  $2=bash  $3=<script>
#       ps -eo args      ->  $1=bash $2=<script>
#    Dung nham cot thi lenh kill khong giet gi ma van thoat 0. Sang 27/08 dieu do
#    lam sinh ra driver THU HAI tren cung mot may; hai job tranh mot GPU va OOM,
#    phai setup lai tu dau.
#
# Quy tac: LUON dung `ps -eo pid,args`, LUON loc theo $3, va LUON kiem tra lai
# sau khi kill.
set -uo pipefail

mvd_pids() {  # $1 = regex khop TEN SCRIPT (cot $3)
  ps -eo pid,args --no-headers 2>/dev/null | awk -v re="$1" '$3 ~ re {print $1}'
}
mvd_job_pids() {  # tien trinh python huan luyen
  ps -eo pid,args --no-headers 2>/dev/null | awk '$0 ~ /src\/train_transfer\.py|src\/train_baseline\.py/ {print $1}'
}
mvd_gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null; }

# LUU Y: co HAI loai driver — `day42_machine.sh` (chay ca hang doi cua mot may)
# va `day42_fold.sh` (nap them mot fold cho may da xong). Bat cu cho nao dem
# driver de quyet dinh "may nay ranh chua" deu phai dem CA HAI, neu khong se huy
# nham mot may dang chay viec nap them.
mvd_stop() {  # dung toan bo driver + job cua run nay, roi XAC NHAN da dung
  mvd_pids 'day42_machine|day42_fold|day42\.sh|matrix\.sh' | xargs -r kill    2>/dev/null; sleep 2
  mvd_job_pids                                  | xargs -r kill -9 2>/dev/null; sleep 3
  local left; left=$( { mvd_pids 'day42_machine|day42_fold|day42\.sh|matrix\.sh'; mvd_job_pids; } | sort -u | wc -l )
  echo "sau khi dung: $left tien trinh con lai"
  [[ "$left" == "0" ]]
}
mvd_reap_gpu() {  # thu hoi tien trinh mo coi con giu VRAM
  # CHI giet tien trinh CUA CHINH MINH VA dung job cua run nay. May local la
  # server dung chung: bo loc cu ("root hoac ntat") se giet ca tien trinh root
  # cua nguoi khac. Da kiem tra that — GPU local dang co job cua user anhnd_02,
  # phai khong bao gio cham toi.
  local me p owner cmd
  me=$(id -un)
  for p in $(mvd_gpu_pids); do
    owner=$(ps -p "$p" -o user= 2>/dev/null | tr -d ' ')
    [ "$owner" = "$me" ] || continue
    cmd=$(ps -p "$p" -o args= 2>/dev/null)
    case "$cmd" in
      *src/train_transfer.py*|*src/train_baseline.py*) kill -9 "$p" 2>/dev/null ;;
    esac
  done
  sleep 3
  nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null
}
mvd_count_drivers() { mvd_pids 'day42_machine|day42_fold' | wc -l; }   # CA HAI loai driver

# Dem driver KHAC minh. Khong the chi loai `$$`: `$( )` fork mot subshell co
# DONG LENH Y HET tien trinh cha nhung pid khac, nen script tu dem chinh no va
# bao "da co 2 driver" trong khi may hoan toan trong. Loai theo PROCESS GROUP:
# subshell cua ta cung pgid voi ta, con driver that phong bang setsid co pgid
# rieng.
mvd_other_drivers() {   # dem CA day42_machine LAN day42_fold
  local mypg; mypg=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
  ps -eo pid,pgid,args --no-headers 2>/dev/null \
    | awk -v pg="$mypg" '$4 ~ /day42_machine|day42_fold/ && $2 != pg {print $1}'
}
