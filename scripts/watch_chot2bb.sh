#!/usr/bin/env bash
# Cron 10 phut. Phong lai run/chot2bb.sh neu no chet ma khoi CHUA du o.
# KHONG giet gi, KHONG xoa gi. Lock cua chinh chot2bb.sh chan chay hai chuoi mot GPU.
export HOME="${HOME:-/home/ntat}"; export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/home/ntat/miniconda3/envs/vdenv/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD; cd "$ROOT" || exit 1
LOG=log/watch_chot2bb.log
ts(){ date -u '+%F %T'; }
NEED=30                       # 2 backbone x (1 baseline + 2 nhanh) x 5 fold
have=$(find results/chot_t5p results/chot_codebert -name 'fold*.json' 2>/dev/null | wc -l)
# HOI LOCK, khong dem tien trinh: `ps|grep` bat luon dong lenh cua chinh watchdog nen
# lan dau no bao driver=4 trong khi chi co MOT. CLAUDE.md muc 8. chot2bb.sh giu
# /tmp/multivd_opt1.lock suot thoi gian chay, nen lay duoc lock = driver DA CHET.
if flock -n /tmp/multivd_opt1.lock -c true 2>/dev/null; then alive=0; else alive=1; fi
job=$(ps -eo comm=,args= | grep -cE '^python[0-9.]*[[:space:]].*src/train_')  # dem theo TEN CHUONG TRINH
echo "$(ts) | o=$have/$NEED driver=$alive job=$job" >> "$LOG"
(( have >= NEED )) && { echo "$(ts) | DU $NEED o — dung watchdog" >> "$LOG"; exit 0; }
(( alive > 0 )) && exit 0
# driver chet ma chua du o: phong lai. Lock cua chinh script chan trung lap.
echo "$(ts) | driver chet, con $((NEED-have)) o — PHONG LAI" >> "$LOG"
setsid nohup bash run/chot2bb.sh >> log/chot2bb.log 2>&1 </dev/null & disown
