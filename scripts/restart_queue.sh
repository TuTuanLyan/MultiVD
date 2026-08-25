#!/usr/bin/env bash
# Khoi dong lai hang doi VOI thu tu buoc moi. Chay TREN may vast.
# Bien vao: RUN_NAME BACKBONES STEPS DISK_FLOOR_GB
set -uo pipefail
cd /workspace/MultiVD
STATE="/workspace/overnight_${RUN_NAME}"
export PYTHON=/venv/main/bin/python HF_HOME=/workspace/hf
export SEED=42 FOLDS="1 2 3 4 5" EXTRA_SEEDS="" STATE
export RUN_NAME BACKBONES STEPS DISK_FLOOR_GB

# 1. Watchdog truoc — neu khong no se bat lai hang doi cu ngay khi ta giet.
#    Loc theo VI TRI, khong dung pkill -f: dong lenh SSH cua chinh phien nay
#    cung chua chuoi "watchdog", va pkill -f se giet luon phien do.
for pid in $(ps -eo pid,args --no-headers | awk '$2=="bash" && $3 ~ /watchdog/ {print $1}'); do
  kill -9 "$pid" 2>/dev/null && echo "  giet watchdog PID $pid"
done
# 2. Hang doi, qua chinh file khoa PID cua no.
if [[ -f "$STATE/queue.pid" ]]; then
  QP=$(cat "$STATE/queue.pid"); kill -9 "$QP" 2>/dev/null && echo "  giet hang doi PID $QP"
  rm -f "$STATE/queue.pid"
fi
for pid in $(ps -eo pid,args --no-headers | awk '$2=="bash" && $3 ~ /overnight/ {print $1}'); do
  kill -9 "$pid" 2>/dev/null && echo "  giet overnight con sot PID $pid"
done
# 3. Hai tang con: matrix.sh roi moi den python.
for pid in $(ps -eo pid,args --no-headers | awk '$2=="bash" && $3 ~ /run\/matrix\.sh/ {print $1}'); do
  kill -9 "$pid" 2>/dev/null && echo "  giet matrix.sh PID $pid"
done
for pid in $(pgrep -f 'src/train_transfer\.py|src/train_baseline\.py' 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null && echo "  giet job PID $pid"
done
sleep 5
rm -f "$STATE/ALL_DONE"

# 4. Ghi lai chinh xac moi truong da phong.
{ echo "RUN_NAME=$RUN_NAME"; echo "BACKBONES=$BACKBONES"; echo "STEPS=$STEPS"
  echo "DISK_FLOOR_GB=$DISK_FLOOR_GB"; echo "phong luc $(date -u '+%F %T') UTC"; } > "$STATE/launch.env"

nohup bash run/overnight.sh >> "/workspace/overnight_${RUN_NAME}.log" 2>&1 &
echo "  hang doi moi PID $!"
sleep 2
nohup bash scripts/watchdog.sh > /dev/null 2>&1 &
echo "  watchdog moi PID $!"
