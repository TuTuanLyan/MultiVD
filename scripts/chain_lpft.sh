#!/usr/bin/env bash
# Cho khoi feat3 tren MAY NAY xong han roi moi phong lpft3. Khong bao gio chay hai chuoi mot GPU.
# Dieu kien phong: driver feat3 KHONG con tien trinh nao. Neu feat3 chet giua chung thi van phong
# (khoi sau doc lai worklist rieng cua no; o thieu cua feat3 se hien ra khi doc bao cao).
cd "$1" || exit 1; shift
BBV="$1"; LOGF="$2"
for i in $(seq 1 240); do            # toi da 4 gio
  pgrep -f "bash run/feat3.sh" >/dev/null || break
  sleep 60
done
pgrep -f "bash run/feat3.sh" >/dev/null && { echo "feat3 van chay sau 4h — KHONG phong lpft3"; exit 1; }
sleep 30
exec env BB="$BBV" bash run/lpft3.sh >> "$LOGF" 2>&1
