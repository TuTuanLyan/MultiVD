#!/bin/bash
# Làm mới bản ghi tiến độ (meta/dbrows/<run>__f<F>.json) cho fold đang chạy ở 161 (local) và các máy từ xa của khối mwonly5.
cd "$(dirname "$0")/.." || exit 1
PY=/home/ntat/miniconda3/envs/vdenv/bin/python
read -r RUN F < <(awk '/ bắt đầu$/ {r=$3; f=$4} / xong rc=/ {r=""} END {print r, substr(f,2)}' state/driver_161.log 2>/dev/null)
[ -n "${RUN:-}" ] && $PY scripts/fpe.py live "$RUN" "$F" "logs/$RUN/fold$F.log" 161
# 05/10 13:1x: paper_mw, paper_mw2 đã destroy
for h in 158 paper_night; do bash scripts/live_remote.sh $h; done   # 07/10 03:3x thêm paper_night (54537908); 06/10 23:00: paper_mw4 (54468293) destroy; 17:5x thêm paper_mw4; paper_mw3 destroy 14:2x (05/10 17:0x thêm cho task SVEN → JS)
# bỏ trường null (epochs/eta/best chưa có) để lệnh update của artifact không đè giá trị gieo sẵn (epochs 16/30, planned_host)
$PY - <<'PYEOF'
import glob, json
for p in glob.glob("meta/dbrows/*.json"):
    d = json.load(open(p, encoding="utf-8"))
    d = {k: v for k, v in d.items() if not (k in ("epochs", "eta_min", "eta_max", "best_epoch", "best_val_roc") and v is None)}
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False)
PYEOF
