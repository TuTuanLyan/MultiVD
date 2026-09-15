#!/usr/bin/env bash
# DAY CHUYEN DEM tren 158 — nguoi dung duyet 15/09:
#   gate1 n=3  ->  (dat nguong da ghi truoc)  ->  gate1 fold 4,5  ->  ctx 256/512/1024
# Khong dat nguong thi BO QUA n=5, di thang sang ctx. Het viec thi DUNG (158 la may cua
# truong, khong tinh tien, nen KHONG huy gi — khac voi may thue).
#
# KHONG kill bat cu thu gi. Cho bang flock + PID file cua chinh minh.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H=tranmanhcuong@112.137.129.158; R=/data/ntat/MultiVD
PY=/data/ntat/envs/vdenv/bin/python
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25"
LOCALPY=/home/ntat/miniconda3/envs/vdenv/bin/python
log(){ echo "$(date -u '+%F %T') | $*"; }

wait_idle() {   # $1 = ten lock, $2 = so vong toi da (90s/vong)
  local lk="$1" n="${2:-160}"
  for i in $(seq 1 "$n"); do
    if $SSH $H "flock -n /tmp/$lk -c true" 2>/dev/null; then return 0; fi
    sleep 90
  done
  log "!! qua han cho $lk"; return 1
}
pull() {
  rsync -az -e "$SSH" "$H:$R/results/" results/ 2>/dev/null
  log "keo ve: $(find results/gate1_* results/ctx*_t5p -name 'fold*.json' 2>/dev/null | wc -l) o"
}
gpu_free() { $SSH $H "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" 2>/dev/null | tr -d ' '; }

log "=== cho gate1 n=3 xong ==="
wait_idle mvd_gate1.lock 160 || exit 1
pull
log "=== quyet dinh leo bac ==="
$LOCALPY tools/gate1_decide.py results/gate1_codebert results/gate1_t5p; RC=$?
log "gate1_decide ma thoat = $RC  (0=leo n=5, 1=dung, 2=thieu du lieu)"

if [ "$RC" = 0 ]; then
  F=$(gpu_free); log "GPU trong ${F} MiB"
  if [ "${F:-0}" -ge 9000 ]; then
    log "=== phong gate1 fold 4,5 ==="
    $SSH $H "cd $R && FOLDS='4 5' PYTHON=$PY PIDFILE=/data/ntat/gate1.pid \
      setsid nohup bash run/gate1.sh >> log/gate1.log 2>&1 </dev/null & disown"
    sleep 20; wait_idle mvd_gate1.lock 120 || true; pull
    $LOCALPY tools/gate1_decide.py results/gate1_codebert results/gate1_t5p || true
  else
    log "!! GPU chi con ${F} MiB — bo qua n=5 de khong lam hong job cua tranmanhcuong"
  fi
else
  log "khong dat nguong bac 1 -> BO QUA n=5, di thang sang ctx"
fi

log "=== khoi ctx (truc ngu canh) ==="
rsync -az --delete -e "$SSH" run/ "$H:$R/run/" 2>/dev/null
F=$(gpu_free); log "GPU trong ${F} MiB"
if [ "${F:-0}" -ge 9000 ]; then
  $SSH $H "cd $R && FOLDS='1 2 3' PYTHON=$PY PIDFILE=/data/ntat/ctx.pid \
    setsid nohup bash run/ctx.sh >> log/ctx.log 2>&1 </dev/null & disown"
  sleep 20; wait_idle mvd_ctx.lock 120 || true; pull
else
  log "!! GPU chi con ${F} MiB — KHONG phong ctx"
fi
log "=== DAY CHUYEN DEM KET THUC ==="
