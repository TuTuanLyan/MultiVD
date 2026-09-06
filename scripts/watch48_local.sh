#!/usr/bin/env bash
# Giam sat NIGHT48 tren HAI MAY LOCAL. Goi tu cron moi 10 phut:
#   */10 * * * * /usr/bin/flock -n -o /tmp/mvd_w48l.lock /bin/bash .../scripts/watch48_local.sh
#
#   161 -> seed 7      lock /tmp/multivd_night48.lock       ket qua tai cho: results/n48_t5p
#   158 -> seed 1234   lock /tmp/multivd_night48_ntat.lock  keo ve results_night48_158/
#
# KHONG BAO GIO huy hay giet gi o day. Hai may nay dung chung voi nguoi khac: chi
# phong lai job CUA MINH khi lock da nha, va chi ghi nhan khi may ban.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

PY161=/home/ntat/miniconda3/envs/vdenv/bin/python
PY158=/data/ntat/envs/vdenv/bin/python
H158=tranmanhcuong@112.137.129.158; R158=/data/ntat/MultiVD
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i /home/ntat/.ssh/id_ed25519"
LOG=log/watch48_local.log; mkdir -p log results_night48_158
say(){ echo "$(date -u '+%F %T') $*" >> "$LOG"; }

# ---------------- 161, seed 7 ----------------
L161=/tmp/multivd_night48.lock
BUSY=$(flock -n "$L161" -c true 2>/dev/null && echo 0 || echo 1)
FIN=$(awk '/NIGHT48 bat dau/{c=0} /NIGHT48 xong/{c++} END{print c+0}' log/night48_161.log 2>/dev/null); FIN="${FIN:-0}"
N=$(ls results/n48_t5p/transfer_latent_bottleneck_*_l0p05_r*/seed_7/fold*.json 2>/dev/null | wc -l)
B=$(ls results/n48_t5p/baseline/seed_7/fold*.json 2>/dev/null | wc -l)
CK=$(ls model/n48/phase1/*/seed_7/best.pt 2>/dev/null | wc -l)
FREE=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) ))
say "[161 seed7] $N/30 o + $B/5 baseline | $CK/3 ckpt | driver=$BUSY xong=$FIN | VRAM trong ${FREE}MB"
# Hai truong hop deu phai phong lai, va deu phai NHUONG khi may ban:
#   FIN=0 -> driver chet giua chung
#   FIN>=1 nhung N<30 -> driver da in "xong" nhung thieu o. 05/09 chuyen nay xay ra
#     that: cuongtm chiem 5,6 GB VRAM tren 161, con trong ~10 GB < 12,6 GB job t5p can,
#     nen 16 o cua seed 7 OOM roi driver van in "xong" voi 14/30. Neu chi bam vao dong
#     "xong" thi 16 o do thanh o TRONG vinh vien.
# matrix.sh bo qua o da co nen phong lai chi chay phan thieu — khong ton gi.
PASSF161=log/n48_161_passes; [[ -f "$PASSF161" ]] || echo 0 > "$PASSF161"
P161=$(cat "$PASSF161" 2>/dev/null); P161="${P161:-0}"
NEED161=30
RELAUNCH=0
(( BUSY == 0 )) && (( FIN == 0 )) && RELAUNCH=1
(( BUSY == 0 )) && (( FIN >= 1 )) && (( N < NEED161 )) && (( P161 < 12 )) && RELAUNCH=2
if (( RELAUNCH > 0 )); then
  if (( FREE < 13000 )); then
    say "[161] can phong lai (ly do $RELAUNCH) nhung VRAM trong ${FREE}MB — NHUONG nguoi khac, thu lai vong sau"
  else
    (( RELAUNCH == 2 )) && { echo $(( P161 + 1 )) > "$PASSF161"; say "[161] driver in xong nhung moi $N/$NEED161 o — lap lo, lan $(( P161 + 1 ))/12"; }
    (( RELAUNCH == 1 )) && say "[161] driver CHET ma chua in dong ket thuc — phong lai"
    setsid nohup nice -n 5 env SEEDS='7' PYTHON="$PY161" MVD_LOCK="$L161" \
      bash run/night48.sh >> log/night48_161.log 2>&1 </dev/null & disown
  fi
fi

# ---------------- 158, seed 1234 ----------------
L158=/tmp/multivd_night48_ntat.lock
OUT=$(timeout 60 $SSH $H158 "
  echo BUSY=\$(flock -n $L158 -c true 2>/dev/null && echo 0 || echo 1)
  echo FIN=\$(awk '/NIGHT48 bat dau/{c=0} /NIGHT48 xong/{c++} END{print c+0}' $R158/log/night48_158.log 2>/dev/null)
  echo N=\$(ls $R158/results/n48_t5p/transfer_latent_bottleneck_*_l0p05_r*/seed_1234/fold*.json 2>/dev/null | wc -l)
  echo B=\$(ls $R158/results/n48_t5p/baseline/seed_1234/fold*.json 2>/dev/null | wc -l)
  echo CK=\$(ls $R158/model/n48/phase1/*/seed_1234/best.pt 2>/dev/null | wc -l)
  echo FREE=\$(( \$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) - \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) ))
  echo DISK=\$(df -P /data | tail -1 | awk '{print \$4}')
" 2>/dev/null)
if [[ -z "$OUT" ]]; then
  say "[158] khong SSH duoc — thu lai vong sau"
else
  eval "$OUT"
  say "[158 seed1234] ${N:-?}/30 o + ${B:-?}/5 baseline | ${CK:-?}/3 ckpt | driver=${BUSY:-?} xong=${FIN:-?} | VRAM trong ${FREE:-?}MB | dia $(( ${DISK:-0}/1048576 ))GB"
  rsync -az -e "$SSH" --include='n48_*/' --include='n48_*/**' --exclude='*' \
        "$H158:$R158/results/" results_night48_158/ 2>/dev/null
  PASSF158=log/n48_158_passes; [[ -f "$PASSF158" ]] || echo 0 > "$PASSF158"
  P158=$(cat "$PASSF158" 2>/dev/null); P158="${P158:-0}"
  R8=0
  (( ${BUSY:-1} == 0 )) && (( ${FIN:-0} == 0 )) && R8=1
  (( ${BUSY:-1} == 0 )) && (( ${FIN:-0} >= 1 )) && (( ${N:-0} < 30 )) && (( P158 < 12 )) && R8=2
  if (( R8 > 0 )); then
    if (( ${FREE:-0} < 13000 )); then
      say "[158] can phong lai (ly do $R8) nhung VRAM trong ${FREE}MB — NHUONG nguoi khac"
    else
      (( R8 == 2 )) && { echo $(( P158 + 1 )) > "$PASSF158"; say "[158] driver in xong nhung moi ${N}/30 o — lap lo, lan $(( P158 + 1 ))/12"; }
      (( R8 == 1 )) && say "[158] driver CHET ma chua in dong ket thuc — phong lai"
      timeout 30 $SSH $H158 "cd $R158 && SEEDS='1234' PYTHON=$PY158 MVD_LOCK=$L158 setsid nohup nice -n 5 bash run/night48.sh >> log/night48_158.log 2>&1 </dev/null & disown" 2>/dev/null
    fi
  fi
fi
