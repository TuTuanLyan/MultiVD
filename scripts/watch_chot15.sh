#!/usr/bin/env bash
# Cron 10 phut. Quan HAI MAY LOCAL cho khoi bo sung n=15 (seed 7 + 1234).
#   161 -> codebert rho 0.1  (results/chot161_codebert)   70 o
#   158 -> t5p      rho 2.0  (results/chot158_t5p)        70 o
#
# --- LUAT DEM 10/09 (nguoi dung dat) ---
# * BI CHIEM GPU (nguoi khac vao truoc luc chuyen pha/fold) -> BAO NGAY, khong cho.
#   Nguoi dung 10/09: "Bi chiem thi van thue vast ngay". GRACE = 0.
# * 158 chi DISCONNECT / TU KHOI DONG LAI / REBOOT -> CHO no len roi chay lai, KHONG thue vast.
#   Han cho: 20 PHUT. Duoi 20 phut la "dang len lai", khong phai vo.
# * Van tiep tuc go cua may da mat: khi no ranh lai thi nhay vao chay tiep va chia lai viec.
# * TOI DA 3 GPU cung luc = 161 + 158 + DUNG MOT vast. Vo ca hai cung chi MOT vast.
# * Script nay KHONG tu thue: no chi dung co VO de monitor bao ve. Viec thue lam tay, vi phai
#   kiem gia va cai dat, va vi dung bo phi vast la bi phat.
export HOME="${HOME:-/home/ntat}"; export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/home/ntat/miniconda3/envs/vdenv/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD; cd "$ROOT" || exit 1
LOG=log/watch_chot15.log
ST=log/chot15_state              # "may:tinh_trang:tu_luc" moi dong
ts(){ date -u '+%F %T'; }
now(){ date -u +%s; }
H=tranmanhcuong@112.137.129.158
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15"
CB='codebert=microsoft/codebert-base:cls'
T5='t5p=Salesforce/codet5p-220m-bimodal:mean'
A_CB='r0p1|recadam|--sam_rho 0.1 --sam_variant asam'
A_T5='r2p0|recadam|--sam_rho 2.0 --sam_variant asam'
B='plain|adamw|--sam_rho 0'
NEED=70
# Hai loai su co, hai han khac nhau:
GRACE_SSH=1200                   # 158 mat ket noi / reboot: cho 20 phut roi moi coi la vo
GRACE_GPU=0                      # bi chiem GPU: bao NGAY, khong cho
grace_of(){ case "$1" in gpu-bi-chiem) echo $GRACE_GPU;; *) echo $GRACE_SSH;; esac; }

touch "$ST"
get_since(){ awk -F: -v m="$1" '$1==m{print $3}' "$ST" | tail -1; }
set_state(){ # may tinh_trang
  local m=$1 s=$2 old_s old_t t
  old_s=$(awk -F: -v m="$m" '$1==m{print $2}' "$ST" | tail -1)
  old_t=$(get_since "$m")
  if [[ "$s" == "ok" ]]; then grep -v "^$m:" "$ST" > "$ST.t" 2>/dev/null; mv "$ST.t" "$ST"; return; fi
  if [[ "$old_s" == "$s" && -n "$old_t" ]]; then t="$old_t"; else t=$(now); fi
  grep -v "^$m:" "$ST" > "$ST.t" 2>/dev/null; mv "$ST.t" "$ST"
  echo "$m:$s:$t" >> "$ST"
  local g mins=$(( ( $(now) - t ) / 60 ))
  g=$(grace_of "$s")
  if (( $(now) - t >= g )); then
    echo "$(ts) | !! $m VO ($s, $mins phut) — DUOC PHEP THUE MOT VAST A4000 (toi da 3 GPU)" >> "$LOG"
  else
    echo "$(ts) | $m: $s da $mins phut (con $(( (g - ($(now)-t)) / 60 )) phut truoc khi coi la vo)" >> "$LOG"
  fi
}

# ---------- 161 ----------
n=$(find results/chot161_codebert -name 'fold*.json' 2>/dev/null | wc -l)
if flock -n /tmp/mvd_chot15.lock -c true 2>/dev/null; then alive=0; else alive=1; fi
train=$(ps -u "$(id -un)" -o comm=,args= | grep -c 'train_\(transfer\|baseline\)\.py' || true)
free161=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1) \
          - $(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1) ))
echo "$(ts) | 161/codebert o=$n/$NEED alive=$alive train=$train freeVRAM=${free161}MiB" >> "$LOG"
if (( n >= NEED )); then
  set_state 161 ok
elif (( alive == 0 )); then
  echo "$(ts) | 161 | driver chet o $n/$NEED — phong lai" >> "$LOG"
  set_state 161 ok
  RUN=chot161 BB="$CB" CFG_A="$A_CB" CFG_B="$B" SEED_LIST="7 1234" \
    setsid nohup bash run/chot15.sh >> log/chot15_161.log 2>&1 </dev/null & disown
elif (( train == 0 && free161 < 8500 )); then
  set_state 161 gpu-bi-chiem          # driver song nhung ngoi cho VRAM -> bao NGAY
else
  set_state 161 ok
fi

# ---------- 158 ----------
out=$(timeout 60 $SSH $H "cd /data/ntat/MultiVD 2>/dev/null || exit 1
echo n=\$(find results/chot158_t5p -name 'fold*.json' 2>/dev/null | wc -l)
echo t=\$(ps -u \$(id -un) -o args= | grep -c 'train_\(transfer\|baseline\)\.py')
echo f=\$(( \$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1) - \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1) ))
if flock -n /tmp/mvd_chot15.lock -c true 2>/dev/null; then echo a=0; else echo a=1; fi" 2>/dev/null)
if [[ -z "$out" ]]; then
  echo "$(ts) | 158 | KHONG SSH DUOC" >> "$LOG"
  set_state 158 khong-ssh-duoc       # co the chi la reboot — cho het GRACE moi coi la vo
  exit 0
fi
m=$(sed -n 's/^n=//p' <<<"$out"); b=$(sed -n 's/^a=//p' <<<"$out")
tt=$(sed -n 's/^t=//p' <<<"$out"); ff=$(sed -n 's/^f=//p' <<<"$out")
echo "$(ts) | 158/t5p      o=$m/$NEED alive=$b train=$tt freeVRAM=${ff}MiB" >> "$LOG"
if (( ${m:-0} >= NEED )); then
  set_state 158 ok
elif (( ${b:-1} == 0 )); then
  echo "$(ts) | 158 | driver chet o $m/$NEED — phong lai" >> "$LOG"
  set_state 158 ok
  timeout 40 $SSH $H "cd /data/ntat/MultiVD && RUN=chot158 BB='$T5' CFG_A='$A_T5' CFG_B='$B' \
    SEED_LIST='7 1234' PYTHON=/data/ntat/envs/vdenv/bin/python \
    setsid nohup bash run/chot15.sh >> log/chot15_158.log 2>&1 </dev/null & disown" >/dev/null 2>&1
elif (( ${tt:-0} == 0 && ${ff:-99999} < 13000 )); then
  set_state 158 gpu-bi-chiem
else
  set_state 158 ok
fi
