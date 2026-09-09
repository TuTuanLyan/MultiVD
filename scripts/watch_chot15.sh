#!/usr/bin/env bash
# Cron 10 phut. Quan HAI MAY LOCAL cho khoi bo sung n=15 (seed 7 + 1234).
#   161 -> codebert rho 0.1   (results/chot161_codebert)
#   158 -> t5p      rho 2.0   (results/chot158_t5p)
# Moi may 70 o = 2 seed x 5 fold x (1 baseline + 3 nguon x 2 cau hinh).
#
# KHONG giet gi. Hoi LOCK chu khong dem tien trinh (ps|grep bat ca dong lenh cua chinh no).
# Vast: khong dung may nao — neu can thue thi nguoi dung quyet.
export HOME="${HOME:-/home/ntat}"; export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/home/ntat/miniconda3/envs/vdenv/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD; cd "$ROOT" || exit 1
LOG=log/watch_chot15.log
ts(){ date -u '+%F %T'; }
H=tranmanhcuong@112.137.129.158
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15"
CB='codebert=microsoft/codebert-base:cls'
T5='t5p=Salesforce/codet5p-220m-bimodal:mean'
A_CB='r0p1|recadam|--sam_rho 0.1 --sam_variant asam'
A_T5='r2p0|recadam|--sam_rho 2.0 --sam_variant asam'
B='plain|adamw|--sam_rho 0'
NEED=70

# ---- 161 / codebert ----
n=$(find results/chot161_codebert -name 'fold*.json' 2>/dev/null | wc -l)
if flock -n /tmp/mvd_chot15.lock -c true 2>/dev/null; then a=0; else a=1; fi
echo "$(ts) | 161/codebert o=$n/$NEED alive=$a" >> "$LOG"
if (( a == 0 )) && (( n < NEED )); then
  echo "$(ts) | 161 | driver chet o $n/$NEED — phong lai" >> "$LOG"
  RUN=chot161 BB="$CB" CFG_A="$A_CB" CFG_B="$B" SEED_LIST="7 1234" \
    setsid nohup bash run/chot15.sh >> log/chot15_161.log 2>&1 </dev/null & disown
fi

# ---- 158 / t5p ----
out=$(timeout 60 $SSH $H "cd /data/ntat/MultiVD 2>/dev/null || exit 1
echo n=\$(find results/chot158_t5p -name 'fold*.json' 2>/dev/null | wc -l)
if flock -n /tmp/mvd_chot15.lock -c true 2>/dev/null; then echo a=0; else echo a=1; fi" 2>/dev/null)
if [[ -z "$out" ]]; then
  echo "$(ts) | 158 | KHONG SSH DUOC — KHONG ket luan may chet" >> "$LOG"; exit 0
fi
m=$(sed -n 's/^n=//p' <<<"$out"); b=$(sed -n 's/^a=//p' <<<"$out")
echo "$(ts) | 158/t5p      o=$m/$NEED alive=$b" >> "$LOG"
if (( ${b:-1} == 0 )) && (( ${m:-0} < NEED )); then
  echo "$(ts) | 158 | driver chet o $m/$NEED — phong lai" >> "$LOG"
  timeout 40 $SSH $H "cd /data/ntat/MultiVD && RUN=chot158 BB='$T5' CFG_A='$A_T5' CFG_B='$B' \
    SEED_LIST='7 1234' PYTHON=/data/ntat/envs/vdenv/bin/python \
    setsid nohup bash run/chot15.sh >> log/chot15_158.log 2>&1 </dev/null & disown" >/dev/null 2>&1
fi
