#!/usr/bin/env bash
# Cron 10 phut. Quan HAI may cho khoi `chot`:
#   161            -> backbone t5p
#   vast ntat 5060 -> backbone codebert   (mot backbone tron mot may, CLAUDE.md muc 4)
# Moi may: GD1 = 4cwe+com (25 o = 5 fold x (1 baseline + 2 nguon x 2 cau hinh)),
#          GD2 = full     (10 o = 5 fold x 2 cau hinh; baseline da co).
# Xong GD1 thi TU phong GD2 cho chinh may do — hai may xong lech nhau nen khong dung
# mot script chuoi chung.
# KHONG giet gi. Hoi LOCK chu khong dem tien trinh (ps|grep bat ca dong lenh cua chinh no).
export HOME="${HOME:-/home/ntat}"; export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/home/ntat/miniconda3/envs/vdenv/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD; cd "$ROOT" || exit 1
LOG=log/watch_chot.log
ts(){ date -u '+%F %T'; }
VE="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p 47566"
VH=root@171.227.33.18
T5P='t5p=Salesforce/codet5p-220m-bimodal:mean'
CB='codebert=microsoft/codebert-base:cls'

# ---- 161 / t5p ----
n=$(find results/chot_t5p -name 'fold*.json' 2>/dev/null | wc -l)
if flock -n /tmp/mvd_chot2bb.lock -c true 2>/dev/null; then alive=0; else alive=1; fi
echo "$(ts) | 161/t5p  o=$n alive=$alive" >> "$LOG"
if (( alive == 0 )); then
  if   (( n < 25 )); then
    echo "$(ts) | 161 | GD1 con thieu ($n/25) — phong lai" >> "$LOG"
    BBS="$T5P" SOURCES_LIST="4cwe com" setsid nohup bash run/chot2bb.sh >> log/chot_t5p.log 2>&1 </dev/null & disown
  elif (( n < 35 )); then
    echo "$(ts) | 161 | GD1 xong, sang GD2 full ($n/35)" >> "$LOG"
    BBS="$T5P" SOURCES_LIST="full" setsid nohup bash run/chot2bb.sh >> log/chot_t5p.log 2>&1 </dev/null & disown
  else
    echo "$(ts) | 161 | DU 35 o — xong" >> "$LOG"
  fi
fi

# ---- vast / codebert ----
out=$(timeout 60 $VE $VH "cd /workspace/MultiVD 2>/dev/null || exit 1
echo n=\$(find results/chot_codebert -name 'fold*.json' 2>/dev/null | wc -l)
if flock -n /tmp/mvd_chot2bb.lock -c true 2>/dev/null; then echo alive=0; else echo alive=1; fi" 2>/dev/null)
if [[ -z "$out" ]]; then
  echo "$(ts) | vast | KHONG SSH DUOC — KHONG ket luan may chet" >> "$LOG"; exit 0
fi
vn=$(sed -n 's/^n=//p' <<<"$out"); va=$(sed -n 's/^alive=//p' <<<"$out")
echo "$(ts) | vast/cb o=$vn alive=$va" >> "$LOG"
(( ${va:-1} == 1 )) && exit 0
if   (( ${vn:-0} < 25 )); then
  echo "$(ts) | vast | GD1 con thieu ($vn/25) — phong lai" >> "$LOG"
  timeout 40 $VE $VH "cd /workspace/MultiVD && BBS='$CB' SOURCES_LIST='4cwe com' PYTHON=/venv/main/bin/python setsid nohup bash run/chot2bb.sh >> log/chot_cb.log 2>&1 </dev/null & disown" >/dev/null 2>&1
elif (( ${vn:-0} < 35 )); then
  echo "$(ts) | vast | GD1 xong, sang GD2 full ($vn/35)" >> "$LOG"
  timeout 40 $VE $VH "cd /workspace/MultiVD && BBS='$CB' SOURCES_LIST='full' PYTHON=/venv/main/bin/python setsid nohup bash run/chot2bb.sh >> log/chot_cb.log 2>&1 </dev/null & disown" >/dev/null 2>&1
else
  echo "$(ts) | vast | DU 35 o — xong" >> "$LOG"
fi
