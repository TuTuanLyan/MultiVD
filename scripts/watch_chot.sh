#!/usr/bin/env bash
# Cron 10 phut. Quan HAI may cho khoi `chot`, CHIA VIEC theo FOLD TRON VEN.
#
# Nguoi dung 09/09 17:20 VN: "Tan 4 task chay cho 2 backbone * 2 setting hon nua con 2
# source va 5 folds thi cu chia bot ma chay dung phi vast la duoc."
#
# Do that (phut/fold, ca hai giai doan):
#   vast 5060Ti  codebert  19.1 (GD1) + 16.2 (GD2)
#   161  A4000   t5p       61.9 (GD1) + 52.8 (GD2)   <-- nut co chai
# Neu 161 om ca 5 fold t5p thi no chay den 02:18 VN trong khi vast ranh tu 19:55 VN
# => 6.4 gio vast nam khong. Chia fold 4-5 sang vast thi ca hai cung xong ~22:00-22:30 VN.
#
# CLAUDE.md muc 4 cho phep dung mot cach chia duy nhat: chia theo FOLD TRON VEN. Baseline
# va MOI nhanh cua mot fold nam cung may, nen Delta ghep cap trong fold do van sach; khong
# bao gio lay hieu giua hai may. Cay ket qua cua vast dat ten KHAC (`chotv_t5p`) de
# `tools/chot_report.py` khong bao gio bac cau qua may — no khoa o theo (cay, bb, seed, fold).
#
# THU TU tren vast: codebert TRUOC cho het (nguoi dung: "uu tien xong backbone codebert
# truoc"), roi moi nhan fold 4-5 cua t5p.
#
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

# ---- 161 / t5p, CHI fold 1 2 3 ----
# GD1 = 3 fold x 5 o = 15 ; GD2 (`full`) = +3 fold x 2 o = 21
n=$(find results/chot_t5p -name 'fold*.json' 2>/dev/null | wc -l)
if flock -n /tmp/mvd_chot2bb.lock -c true 2>/dev/null; then alive=0; else alive=1; fi
echo "$(ts) | 161/t5p  o=$n/21 (fold 1-3) alive=$alive" >> "$LOG"
if (( alive == 0 )); then
  if   (( n < 15 )); then
    echo "$(ts) | 161 | GD1 con thieu ($n/15) — phong lai" >> "$LOG"
    BBS="$T5P" SOURCES_LIST="4cwe com" FOLD_LIST="1 2 3" \
      setsid nohup bash run/chot2bb.sh >> log/chot_t5p.log 2>&1 </dev/null & disown
  elif (( n < 21 )); then
    echo "$(ts) | 161 | GD1 xong, sang GD2 full ($n/21)" >> "$LOG"
    BBS="$T5P" SOURCES_LIST="full" FOLD_LIST="1 2 3" \
      setsid nohup bash run/chot2bb.sh >> log/chot_t5p.log 2>&1 </dev/null & disown
  else
    echo "$(ts) | 161 | DU 21 o — xong phan cua 161" >> "$LOG"
  fi
fi

# ---- vast: codebert (5 fold) TRUOC, roi t5p fold 4-5 ----
# codebert: GD1 25 -> GD2 35 ;  t5p fold 4-5: GD1 10 -> GD2 14  (cay `chotv_t5p`)
out=$(timeout 60 $VE $VH "cd /workspace/MultiVD 2>/dev/null || exit 1
echo cb=\$(find results/chot_codebert  -name 'fold*.json' 2>/dev/null | wc -l)
echo tp=\$(find results/chotv_t5p      -name 'fold*.json' 2>/dev/null | wc -l)
if flock -n /tmp/mvd_chot2bb.lock -c true 2>/dev/null; then echo alive=0; else echo alive=1; fi" 2>/dev/null)
if [[ -z "$out" ]]; then
  echo "$(ts) | vast | KHONG SSH DUOC — KHONG ket luan may chet" >> "$LOG"; exit 0
fi
vc=$(sed -n 's/^cb=//p' <<<"$out"); vt=$(sed -n 's/^tp=//p' <<<"$out"); va=$(sed -n 's/^alive=//p' <<<"$out")
echo "$(ts) | vast  codebert=$vc/35  t5p(f4-5)=$vt/14  alive=$va" >> "$LOG"
(( ${va:-1} == 1 )) && exit 0

launch(){ timeout 40 $VE $VH "cd /workspace/MultiVD && $1 PYTHON=/venv/main/bin/python \
  setsid nohup bash run/chot2bb.sh >> $2 2>&1 </dev/null & disown" >/dev/null 2>&1; }

if   (( ${vc:-0} < 25 )); then
  echo "$(ts) | vast | codebert GD1 con thieu ($vc/25) — phong lai" >> "$LOG"
  launch "BBS='$CB' SOURCES_LIST='4cwe com'" log/chot_cb.log
elif (( ${vc:-0} < 35 )); then
  echo "$(ts) | vast | codebert GD1 xong, sang GD2 full ($vc/35)" >> "$LOG"
  launch "BBS='$CB' SOURCES_LIST='full'" log/chot_cb.log
elif (( ${vt:-0} < 10 )); then
  echo "$(ts) | vast | codebert XONG — nhan t5p fold 4-5 GD1 ($vt/10)" >> "$LOG"
  launch "RUN=chotv BBS='$T5P' SOURCES_LIST='4cwe com' FOLD_LIST='4 5'" log/chotv_t5p.log
elif (( ${vt:-0} < 14 )); then
  echo "$(ts) | vast | t5p fold 4-5 GD1 xong, sang GD2 full ($vt/14)" >> "$LOG"
  launch "RUN=chotv BBS='$T5P' SOURCES_LIST='full' FOLD_LIST='4 5'" log/chotv_t5p.log
else
  echo "$(ts) | vast | DU 35 codebert + 14 t5p — HET VIEC, cho lenh huy" >> "$LOG"
fi
