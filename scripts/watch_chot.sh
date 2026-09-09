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
# Nhanh A = TOT NHAT CUA CHINH BACKBONE DO (nguoi dung: "chay 2 backbone voi cai tot nhat
# cua no"). rho la sieu tham so Pha 2 nen khac nhau giua hai backbone — FACTS §32:
#   t5p      rho 2.0 -> dROC +0.0352 (52/57)
#   codebert rho 0.1 -> dROC +0.0244 (40/40);  rho 2.0 cho -0.0439 (6/12)
A_T5P='r2p0|recadam|--sam_rho 2.0 --sam_variant asam'
A_CB='r0p1|recadam|--sam_rho 0.1 --sam_variant asam'

# ---- 161 / t5p, CHI fold 1 2 3 ----
# GD1 = 3 fold x 5 o = 15 ; GD2 (`full`) = +3 fold x 2 o = 21
n=$(find results/chot_t5p -name 'fold*.json' 2>/dev/null | wc -l)
if flock -n /tmp/mvd_chot2bb.lock -c true 2>/dev/null; then alive=0; else alive=1; fi
echo "$(ts) | 161/t5p  o=$n/28 (fold 1,2,3,5) alive=$alive" >> "$LOG"
if (( alive == 0 )); then
  if   (( n < 20 )); then
    echo "$(ts) | 161 | GD1 con thieu ($n/20) — phong lai" >> "$LOG"
    BBS="$T5P" CFG_A="$A_T5P" SOURCES_LIST="4cwe com" FOLD_LIST="1 2 3 5" \
      setsid nohup bash run/chot2bb.sh >> log/chot_t5p.log 2>&1 </dev/null & disown
  elif (( n < 28 )); then
    echo "$(ts) | 161 | GD1 xong, sang GD2 full ($n/28)" >> "$LOG"
    BBS="$T5P" CFG_A="$A_T5P" SOURCES_LIST="full" FOLD_LIST="1 2 3 5" \
      setsid nohup bash run/chot2bb.sh >> log/chot_t5p.log 2>&1 </dev/null & disown
  else
    echo "$(ts) | 161 | DU 28 o — xong phan cua 161" >> "$LOG"
  fi
fi

# ---- vast: DA HUY 09/09 15:29 UTC sau khi doi chieu byte + md5 (xem CURRENT_RUN.md).
# Khong con may nao de hoi. Giu khoi 161 o tren; khi 161 du 28 o thi watchdog nay het viec.
exit 0
