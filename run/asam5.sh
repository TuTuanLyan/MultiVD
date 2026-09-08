#!/usr/bin/env bash
# ASAM5 — truc rho, muc 8.0. Chi so CHINH khai bao truoc: ROC-AUC.
# Ghi vao CUNG cay `asam1_t5p` nen doi chung rho=0 ghep cap duoc ngay.
#
# CAU HOI: duong cong rho da co DINH NOI TAI chua? Do duoc 08/09 (ghep cap tung fold,
# doi chung rho=0, 15 o = 5 fold x 3 nguon, t5p/latent_bottleneck/lambda=0.05):
#   rho   0.1    0.2    0.5    1.0    2.0    4.0
#   PR   -0.0013 +0.0020 +0.0085 +0.0159 +0.0155 -0.1177(n=8)
#   ROC  -0.0080 -0.0024 +0.0033 +0.0065 +0.0088 -0.1188(n=8)
# Len den 2.0 roi TUT MANH o 4.0 -> co dinh trong khoang 1.0-2.0. rho=8.0 de xac nhan
# nhanh giam la that chu khong phai mot o ngoai le cua 4.0.
#
# BAY DA MAC 08/09 (ban cu cua chinh file nay): dong echo co dau nhay LECH
#   echo "  truc rho: 8.0 — ... thi "cang lon cang tot" va phai noi ro dieu do
# Dau " thu hai DONG chuoi, phan sau thanh lenh, va dau " tiep theo o DONG SAU lai mo
# chuoi moi -> ca vong `for` bi nuot. `bash -n` van bao OK vi mot dau nhay o duoi can lai.
# Chay that thi `recadam: command not found`, `CONFIGS: unbound variable`, 0 o sinh ra,
# nhung driver van ghi vao worklist.done. Bai hoc: bash -n KHONG du — phai chay thu voi
# loi goi huan luyen thay bang echo.
#
# r0 nam trong CONFIGS co CHU DICH: matrix.sh bo qua o da co (dong 371), nen tren may
# da co r0 o fold 1/2/4 thi day chi chay bu fold 3/5 — dung 6 o, va no la thu MO KHOA
# phep ghep cap cho ca r4p0 lan r8p0 tren may nay (CLAUDE.md muc 4: doi chung phai
# cung may). Khong co no thi 4 o r4p0 dang bi bo roi.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( [ -x /venv/main/bin/python ] && echo /venv/main/bin/python \
  || echo /home/ntat/miniconda3/envs/vdenv/bin/python )}"
export PYTHON
SOURCES_LIST="${SOURCES_LIST:-4cwe com}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
SEED="${SEED:-42}"
echo "########## ASAM5 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  truc rho: 8.0 + bu doi chung r0 cac fold con thieu"
echo "  nguon: $SOURCES_LIST | fold: $FOLD_LIST | seed: $SEED"
for FOLD in $FOLD_LIST; do
  for SRC in $SOURCES_LIST; do
    echo "===== $(date -u '+%F %T') | fold $FOLD | nguon $SRC ====="
    RUN=asam1 SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 \
    CONFIGS="r0|recadam|--sam_rho 0
r8p0|recadam|--sam_rho 8.0 --sam_variant asam" \
    bash run/opt1.sh
  done
done
echo "########## ASAM5 xong $(date -u '+%F %T') ##########"
