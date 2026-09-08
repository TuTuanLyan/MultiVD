#!/usr/bin/env bash
# ASAM_AW — cau hoi CHUA AI DO: truc rho do duoc (FACTS §21) nam TREN NEN RecAdam,
# nhung RecAdam da null o moi gamma (§20 + RESEARCH §10-12) nen phuong phap chot se
# dung AdamW. ASAM rho=2.0 co con giup khi bo neo di khong?
#
# Neu CO: cau hinh chot la AdamW + head + ASAM rho=2.0, va ASAM la dong gop doc lap.
# Neu KHONG: loi ich cua rho=2.0 la mot TUONG TAC voi neo RecAdam — van dang viet,
#            nhung phai phat bieu khac han.
# Khong do thi khong chot duoc cau hinh nao ca. Day la o quyet dinh.
#
# BAC 1 (sang loc, n=3 fold, seed 42). Tot moi len n=5.
# Doi chung `aw_r0` chay CUNG MAY CUNG PHIEN (CLAUDE.md muc 4) — khong so voi r0 cua
# khoi asam1 vi khoi do dung RecAdam va chay tren may khac.
#
# Tag co tien to `aw_` co chu dich: matrix.sh them hau to `_adamw` khi optimizer khac
# recadam, ma report2.py lai CAT hau to do khi doc -> `r0` cua adamw se DUNG TEN voi
# `r0` cua recadam neu hai khoi roi vao cung cay. Tien to `aw_` chan va cham do.
#
#   FOLD_LIST="1 2 3" bash run/asam_aw.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
SOURCES_LIST="${SOURCES_LIST:-4cwe com full}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
SEED="${SEED:-42}"
echo "########## ASAM_AW bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  ASAM rho=2.0 tren nen AdamW (khong neo). Doi chung aw_r0 cung may cung phien."
echo "  chi so CHINH khai bao truoc: ROC-AUC. nguon: $SOURCES_LIST | fold: $FOLD_LIST"
for FOLD in $FOLD_LIST; do
  for SRC in $SOURCES_LIST; do
    echo "===== $(date -u '+%F %T') | fold $FOLD | nguon $SRC ====="
    RUN=asamaw SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 \
    CONFIGS="aw_r0|adamw|--sam_rho 0
aw_r2p0|adamw|--sam_rho 2.0 --sam_variant asam" \
    bash run/opt1.sh
  done
done
echo "########## ASAM_AW xong $(date -u '+%F %T') ##########"
