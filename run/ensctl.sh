#!/usr/bin/env bash
# ENSCTL — DOI CHUNG cho phat hien "tron baseline + chuyen giao" (FACTS §25).
#
# CAU HOI CHUA BIET: ban tron vuot CA HAI dau mut (ROC +0.0165 so voi chuyen giao
# thuan, 59/83, p=0.0002). Phan bien hien nhien: "tron hai mo hinh bat ky cung loi,
# do la trung binh hoa phuong sai chu khong phai chuyen giao". Neu dung vay thi tron
# BASELINE(seed 42) voi BASELINE(seed 7) — cung kien truc, cung du lieu, chi khac seed
# — phai cho DUNG cai loi do. Neu no cho ~0 thi loi ich den tu Pha 1, khong tu phep tron.
#
# KHONG DO OFFLINE DUOC: 21 cap baseline da-seed duy nhat trong kho la cua khoi 47,
# chay TRUOC khi luu xac suat tung mau -> khong co `test_probabilities`. Phai chay lai.
#
# RE: chi baseline, khong Pha 1, khong Pha 2 chuyen giao. ~5 phut/o tren t5p.
# Ghi vao DUNG cay `asamaw_t5p` da co baseline seed 42 + moi nhanh chuyen giao,
# nen phep tron ghep cap trong CUNG may CUNG cay CUNG fold (CLAUDE.md muc 4).
#
# BAY: matrix.sh dung `MODES="${MODES:-...}"` (co hai cham) nen MODES="" bi thay bang
# mac dinh 4 nhanh. Phai truyen MOT DAU CACH: no khac rong nen sang duoc cong `:-`,
# roi `for MODE in $MODES` tach thanh KHONG tu nao. Da kiem hai chieu bang stub dem.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
RUN="${RUN:-asamaw}"
SEEDS="${SEEDS:-7 1234}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
# BB PHAI TRUNG KHIT voi BB da sinh ra baseline seed 42 trong cay asamaw_t5p, neu khong
# Delta khong ghep cap duoc (CLAUDE.md muc 4). Doc tu chinh o da co:
#   model_name = Salesforce/codet5p-220m-bimodal , pooling = mean
# BAY DA MAC 09/09 04:21: dat ':enc' -> train_baseline.py chi nhan (cls, mean) nen no
# tu choi ngay, ca ba fold hong, muc ghi `done` voi 0 o va ntat2 nam khong. Cong dem
# hien vat da bat duoc va ghi vao worklist.noop — nhung phai kiem THAM SO chu khong chi
# kiem file co ton tai.
BB="${BB:-t5p=Salesforce/codet5p-220m-bimodal:mean}"
echo "########## ENSCTL bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  chi BASELINE, seed $SEEDS, fold $FOLD_LIST, cay ${RUN}_t5p"
for FOLD in $FOLD_LIST; do
  for S in $SEEDS; do
    echo "===== $(date -u '+%F %T') | fold $FOLD | seed $S | chi baseline ====="
    RUN_NAME="$RUN" SEED="$S" FOLDS="$FOLD" \
    BACKBONES="$BB" MODES=" " OPTIMIZERS="adamw" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PYTHON" bash run/matrix.sh 8>&-
  done
done
echo "########## ENSCTL xong $(date -u '+%F %T') ##########"
