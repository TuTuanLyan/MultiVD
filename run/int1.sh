#!/usr/bin/env bash
# INT1 — theta_Pha1 (+) theta_Pha2: mo hinh nguon co DONG GOP gi cho diem dich khong?
#
# VI SAO (RESEARCH §11): ba khoi lien tiep (OPT1, RET1, SPD1) deu noi cung mot dieu —
# cai NEO khong lam gi. Va §11.1 giai thich duoc vi sao: neo chi LAM CHAM viec roi khoi
# theta*, no khong mang them mot bit nao tu nguon vao dich. Toan bo tri thuc nguon di vao
# qua DIEM XUAT PHAT theta*.
#
# Khoi nay hoi thang cau hoi con lai: ngoai vai tro diem xuat phat, mo hinh nguon co con
# dong gop duoc gi nua khong. Tron  theta(alpha) = alpha*theta_Pha2 + (1-alpha)*theta_Pha1,
# CHON alpha tren VAL, bao test tai alpha do.
#
#   alpha toi uu nam han trong (0,1)  ->  CO. Nguon dong gop truc tiep vao diem dich, va
#                                          do la mot phat bieu transfer that su.
#   alpha toi uu = 1.0 o moi o        ->  KHONG. Gia thuyet bi bac, chi phi ~2 phut/o.
#
# Ca hai chieu deu dung viec — do la ly do khoi nay dang chay truoc D4/D5 (dat hon nhieu).
#
# KHAC `--source_interpolation` da co (cap theta_Pha1 <-> pretrained, ap TRUOC Pha 2,
# da bac o DEAD_ENDS #3). Day la CAP KHAC va chua thu bao gio.
#
# BAT BUOC nam trong pha `test`: run/matrix.sh xoa checkpoint Pha 2 sau moi o, nen khong
# tinh nguoc duoc cho o da chay — dung cai bay da mat 207 o cua phep tach nhom ro ri.
#
#   FOLD_LIST="1 2" bash run/int1.sh     # 161
#   FOLD_LIST="3"   bash run/int1.sh     # 158
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Tu chon env cua may. KHONG de mac dinh `python`: 08/09 chay thieu bien nay, `python` la
# conda base khong co numpy, cong tham do Pha 1 that bai va matrix.sh XOA hai checkpoint
# Pha 1 tot. Da vá cong o run/matrix.sh, day la lop chan thu hai.
PYTHON="${PYTHON:-$( [ -x /data/ntat/envs/vdenv/bin/python ] && echo /data/ntat/envs/vdenv/bin/python || echo /home/ntat/miniconda3/envs/vdenv/bin/python )}"
export PYTHON
SOURCES_LIST="${SOURCES_LIST:-4cwe com}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
SEED="${SEED:-42}"
GRID="${GRID:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"

echo "########## INT1 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  nguon: $SOURCES_LIST | fold: $FOLD_LIST | seed: $SEED | luoi alpha: $GRID"

data_of(){ case "$1" in
  4cwe) echo "data/phase1_4cwe.jsonl" ;;
  com)  echo "data/phase1_common.jsonl" ;;
  full) echo "data/phase1_full.jsonl" ;;
  *) echo "" ;; esac; }

for FOLD in $FOLD_LIST; do
  for SRC in $SOURCES_LIST; do
    DATA="$(data_of "$SRC")"
    [[ -n "$DATA" ]] || { echo "  !! nguon la: $SRC"; continue; }
    echo "===== $(date -u '+%F %T') | fold $FOLD | nguon $SRC ====="
    # Hai cau hinh: KHONG neo (plain) va CO neo (gamma=50). Neu alpha toi uu khac nhau
    # giua hai cai thi cai neo co anh huong toi vi tri diem ngot — cung la mot phat hien.
    RUN=int1 SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 \
    SOURCE_EVAL_DATA="$DATA" SOURCE_INTERP_GRID="$GRID" \
    CONFIGS="plain|adamw|--sam_rho 0
c50_t0p05|recadam|--sam_rho 0 --pretrain_cof 50" \
    bash run/opt1.sh
  done
done
echo "########## INT1 xong $(date -u '+%F %T') ##########"
