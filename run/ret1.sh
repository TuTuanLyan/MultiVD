#!/usr/bin/env bash
# RET1 — do GIU LAI TRI THUC NGUON. Khoi dau tien tra loi cau hoi "cai neo lam gi".
#
# VI SAO KHOI NAY TON TAI (do duoc 07/09 tren 10 o ghep cap day du, bang 2 cua
# tools/opt1_report.py): tren TAP DICH, RecAdam o moi gamma da thu (0.5 / 5 / 50 / 500
# / 5000) deu KHONG khac AdamW thuan qua duoc san nhieu 0.010 — cheh lech lon nhat
# +0.0065, p >= 0.2. Cau chuyen "neo nang san" dua vao DUNG MOT o (4cwe/fold3); bo o do
# ra thi do tan chi con 0.0188 -> 0.0126, va tren codebert no khong lap lai.
# Nen truc F1-dich da het cho de toi uu. Nhung do KHONG phai cau hoi ma RecAdam sinh ra
# de tra loi: no sinh ra de GIU tri thuc nguon. Dieu do chua he duoc do lan nao.
#
# Phep do: cham chinh mo hinh Pha 2 tren DUNG tap val cua Pha 1, tai lap bang
# split_source_records(records, seed) — cung ham cung seed Pha 1 da dung.
# Cong hai chieu da chay 07/09: cham checkpoint Pha 1 tren tap nay ra 0.697621, TRUNG
# KHIT best_val_macro_f1 da ghi trong checkpoint; cham no tren nguon khac ra 0.5674.
#
# Ba muc so sanh, tat ca CUNG MAY CUNG PHIEN:
#   plain (adamw)       fine-tune hai lan — bao nhieu tri thuc nguon con lai khi khong neo
#   c5000_t0p05         RecAdam mac dinh cua bai goc
#   c50_t0p05           gamma tot nhat do duoc o khoi OPT1
#   baseline            chua he thay nguon -> SAN tham chieu (cham tren ca ba nguon)
#
# Ket qua doc duoc theo huong nao cung dung viec:
#   neo giu duoc nguon, AdamW quen  -> khac biet CO THAT, do duoc, va dung la ly do
#                                      phuong phap can RecAdam chu khong phai finetune 2 lan
#   ca hai deu quen nhu nhau        -> cai neo khong lam gi that; phai doi huong khac
#
# BAC 1 (kiem chung, CLAUDE.md muc 1): 3 fold, seed 42. Tot moi len n=5.
#
#   FOLD_LIST="1 2 3" bash run/ret1.sh
#
# Ket qua: results/ret1_t5p/... (cay RIENG, nen baseline va doi chung deu chay lai
# cung phien — khong muon o nao bi bo qua vi "da co" ben cay opt1).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCES_LIST="${SOURCES_LIST:-4cwe com}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
SEED="${SEED:-42}"
# Baseline chua he thay nguon nao, nen cham no tren CA BA — mot lan, gia gan bang 0.
# Khoang trang o day AN TOAN: da thu hai chieu 07/09 —  V="a b c"; set -- ${V:+--flag "$V"}
# cho DUNG 2 tham so, dau nhay ben trong ${var:+...} van duoc giu.
ALL_SRC="data/phase1_4cwe.jsonl data/phase1_common.jsonl data/phase1_full.jsonl"

data_of(){ case "$1" in
  4cwe) echo "data/phase1_4cwe.jsonl" ;;
  com)  echo "data/phase1_common.jsonl" ;;
  full) echo "data/phase1_full.jsonl" ;;
  *) echo "" ;; esac; }

echo "########## RET1 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  nguon: $SOURCES_LIST | fold: $FOLD_LIST | seed: $SEED"

# Fold la vong NGOAI (CLAUDE.md muc 1): xong fold 1 la da co mot lat cat so duoc ngay.
for FOLD in $FOLD_LIST; do
  for SRC in $SOURCES_LIST; do
    DATA="$(data_of "$SRC")"
    [[ -n "$DATA" ]] || { echo "  !! nguon la: $SRC"; continue; }
    echo "===== $(date -u '+%F %T') | fold $FOLD | nguon $SRC ====="
    RUN=ret1 SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 \
    SOURCE_EVAL_DATA="$DATA" BASELINE_SOURCE_EVAL="$ALL_SRC" \
    CONFIGS="plain|adamw|--sam_rho 0
c5000_t0p05|recadam|--sam_rho 0
c50_t0p05|recadam|--sam_rho 0 --pretrain_cof 50" \
    bash run/opt1.sh
  done
done
echo "########## RET1 xong $(date -u '+%F %T') ##########"
