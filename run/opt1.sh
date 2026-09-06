#!/usr/bin/env bash
# OPT1 — toi uu RecAdam, khoi 1 (RESEARCH_2026-09-06_recadam.md §5.5).
#
# CHI DOI PHA 2. Dung lai DUNG checkpoint Pha 1 cua NIGHT48
# (model/n48/phase1/t5p__latent_bottleneck_<src>_l0p05/seed_<seed>/best.pt).
# Thieu checkpoint thi BO QUA nguon do va in loi to — KHONG BAO GIO huan luyen Pha 1
# moi o day, vi khi do phep so doi hai bien (CLAUDE.md muc 5).
#
# 12 cau hinh Pha 2 cho moi (fold, nguon), hai doi chung CHAY TRUOC trong moi nhom:
#   c5000_t0p05        RecAdam mac dinh (gamma 5000, t0 = 5% buoc)   <- doi chung 1, cung phien
#   plain (adamw)      AdamW thuan = fine-tune hai lan                 <- doi chung 2, cung phien
#   c500/c50/c5/c0p5_t0p05     truc GAMMA o lich mac dinh: 500, 50, 5, 0.5
#   c5000_t0p5                 BANG CHUNG dong bang (xem duoi), giu DUNG MOT nhanh
#   c5000_t0p2k02, c50_t0p2k02 neo BEN THAT SU: t0 = 20% (174 buoc), k = 0.02
#   nohead_c5000_t0p05         RecAdam mac dinh nhung KHONG neo vul_head (nhu bai goc)
#   pre_c20_t0p2k02            neo ve PRETRAINED (archive §40.5), o lich t0p2k02
#   warm (adamw)               AdamW + lich lambda(t) tren lr, KHONG neo (tach warmup khoi neo)
#
# BAY DA MAC (06/09, do duoc tu val_history): sigmoid lambda(t)=1/(1+exp(-k(t-t0))) co
# hai tham so KHONG DOC LAP. Dat t0 = 50% (435 buoc) ma giu k = 0.05 cho k*t0 = 21.7,
# nen lambda(1) ~ 4e-10 va sau 7 epoch moi len 9e-06: buoc task bi nhan voi ~0, con luc
# keo neo cung ~0 vi theta van bang theta*. Mo hinh DUNG NGUYEN tai checkpoint Pha 1 ->
# ca ba gamma 5000/500/50 cho TRUNG KHIT test F1 0.4738, val F1 0.5131 dung im 7 epoch.
# Do la phep do "Pha 1 ap thang len Python", khong phai "neo ben". Giu DUNG MOT nhanh
# (c5000_t0p5) lam bang chung, phan con lai chuyen sang t0=20% k=0.02:
#   lambda ep1/ep6/ep12 = 0.052 / 0.500 / 0.970  (so voi t0p05: 0.332 / 0.999 / 1.000)
# Quy tac rut ra: k phai ti le nghich voi t0, giu k*t0 ~ 3-4.
#
# Vi sao hai doi chung phai chay lai o day du da co o khoi A/B: CLAUDE.md muc 4 —
# nhanh doi chung phai cung may cung phien voi nhanh no doi chung. Khoi A/B chay tren
# may khac; chenh lech may do duoc toi 0.118 tren mot fold (RESEARCH §5.3).
#
# THU TU: fold la vong ngoai, nguon vong giua, cau hinh vong trong (CLAUDE.md muc 1).
# Het fold 1 la da co mot lat cat 20 o + baseline so duoc ngay.
#
#   FOLD_LIST="1 2"   bash run/opt1.sh     # may 161
#   FOLD_LIST="3 4 5" bash run/opt1.sh     # may 158  (chia theo FOLD TRON VEN, muc 4)
#
# Ket qua: results/opt1_t5p/{baseline,transfer_latent_bottleneck_<src>_l0p05_<tag>[_adamw]}/seed_<seed>/fold<k>.json
# Log job: log/opt1/. Bao cao: python3 tools/opt1_report.py
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS="${SEEDS:-42}"
SOURCES="${SOURCES:-4cwe com}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
LAM="${LAM:-0.05}"
BB="${BB:-t5p=Salesforce/codet5p-220m-bimodal:mean}"
LTAG="l$(printf '%s' "$LAM" | tr '.' 'p')"
LABEL="${BB%%=*}"
P1STORE="${P1STORE:-model/n48/phase1}"
RUN="${RUN:-opt1}"

# "tag|optimizer|cac co Pha 2". Doi chung dung dau.
CONFIGS="${CONFIGS:-\
c5000_t0p05|recadam|--sam_rho 0
plain|adamw|--sam_rho 0
c500_t0p05|recadam|--sam_rho 0 --pretrain_cof 500
c50_t0p05|recadam|--sam_rho 0 --pretrain_cof 50
c5_t0p05|recadam|--sam_rho 0 --pretrain_cof 5
c0p5_t0p05|recadam|--sam_rho 0 --pretrain_cof 0.5
c5000_t0p5|recadam|--sam_rho 0 --anneal_t0_ratio 0.5
c5000_t0p2k02|recadam|--sam_rho 0 --anneal_t0_ratio 0.2 --anneal_k 0.02
c50_t0p2k02|recadam|--sam_rho 0 --pretrain_cof 50 --anneal_t0_ratio 0.2 --anneal_k 0.02
nohead_c5000_t0p05|recadam|--sam_rho 0 --recadam_anchor_head none
pre_c20_t0p2k02|recadam|--sam_rho 0 --recadam_anchor pretrained --pretrain_cof 20 --anneal_t0_ratio 0.2 --anneal_k 0.02
warm|adamw|--sam_rho 0 --adamw_anneal_lr}"

data_of(){ case "$1" in
  4cwe) echo "data/phase1_4cwe.jsonl fixed4" ;;
  com)  echo "data/phase1_common.jsonl precomputed" ;;
  full) echo "data/phase1_full.jsonl precomputed" ;;
  *) echo "" ;; esac; }

LOCK="${MVD_LOCK:-/tmp/multivd_opt1.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver opt1 dang chay"; exit 3; }
mkdir -p log

run_cell(){ # $1=src $2=data $3=vocab $4=seed $5=folds $6=tag $7=opt $8=p2extra
  RUN_NAME="$RUN" SEED="$4" FOLDS="$5" \
  BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS="$7" \
  PHASE1_DATA_PATH="$2" CWE_VOCAB="$3" \
  ARM_TAG="_${1}_${LTAG}_${6}" PHASE1_TAG="_${1}_${LTAG}" PHASE1_STORE="$P1STORE" \
  LAMBDA_CWE="$LAM" PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 \
  PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="$8" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  bash run/matrix.sh 9>&-
}

NS=$(echo $SOURCES|wc -w); NE=$(echo $SEEDS|wc -w); NF=$(echo $FOLD_LIST|wc -w)
NC=$(printf '%s\n' "$CONFIGS" | grep -c '|')
EXP=$(( NS*NE*NF*NC )); EXPB=$(( NE*NF ))
echo "########## OPT1 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  $LABEL | latent_bottleneck | lambda $LAM | Pha 1 dung lai tu $P1STORE"
echo "  nguon: $SOURCES | seed: $SEEDS | fold: $FOLD_LIST | $NC cau hinh"
echo "  ky vong: ${NS}x${NE}x${NF}x${NC} = $EXP o Pha 2 + $EXPB baseline"

for SEED in $SEEDS; do
  # Kiem checkpoint Pha 1 TRUOC, in ro cai nao thieu — o thieu phai thay ngay tu dau log.
  for SRC in $SOURCES; do
    CKPT="$P1STORE/${LABEL}__latent_bottleneck_${SRC}_${LTAG}/seed_${SEED}/best.pt"
    if [[ -f "$CKPT" ]]; then
      V=$(PYTHONWARNINGS=ignore ${PYTHON:-python} - "$CKPT" <<'PY' 2>/dev/null
import sys,torch
try:
    c=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
    print("val=%.4f ep=%s lambda=%s"%(float(c.get("best_val_macro_f1") or 0),c.get("best_epoch"),(c.get("training_args") or {}).get("lambda_cwe")))
except Exception as e: print("KHONG DOC DUOC: %s"%e)
PY
)
      echo "  [seed $SEED / $SRC] Pha 1 OK — $V — $(stat -c %s "$CKPT") B"
    else
      echo "  [seed $SEED / $SRC] !! THIEU checkpoint Pha 1 $CKPT — $(( NF*NC )) o cua nguon nay se TRONG. KHONG tu huan luyen."
    fi
  done

  for FOLD in $FOLD_LIST; do
    for SRC in $SOURCES; do
      read -r DATA VOCAB <<< "$(data_of "$SRC")"
      CKPT="$P1STORE/${LABEL}__latent_bottleneck_${SRC}_${LTAG}/seed_${SEED}/best.pt"
      [[ -f "$CKPT" ]] || continue
      while IFS='|' read -r TAG OPT EXTRA; do
        [[ -z "$TAG" ]] && continue
        echo "===== $(date -u '+%F %T') | seed $SEED | fold $FOLD | nguon $SRC | $TAG ($OPT) | $EXTRA ====="
        run_cell "$SRC" "$DATA" "$VOCAB" "$SEED" "$FOLD" "$TAG" "$OPT" "$EXTRA"
      done <<< "$CONFIGS"
    done
    N=$(ls results/${RUN}_${LABEL}/transfer_latent_bottleneck_*_${LTAG}_*/seed_${SEED}/fold${FOLD}.json 2>/dev/null | wc -l)
    B=$(ls results/${RUN}_${LABEL}/baseline/seed_${SEED}/fold${FOLD}.json 2>/dev/null | wc -l)
    echo "########## OPT1 seed $SEED fold $FOLD xong $(date -u '+%F %T') | $N/$(( NS*NC )) o + $B/1 baseline ##########"
  done
done

TOT=0; TOTB=0
for SEED in $SEEDS; do for FOLD in $FOLD_LIST; do
  TOT=$(( TOT + $(ls results/${RUN}_${LABEL}/transfer_latent_bottleneck_*_${LTAG}_*/seed_${SEED}/fold${FOLD}.json 2>/dev/null | wc -l) ))
  TOTB=$(( TOTB + $(ls results/${RUN}_${LABEL}/baseline/seed_${SEED}/fold${FOLD}.json 2>/dev/null | wc -l) ))
done; done
echo "########## OPT1 xong $(date -u '+%F %T') | $(hostname) | $TOT/$EXP o Pha 2 + $TOTB/$EXPB baseline ##########"
