#!/usr/bin/env bash
# Ngay 29/08 — ASAM o PHASE 2, seed 42, RecAdam.
#
#   bash run/day43.sh <nguon> <backbone-spec>
#   vi du: bash run/day43.sh 4cwe codebert=microsoft/codebert-base:cls
#
# PHEP SO LA GI: nhanh `transfer_<mode>_<nguon>_asam` so voi `transfer_<mode>_<nguon>`
# da chay 27-28/08. Cung Phase-1 checkpoint, cung fold, cung RecAdam, cung seed.
# Khac DUNG MOT BIEN: ASAM bat hay tat o Phase 2.
#
# VI SAO PHASE1_TAG PHAI TACH KHOI ARM_TAG:
# Neu de khoa kho Phase 1 di theo ten nhanh thi nhanh `_asam` se khong tim thay
# kho, TU HUAN LUYEN mot Phase 1 KHAC, va phep so "chi doi ASAM" hoa ra doi hai
# bien — vua diem xuat phat vua duong di. Da xay ra that o run `same_emb`.
# Nen: ARM_TAG="_<nguon>_asam" (ten nhanh moi) nhung PHASE1_TAG="_<nguon>" (kho cu).
#
# RHO CUA ASAM KHONG CUNG THANG DO VOI RHO CUA SAM.
# Ban kinh cua ASAM do bang don vi |w| chu khong phai khoang cach L2 tuyet doi.
# Kwon et al. quet {5e-5 ... 2.0} va chon 0.5 (CIFAR-10), 1.0 (CIFAR-100,
# ImageNet), trong khi SAM dung 0.05/0.1/0.05. Thi nghiem transformer duy nhat
# cua ho (IWSLT'14 DE-EN, Adam) dung 0.1 cho SAM va 0.2 cho ASAM.
# Mac dinh o day 0.1 (nguoi dung chon, 29/08 — thap hon ca gia tri transformer
# 0.2 cua bai bao, co y an toan sau vu SAM rho=0.05 giet codebert). Dat
# ASAM_RHO de doi. Quet kiem chung o run/asam_rho_sweep.sh. Bung 0.05 vao la GAN NHU KHONG LAM GI
# — neu ket qua trung khit nhanh khong-ASAM thi kiem tra rho truoc tien.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC="${1:?nguon: full | com | 4cwe}"
BB="${2:?vi du: codebert=microsoft/codebert-base:cls}"

case "$SRC" in
  full) DATA=data/phase1_full.jsonl;   VOCAB=precomputed; M="none latent_bottleneck latent_proto" ;;
  com)  DATA=data/phase1_common.jsonl; VOCAB=precomputed; M="none latent_bottleneck latent_proto" ;;
  4cwe) DATA=data/phase1_4cwe.jsonl;   VOCAB=fixed4;      M="none cwe latent_bottleneck latent_proto" ;;
  *) echo "nguon khong hop le: $SRC"; exit 2 ;;
esac

# CONTROL=1 -> nhanh doi chung KHONG ASAM, chay CUNG MAY CUNG PHIEN voi nhanh
# ASAM. Bat buoc phai co: nhanh khong-ASAM cu (27-28/08) chay tren MAY KHAC, ma
# chenh lech giua may do duoc la 0.028 Macro-F1 — lon hon hieu ung ASAM trong
# tai lieu. Neu so ASAM(may moi) voi khongASAM(may cu) thi Delta = hieu ung ASAM
# + hieu ung phan cung, khong tach duoc.
if [[ "${CONTROL:-0}" == "1" ]]; then
  TAGSUF="_ctl"; P2X="--sam_rho 0"
else
  # Ten nhanh PHAI mang rho: chay hai rho khac nhau ma cung ten `_asam` thi
  # matrix.sh thay ket qua da co va BO QUA, nen lan hai khong chay gi ma trong
  # nhu da chay. rho=0.1 giu ten `_asam` cho khop ket qua da sinh; rho khac
  # duoc hau to (vi du `_asam_r05`).
  R="${ASAM_RHO:-0.1}"
  if [[ "$R" == "0.1" ]]; then TAGSUF="_asam"; else TAGSUF="_asam_r$(echo "$R" | tr -d '.')"; fi
  P2X="--sam_rho $R --sam_variant asam --asam_eta ${ASAM_ETA:-0.01}"
fi

RUN_NAME="${RUN_NAME:-s42}" \
SEED=42 \
FOLDS="${FOLDS:-1 2 3 4 5}" \
BACKBONES="$BB" \
MODES="${MODES:-$M}" \
OPTIMIZERS="${OPTIMIZERS:-recadam}" \
PHASE1_DATA_PATH="$DATA" \
CWE_VOCAB="$VOCAB" \
ARM_TAG="_${SRC}${TAGSUF}" \
PHASE1_TAG="_${SRC}" \
LAMBDA_CWE=0.05 \
PHASE1_EXTRA="--sam_rho 0" \
PHASE2_EXTRA="$P2X" \
DATA_ROOT="${DATA_ROOT:-data/sven_python_folds_norm}" \
TARGET_LANG=python \
bash run/matrix.sh
