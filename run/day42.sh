#!/usr/bin/env bash
# Ngay 27/08 — seed 42 va CHI 42. Muc tieu: phuong phap nao voi setting nao tot nhat.
#
#   3 backbone : codebert, unixcoder, t5p (codet5p-220m-bimodal)
#   3 nguon    : full | com | 4cwe  — MOI nguon deu la ccpp + js GOP LAI
#   4 nhanh    : none, cwe, latent_bottleneck, latent_proto
#                cwe CHI chay tren nguon 4cwe (chi o do nhan 4 lop moi dinh nghia duoc)
#   2 optimizer: adamw, recadam — nam canh nhau trong cung fold de ghep cap duoc
#   lambda 0.05, SAM rho o Phase 1 = $SAM_RHO (MAC DINH 0 = TAT)
#
#   SAM da duoc chuyen sang CHI Phase 2. Ly do (27/08):
#   - Phase 1 la pha THU NAP bieu dien; SAM la bo dieu chuan, dat no o day la
#     danh doi do khop lay do ben dung cho ta dang can khop toi da.
#   - O Phase 1, epsilon tinh tu gradient cua TONG L_binary + lambda*L_aux nen
#     huong day bi chi phoi boi loss nao co chuan gradient lon hon (SAMO;
#     "Beyond Losses Reweighting" ICCV 2025). Phase 2 chi mot loss -> het benh.
#   - Bang chung pro-SAM sat scale nhat (Bahri et al. ACL 2022) ap SAM luc
#     FINE-TUNE, khong phai pretrain. Bai chong lung cho SAM-luc-pretrain
#     (Watts et al. 2605.02105) co ca 3 tuyen bo headline TRUOT kiem chung 0-3.
#   - SAM ton 2x. O Phase 1 cai 2x do danh vao 7598 hang; o Phase 2 vao 760 hang.
#
#   CANH BAO cho khoi SAM@Phase2: TRAM (ICLR 2024) do duoc viec cong mot
#   regularizer NEO vao loss ma SAM dang nhieu lam KET QUA TE HON Adam tron
#   (M2D2 S2ORC: Adam 27.4, ASAM 26.8, ASAM+TRPO 30.2). RecAdam dung ho do.
#   Vi vay SAM x optimizer phai la MOT TRUC, chay ca adamw lan recadam.
#
#   rho KHONG chuyen duoc giua cac backbone. Do truc tiep sang 27/08 tren nguon
#   4cwe, nhanh `none`, cung rho=0.05:
#       t5p       train loss 0.35   val macro-F1 0.6614  -> lanh manh
#       unixcoder                   val macro-F1 0.6476  -> lanh manh
#       codebert  train loss 0.695-0.705 = ln2, dung yen 13 epoch -> SAP
#   Nen codebert chay SAM_RHO=0.01 (FACTS §5c: cwe 0.6209, bottleneck 0.6423,
#   proto 0.6123 -- deu qua cong 0.55). Day dung la cach tai lieu lam: Bahri et
#   al. (ACL 2022) phai doi rho 3 lan trong cung mot ho mo hinh theo kich thuoc.
#
# Dung:
#   bash run/day42.sh <nguon> phase1     # chi Phase 1 (khong phu thuoc tap dich)
#   bash run/day42.sh <nguon> phase2     # Phase 1 (dung lai) + Phase 2 moi fold
#
# CANH BAO da biet truoc, xem FACTS.md §5c: codebert + SAM rho=0.05 o Phase 1 da
# tung ket o train loss = ln2, val macro-F1 0.3333. Cong `phase1_usable` trong
# matrix.sh se tu chi checkpoint do (val < 0.55) va dat ten .rejected thay vi de
# no dau doc Phase 2. Neu thay .rejected xuat hien thi DUNG, dung ha rho.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC="${1:?nguon: full | com | 4cwe}"
STAGE="${2:-phase1}"

case "$SRC" in
  full) DATA=data/phase1_full.jsonl;   VOCAB=precomputed; M="none latent_bottleneck latent_proto";      TAG=_full ;;
  com)  DATA=data/phase1_common.jsonl; VOCAB=precomputed; M="none latent_bottleneck latent_proto";      TAG=_com  ;;
  4cwe) DATA=data/phase1_4cwe.jsonl;   VOCAB=fixed4;      M="none cwe latent_bottleneck latent_proto"; TAG=_4cwe ;;
  *) echo "nguon khong hop le: $SRC"; exit 2 ;;
esac

case "$STAGE" in
  phase1) F="" ;;
  phase2) F="${FOLDS:-1 2 3 4 5}" ;;
  *) echo "stage khong hop le: $STAGE"; exit 2 ;;
esac

RUN_NAME="${RUN_NAME:-s42}" \
SEED=42 \
FOLDS="$F" \
BACKBONES="${BACKBONES:-codebert=microsoft/codebert-base:cls unixcoder=microsoft/unixcoder-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean}" \
MODES="$M" \
OPTIMIZERS="adamw recadam" \
PHASE1_DATA_PATH="$DATA" \
CWE_VOCAB="$VOCAB" \
ARM_TAG="$TAG" \
LAMBDA_CWE=0.05 \
PHASE1_EXTRA="--sam_rho ${SAM_RHO:-0}" \
DATA_ROOT="${DATA_ROOT:-data/sven_python_folds_norm}" \
TARGET_LANG=python \
bash run/matrix.sh
