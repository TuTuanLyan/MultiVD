#!/usr/bin/env bash
# Lap O TRONG duy nhat con lai: nhanh ASAM+RecAdam cua dao chieu, dich `js_full`, backbone
# codebert, fold 1. O nay OOM luc 09:39 tren 161 vi user `cuongtm` no VRAM giua chung
# (GPU con 3 MiB trong) — xem FACTS §41.2. Mot o bi chan la mot o TRONG, va o trong khong
# viet duoc gi vao bai (CLAUDE.md muc 3).
#
# 1 o. Baseline cua dich do da co san trong cung cay nen ghep cap duoc ngay.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-/venv/main/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "########## LAP O TRONG bat dau $(date -u '+%F %T') | $(hostname) ##########"

bad=0
for f in run/rev1.sh run/matrix.sh data/phase1_python.jsonl \
         model/rev1/phase1/codebert__latent_bottleneck_py_l0p05/seed_42/best.pt \
         data/js_full_folds/fold1/train.jsonl data/js_full_folds/fold1/val.jsonl \
         data/js_full_folds/fold1/test.jsonl; do
  [ -e "$f" ] || { echo "  !! THIEU $f"; bad=1; }
done
(( bad )) && { echo "!! thieu dau vao — DUNG"; exit 2; }
echo "  moi duong dan DA CO"

# CHI nhanh ASAM (rho 0.1 cua codebert); `plain` va `baseline` da co san.
BB="codebert=microsoft/codebert-base:cls" \
RUN=rev1full FOLDS=1 TGT_ROOT=data/js_full_folds \
CONFIGS="r0p1|recadam|--sam_rho 0.1 --sam_variant asam" \
REV_LOCK=/tmp/mvd_fillgap.lock \
bash run/rev1.sh

echo "########## LAP O TRONG xong $(date -u '+%F %T') | rev1full_codebert co $(find results/rev1full_codebert -name 'fold*.json' 2>/dev/null | wc -l)/3 o ##########"
