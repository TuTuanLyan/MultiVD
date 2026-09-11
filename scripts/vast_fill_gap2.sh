#!/usr/bin/env bash
# Chay not nhanh `plain` cua dao chieu js_full x codebert TREN VAST, de CA HANG nam tron
# tren MOT may.
#
# Vi sao can: `scripts/vast_fill_gap.sh` chay baseline + ASAM tren vast, trong khi `plain`
# cua hang nay do 161 chay hom truoc. Δ cua ASAM khi do ghep cap voi baseline VAST con Δ
# cua plain ghep cap voi baseline 161 — moi Δ deu hop le, nhung so ASAM voi plain giua
# chung la so qua HAI LOAI GPU, san nhieu 0.028 (CLAUDE.md muc 4: nhanh doi chung phai
# chay cung may cung phien voi nhanh no doi chung).
#
# 1 o. Sau buoc nay hang js_full x codebert co du baseline + plain + ASAM tren cung vast.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-/venv/main/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "########## LAP O TRONG (2/2) bat dau $(date -u '+%F %T') | $(hostname) ##########"
BB="codebert=microsoft/codebert-base:cls" \
RUN=rev1full FOLDS=1 TGT_ROOT=data/js_full_folds \
CONFIGS="plain|adamw|--sam_rho 0" \
REV_LOCK=/tmp/mvd_fillgap2.lock \
bash run/rev1.sh
echo "########## LAP O TRONG (2/2) xong $(date -u '+%F %T') | rev1full_codebert: $(find results/rev1full_codebert -name 'fold*.json' 2>/dev/null | wc -l) o ##########"
