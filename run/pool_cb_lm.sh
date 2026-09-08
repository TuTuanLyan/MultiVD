#!/usr/bin/env bash
# POOL_CB_LM — luoi NGON NGU (lm*) tren backbone THU HAI (codebert).
#
# VI SAO: B.2/B.2b/B.2c cua RESEARCH_2026-09-08_transfer.md deu dua tren luoi `lm`
# do DUY NHAT tren t5p. Phat bieu manh nhat cua ca nhanh — "dong lech CWE gay nhieu
# chu dong, 697 dong dung CWE + 233 dong lech THUA 232 dong dung CWE mot minh" — se
# chi la mot dac tinh cua t5p neu khong lap duoc tren backbone khac.
#
# `pool_cb.sh` CHI chay cac pool `pur*`, tuc luoi con lan ti le ngon ngu (RESEARCH B.1).
# Khoi nay chay dung 5 pool `lm*` da ghim js o 0.87.
#
# BAC 1 (sang loc, n=3 fold, seed 42). Doi chung lm100_n930 CUNG MAY CUNG PHIEN.
# Phai huan luyen 5 Pha 1 codebert moi — matrix.sh tu lam khi thieu checkpoint.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POOLS="lm100_n930 lm75_n930 lm50_n930 lm25_n930 lm12_n930" exec bash run/pool_cb.sh
