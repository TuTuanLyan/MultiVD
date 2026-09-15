#!/usr/bin/env bash
# Bao cao day du cho khoi gate1 — chay MOT lenh, ra tat ca thu De xuat 1 doi.
#   bash scripts/gate1_report.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/ntat/miniconda3/envs/vdenv/bin/python}"
R1=results/gate1_codebert; R2=results/gate1_t5p
B=transfer_none_com_real

echo "######## 1. SO O ########"
for r in $R1 $R2; do echo "  $r: $(find $r -name 'fold*.json' 2>/dev/null | wc -l) o"; done

echo; echo "######## 2. QUYET DINH LEO BAC (nguong da ghi truoc) ########"
"$PY" tools/gate1_decide.py $R1 $R2; echo "  ma thoat = $?"

echo; echo "######## 3. GHEP vs TUNG MODEL — bon chi so, tung backbone ########"
for r in $R1 $R2; do
  echo "--- $(basename $r)"
  "$PY" tools/late_fusion_gate.py --a baseline --b $B --modes g05,logreg "$r" 2>&1 | sed -n '3,30p'
done

echo; echo "######## 4. DOI CHUNG ENSEMBLE THUAN: ghep(base42, base7) ########"
echo "  (cong cu ghep theo tag; hai baseline khac SEED nen doc o muc 2, D_NGUON)"

echo; echo "######## 5. PHAN BO g THEO CWE + he so cong (A.3) ########"
"$PY" tools/gate_readout.py --a baseline --b $B $R1 $R2 2>&1 | tail -40

echo; echo "######## 6. TACH THEO NHOM RO RI (cong 3) ########"
"$PY" tools/leak_groups_auc.py --a $B --b baseline $R1 $R2 2>&1 | sed -n '3,9p'
