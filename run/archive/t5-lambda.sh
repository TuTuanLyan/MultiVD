#!/usr/bin/env bash
# Giải pháp thử cho họ T5: hạ trọng số task phụ.
#
# §34.2 chỉ ra chỗ hỏng cụ thể trên CodeT5+ / Python, theo từng lớp CWE:
#
#   CWE-079 ( 82 mẫu)  +0.1202   <- head phụ CÓ tác dụng, rất mạnh
#   CWE-022 ( 66 mẫu)  +0.0201
#   CWE-078 (204 mẫu)  -0.0256   <- và nó LÀM HỎNG lớp này
#   CWE-089 (408 mẫu)  -0.0000
#
# Lợi ở lớp hiếm bị hại ở lớp phổ biến trung hoà gần hết: trung bình có trọng số
# chỉ còn +0.0078. Vấn đề không phải "head phụ vô dụng" mà là **nó vừa giúp vừa
# phá, và hai thứ triệt tiêu nhau**.
#
# Giả thuyết: λ = 0.2 quá mạnh với một backbone vốn đã biểu diễn tốt. Mục tiêu phụ
# ép tách CWE đến mức làm méo lớp phổ biến mà backbone đang làm tốt. Hạ λ có thể
# giữ phần lợi ở lớp hiếm mà bỏ phần hại.
#
# Đáng thử vì **λ chưa từng được đo**. 0.2 là giá trị đặt từ đầu và giữ nguyên qua
# mọi thí nghiệm; không có gì bảo đảm nó tối ưu, và càng không có gì bảo đảm cùng
# một λ hợp cho cả backbone yếu lẫn backbone mạnh.
#
# Ba nhánh dùng chung một baseline, cùng máy, cùng fold:
#   none        -> mốc dưới
#   cwe λ=0.2   -> bản hiện tại
#   cwe λ=0.05  -> bản hạ trọng số
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"
export PYTHON
SEED="${SEED:-42}"; export SEED
BB="${BB:-t5p=Salesforce/codet5p-220m:cls}"
LABEL="${BB%%=*}"
LOW_LAMBDA="${LOW_LAMBDA:-0.05}"

echo "################ Buoc 1: baseline + none + cwe (lambda 0.2) ################"
RUN_NAME=lam_ref BACKBONES="$BB" MODES="none cwe" \
  DATA_ROOT=data/sven_python_folds_norm \
  PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
  LAMBDA_CWE=0.2 GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

echo ""
echo "################ Buoc 2: cwe voi lambda = $LOW_LAMBDA ################"
mkdir -p "results/lam_low_${LABEL}/baseline/seed_$SEED"
cp results/lam_ref_${LABEL}/baseline/seed_$SEED/fold*.json \
   "results/lam_low_${LABEL}/baseline/seed_$SEED/" 2>/dev/null
RUN_NAME=lam_low BACKBONES="$BB" MODES="cwe" \
  DATA_ROOT=data/sven_python_folds_norm \
  PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
  LAMBDA_CWE="$LOW_LAMBDA" GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

echo ""
echo "################ So sanh ################"
$PYTHON - <<PY
import glob, json, statistics
L = "$LABEL"; S = $SEED
def load(d, met):
    o = {}
    for p in glob.glob(d + "/fold*.json"):
        r = json.load(open(p))
        if r.get(met) is not None: o[int(r["fold"])] = float(r[met])
    return o
for met in ("test_macro_f1_at_0.5", "test_roc_auc"):
    b = load("results/lam_ref_%s/baseline/seed_%d" % (L, S), met)
    n = load("results/lam_ref_%s/transfer_none/seed_%d" % (L, S), met)
    arms = {"cwe lambda=0.2": load("results/lam_ref_%s/transfer_cwe/seed_%d" % (L, S), met),
            "cwe lambda=$LOW_LAMBDA": load("results/lam_low_%s/transfer_cwe/seed_%d" % (L, S), met)}
    print("[%s]" % met)
    for name, m in arms.items():
        k = sorted(set(m) & set(n) & set(b))
        if not k: print("   %-18s chua co" % name); continue
        add = [(m[f]-b[f]) - (n[f]-b[f]) for f in k]
        print("   %-18s n=%d  head phu cong them %+.4f  (duong %d/%d, bo fold tot nhat %+.4f)"
              % (name, len(k), statistics.mean(add), sum(x>0 for x in add), len(add),
                 statistics.mean(sorted(add)[:-1]) if len(add)>1 else float("nan")))
    print()
PY
touch /workspace/T5LAMBDA_DONE
