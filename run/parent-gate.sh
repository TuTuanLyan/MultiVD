#!/usr/bin/env bash
# Cổng 1 cho tín hiệu phụ ở mức PILLAR: gộp CWE của source theo parent gốc.
#
# Câu hỏi: §34.3 đo được giá trị gia tăng của head phụ, Δ(cwe) − Δ(none), gần như
# bằng 0 trên backbone mạnh. Cách giải thích là tín hiệu CWE chi tiết đã nằm sẵn
# trong biểu diễn của backbone, nên head phụ lặp lại thứ nó đã biết. Nhãn pillar
# ở mức trừu tượng cao hơn: nó nói *kiểu sai lầm* (kiểm soát tài nguyên, trung hoà
# đầu vào, kiểm soát truy cập...) chứ không nói *lỗ hổng nào*. Đó là thông tin
# khác chứ không phải cùng thông tin ở độ phân giải thấp hơn.
#
# VÌ SAO KHÔNG DÙNG train_ccpp_js: nguồn đó chỉ có 4 CWE và gộp parent ra đúng
# HAI lớp, tỉ lệ 90/10. Tệ hơn nữa, §34.2 đo được CWE-079 được +0.1202 còn CWE-078
# bị −0.0256 trên CodeT5+, mà hai lớp đó cùng thuộc pillar CWE-707 — gộp lại là
# xoá đúng phần phân biệt tạo ra lợi ích. Chạy ở đó sẽ hỏng vì lý do không nói lên
# điều gì về ý tưởng.
#
# ccpp_primevul_paired_common có 73 CWE, gộp ra 9 pillar, 100% dòng dùng được:
#   CWE-707 30.2% | CWE-664 28.2% | CWE-703 16.4% | CWE-682 12.1% | CWE-284 6.1%
#   CWE-691 4.5% | CWE-693 2.3% | CWE-697 0.2% | CWE-435 0.1%
#
# Ba nhánh dùng CHUNG một baseline vì baseline không hề thấy dữ liệu source:
#   none            -> pretrain trần                        (mốc dưới)
#   cwe / source    -> head phụ 73 lớp CWE chi tiết         (mốc so sánh)
#   cwe / pillar    -> head phụ 9 lớp pillar                (bản đề xuất)
#
# Giữa nhánh 2 và 3 chỉ đổi ĐÚNG MỘT biến: độ phân giải của taxonomy. Cùng dòng
# dữ liệu, cùng backbone, cùng fold, cùng seed, cùng máy.
#
# Số cần nhìn KHÔNG phải Δ tuyệt đối mà là Δ(pillar) − Δ(none) so với
# Δ(cwe source) − Δ(none).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/workspace/hf
PYTHON="${PYTHON:-/venv/main/bin/python}"
export PYTHON

SEED="${SEED:-42}"; export SEED
BB="${BB:-t5pm=Salesforce/codet5p-220m:mean}"
LABEL="${BB%%=*}"
FINE_SRC="${FINE_SRC:-data/ccpp_primevul_paired_common.jsonl}"
PILLAR_SRC="${PILLAR_SRC:-data/ccpp_common_parent.jsonl}"

if [[ ! -f "$PILLAR_SRC" ]]; then
  echo "chua co $PILLAR_SRC — dung build_parent_labels.py"; exit 1
fi

echo "################ BUOC 1: baseline + none + cwe 73 lop ################"
RUN_NAME=parent_ref BACKBONES="$BB" MODES="none cwe" \
  DATA_ROOT=data/sven_python_folds_norm \
  PHASE1_DATA_PATH="$FINE_SRC" CWE_VOCAB=source \
  GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

echo ""
echo "################ BUOC 2: cwe 9 lop pillar ################"
# Baseline sao chép từ bước 1: cùng máy, cùng seed, cùng fold, cùng backbone, và
# baseline không đọc dữ liệu source, nên huấn luyện lại chỉ tốn GPU mà ra đúng
# con số cũ.
mkdir -p "results/parent_new_${LABEL}/baseline/seed_$SEED"
cp results/parent_ref_${LABEL}/baseline/seed_$SEED/fold*.json \
   "results/parent_new_${LABEL}/baseline/seed_$SEED/" 2>/dev/null
RUN_NAME=parent_new BACKBONES="$BB" MODES="cwe" \
  DATA_ROOT=data/sven_python_folds_norm \
  PHASE1_DATA_PATH="$PILLAR_SRC" CWE_VOCAB=precomputed \
  GATE_FOLDS="1 2 3" REST_FOLDS="4 5" \
  bash run/gated.sh

echo ""
echo "################ SO SANH ################"
$PYTHON - <<PY
import glob, json, statistics
L = "$LABEL"; S = $SEED
def load(d):
    o = {}
    for p in glob.glob(d + "/fold*.json"):
        r = json.load(open(p))
        o[int(r["fold"])] = (r["test_macro_f1_at_0.5"], r["test_roc_auc"])
    return o
base = load("results/parent_ref_%s/baseline/seed_%d" % (L, S))
none = load("results/parent_ref_%s/transfer_none/seed_%d" % (L, S))
arms = {"cwe 73 lop":  load("results/parent_ref_%s/transfer_cwe/seed_%d" % (L, S)),
        "pillar 9 lop": load("results/parent_new_%s/transfer_cwe/seed_%d" % (L, S))}
for i, met in enumerate(("macro_f1", "roc_auc")):
    print("[%s]" % met)
    for name, m in arms.items():
        k = sorted(set(m) & set(none) & set(base))
        if not k:
            print("   %-14s chua co" % name); continue
        add = [(m[f][i] - base[f][i]) - (none[f][i] - base[f][i]) for f in k]
        print("   %-14s n=%d  head phu cong them %+.4f  duong %d/%d   tung fold %s"
              % (name, len(k), statistics.mean(add), sum(x > 0 for x in add), len(add),
                 ", ".join("f%d%+.4f" % (f, v) for f, v in zip(k, add))))
    print()
print("Nguong nhieu 0.005. Ba fold chi du de DUNG; muon KET LUAN phai du 5 fold.")
PY
touch /workspace/PARENT_GATE_DONE
