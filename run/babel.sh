#!/usr/bin/env bash
# MW_ASSEMBLE_BABEL — hoc vien co so la BABEL (dong lenh + hai do thi dong nhat),
# con khung `assemble` thi giu nguyen: hai model dong cung, tron xac suat hoc tren val.
#
#   babel   : dong lenh + 2 GCN, khoi tao tu backbone GOC
#   babelTR : GIONG HET, khac DUNG MOT thu la --init_ckpt Pha 1
#   assemble: 0 GPU, tinh sau bang tools/mw_n5_table.py
#
# BASELINE **KHONG** chay lai (nguoi dung 17/09): kho local co 20 cay baseline codebert
# du 5 fold, seed 42, cung chu ky sieu tham so. tools/mw_n5_table.py tu lay trung vi.
#
# Bac 2 — XAC NHAN, n=5 fold, seed 42. Fold la vong NGOAI.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python3}"
RN="${RUN_NAME:-mw_assemble_babel}"
MODEL="${MODEL_NAME:-microsoft/codebert-base}"
POOL="${POOLING:-cls}"
CKPT="${INIT_CKPT:-model/shuf1/phase1/codebert__none_com_real/seed_42/best.pt}"
FOLDS_LIST="${FOLDS:-1 2 3 4 5}"
SEED="${SEED:-42}"
ARMS="${ARMS:-babel babelTR}"   # nhanh nao KET THUC bang TR thi duoc nap Pha 1
MAXLINES="${MAX_LINES:-150}"
MAXLTOK="${MAX_LINE_TOKENS:-48}"
GHID="${GRAPH_HIDDEN:-256}"
GLR="${GRAPH_LR:-5e-4}"
SYMB="${SYMBOLIZE:-none}"
NMODE="${NODE_MODE:-line}"        # line = moi DONG mot dinh | chunk = gop thanh KHUC truoc
CBUD="${CHUNK_BUDGET:-96}"        # chi dung khi NODE_MODE=chunk
NEED_FREE="${NEED_FREE:-8000}"
WAIT_MAX="${WAIT_MAX:-14400}"

exec 4>"${LOCKF:-/tmp/mvd_babel.lock}" || exit 1
flock -n 4 || { echo "DA CO babel dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/tmp/babel.pid}"; echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

[ -f src/train_babel.py ] || { echo "!! THIEU src/train_babel.py"; exit 5; }
[ -f src/babel_graph.py ] || { echo "!! THIEU src/babel_graph.py"; exit 5; }
[ -f data/sven_python_folds_norm/fold5/test.jsonl ] || { echo "!! THIEU du lieu dich"; exit 5; }
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__)" || {
  echo "!! PYTHON='$PY' khong import duoc torch"; exit 5; }
MODEL="$MODEL" "$PY" - <<'PYCHECK' || { echo "!! KHONG nap duoc model offline"; exit 5; }
import os, sys; sys.path.insert(0, "src")
from transformers import AutoTokenizer
from model import build_backbone
m = os.environ["MODEL"]
AutoTokenizer.from_pretrained(m); build_backbone(m)
print("  HF offline OK:", m)
PYCHECK
# Cong do thi: dung lai roi doi chieu voi so da do o local, de biet ngay neu du lieu khac.
NMODE="$NMODE" CBUD="$CBUD" MODEL="$MODEL" "$PY" - <<'PYG' || { echo "!! do thi dung KHONG dat"; exit 5; }
import sys, json, os; sys.path.insert(0, "src")
import numpy as np
from babel_graph import build, build_chunked
mode, bud = os.environ["NMODE"], int(os.environ["CBUD"])
tok = None
if mode == "chunk":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ["MODEL"])
rows = [json.loads(l) for l in open("data/sven_python_folds_norm/fold1/test.jsonl")]
nl, dd = [], []
for r in rows:
    raw, Ad, Ac = (build_chunked(r["code"], tok, bud, 48) if mode == "chunk"
                   else build(r["code"], 150, 0))
    n = len(raw); nl.append(n); dd.append((Ad.sum() - n) / max(n, 1))
one = 100 * np.mean(np.array(nl) == 1)
print(f"  do thi OK: mode={mode} dinh/ham TB {np.mean(nl):.1f} | 1 dinh {one:.1f}% | canh DATA/dinh TB {np.mean(dd):.2f}")
# Nguong khac nhau cho hai che do — do o local: DONG ~20.7 dinh / 2.81 canh;
# KHUC 96 token ~5.5 dinh / 2.46 canh, 14.8% ham chi con 1 dinh.
if mode == "chunk":
    assert 2.5 < np.mean(nl) < 15, "so khuc/ham lech xa muc da do (~5.5)"
    assert one < 35, f"{one:.0f}% ham chi con MOT dinh — khong con do thi de hoc"
else:
    assert 8 < np.mean(nl) < 60, "so dong/ham lech xa muc da do (~20)"
assert 0.5 < np.mean(dd) < 12, "bac do thi lech xa muc da do"
PYG

if [ "${INIT_CKPT_REQUIRED:-1}" = 1 ] && [ ! -f "$CKPT" ]; then
  echo "!! THIEU checkpoint Pha 1: $CKPT — nhanh babelTR se khong chay duoc"; exit 5
fi

wait_gpu(){
  local t=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -n "${free:-}" ] || { echo "  khong doc duoc nvidia-smi — phong lieu"; return 0; }
    (( free >= NEED_FREE )) && { echo "  GPU trong ${free} MiB — phong"; return 0; }
    (( t >= WAIT_MAX )) && { echo "  !! cho ${t}s ma card van chi con ${free} MiB — BO"; return 1; }
    echo "  $(ts) card con ${free} MiB (<${NEED_FREE}) — cho 120s"; sleep 120; t=$((t+120))
  done
}

N_EXP=0; for f in $FOLDS_LIST; do for x in $ARMS; do N_EXP=$((N_EXP+1)); done; done
echo "########## BABEL bat dau $(ts) | $(hostname) | fold: $FOLDS_LIST | arms: $ARMS ##########"
echo "  model $MODEL | pooling $POOL | max_lines $MAXLINES | max_line_tokens $MAXLTOK"
echo "  graph_hidden $GHID | graph_lr $GLR | symbolize $SYMB | node_mode $NMODE | chunk_budget $CBUD"
echo "  ckpt TR: $CKPT"
for FOLD in $FOLDS_LIST; do                       # FOLD la vong NGOAI
  for NAME in $ARMS; do
    OUT="results/${RN}/${NAME}/seed_${SEED}/fold${FOLD}.json"
    if [ -f "$OUT" ]; then echo "=== $(ts) | fold $FOLD | $NAME | da co ==="; continue; fi
    wait_gpu || exit 1
    EXTRA=(); case "$NAME" in *TR) EXTRA=(--init_ckpt "$CKPT");; esac
    echo "===== $(ts) | FOLD $FOLD | $NAME ${EXTRA[*]:-} ====="
    mkdir -p "log/${RN}"
    "$PY" -u src/train_babel.py \
      --run_name "$RN" --method_name "$NAME" --fold "$FOLD" --seed "$SEED" \
      --model_name "$MODEL" --pooling "$POOL" "${EXTRA[@]}" \
      --max_lines "$MAXLINES" --max_line_tokens "$MAXLTOK" --graph_hidden "$GHID" \
      --symbolize "$SYMB" --max_degree 0 --node_mode "$NMODE" --chunk_budget "$CBUD" \
      --batch_size 4 --eval_batch_size 8 --micro 64 \
      --learning_rate 2e-5 --graph_lr "$GLR" --weight_decay 0.01 --warmup_ratio 0.10 \
      --epochs 30 --min_epochs 3 --patience 8 --selection_metric roc_auc --sam_rho 0.02 \
      >> "log/${RN}/${NAME}_fold${FOLD}.log" 2>&1
    if [ -f "$OUT" ]; then echo "  $(ts) $NAME fold $FOLD XONG"
    else echo "  !! $NAME fold $FOLD THAT BAI — log/${RN}/${NAME}_fold${FOLD}.log"
         tail -5 "log/${RN}/${NAME}_fold${FOLD}.log" | sed 's/^/       /'; fi
  done
  echo "----- $(ts) | het fold $FOLD | o: $(find results/${RN} -name 'fold*.json' 2>/dev/null | wc -l)/${N_EXP} -----"
done
HAVE=$(find results/${RN} -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## BABEL xong $(ts) | $HAVE/$N_EXP o ##########"
[ "$HAVE" = "$N_EXP" ] || echo "  !! THIEU $((N_EXP-HAVE)) o — doi chieu truoc khi bao ket qua"
