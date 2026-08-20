#!/usr/bin/env bash
# Is +0.042 a property of the method, or of one source model?
#
# Every transfer number in RESULT.md comes from a single Phase-1 checkpoint,
# reused across all five folds. The folds resample the target split; nothing in
# the design ever resamples the source model. And the eight valid Phase-1
# checkpoints trained on this corpus spread over sd 0.0202 on source validation,
# the same order as the 0.042 effect being reported.
#
# So the reported claim is really "with this source model, the method gains
# 0.042 over five folds" -- not "the method gains 0.042". This separates them.
#
# run/determinism.sh leaves four Phase-1 checkpoints behind: three repeats at
# seed 36 and one draw each at seeds 7, 12, 18. This reuses them as source
# models and holds everything downstream fixed -- Phase-2 seed 36, the same
# three folds, the same baseline. The source checkpoint is the only thing that
# moves, so the spread of the deltas is attributable to it alone.
#
#   deltas cluster       -> the effect belongs to the method, and the single-draw
#                           tables stand as written.
#   deltas swing         -> the effect belongs to a draw, every delta in the
#                           document needs a source-model interval around it, and
#                           the method has to be reported as an average over draws.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls
export CWE_VOCAB=fixed4
export SEED=36
export PHASE1_DATA_PATH=data/train_ccpp_js.jsonl

# method_name inside the determinism run -> the seed its Phase 1 was trained at
declare -A DRAWS=(
  [same36_rep1]=36
  [draw_seed7]=7
  [draw_seed12]=12
  [draw_seed18]=18
)

for NAME in "${!DRAWS[@]}"; do
  SRC="model/determinism/$NAME/seed_${DRAWS[$NAME]}/source/best.pt"
  RUN="draw_$NAME"
  if [[ ! -f "$SRC" ]]; then
    echo "missing $SRC -- run/determinism.sh has to finish first"; continue
  fi
  mkdir -p "model/$RUN/transfer_cwe/seed_$SEED/source"
  cp "$SRC" "model/$RUN/transfer_cwe/seed_$SEED/source/best.pt"

  # The baseline never sees source data, so it is identical across draws and is
  # reused rather than retrained four times.
  mkdir -p "results/$RUN/baseline/seed_$SEED"
  cp results/twin_ccppjs/baseline/seed_$SEED/fold[123].json \
     "results/$RUN/baseline/seed_$SEED/" 2>/dev/null

  echo "=== source draw $NAME (Phase 1 trained at seed ${DRAWS[$NAME]}) ==="
  RUN_NAME="$RUN" FOLDS="1 2 3" MODES="cwe" bash run/fold-major.sh
done

echo
echo "=== delta per source draw, Phase 2 held at seed $SEED, folds 1-3 ==="
python - <<'PY'
import glob, json, statistics
from pathlib import Path

METRIC = "test_macro_f1_at_0.5"


def load(directory):
    scores = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        record = json.load(open(path))
        if record.get(METRIC) is not None:
            scores[int(record["fold"])] = float(record[METRIC])
    return scores


deltas = {}
for base_dir in sorted(glob.glob("results/draw_*/baseline/seed_36")):
    run = Path(base_dir).parts[1]
    base = load(base_dir)
    method = load(f"results/{run}/transfer_cwe/seed_36")
    folds = sorted(set(base) & set(method))
    if not folds:
        continue
    deltas[run] = statistics.mean(method[f] - base[f] for f in folds)
    print(f"{run:<24}n={len(folds)}  baseline {statistics.mean(base[f] for f in folds):.4f}"
          f"  transfer {statistics.mean(method[f] for f in folds):.4f}"
          f"  delta {deltas[run]:+.4f}")

if len(deltas) > 1:
    values = list(deltas.values())
    spread = max(values) - min(values)
    print(f"\nacross {len(values)} source draws: delta {min(values):+.4f} to {max(values):+.4f}, "
          f"spread {spread:.4f}, sd {statistics.stdev(values):.4f}")
    print(f"every draw agrees on the sign: {all(v > 0 for v in values) or all(v < 0 for v in values)}")
    print("\nCompare the spread against the 0.042 headline. A spread of that size means")
    print("the headline is one sample from this distribution, not an estimate of its centre.")
PY
touch /workspace/SOURCEDRAWS_DONE
