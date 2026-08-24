#!/usr/bin/env bash
# Does Phase 1 reproduce at a fixed seed?
#
# Two Phase-1 checkpoints in this project were trained from identical arguments
# -- same source file, same seed 36, same 15 epochs, same lr, lora_rank 0,
# lp_epochs 0, source_interpolation 1.0, max_train_samples None -- and landed on
# different weights:
#
#   frac114_codebert-base/transfer_cwe  best epoch 3   source val macro-F1 0.5450
#   frac228_codebert-base/transfer_cwe  best epoch 13  source val macro-F1 0.6654
#
# 0.12 apart, while the transfer effect being measured is 0.04. If that spread is
# real, one Phase-1 draw is a larger source of variance than the method, and the
# fold-major design makes it worse rather than better: all five folds share one
# Phase-1 checkpoint, so they are five measurements of a single draw, not five
# independent samples of the method. Every paired test across folds in RESULT.md
# would then understate the uncertainty about the method itself.
#
# src/train_transfer.py already sets manual_seed, cuda.manual_seed_all,
# cudnn.deterministic=True, cudnn.benchmark=False, num_workers=0 and seeds the
# loader generator, so the expectation is that it reproduces. This runs the same
# Phase 1 three times into separate directories and prints the source validation
# score and a weight fingerprint for each.
#
#   identical fingerprints -> Phase 1 is reproducible, and the frac114/frac228
#                             gap came from something else that must be found.
#   differing fingerprints -> the seed does not pin Phase 1, single-draw
#                             comparisons are unsound, and every arm must share
#                             one checkpoint or average over several draws.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf

# Two questions, not one.
#
#   REPEATS at a fixed seed  -> can a reported number be reproduced at all?
#   One run at each of SEEDS -> how far apart do independent Phase-1 draws land?
#
# The second is what actually sizes the uncertainty. Even a perfectly
# reproducible Phase 1 leaves the design problem intact: one draw is shared by
# all five folds, so the folds resample the split and never the source model.
# The spread across seeds is the noise floor that a 0.04 effect has to clear.
SEED="${SEED:-36}"
REPEATS="${REPEATS:-3}"
SEEDS="${SEEDS:-7 12 18}"

run_phase1() {  # $1 = method_name (output dir), $2 = seed
  python -u src/train_transfer.py --phase phase1 \
    --run_name determinism --method_name "$1" \
    --data_path data/train_ccpp_js.jsonl --data_root data/sven_python_twin \
    --model_name microsoft/codebert-base --pooling cls \
    --aux_mode cwe --cwe_vocab fixed4 --num_latent 8 \
    --seed "$2" --fold 1 --epochs 15 --min_epochs 3 --patience 5 \
    --batch_size 16 --eval_batch_size 16 --max_length 512 \
    --truncation_strategy head_middle_tail --learning_rate 2e-5 \
    --weight_decay 0.01 --lambda_cwe 0.2 --max_grad_norm 1.0 --num_workers 0 \
    2>&1 | tail -3
}

for i in $(seq 1 "$REPEATS"); do
  echo "=== same seed $SEED, repeat $i/$REPEATS ==="
  run_phase1 "same${SEED}_rep$i" "$SEED"
done

for s in $SEEDS; do
  echo "=== independent draw, seed $s ==="
  run_phase1 "draw_seed$s" "$s"
done

echo
echo "=== source val macro-F1 and weight fingerprint per run ==="
python - <<'PY'
import glob, hashlib, statistics, torch

repeats, draws = {}, {}
for path in sorted(glob.glob("model/determinism/*/seed_*/source/best.pt")):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = next(v for k, v in checkpoint.items() if k.endswith("state_dict"))
    first = next(iter(state))
    fingerprint = hashlib.md5(state[first].float().numpy().tobytes()).hexdigest()[:8]
    name = path.split("/")[2]
    score = checkpoint["best_val_macro_f1"]
    print(f"{name:<18} best epoch {checkpoint['best_epoch']:<3} "
          f"source val macro-F1 {score:.4f}  weights {fingerprint}")
    (repeats if name.startswith("same") else draws)[name] = (score, fingerprint)

print()
prints = {f for _, f in repeats.values()}
if len(prints) == 1:
    print(f"Fixed seed, {len(repeats)} repeats: one fingerprint. Phase 1 reproduces.")
    print("The frac114/frac228 gap then came from something other than the seed,")
    print("and that cause still has to be found before those runs can be compared.")
else:
    scores = [s for s, _ in repeats.values()]
    print(f"Fixed seed, {len(repeats)} repeats: {len(prints)} fingerprints, "
          f"val spread {max(scores) - min(scores):.4f}. The seed does not pin Phase 1.")

if len(draws) > 1:
    scores = [s for s, _ in draws.values()]
    spread = max(scores) - min(scores)
    print(f"\nIndependent draws across {len(draws)} seeds: val "
          f"{min(scores):.4f}-{max(scores):.4f}, spread {spread:.4f}, "
          f"sd {statistics.stdev(scores):.4f}")
    print("This spread is the noise floor a 0.04 transfer effect has to clear, and")
    print("no amount of fold-splitting samples it -- all five folds share one draw.")
PY
touch /workspace/DETERMINISM_DONE
