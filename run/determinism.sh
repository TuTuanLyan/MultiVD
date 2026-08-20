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

SEED="${SEED:-36}"
REPEATS="${REPEATS:-3}"
for i in $(seq 1 "$REPEATS"); do
  echo "=== Phase 1 repeat $i/$REPEATS, seed $SEED ==="
  python -u src/train_transfer.py --phase phase1 \
    --run_name determinism --method_name "rep$i" \
    --data_path data/train_ccpp_js.jsonl --data_root data/sven_python_twin \
    --model_name microsoft/codebert-base --pooling cls \
    --aux_mode cwe --cwe_vocab fixed4 --num_latent 8 \
    --seed "$SEED" --fold 1 --epochs 15 --min_epochs 3 --patience 5 \
    --batch_size 16 --eval_batch_size 16 --max_length 512 \
    --truncation_strategy head_middle_tail --learning_rate 2e-5 \
    --weight_decay 0.01 --lambda_cwe 0.2 --max_grad_norm 1.0 --num_workers 0 \
    2>&1 | tail -3
done

echo
echo "=== source val macro-F1 and first-tensor fingerprint per repeat ==="
python - <<'PY'
import glob, hashlib, torch

seen = {}
for path in sorted(glob.glob("model/determinism/*/seed_*/source/best.pt")):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = next(v for k, v in checkpoint.items() if k.endswith("state_dict"))
    first = next(iter(state))
    fingerprint = hashlib.md5(state[first].float().numpy().tobytes()).hexdigest()[:8]
    name = path.split("/")[2]
    print(f"{name:<8} best epoch {checkpoint['best_epoch']:<3} "
          f"source val macro-F1 {checkpoint['best_val_macro_f1']:.4f}  weights {fingerprint}")
    seen.setdefault(fingerprint, []).append(name)

if len(seen) == 1:
    print("\nOne fingerprint across every repeat: Phase 1 is reproducible at a fixed seed.")
else:
    print(f"\n{len(seen)} distinct fingerprints: the seed does not pin Phase 1.")
    print("Arms that trained their own Phase 1 are not comparable to arms that")
    print("reused another one, and a single draw cannot support a 0.04 effect.")
PY
touch /workspace/DETERMINISM_DONE
