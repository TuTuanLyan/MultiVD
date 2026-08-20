#!/usr/bin/env bash
# Does the latent bottleneck survive the Phase-1 draw lottery the way cwe did?
#
# At n=10 latent_bottleneck is the strongest arm: the only one clearing 0.05 on
# both tests (Macro-F1 Wilcoxon 0.0020, corrected t 0.027) and the only one
# clearing the ranking metric cwe misses (ROC-AUC 0.0059 against 0.0840). It also
# removes the four-CWE lock-in, so it is now the arm the argument rests on.
#
# But section 26 established robustness across Phase-1 draws for cwe only, and
# that check mattered: Phase 1 does not reproduce, and source validation swings
# with sd 0.0521. cwe came through with 5/5 draws positive at sd 0.0085. Nothing
# says the latent head has to behave the same way -- its Phase 1 optimises a
# different objective, routing through a K-dimensional bottleneck, and could
# plausibly be more sensitive to which draw it lands on, not less.
#
# Four independent Phase-1 checkpoints, then Phase 2 held at seed 36 on the same
# three folds against the same reused baseline, exactly as source-draws.sh did.
# The draw is the only thing that moves.
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

DRAW_SEEDS="${DRAW_SEEDS:-36 7 12 18}"

for S in $DRAW_SEEDS; do
  DST="model/latentdraw_$S/transfer_latent_bottleneck/seed_$SEED/source"
  mkdir -p "$DST"
  if [[ ! -f "$DST/best.pt" ]]; then
    echo "=== Phase 1 latent_bottleneck, draw seed $S ==="
    python -u src/train_transfer.py --phase phase1 \
      --run_name "latentdraw_$S" --method_name transfer_latent_bottleneck \
      --data_path "$PHASE1_DATA_PATH" --data_root "$DATA_ROOT" \
      --model_name "$MODEL_NAME" --pooling "$POOLING" \
      --aux_mode latent_bottleneck --cwe_vocab fixed4 --num_latent 8 \
      --latent_temperature 0.1 --freeze_prototypes_steps 0 \
      --seed "$S" --fold 1 --epochs 15 --min_epochs 3 --patience 5 \
      --batch_size 16 --eval_batch_size 16 --max_length 512 \
      --truncation_strategy head_middle_tail --learning_rate 2e-5 \
      --weight_decay 0.01 --lambda_cwe 0.2 --max_grad_norm 1.0 --num_workers 0 \
      --checkpoint_path "$DST/best.pt" 2>&1 | tail -3
  fi

  # The baseline never sees source data, so it is reused rather than retrained.
  mkdir -p "results/latentdraw_$S/baseline/seed_$SEED"
  cp results/twin_ccppjs/baseline/seed_$SEED/fold[123].json \
     "results/latentdraw_$S/baseline/seed_$SEED/" 2>/dev/null

  echo "=== Phase 2 from draw seed $S ==="
  RUN_NAME="latentdraw_$S" FOLDS="1 2 3" MODES="latent_bottleneck" bash run/fold-major.sh
done

echo
echo "=== latent_bottleneck across Phase-1 draws, Phase 2 held at seed $SEED ==="
python - <<'PY'
import glob, hashlib, json, statistics, torch
from pathlib import Path

METRICS = ["test_macro_f1_at_0.5", "test_roc_auc"]


def load(directory, metric):
    out = {}
    for path in glob.glob(f"{directory}/fold*.json"):
        record = json.load(open(path))
        if record.get(metric) is not None:
            out[int(record["fold"])] = float(record[metric])
    return out


runs = sorted(glob.glob("results/latentdraw_*/transfer_latent_bottleneck/seed_36"))
for metric in METRICS:
    deltas, seen = {}, {}
    for run_dir in runs:
        run = Path(run_dir).parts[1]
        base = load(f"results/{run}/baseline/seed_36", metric)
        meth = load(run_dir, metric)
        folds = sorted(set(base) & set(meth))
        if len(folds) < 3:
            continue
        deltas[run] = statistics.mean(meth[f] - base[f] for f in folds)
        ckpt = f"model/{run}/transfer_latent_bottleneck/seed_36/source/best.pt"
        try:
            c = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = next(v for k, v in c.items() if k.endswith("state_dict"))
            first = next(iter(state))
            seen[run] = (c["best_val_macro_f1"],
                         hashlib.md5(state[first].float().numpy().tobytes()).hexdigest()[:8])
        except Exception:
            seen[run] = (float("nan"), "--------")
    if not deltas:
        continue
    print(f"\n### {metric}")
    for run, d in sorted(deltas.items(), key=lambda kv: kv[1]):
        val, fp = seen[run]
        print(f"  {run:<20} val nguon {val:.4f}  weights {fp}  delta {d:+.4f}")
    values = list(deltas.values())
    if len(values) > 1:
        print(f"  -> {len(values)} lan rut: {min(values):+.4f} .. {max(values):+.4f}, "
              f"sd {statistics.stdev(values):.4f}, duong {sum(v > 0 for v in values)}/{len(values)}")
PY
touch /workspace/LATENTDRAWS_DONE
