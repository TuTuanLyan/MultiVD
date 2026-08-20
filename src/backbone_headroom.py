"""Is the transfer backbone-dependent, or is the backbone just running out of room?

Three interventions aimed at protecting a strong backbone during Phase 2 all
failed, which leaves the raw pattern unexplained: the gain shrinks as the
backbone gets stronger. A shrinking absolute gain is what a ceiling looks like,
so the same numbers are re-read as a fraction of the error still available to
remove.
"""
import glob, json, statistics
from pathlib import Path

ROOTS = {
    "CodeBERT  (orig folds)": ("results_vast/auxmatrix_ccppjs", "transfer_cwe"),
    "CodeBERT  (twin folds)": ("results_vast/twin_ccppjs", "transfer_cwe"),
    "CodeT5                ": ("results_vast/backbone_codet5", "transfer"),
    "CodeT5+               ": ("results_vast/backbone_codet5p", "transfer"),
    "CodeT5+   (twin folds)": ("results_vast/t5p_twin", "transfer_cwe"),
}
METRIC = "test_macro_f1_at_0.5"

def load(d):
    out = {}
    for p in glob.glob(f"{d}/seed_*/fold*.json"):
        r = json.load(open(p))
        if r.get(METRIC) is not None:
            out[(Path(p).parent.name, int(r["fold"]))] = float(r[METRIC])
    return out

print(f"{'backbone':<24}{'n':>3}{'baseline':>10}{'transfer':>10}{'gain':>9}"
      f"{'headroom':>10}{'% of error removed':>20}")
print("-" * 86)
rows = []
for name, (root, method) in ROOTS.items():
    base, meth = load(f"{root}/baseline"), load(f"{root}/{method}")
    keys = sorted(set(base) & set(meth))
    if not keys:
        continue
    b = statistics.mean(base[k] for k in keys)
    m = statistics.mean(meth[k] for k in keys)
    # Per fold, then average: the fraction of the still-missing F1 that the
    # transfer recovers. Averaging the ratio is not the ratio of the averages.
    frac = statistics.mean((meth[k] - base[k]) / (1.0 - base[k]) for k in keys)
    rows.append((name, len(keys), b, m, m - b, 1 - b, frac))
    print(f"{name:<24}{len(keys):>3}{b:>10.4f}{m:>10.4f}{m-b:>+9.4f}"
          f"{1-b:>10.4f}{100*frac:>19.1f}%")

if len(rows) > 1:
    gains = [r[4] for r in rows]
    fracs = [r[6] for r in rows]
    print(f"\nspread of the absolute gain          : {max(gains)-min(gains):.4f}")
    print(f"spread of the error-removed fraction : {max(fracs)-min(fracs):.4f}")
    print("\nIf the second spread is the smaller one, the backbones differ in how much\n"
          "room they leave, not in how well the transfer uses it.")
