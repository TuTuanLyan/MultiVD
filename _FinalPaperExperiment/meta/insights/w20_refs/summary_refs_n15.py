"""Tổng kết hai dòng mốc mới của tab warmup 0,2 (khối 158, 09/10): baseline_w20* (baseline warmup 0,2, không dừng sớm) và
nop1_adamw* (MW only, seed 42 = nop1_adamw ở tab Kết quả gốc), n=15, ghép cặp theo (seed, fold) - và ba dòng SOTA rasam_w20_* so với từng mốc.
p: Wilcoxon trên 5 trung bình theo FOLD (các ô cùng fold không độc lập - [[repeated-measures-p-value-per-fold]]).
Chạy: /home/ntat/miniconda3/envs/vdenv/bin/python meta/insights/w20_refs/summary_refs_n15.py (cần scipy)."""
import json, os
from scipy.stats import wilcoxon
M = [("test_roc_auc", "ROC"), ("test_pr_auc", "PR"), ("test_macro_f1_at_0.5", "F1@0.5"), ("test_macro_f1_at_valcal", "F1@val")]
SEEDS = [(42, ""), (1234, "_s1234"), (7, "_s7")]
SRC = ["common_ext", "4cwe", "full"]
REF = {
    "baseline gốc": lambda s, sfx: f"baseline{sfx}",
    "baseline warmup 0,2": lambda s, sfx: f"baseline_w20{sfx}",
    "MW only": lambda s, sfx: "nop1_adamw" if s == 42 else f"nop1_adamw{sfx}",
}
def load(run, k):
    p = f"results/{run}/fold{k}.json"
    return json.load(open(p)) if os.path.exists(p) else None
def pairs(arm, ref):
    """arm(s, sfx) -> danh sách tên run (một hoặc ba nguồn); ref(s, sfx) -> tên run mốc."""
    R = []
    for k in range(1, 6):
        for s, sfx in SEEDS:
            B = load(ref(s, sfx), k)
            for run in arm(s, sfx):
                A = load(run, k)
                if A and B: R.append(dict(k=k, s=s, run=run, A=A, B=B))
    return R
def block(rows, title, la, lb):
    print(f"== {title} (n={len(rows)} ô ghép cặp)")
    if not rows: return
    for m, l in M:
        d = [r["A"][m] - r["B"][m] for r in rows]
        a = sum(r["A"][m] for r in rows) / len(rows); b = sum(r["B"][m] for r in rows) / len(rows)
        fm = [sum(r["A"][m] - r["B"][m] for r in rows if r["k"] == k) / max(1, sum(1 for r in rows if r["k"] == k)) for k in range(1, 6)]
        try: p = wilcoxon(fm).pvalue
        except ValueError: p = float("nan")
        print(f"  {l:7s} {la} {a:.4f} / {lb} {b:.4f} | Δ TB {sum(d)/len(d):+.4f} trung vị {sorted(d)[len(d)//2]:+.4f} "
              f"+{sum(x > 0.001 for x in d)}/-{sum(x < -0.001 for x in d)} min {min(d):+.3f} max {max(d):+.3f} | "
              f"theo fold {' '.join(f'{x:+.4f}' for x in fm)} ({sum(x > 0 for x in fm)}/5 dương) p={p:.3f}")
# 1. hai mốc mới so với baseline gốc
block(pairs(lambda s, sfx: [f"baseline_w20{sfx}"], REF["baseline gốc"]), "Baseline warmup 0,2 - baseline gốc", "w20", "gốc")
R = pairs(lambda s, sfx: [f"baseline_w20{sfx}"], REF["baseline gốc"])
block([r for r in R if not (r["s"] == 7 and r["k"] == 1)], "  như trên, BỎ ô sập seed 7 fold 1 của baseline gốc", "w20", "gốc")
block(pairs(lambda s, sfx: ["nop1_adamw" if s == 42 else f"nop1_adamw{sfx}"], REF["baseline gốc"]), "MW only - baseline gốc", "MW", "gốc")
block(pairs(lambda s, sfx: ["nop1_adamw" if s == 42 else f"nop1_adamw{sfx}"], REF["baseline warmup 0,2"]), "MW only - baseline warmup 0,2", "MW", "w20")
# 2. ba dòng SOTA so với từng mốc
for name, ref in REF.items():
    sota = lambda s, sfx: [f"rasam_w20_jspy41jsval_{src}{sfx}" for src in SRC]
    block(pairs(sota, ref), f"SOTA gộp 3 nguồn - {name}", "SOTA", "mốc")
    for src in SRC:
        block(pairs(lambda s, sfx, src=src: [f"rasam_w20_jspy41jsval_{src}{sfx}"], ref), f"  {src} - {name}", src, "mốc")
