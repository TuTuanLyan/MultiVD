"""Tổng kết tab warmup 0,2 + không patience (rasam_w20_*) so với tab paper đầu (rasam_jspy41jsval_*), n=15/nguồn, ghép cặp (nguồn, seed, fold).
p: Wilcoxon trên 5 trung bình theo FOLD (3 seed × 3 nguồn = 9 ô/fold không độc lập - [[repeated-measures-p-value-per-fold]])."""
import json, os
from scipy.stats import wilcoxon
M = [("test_roc_auc","ROC"),("test_pr_auc","PR"),("test_macro_f1_at_0.5","F1@0.5"),("test_macro_f1_at_valcal","F1@val")]
R = []
for k in range(1,6):
  for src in ["common_ext","4cwe","full"]:
    for s,sfx in [(42,""),(1234,"_s1234"),(7,"_s7")]:
      a=f"results/rasam_w20_jspy41jsval_{src}{sfx}/fold{k}.json"; b=f"results/rasam_jspy41jsval_{src}{sfx}/fold{k}.json"
      if os.path.exists(a) and os.path.exists(b):
        A=json.load(open(a)); B=json.load(open(b)); R.append(dict(k=k,src=src,s=s,A=A,B=B))
print("n ô ghép cặp =", len(R))
def block(rows, title):
    print(f"== {title} (n={len(rows)})")
    for m,l in M:
        d=[r["A"][m]-r["B"][m] for r in rows]; a=sum(r["A"][m] for r in rows)/len(rows); b=sum(r["B"][m] for r in rows)/len(rows)
        folds=sorted({r["k"] for r in rows}); fm=[sum(r["A"][m]-r["B"][m] for r in rows if r["k"]==k)/sum(1 for r in rows if r["k"]==k) for k in folds]
        try: p=wilcoxon(fm).pvalue if len(fm)>=5 else float('nan')
        except ValueError: p=float('nan')
        print(f"  {l:7s} mới {a:.4f} / paper {b:.4f} | Δ TB {sum(d)/len(d):+.4f} trung vị {sorted(d)[len(d)//2]:+.4f} +{sum(x>0.001 for x in d)}/-{sum(x<-0.001 for x in d)} "
              f"min {min(d):+.3f} max {max(d):+.3f} | theo fold {' '.join(f'{x:+.4f}' for x in fm)} ({sum(x>0 for x in fm)}/5 fold dương) p={p:.3f}")
block(R, "Gộp 3 nguồn")
for src in ["common_ext","4cwe","full"]: block([r for r in R if r["src"]==src], src)
for s in [42,1234,7]: block([r for r in R if r["s"]==s], f"seed {s}")
