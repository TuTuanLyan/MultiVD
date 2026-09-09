#!/usr/bin/env python3
"""rho_paired.py — so HAI muc rho GHEP CAP TRONG CUNG O.

Khoi `chot` tinh co tao ra dung phep so manh nhat cho cau hoi "rho nao hop codebert":
8 o o rho=2.0 va cac o o rho=0.1 nam trong CUNG mot cay, CUNG fold, CUNG nguon, CUNG
baseline, CUNG may, CUNG phien. Nen thay vi lay hieu cua hai Delta tinh rieng o hai khoi
khac nhau (nhu §32 buoc phai lam, va do la cho yeu cua §32), o day ghep cap duoc that.

Day dung la bai hoc §25.9: hai so co n khac nhau thi dung tru nhau. Ghep cap thi khong con
van de do.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

MET = [("F1@0.5","test_macro_f1_at_0.5"), ("F1@val","test_macro_f1_at_valcal"),
       ("ROC-AUC","test_roc_auc"), ("PR-AUC","test_pr_auc")]

def load(root, tags):
    cells = defaultdict(dict)
    for dp,_,fns in os.walk(root):
        for fn in fns:
            m = re.fullmatch(r"fold(\d+)\.json", fn)
            if not m: continue
            p = os.path.join(dp,fn)
            arm = os.path.basename(os.path.dirname(os.path.dirname(p)))
            tag = next((t for t in tags if arm.endswith("_"+t)), None)
            if tag is None: continue
            src = next((s for s in ("4cwe","com","full") if f"_{s}_" in arm), "-")
            seed = os.path.basename(os.path.dirname(p))
            cells[(src, seed, int(m.group(1)))][tag] = json.load(open(p))
    return cells

def main():
    root = sys.argv[1] if len(sys.argv)>1 else "results/chot_codebert"
    a, b = (sys.argv[2], sys.argv[3]) if len(sys.argv)>3 else ("r2p0","r0p1")
    cells = load(root, (a,b))
    pairs = [(v[a], v[b], k) for k,v in cells.items() if a in v and b in v]
    print(f"# {root} | {a} vs {b} | {len(pairs)} o ghep cap duoc "
          f"(co {sum(1 for v in cells.values() if a in v)} o {a}, "
          f"{sum(1 for v in cells.values() if b in v)} o {b})")
    if not pairs: print("# chua du de so"); return
    srcs = sorted({k[0] for _,_,k in pairs})
    print(f"\n{'nguon':<8}{'n':>3}  " + "".join(f"{m:>26}" for m,_ in MET))
    for grp in srcs + (["GOP"] if len(srcs)>1 else []):
        sub = pairs if grp=="GOP" else [p for p in pairs if p[2][0]==grp]
        line = f"{grp:<8}{len(sub):>3}  "
        for _,f in MET:
            v = np.array([x[f]-y[f] for x,y,_ in sub if x.get(f) is not None and y.get(f) is not None])
            if not len(v): line += f"{'-':>26}"; continue
            nz = v[v!=0]; pos = int((nz>0).sum())
            p = binomtest(pos,len(nz),.5).pvalue if len(nz) else float("nan")
            line += f"{v.mean():>+10.4f} {pos:>2d}/{len(v):<3d} p={p:<6.4f}"
        print(line)
    print(f"\n# Dau DUONG nghia la {a} hon {b}. San nhieu 0.010 (cung loai GPU).")
    print(f"# n={len(pairs)}: san cua phep thu dau la p={2**-(len(pairs)-1):.4f}" if len(pairs)<=8 else "")

if __name__ == "__main__": main()
