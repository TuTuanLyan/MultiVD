#!/usr/bin/env python3
"""chot_report.py — bao cao RIENG cho khoi `chot` (cau hinh chot tren hai backbone).

Nguoi dung 09/09: "cai nay se tong hop rieng ki nhe."

Khoi nay co dung HAI dieu kien, nen bao cao phai tra loi ba cau, khong phai mot:
  1. A vs baseline   — day du optimizer co hon mo hinh chi-dich khong?
  2. B vs baseline   — tat ca hai optimizer thi con hon khong?
  3. A vs B          — RIENG optimizer dong gop bao nhieu? (ghep cap trong CUNG o)
Cau 3 la cau moi: no tach phan cua optimizer khoi phan cua Pha 1 + head.

Moi Delta ghep cap theo (backbone, nguon, seed, fold) va CHI trong cung cay ket qua —
khong bao gio bac cau qua may (CLAUDE.md muc 2 va muc 4).
Luon in DU BON chi so kem so fold cung dau (muc 2b).
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest

MET = [("F1@0.5","test_macro_f1_at_0.5"), ("F1@val","test_macro_f1_at_valcal"),
       ("ROC-AUC","test_roc_auc"), ("PR-AUC","test_pr_auc")]
CWEN = {0:"CWE-022",1:"CWE-078",2:"CWE-079",3:"CWE-089"}

def load(roots):
    cells = defaultdict(dict)
    for root in roots:
        if not os.path.isdir(root): continue
        bb = "codebert" if "codebert" in root else ("t5p" if "t5p" in root else "?")
        for dp,_,fns in os.walk(root):
            for fn in fns:
                m = re.fullmatch(r"fold(\d+)\.json", fn)
                if not m: continue
                parts = os.path.relpath(os.path.join(dp,fn), root).split(os.sep)
                if len(parts) < 3: continue
                arm, seed, fold = parts[-3], parts[-2], int(m.group(1))
                with open(os.path.join(dp,fn)) as f: d = json.load(f)
                src = "-"
                for s in ("4cwe","com","full"):
                    if f"_{s}_" in arm: src = s; break
                tag = "baseline" if arm=="baseline" else ("r2p0" if "r2p0" in arm else
                      ("plain" if "plain" in arm else arm))
                cells[(root,bb,src,seed,fold)][tag] = d
    return cells

def stat(v):
    v=np.asarray(v,float); v=v[~np.isnan(v)]
    if not len(v): return None
    nz=v[v!=0]; pos=int((nz>0).sum())
    p=binomtest(pos,len(nz),.5).pvalue if len(nz) else float("nan")
    return v.mean(), pos, len(v), p

def fmt(s):
    if not s: return f"{'-':>24}"
    d,pos,n,p = s
    return f"{d:>+9.4f} {pos:>3d}/{n:<3d} p={p:<6.4f}"

def show(title, pairs):
    """pairs: {(bb,src): [ (a_cell, b_cell) ]}"""
    keys = sorted({k for k in pairs})
    if not keys: return
    print(f"\n=== {title} ===")
    print(f"{'backbone':<11}{'nguon':<8}{'n':>3}  " + "".join(f"{m:>24}" for m,_ in MET))
    for bb in sorted({k[0] for k in keys}):
        rows = [(k,v) for k,v in pairs.items() if k[0]==bb]
        allv = defaultdict(list)
        for (b,src),lst in sorted(rows):
            line = f"{b:<11}{src:<8}{len(lst):>3}  "
            for mn,f in MET:
                vs=[a[f]-c[f] for a,c in lst if a.get(f) is not None and c.get(f) is not None]
                for x in vs: allv[mn].append(x)
                line += fmt(stat(vs))
            print(line)
        if len(rows) > 1:
            line = f"{'':<11}{'GOP':<8}{sum(len(v) for _,v in rows):>3}  "
            for mn,_ in MET: line += fmt(stat(allv[mn]))
            print(line)

def main():
    roots = sys.argv[1:] or ["results/chot_t5p","results/chotv_t5p","results/chot_codebert"]
    cells = load(roots)
    have = sum(len(v) for v in cells.values())
    print(f"# {have} o trong {len(cells)} khoi (backbone, nguon, seed, fold) | cay: {', '.join(roots)}")
    have_tags = defaultdict(int)
    for v in cells.values():
        for t in v: have_tags[t]+=1
    print(f"# theo nhanh: {dict(have_tags)}")
    # Baseline nam o arm "baseline" nen src cua no la "-", trong khi hai nhanh mang
    # src that (4cwe/com/full). Neu khoa o gom ca src thi baseline KHONG BAO GIO gap
    # nhanh nao ca — bang A va B ra rong, va rong mot cach im lang. Baseline la CHUNG
    # cho moi nguon trong cung (cay, backbone, seed, fold), nen tra theo dung khoa do.
    # Giu `root` trong khoa de khong bao gio ghep cap bac cau qua may (CLAUDE.md muc 4).
    base_of = {}
    for (root,bb,src,seed,fold),v in cells.items():
        if "baseline" in v: base_of[(root,bb,seed,fold)] = v["baseline"]
    A=defaultdict(list); B=defaultdict(list); AB=defaultdict(list)
    for (root,bb,src,seed,fold),v in cells.items():
        if src == "-": continue                      # o baseline khong tu ghep voi chinh no
        b = base_of.get((root,bb,seed,fold))
        if b is not None:
            if "r2p0"  in v: A[(bb,src)].append((v["r2p0"], b))
            if "plain" in v: B[(bb,src)].append((v["plain"], b))
        if "r2p0" in v and "plain" in v: AB[(bb,src)].append((v["r2p0"], v["plain"]))
    thieu = sum(1 for (root,bb,src,seed,fold),v in cells.items()
                if src != "-" and base_of.get((root,bb,seed,fold)) is None)
    if thieu: print(f"# CANH BAO: {thieu} o co nhanh nhung KHONG co baseline cung fold — da bo")
    show("A  (RecAdam + ASAM 2.0)  −  baseline", A)
    show("B  (AdamW, khong SAM)    −  baseline", B)
    show("A − B   RIENG phan optimizer dong gop (ghep cap trong CUNG o)", AB)

    # per-CWE cho cau hoi chinh cua ca du an
    from sklearn.metrics import roc_auc_score
    print("\n=== A − baseline, DROC-AUC theo CWE ===")
    print(f"{'backbone':<11}" + "".join(f"{CWEN[c]:>22}" for c in sorted(CWEN)))
    for bb in sorted({k[0] for k in A}):
        acc=defaultdict(list)
        for (b,src),lst in A.items():
            if b!=bb: continue
            for a,base in lst:
                if "test_probabilities" not in a or "test_cwe_classes" not in a: continue
                y=np.asarray(a["test_labels"]); cw=np.asarray(a["test_cwe_classes"])
                pa=np.asarray(a["test_probabilities"],float); pb=np.asarray(base["test_probabilities"],float)
                for c in sorted(set(cw.tolist())):
                    m=cw==c
                    if len(set(y[m].tolist()))<2: continue
                    acc[c].append(roc_auc_score(y[m],pa[m])-roc_auc_score(y[m],pb[m]))
        line=f"{bb:<11}"
        for c in sorted(CWEN):
            s=stat(acc.get(c,[]))
            line += (f"{s[0]:>+9.4f} {s[1]:>3d}/{s[2]:<3d}" if s else f"{'-':>22}")
        print(line)

if __name__ == "__main__":
    main()
