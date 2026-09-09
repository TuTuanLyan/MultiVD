#!/usr/bin/env python3
"""chot_export.py — xuat toan bo khoi `chot` ra JSON cho trang ket qua rieng cua khoi.

Moi Delta ghep cap voi `baseline` CUNG (cay, backbone, seed, fold). Nhanh A la nhanh ASAM
o rho tot nhat CUA CHINH backbone do (t5p r2p0, codebert r0p1) — 8 o codebert o rho=2.0 la
BANG CHUNG cua mot lua chon da bo, xuat rieng duoi nhan `evid`.
"""
import json, os, re, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

MET = [("f1","test_macro_f1_at_0.5"), ("f1v","test_macro_f1_at_valcal"),
       ("roc","test_roc_auc"), ("pr","test_pr_auc")]
CWEN = {0:"CWE-022", 1:"CWE-078", 2:"CWE-079", 3:"CWE-089"}
KEEP = {"codebert":"r0p1", "t5p":"r2p0"}

def main():
    roots = ["results/chot_t5p","results/chotv_t5p","results/chot_codebert"]
    cells = defaultdict(dict)
    for root in roots:
        bb = "codebert" if "codebert" in root else "t5p"
        for dp,_,fns in os.walk(root):
            for fn in fns:
                m = re.fullmatch(r"fold(\d+)\.json", fn)
                if not m: continue
                p = os.path.join(dp,fn)
                arm = os.path.basename(os.path.dirname(os.path.dirname(p)))
                src = next((s for s in ("4cwe","com","full") if f"_{s}_" in arm), "-")
                mr = re.search(r"_(r\d+p\d+)$", arm)
                tag = ("baseline" if arm=="baseline"
                       else "A" if (mr and mr.group(1)==KEEP[bb])
                       else "evid" if mr
                       else "B" if "plain" in arm else None)
                if tag is None: continue
                d = json.load(open(p))
                cells[(bb,src,int(m.group(1)))][tag] = d
                if tag == "evid": d["_rho"] = mr.group(1)
    base = {(k[0],k[2]): v["baseline"] for k,v in cells.items() if "baseline" in v}

    rows, cwe, absv = [], [], []
    for (bb,src,fold),v in sorted(cells.items()):
        b = base.get((bb,fold))
        for tag in ("A","B","evid"):
            if tag not in v or b is None or src == "-": continue
            a = v[tag]
            e = dict(bb=bb, src=src, fold=fold, arm=tag,
                     rho=a.get("_rho") or (KEEP[bb] if tag=="A" else "r0"),
                     p1val=a.get("phase1_val_macro_f1"), ep=a.get("best_epoch"))
            for k,f in MET:
                e[k] = None if a.get(f) is None or b.get(f) is None else round(a[f]-b[f], 9)
                e[k+"_a"] = None if a.get(f) is None else round(a[f], 9)
                e[k+"_b"] = None if b.get(f) is None else round(b[f], 9)
            rows.append(e)
            # per-CWE
            y = np.asarray(a["test_labels"]); cw = np.asarray(a["test_cwe_classes"])
            pa = np.asarray(a["test_probabilities"], float); pb = np.asarray(b["test_probabilities"], float)
            for c in sorted(set(cw.tolist())):
                mm = cw == c
                if len(set(y[mm].tolist())) < 2: continue
                cwe.append(dict(bb=bb, src=src, fold=fold, arm=tag, cwe=CWEN.get(c, str(c)),
                                n=int(mm.sum()),
                                d=round(roc_auc_score(y[mm],pa[mm]) - roc_auc_score(y[mm],pb[mm]), 9)))
        if b is not None:
            e = dict(bb=bb, src="-", fold=fold, arm="baseline", rho="-", p1val=None, ep=b.get("best_epoch"))
            for k,f in MET: e[k]=None; e[k+"_a"]=round(b[f],9) if b.get(f) is not None else None; e[k+"_b"]=e[k+"_a"]
            absv.append(e)
    # so hang test moi CWE
    any_a = next(iter(cells.values()))["A"]
    cwn = {CWEN.get(c,str(c)): int((np.asarray(any_a["test_cwe_classes"])==c).sum())
           for c in sorted(set(np.asarray(any_a["test_cwe_classes"]).tolist()))}
    out = dict(rows=rows, cwe=cwe, base=absv, cwe_n=cwn,
               ntest=len(any_a["test_labels"]),
               gen=__import__("time").strftime("%Y-%m-%d %H:%M UTC", __import__("time").gmtime()))
    sys.stderr.write(f"# {len(rows)} dong nhanh | {len(absv)} baseline | {len(cwe)} dong per-CWE\n")
    json.dump(out, sys.stdout)

if __name__ == "__main__": main()
