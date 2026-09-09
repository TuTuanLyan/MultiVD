#!/usr/bin/env python3
"""ensemble.py — TRON XAC SUAT giua baseline (chi-dich) va mo hinh chuyen giao.

CAU HOI: §23 do duoc chuyen giao chi an tren HAI CWE HIEM (022 +0.161, 079 +0.165)
va LO nhe tren hai CWE thuong (078 -0.022, 089 -0.024). Neu dung vay thi mot phep
tron co the giu phan an va tra lai phan lo -> tong loi hon HAN ca hai dau mut.
Do la mot PHUONG THUC (khong phu thuoc backbone, khong can nhan CWE luc suy luan),
khong phai mot sieu tham so.

KHONG TON GPU: moi o da luu `test_probabilities`, `test_labels`, `test_cwe_classes`.
Ghep cap theo (cay ket qua, nhanh, seed, fold) — dung nguyen tac CLAUDE.md muc 2,
va CHI ghep trong CUNG cay (muc 4).

GIOI HAN PHAI NEU RO: khong o nao luu xac suat tren tap VAL, nen NGUONG hieu chinh
cua ban tron khong tinh duoc ngoai tuyen. Bao cao F1@0.5, ROC-AUC, PR-AUC (ba chi so
KHONG phu thuoc nguong hieu chinh) va bo trong F1@val — bia mot nguong tu tap TEST
la ro ri. Muon co F1@val phai chay lai co luu xac suat val.
"""
import argparse, json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

BACKBONES = ("t5pe", "t5p", "codebert", "unixcoder", "roberta")

def backbone_of(cell, root, run):
    """t5pe phai dung TRUOC t5p: 'asam1_t5pe' chua ca hai chuoi."""
    hay = f"{cell.get('experiment_name','')} {root} {run}"
    for b in BACKBONES:
        if b in hay: return b
    return "?"

def signtest(v):
    v = np.asarray(v); v = v[~np.isnan(v)]
    nz = v[v != 0]
    if len(nz) == 0: return float("nan"), 0, 0
    pos = int((nz > 0).sum())
    return binomtest(pos, len(nz), 0.5).pvalue, pos, len(nz)

CWE_OF_CLASS = None  # suy ra tu per_cwe + so mau

def macro_f1(y, p, thr=0.5):
    return f1_score(y, (np.asarray(p) >= thr).astype(int), average="macro", zero_division=0)

def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    out = {"f1@0.5": macro_f1(y, p)}
    out["roc"] = roc_auc_score(y, p) if len(set(y.tolist())) > 1 else float("nan")
    out["pr"] = average_precision_score(y, p) if len(set(y.tolist())) > 1 else float("nan")
    return out

def load(fp):
    with open(fp) as f: d = json.load(f)
    if "test_probabilities" not in d or "test_labels" not in d: return None
    return d

def scan(roots):
    """-> {block_key: {arm: cell}}  block_key = (tree, run, seed, fold)"""
    blocks = defaultdict(dict)
    for root in roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                m = re.fullmatch(r"fold(\d+)\.json", fn)
                if not m: continue
                fp = os.path.join(dp, fn)
                rel = os.path.relpath(fp, root)
                parts = rel.split(os.sep)
                if len(parts) < 3: continue
                fold = int(m.group(1))
                seed = parts[-2]                       # seed_42
                arm_ix = len(parts) - 3                # thanh phan ngay truoc seed_*
                arm = parts[arm_ix]
                run = os.sep.join(parts[:arm_ix])      # '' hoac 'asam1_t5p'
                d = load(fp)
                if d is None: continue
                blocks[(root, run, seed, fold)][arm] = d
    return blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--percwe", action="store_true")
    ap.add_argument("--arm-filter", default="")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]

    blocks = scan(a.roots)
    rows = []          # (alpha, metric) -> list of delta vs baseline
    per_alpha = defaultdict(lambda: defaultdict(list))
    per_alpha_cwe = defaultdict(lambda: defaultdict(list))
    n_pairs = 0; n_skip_align = 0; n_no_base = 0
    for key, arms in sorted(blocks.items()):
        base = arms.get("baseline")
        if base is None: n_no_base += len(arms); continue
        yb = base["test_labels"]; pb = np.asarray(base["test_probabilities"], float)
        cw = base.get("test_cwe_classes")
        for arm, d in sorted(arms.items()):
            if arm == "baseline": continue
            if a.arm_filter and a.arm_filter not in arm: continue
            if d["test_labels"] != yb:            # phai CUNG hang test, CUNG thu tu
                n_skip_align += 1; continue
            pt = np.asarray(d["test_probabilities"], float)
            y = np.asarray(yb)
            mb = metrics(y, pb)
            n_pairs += 1
            for al in alphas:
                pe = (1 - al) * pb + al * pt
                me = metrics(y, pe)
                for k in ("f1@0.5", "roc", "pr"):
                    per_alpha[al][k].append(me[k] - mb[k])
                if a.percwe and cw is not None:
                    cwa = np.asarray(cw)
                    for c in sorted(set(cwa.tolist())):
                        m = cwa == c
                        if len(set(y[m].tolist())) < 2: continue
                        per_alpha_cwe[al][c].append(
                            roc_auc_score(y[m], pe[m]) - roc_auc_score(y[m], pb[m]))

    print(f"# o ghep cap: {n_pairs} | bo vi lech hang test: {n_skip_align} | bo vi khong co baseline cung khoi: {n_no_base}")
    print(f"# cay: {', '.join(a.roots)}")
    print()
    print("alpha   " + "".join(f"{k:>22}" for k in ("F1@0.5", "ROC-AUC", "PR-AUC")))
    print("        " + "".join(f"{'D (+/n)':>22}" for _ in range(3)))
    for al in alphas:
        line = f"{al:<8.2f}"
        for k in ("f1@0.5", "roc", "pr"):
            v = np.asarray(per_alpha[al][k]); v = v[~np.isnan(v)]
            pos = int((v > 0).sum())
            line += f"{v.mean():>+13.4f} {pos:>4d}/{len(v):<4d}"
        print(line)
    if a.percwe:
        print("\n# DROC-AUC theo lop CWE (chi so lop trong test_cwe_classes)")
        cls = sorted({c for al in alphas for c in per_alpha_cwe[al]})
        CWEN={0:"CWE-022",1:"CWE-078",2:"CWE-079",3:"CWE-089"}
        print("alpha   " + "".join(f"{CWEN.get(c,'lop '+str(c)):>20}" for c in cls))
        for al in alphas:
            line = f"{al:<8.2f}"
            for c in cls:
                v = np.asarray(per_alpha_cwe[al].get(c, []))
                line += f"{v.mean():>+11.4f} {int((v>0).sum()):>3d}/{len(v):<4d}" if len(v) else f"{'-':>20}"
            print(line)

if __name__ == "__main__":
    main()
