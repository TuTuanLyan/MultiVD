#!/usr/bin/env python3
"""ensemble2.py — ban chat che cua tools/ensemble.py.

BA cho ban dau lam so dep hon su that, sua het o day:

1. KHONG DOC LAP. 710 o chia nhau chi ~200 baseline; mot baseline yeu lam CA CUM
   nhanh cua no duong. Sign test tren 710 o la thoi phong. -> gop ve MOT Delta cho
   moi khoi (cay, run, seed, fold) bang cach trung binh cac nhanh TRUOC, roi moi
   dem dau tren so khoi. Do la don vi doc lap that su.
2. GOP CA NHANH DA BIET LA HONG. Trong pool co lm12/pur12 (pha loang, B.2e am),
   rho=4/8 (§21.2 am 0/15), Pha 1 suy bien val<0.40. Ho tron vao thi alpha=1.0
   trong xau di vi ly do KHONG lien quan den chuyen giao. -> --only / --exclude.
3. KHONG TACH BACKBONE. Phat bieu ma nguoi dung can la "khong phu thuoc backbone",
   nen phai in tung backbone rieng chu khong phai mot so gop.

alpha=0.5 la lua chon KHONG THAM SO (hai mo hinh mot phieu ngang nhau), khai bao
truoc. Duong alpha day du chi de CHAN DOAN hinh dang — khong duoc chon alpha tren TEST.
"""
import argparse, json, os, re
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

BACKBONES = ("t5pe", "t5p", "codebert", "unixcoder", "roberta")
CWEN = {0: "CWE-022", 1: "CWE-078", 2: "CWE-079", 3: "CWE-089"}
MET = ("f1@0.5", "roc", "pr")

def backbone_of(cell, root, run, arm):
    hay = f"{cell.get('experiment_name','')} {root} {run} {arm}"
    for b in BACKBONES:                       # t5pe TRUOC t5p
        if b in hay: return b
    return "?"

def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    o = {"f1@0.5": f1_score(y, (p >= .5).astype(int), average="macro", zero_division=0)}
    two = len(set(y.tolist())) > 1
    o["roc"] = roc_auc_score(y, p) if two else np.nan
    o["pr"] = average_precision_score(y, p) if two else np.nan
    return o

def sgn(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    nz = v[v != 0]
    if not len(nz): return np.nan, 0, 0, np.nan
    pos = int((nz > 0).sum())
    return v.mean(), pos, len(nz), binomtest(pos, len(nz), 0.5).pvalue

def fmt(v):
    m, pos, n, p = sgn(v)
    if np.isnan(m): return f"{'-':>24}"
    return f"{m:>+9.4f} {pos:>4d}/{n:<4d} p={p:<6.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--only", default="", help="regex nhanh PHAI khop")
    ap.add_argument("--exclude", default="", help="regex nhanh bi loai")
    ap.add_argument("--min-p1val", type=float, default=None, help="loai o co phase1_val duoi nguong")
    ap.add_argument("--percwe", action="store_true")
    ap.add_argument("--by-backbone", action="store_true")
    ap.add_argument("--by-tree", action="store_true",
                    help="tach theo (cay ket qua, run) — kiem xem mot cay co chi phoi khong")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    only = re.compile(a.only) if a.only else None
    excl = re.compile(a.exclude) if a.exclude else None

    blocks = defaultdict(dict)
    for root in a.roots:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                m = re.fullmatch(r"fold(\d+)\.json", fn)
                if not m: continue
                fp = os.path.join(dp, fn)
                parts = os.path.relpath(fp, root).split(os.sep)
                if len(parts) < 3: continue
                i = len(parts) - 3
                try:
                    with open(fp) as f: d = json.load(f)
                except Exception: continue
                if "test_probabilities" not in d: continue
                blocks[(root, os.sep.join(parts[:i]), parts[-2], int(m.group(1)))][parts[i]] = d

    # cell_delta[(bb, alpha, metric)][block] = [delta cua tung nhanh trong khoi]
    cell = defaultdict(lambda: defaultdict(list))
    cwe  = defaultdict(lambda: defaultdict(list))
    n_arm = 0; n_block = 0; n_align = 0
    for key, arms in sorted(blocks.items()):
        base = arms.get("baseline")
        if base is None: continue
        yb = base["test_labels"]; y = np.asarray(yb)
        pb = np.asarray(base["test_probabilities"], float)
        cwv = np.asarray(base["test_cwe_classes"]) if base.get("test_cwe_classes") else None
        mb = metrics(y, pb)
        mbc = {}
        if cwv is not None:
            for c in set(cwv.tolist()):
                msk = cwv == c
                if len(set(y[msk].tolist())) > 1: mbc[c] = roc_auc_score(y[msk], pb[msk])
        used = False
        for arm, d in sorted(arms.items()):
            if arm == "baseline": continue
            if only and not only.search(arm): continue
            if excl and excl.search(arm): continue
            if a.min_p1val is not None and (d.get("phase1_val_macro_f1") or 0) < a.min_p1val: continue
            if d["test_labels"] != yb: n_align += 1; continue
            pt = np.asarray(d["test_probabilities"], float)
            bb = backbone_of(d, key[0], key[1], arm)
            n_arm += 1; used = True
            for al in alphas:
                pe = (1 - al) * pb + al * pt
                me = metrics(y, pe)
                for k in MET:
                    cell[(bb, al, k)][key].append(me[k] - mb[k])
                    cell[("TAT CA", al, k)][key].append(me[k] - mb[k])
                if a.percwe and cwv is not None:
                    for c, b0 in mbc.items():
                        msk = cwv == c
                        cwe[("TAT CA", al, c)][key].append(roc_auc_score(y[msk], pe[msk]) - b0)
                        cwe[(bb, al, c)][key].append(roc_auc_score(y[msk], pe[msk]) - b0)
        n_block += used

    def blockwise(store, k):
        """trung binh cac nhanh TRONG mot khoi -> mot so cho moi khoi"""
        return [float(np.nanmean(v)) for v in store.get(k, {}).values() if len(v)]

    bbs = ["TAT CA"] + ([b for b in BACKBONES if any(k[0] == b for k in cell)] if a.by_backbone else [])
    if a.by_tree:
        # dem so khoi moi cay truoc, roi in cac cay lon nhat
        per_tree = defaultdict(set)
        for k, blks in cell.items():
            if k[0] != "TAT CA": continue
            for blk_key in blks: per_tree[(blk_key[0], blk_key[1])].add(blk_key)
        print("# so KHOI theo cay ket qua (giam dan):")
        for t, v in sorted(per_tree.items(), key=lambda x: -len(x[1]))[:12]:
            print(f"#   {len(v):>3d}  {t[0]}/{t[1] or '.'}")
        print()
    print(f"# nhanh ghep cap: {n_arm} | KHOI doc lap (cay,run,seed,fold): {n_block} | bo vi lech hang test: {n_align}")
    print(f"# only={a.only or '-'} exclude={a.exclude or '-'} min-p1val={a.min_p1val}")
    print("# Delta = tron(alpha) - baseline. Moi KHOI mot so (trung binh cac nhanh), dem dau tren khoi.\n")
    for bb in bbs:
        ns = len(blockwise(cell, (bb, alphas[-1], "roc")))
        if ns == 0: continue
        print(f"===== {bb}  ({ns} khoi) =====")
        print(f"{'alpha':<7}" + "".join(f"{h:>24}" for h in ("F1@0.5", "ROC-AUC", "PR-AUC")))
        for al in alphas:
            print(f"{al:<7.2f}" + "".join(fmt(blockwise(cell, (bb, al, k))) for k in MET))
        # PHEP KIEM QUYET DINH: tron co hon HAI DAU MUT khong?
        # Neu Pha 1 chi la "finetune them" thi hai mo hinh THUA nhau va diem cua ban tron
        # phai nam GIUA hai dau mut. Ban tron VUOT ca hai => hai mo hinh sai o cho KHAC
        # nhau, tuc Pha 1 mang vao thong tin ma mo hinh chi-dich khong co. Do la lap luan
        # chong lai phan bien "hai pha thi tat nhien hon mot pha".
        half = 0.5 if 0.5 in alphas else alphas[len(alphas)//2]
        print(f"{'  vs a=1':<7}" + "".join(
            fmt([x - y for x, y in zip(blockwise(cell, (bb, half, k)),
                                       blockwise(cell, (bb, alphas[-1], k)))]) for k in MET))
        print(f"{'  vs a=0':<7}" + "".join(fmt(blockwise(cell, (bb, half, k))) for k in MET))
        if a.percwe:
            cls = sorted(CWEN)
            print(f"{'alpha':<7}" + "".join(f"{CWEN[c]+' dROC':>24}" for c in cls))
            for al in alphas:
                print(f"{al:<7.2f}" + "".join(fmt(blockwise(cwe, (bb, al, c))) for c in cls))
        print()

if __name__ == "__main__":
    main()
