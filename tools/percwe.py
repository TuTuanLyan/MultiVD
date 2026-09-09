#!/usr/bin/env python3
"""percwe.py — Delta THEO TUNG CWE giua mot nhanh va baseline, ghep cap tung fold.

    python3 tools/percwe.py results/s42_codebert results/s42_t5p ...
    python3 tools/percwe.py --arm none results/...        # doi nhanh can do

CAU HOI: transfer co giup DUNG CHO dich yeu khong? Tap dich Python lech nang —
CWE-089 chiem 54% hang test con CWE-022 chi 9%, CWE-079 11%. Nguon `4cwe` thi
NGUOC LAI: CWE-79 co 692 dong, CWE-22 co 92, con CWE-89 chi 46.
Neu chuyen giao that su mang tri thuc CWE cu the sang, loi ich phai DON vao dung
hai lop hiem ma nguon giau — 022 va 079 — chu khong trai deu.

Doc `per_cwe` (macro-F1 tung lop) va tinh ROC-AUC tung lop tu `test_probabilities`
+ `test_cwe_classes`. Anh xa class 0..3 -> ten CWE lay tu chinh file (sorted), da
doi chieu number_of_samples khop tung lop truoc khi dung.
"""
import argparse, json, glob, os
from collections import defaultdict
from math import comb

def sign_p(k, n):
    if n == 0: return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)

def auc(y, p):
    pair = sorted(zip(p, y)); pos = sum(y); neg = len(y) - pos
    if pos == 0 or neg == 0: return None
    r = {}; i = 0
    while i < len(pair):
        j = i
        while j + 1 < len(pair) and pair[j+1][0] == pair[i][0]: j += 1
        rk = (i + j) / 2 + 1
        for k in range(i, j+1): r[k] = rk
        i = j + 1
    s = sum(r[k] for k in range(len(pair)) if pair[k][1] == 1)
    return (s - pos*(pos+1)/2) / (pos*neg)

def load(roots, arm):
    base, arms = {}, {}
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "baseline", "seed_*", "fold*.json"), recursive=True):
            d = json.load(open(p))
            base[(os.path.dirname(os.path.dirname(os.path.dirname(p))), d.get("seed"), d.get("fold"))] = d
        for p in glob.glob(os.path.join(root, "**", f"transfer_{arm}_*", "seed_*", "fold*.json"), recursive=True):
            d = json.load(open(p))
            nm = os.path.basename(os.path.dirname(os.path.dirname(p)))
            tree = os.path.dirname(os.path.dirname(os.path.dirname(p)))
            arms[(tree, nm, d.get("seed"), d.get("fold"))] = d
    return base, arms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--arm", default="latent_bottleneck")
    a = ap.parse_args()
    base, arms = load(a.roots, a.arm)
    print(f"{len(arms)} o nhanh '{a.arm}', {len(base)} o baseline")

    acc = defaultdict(lambda: defaultdict(list))   # backbone -> (cwe, metric) -> [delta]
    paired = 0
    for (tree, nm, seed, fold), d in arms.items():
        b = base.get((tree, seed, fold))
        if b is None: continue
        paired += 1
        bb = os.path.basename(tree).split("_")[-1]
        for key in ("TAT CA", bb):
            names = sorted(d.get("per_cwe") or {})
            for i, cw in enumerate(names):
                pa, pb = (d.get("per_cwe") or {}).get(cw), (b.get("per_cwe") or {}).get(cw)
                if pa and pb and pa.get("macro_f1") is not None and pb.get("macro_f1") is not None:
                    acc[key][(cw, "F1")].append(pa["macro_f1"] - pb["macro_f1"])
                if "test_probabilities" in d and "test_probabilities" in b and "test_cwe_classes" in d:
                    idx = [j for j, c in enumerate(d["test_cwe_classes"]) if c == i]
                    if len(idx) >= 8:
                        y = [d["test_labels"][j] for j in idx]
                        va, vb = auc(y, [d["test_probabilities"][j] for j in idx]), auc(y, [b["test_probabilities"][j] for j in idx])
                        if va is not None and vb is not None:
                            acc[key][(cw, "ROC")].append(va - vb)
    print(f"{paired} cap ghep duoc\n")
    for key in ["TAT CA"] + sorted(k for k in acc if k != "TAT CA"):
        rows = [(cw, m) for (cw, m) in acc[key]]
        if not rows: continue
        print(f"### {key}")
        print(f"{'CWE':10}{'chi so':>7}{'n':>5}{'Delta':>10}{'cung dau':>11}{'p':>8}")
        for cw in sorted({c for c, _ in rows}):
            for m in ("F1", "ROC"):
                v = acc[key].get((cw, m))
                if not v: continue
                k = sum(1 for x in v if x > 0)
                print(f"{cw:10}{m:>7}{len(v):>5}{sum(v)/len(v):>+10.4f}{k:>6}/{len(v):<4}{sign_p(k,len(v)):>8.3f}")
        print()

if __name__ == "__main__":
    main()
