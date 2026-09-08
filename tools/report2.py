#!/usr/bin/env python3
"""report2.py — bao cao CHUAN cho moi khoi: LUON in CA HAI chi so, khong bao gio mot cai.

    python3 tools/report2.py --a <nhanh A> --b <nhanh B> results_dir [results_dir ...]
    python3 tools/report2.py --list results_int1_ntat          # xem co nhung tag nao

VI SAO CO FILE NAY (08/09/2026): ket luan "ASAM null" giu suot ba tuan duoc tinh tren
macro-F1 VA CHI macro-F1. Do lai tren 190 o ghep cap thi macro-F1 +0.0015 (100/190, p=0.51)
nhung ROC-AUC +0.0037 (119/190, p=0.0006) va PR-AUC +0.0045 (111/190, p=0.024). Mot chi so
noi khong, hai chi so kia noi co. Cong cu nay khong cho phep lap lai loi do: no in ca ba,
va in ca so fold cung dau chu khong chi trung binh.

BA CACH DOC SAI MA CONG CU NAY CHAN:
  1. Chi nhin mot chi so            -> in ca F1@0.5, F1@nguong-val, ROC-AUC, PR-AUC.
  2. Chi nhin trung binh            -> in +/n va p (kiem dau) canh moi trung binh; mot o
                                       cuc tri keo trung binh nhung khong keo duoc dem dau.
  3. Gop o cua nhieu may/khoi       -> ghep cap theo (cay, seed, fold, nguon); o le bi BO
                                       va bao ro so o bi bo, khong nuot im.

Kiem dau la nhi thuc hai phia. CANH BAO in kem: cac fold KHONG doc lap khi dung lai cung
bo fold dich qua nhieu backbone/nguon/seed, nen p lac quan — phan chac la SO FOLD CUNG DAU.
"""
import argparse, json, glob, os, sys
from collections import defaultdict
from math import comb
import statistics as st

M = [("F1@0.5","test_macro_f1_at_0.5"), ("F1@val","test_macro_f1_at_valcal"),
     ("ROC-AUC","test_roc_auc"), ("PR-AUC","test_pr_auc")]

def sign_p(k, n):
    if n == 0: return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)

def load(roots):
    cells = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, "transfer_*", "seed_*", "fold*.json")):
            try: d = json.load(open(p))
            except Exception: continue
            name = os.path.basename(os.path.dirname(os.path.dirname(p)))
            body = name.replace("transfer_latent_bottleneck_", "")
            src, _, tag = body.partition("_l0p05_")
            for suf in ("_adamw", "_spd"):
                if tag.endswith(suf): tag = tag[:-len(suf)]; break
            cells[(root, src, tag, d.get("seed"), d.get("fold"))] = d
    return cells

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--a", help="nhanh can do")
    ap.add_argument("--b", help="nhanh doi chung")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cells = load(args.roots)
    if args.list or not (args.a and args.b):
        tags = defaultdict(int)
        for (_, src, tag, _, _) in cells: tags[(src, tag)] += 1
        print(f"{len(cells)} o. Cac (nguon, tag) co san:")
        for k in sorted(tags): print(f"  {k[0]:6} {k[1]:24} n={tags[k]}")
        return 0

    pairs = defaultdict(list)
    lone = 0
    for (root, src, tag, seed, fold), d in cells.items():
        if tag != args.a: continue
        b = cells.get((root, src, args.b, seed, fold))
        if b is None: lone += 1; continue
        pairs[src].append((d, b))
    if not pairs:
        print(f"khong ghep duoc cap nao giua '{args.a}' va '{args.b}'"); return 1

    print(f"=== {args.a}  −  {args.b}   (ghep cap theo cay/seed/fold/nguon) ===")
    if lone: print(f"    BO QUA {lone} o cua '{args.a}' khong co cap doi chung")
    print(f"{'nguon':6} {'n':>3} | " + " | ".join(f"{n:>22}" for n, _ in M))
    tot = defaultdict(list)
    for src in sorted(pairs) + ["TAT CA"]:
        v = pairs[src] if src != "TAT CA" else [x for s in pairs for x in pairs[s]]
        row = []
        for name, key in M:
            dd = [a.get(key, float("nan")) - b.get(key, float("nan")) for a, b in v
                  if a.get(key) is not None and b.get(key) is not None]
            if not dd: row.append(f"{'-':>22}"); continue
            pos = sum(1 for x in dd if x > 0)
            row.append(f"{st.mean(dd):+7.4f} {pos:>2}/{len(dd):<2} p={sign_p(pos,len(dd)):<5.3f}")
            if src != "TAT CA": tot[name] += dd
        print(f"{src:6} {len(v):>3} | " + " | ".join(row))
    print("\nCANH BAO doc so: cac fold KHONG doc lap neu dung lai cung bo fold dich qua nhieu")
    print("backbone/nguon/seed — p se lac quan. Phan chac la SO FOLD CUNG DAU, khong phai p.")
    print("Mot chi so duong ma chi so kia phang => hieu ung o XEP HANG chu khong o NGUONG 0.5.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
