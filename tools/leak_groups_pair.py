#!/usr/bin/env python3
"""So HAI NHANH BAT KY theo NHOM RO RI — mo rong cua tools/leak_groups.py (von chi so voi baseline).

    python3 tools/leak_groups_pair.py --a <tag A> --b <tag B> <cay ket qua...>

VI SAO CAN. Nhom `train` = hang test co BAN DOI NGHICH (cung code, NGUOC NHAN) nam trong TRAIN.
Doan dung o do nghia la mo hinh phan biet duoc khac biet nho giua ham loi va ham da va — tuc
no hoc DAC TRUNG LO HONG chu khong khop MAU VAN BAN. Day chinh la tieu chi ma reviewer neu khi
yeu cau giu split NGAU NHIEN. Diem TONG tron ca ba nhom nen khong doc duoc dieu do.

Cong cu cu chi so duoc `nhanh vs baseline` trong cung cay; cac cay fusion khong co baseline.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import statistics as st
from math import comb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leak_groups import CACHE, macro_f1          # noqa: E402

def sp(k, n):
    if n == 0: return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)

def arm_tag(name):
    b = name.replace("transfer_latent_bottleneck_", "")
    if "_l0p05_" in b: return b.partition("_l0p05_")[2].removesuffix("_adamw") or "goc"
    if b.endswith("_l0p05_adamw") or b.endswith("_l0p05"): return "goc"
    return b.removesuffix("_adamw")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    if not CACHE.is_file():
        print("Chua co cache — chay `python3 tools/leak_groups.py --build`", file=sys.stderr); return 1
    groups = json.loads(CACHE.read_text())

    cells, skipped = defaultdict(dict), 0
    for root in a.roots:
        root = Path(root)
        for d in sorted(root.glob("*/seed_*/fold*.json")):
            arm = d.parent.parent.name
            tag = "baseline" if arm == "baseline" else arm_tag(arm)
            js = json.loads(d.read_text())
            if "test_probabilities" not in js: skipped += 1; continue
            cells[(str(root), js["seed"], js["fold"])][tag] = js
    print(f"(bo qua {skipped} o khong co test_probabilities)")
    GRP = ("train", "test", "none")
    acc = {g: [] for g in GRP}; acc["TAT CA"] = []
    npair = 0
    for k, arms in sorted(cells.items()):
        if a.a not in arms or a.b not in arms: continue
        lab = groups.get(str(k[2]))
        if lab is None: continue
        A, B = arms[a.a], arms[a.b]
        yA, pA = A["test_labels"], A["test_probabilities"]
        yB, pB = B["test_labels"], B["test_probabilities"]
        if len(lab) != len(yA) or yA != yB:      # phai cung thu tu hang, neu khong la so nham
            print(f"  !! BO {k}: nhan/thu tu khong khop (len {len(lab)}/{len(yA)})"); continue
        npair += 1
        for g in GRP:
            idx = [i for i, x in enumerate(lab) if x == g]
            if len(idx) < 3: continue
            fa = macro_f1([yA[i] for i in idx], [pA[i] for i in idx], a.thr)
            fb = macro_f1([yB[i] for i in idx], [pB[i] for i in idx], a.thr)
            acc[g].append((fa - fb, len(idx)))
        acc["TAT CA"].append((macro_f1(yA, pA, a.thr) - macro_f1(yB, pB, a.thr), len(yA)))
    print(f"\n=== {a.a}  −  {a.b} ===   {npair} cap o (cung cay, cung seed, cung fold)")
    print(f"{'nhom':8} {'so hang TB':>11} {'n o':>4} | {'Δ macro-F1@'+str(a.thr):>22}")
    for g in list(GRP) + ["TAT CA"]:
        v = acc[g]
        if not v: print(f"{g:8} {'-':>11} {0:>4} |"); continue
        dd = [x for x, _ in v]; pos = sum(1 for x in dd if x > 0)
        print(f"{g:8} {st.mean(n for _, n in v):11.1f} {len(v):>4} | "
              f"{st.mean(dd):+8.4f}  {pos}/{len(dd)}  p={sp(pos, len(dd)):.3f}")
    print("\n`train` = co ban DOI NGHICH gan trung trong TRAIN -> doan dung = hoc DAC TRUNG, khong hoc VET")
    print("`none`  = khong co ban doi nghich -> khai quat hoa thuong")
    print("`test`  = ban doi nghich nam trong TEST -> it hang, phuong sai lon, dung ket luan")
    return 0

if __name__ == "__main__":
    sys.exit(main())
