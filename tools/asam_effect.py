#!/usr/bin/env python3
"""ASAM rho co anh huong khong? So GHEP CAP theo fold voi chinh cau hinh rho=0.

Day la phep so mot bien sach nhat co the: cung checkpoint Phase 1, cung lambda,
cung fold, cung may, cung seed. Chi khac buoc nhieu loan ASAM o Phase 2.
"""
import json, glob, sys, statistics, os
from math import comb

src, root = sys.argv[1], sys.argv[2]
LT = {"l0p01": 0.01, "l0p05": 0.05, "l0p2": 0.2}
RT = {"r0":0.0,"r0p05":0.05,"r0p1":0.1,"r0p2":0.2,"r0p5":0.5,"r1p0":1.0,"r2p0":2.0}
NOISE = 0.010

cells = {}   # (lam, rho, fold) -> dict
for p in glob.glob(os.path.join(root, f"results/sw_t5p/transfer_latent_bottleneck_{src}_*/seed_42/fold*.json")):
    tag = p.split("/")[-3].replace(f"transfer_latent_bottleneck_{src}_", "")
    lt, rt = tag.rsplit("_", 1)
    if lt not in LT or rt not in RT:
        continue
    d = json.load(open(p))
    cells[(LT[lt], RT[rt], d["fold"])] = d

def sign_p(pos, n):
    if n == 0: return 1.0
    k = min(pos, n - pos)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def show(label, ds):
    if not ds:
        print(f"  {label:<26} (chua du du lieu)"); return None
    n = len(ds); pos = sum(1 for x in ds if x > 0); m = statistics.mean(ds)
    sd = statistics.stdev(ds) if n > 1 else 0.0
    flag = "  <- duoi san nhieu" if abs(m) < NOISE else ""
    print(f"  {label:<26} n={n:<3} {m:+.4f} ±{sd:.4f}  {pos}/{n} duong  p={sign_p(pos,n):.3f}{flag}")
    return m

lams = sorted({k[0] for k in cells})
rhos = sorted({k[1] for k in cells})
print(f"\n===== NGUON {src} =====")
print("\n[A] Anh huong cua rho, ghep cap voi chinh cau hinh rho=0 (cung lambda, cung fold)\n")
pooled = {r: [] for r in rhos if r > 0}
for lam in lams:
    print(f"  lambda = {lam}")
    for rho in rhos:
        if rho == 0: continue
        ds = [cells[(lam, rho, f)]["test_macro_f1_at_valcal"] - cells[(lam, 0.0, f)]["test_macro_f1_at_valcal"]
              for f in (1, 2, 3) if (lam, rho, f) in cells and (lam, 0.0, f) in cells]
        show(f"    rho {rho} vs rho 0", ds)
        pooled[rho] += ds
    print()

print("[B] Gop ca ba lambda — moi gia tri rho so voi rho=0\n")
allpos = []
for rho in sorted(pooled):
    show(f"  rho {rho} vs rho 0", pooled[rho])
    allpos += pooled[rho]
show("  MOI rho>0 vs rho 0", allpos)

print("\n[C] Anh huong cua lambda (o rho tot nhat cua tung lambda), Delta so voi baseline\n")
base = {}
for p in glob.glob(os.path.join(root, "results/sw_t5p/baseline/seed_42/fold*.json")):
    d = json.load(open(p)); base[d["fold"]] = d["test_macro_f1_at_valcal"]
for lam in lams:
    best, bestrho = None, None
    for rho in rhos:
        v = [cells[(lam, rho, f)]["test_macro_f1_at_valcal"] for f in (1,2,3) if (lam, rho, f) in cells]
        if len(v) < 3: continue
        m = statistics.mean(v)
        if best is None or m > best: best, bestrho = m, rho
    if best is None: continue
    ds = [cells[(lam, bestrho, f)]["test_macro_f1_at_valcal"] - base[f] for f in (1,2,3) if f in base]
    print(f"  lambda {lam}: rho tot nhat = {bestrho}, F1 = {best:.4f}", end="")
    print(f", Delta vs baseline = {statistics.mean(ds):+.4f} ({sum(1 for x in ds if x>0)}/{len(ds)} fold)")
