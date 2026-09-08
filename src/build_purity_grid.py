#!/usr/bin/env python3
"""Dung LUOI NGUON: tach DO TINH KHIET NHAN khoi CO DU LIEU.

Ba nguon hien co lam hai bien dinh chat vao nhau — do tinh khiet va co du lieu tuong quan
NGHICH hoan hao (4cwe 930 hang/100%, com 3744/24.8%, full 7598/12.2%), nen khong the noi
cai nao gay ra cai gi. Luoi nay tach chung ra:

  truc DO TINH KHIET (co CO DINH 930 hang): 100%, 75%, 50%, 25%, 12%
  truc CO DU LIEU (do tinh khiet CO DINH 100%): 232, 465, 930

"Tinh khiet" = ti le hang co CWE nam trong tap CWE cua dich (89, 78, 79, 22).

BAY DA MAC va da vá o day: dich ghi `CWE-089` con nguon ghi `CWE-89`. So chuoi truc tiep
cho ra "0% trung" tren ca ba nguon — sai hoan toan. Phai chuan hoa ve so nguyen truoc khi so.

Giu can bang nhan 50/50 trong moi nguon, va giu ti le ngon ngu cua phan duoc lay.
    python3 src/build_purity_grid.py --out data/pool
"""
import argparse, json, random, re, collections
from pathlib import Path

def norm_cwe(c):
    m = re.search(r"(\d+)", str(c) or "")
    return f"CWE-{int(m.group(1))}" if m else "?"

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def balanced(rows, n, rng):
    """Lay n hang, giu 50/50 nhan. Thieu thi lay het ben it roi bu ben nhieu."""
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    half = n // 2
    take_p, take_n = pos[:half], neg[:half]
    short = n - len(take_p) - len(take_n)
    rest = pos[len(take_p):] + neg[len(take_n):]
    rng.shuffle(rest)
    return take_p + take_n + rest[:max(short, 0)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="data/phase1_full.jsonl")
    ap.add_argument("--target_root", default="data/sven_python_folds_norm")
    ap.add_argument("--out", default="data/pool")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tgt = []
    for k in range(1, 6):
        tgt += load(Path(args.target_root) / f"fold{k}" / "test.jsonl")
    tset = {norm_cwe(r.get("cwe")) for r in tgt}
    print(f"CWE cua dich: {sorted(tset)}")

    rows = load(args.full)
    inn = [r for r in rows if norm_cwe(r.get("cwe")) in tset]
    out = [r for r in rows if norm_cwe(r.get("cwe")) not in tset]
    print(f"kho nguon: {len(rows)} hang | trong taxonomy dich {len(inn)} | ngoai {len(out)}")

    made = []
    # --- truc DO TINH KHIET, co co dinh 930 ---
    N = 930
    for pur in (100, 75, 50, 25, 12):
        n_in = round(N * pur / 100); n_out = N - n_in
        if n_in > len(inn) or n_out > len(out):
            print(f"  bo qua p{pur}: khong du hang"); continue
        sel = balanced(inn, n_in, rng) + (balanced(out, n_out, rng) if n_out else [])
        rng.shuffle(sel)
        name = f"pur{pur}_n{N}"
        write(Path(args.out) / f"{name}.jsonl", sel)
        made.append((name, len(sel), pur, sum(1 for r in sel if r["label"] == 1)))
    # --- truc CO DU LIEU, do tinh khiet co dinh 100% ---
    for n in (232, 465):
        sel = balanced(inn, n, rng); rng.shuffle(sel)
        name = f"pur100_n{n}"
        write(Path(args.out) / f"{name}.jsonl", sel)
        made.append((name, len(sel), 100, sum(1 for r in sel if r["label"] == 1)))

    print(f"\n{'ten':>14} {'n':>5} {'tinh khiet':>11} {'nhan 1':>7} {'ti le 1':>8}")
    for name, n, pur, p1 in made:
        print(f"{name:>14} {n:>5} {pur:>10}% {p1:>7} {p1/n*100:>7.1f}%")
    print(f"\nGhi vao {args.out}/  — {len(made)} nguon ung vien")

if __name__ == "__main__":
    main()
