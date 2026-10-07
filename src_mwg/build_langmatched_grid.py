#!/usr/bin/env python3
"""Luoi tinh khiet GIU NGUYEN TI LE NGON NGU — go nhieu ngon ngu khoi do tinh khiet.

Luoi cu (src/build_purity_grid.py) tach duoc tinh khiet khoi CO du lieu, nhung KHONG tach
duoc khoi NGON NGU: hang trong taxonomy dich chu yeu la js (812/930 = 87.3%) con hang ngoai
chu yeu la ccpp, nen pha loang cung dong thoi doi ti le ngon ngu. Khong biet cai nao gay ra.

Luoi nay giu ti le js/ccpp CO DINH o 87.3/12.7 tren MOI muc tinh khiet: phan pha loang cung
duoc lay theo dung ti le do. Khi ay bien duy nhat con doi la do tinh khiet nhan.

Kha thi den dau: chi co 744 hang js NGOAI taxonomy, nen o n=930 va tinh khiet 12% can 818
hang ngoai, tuc 714 js — vua du. Duoi 12% thi het js va luoi khong giu duoc ti le nua.

Luoi 50/50 ngon ngu KHONG kha thi: chi co 118 hang ccpp trong taxonomy, nen n toi da la 236,
ma muc n=232 da do va cho ket qua AM (-0.0091) — tuc roi vao vung "qua nho".
    python3 src/build_langmatched_grid.py --out data/pool
"""
import argparse, json, random, re, collections
from pathlib import Path

def norm(c):
    m = re.search(r"(\d+)", str(c) or "")
    return f"CWE-{int(m.group(1))}" if m else "?"

def balanced(rows, n, rng):
    """Lay n hang giu 50/50 NHAN."""
    pos = [r for r in rows if r["label"] == 1]; neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    h = n // 2
    out = pos[:h] + neg[:h]
    rest = pos[h:] + neg[h:]; rng.shuffle(rest)
    return out + rest[:max(n - len(out), 0)]

def remap(rows):
    present = sorted({r["cwe_class"] for r in rows if r.get("cwe_class", -100) >= 0})
    m = {c: i for i, c in enumerate(present)}
    for r in rows:
        c = r.get("cwe_class", -100)
        r["cwe_class"] = m[c] if c >= 0 else -100
    return len(present)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="data/phase1_full.jsonl")
    ap.add_argument("--target_root", default="data/sven_python_folds_norm")
    ap.add_argument("--out", default="data/pool")
    ap.add_argument("--n", type=int, default=930)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tset = set()
    for k in range(1, 6):
        for l in open(Path(a.target_root) / f"fold{k}" / "test.jsonl", encoding="utf-8"):
            tset.add(norm(json.loads(l).get("cwe")))
    rows = [json.loads(l) for l in open(a.full, encoding="utf-8") if l.strip()]
    pool = collections.defaultdict(list)
    for r in rows:
        pool[(norm(r.get("cwe")) in tset, r.get("lang"))].append(r)
    inn = pool[(True, "js")] + pool[(True, "ccpp")]
    js_ratio = len(pool[(True, "js")]) / len(inn)
    print(f"ti le js trong phan TRONG taxonomy: {js_ratio:.3f} ({len(pool[(True,'js')])}/{len(inn)})")
    made = []
    for pur in (100, 75, 50, 25, 12):
        n_in = round(a.n * pur / 100); n_out = a.n - n_in
        need_js, need_cc = round(n_out * js_ratio), n_out - round(n_out * js_ratio)
        if need_js > len(pool[(False, "js")]) or need_cc > len(pool[(False, "ccpp")]):
            print(f"  bo qua pur{pur}: can {need_js} js ngoai, chi co {len(pool[(False,'js')])}"); continue
        sel = balanced(inn, n_in, rng)
        if n_out:
            sel += balanced(pool[(False, "js")], need_js, rng) + balanced(pool[(False, "ccpp")], need_cc, rng)
        rng.shuffle(sel)
        k = remap(sel)
        name = f"lm{pur}_n{a.n}"
        p = Path(a.out) / f"{name}.jsonl"; p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for r in sel: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = collections.Counter(r["lang"] for r in sel)
        made.append((name, len(sel), pur, c.get("js", 0) / len(sel), sum(1 for r in sel if r["label"] == 1) / len(sel), k))
    print(f"\n{'ten':>12} {'n':>5} {'tinh khiet':>11} {'ti le js':>9} {'ti le nhan 1':>13} {'lop CWE':>8}")
    for nm, n, pur, js, lb, k in made:
        print(f"{nm:>12} {n:>5} {pur:>10}% {js:>9.3f} {lb:>13.3f} {k:>8}")

if __name__ == "__main__":
    main()
