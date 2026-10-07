#!/usr/bin/env python3
"""Dung pool nguon pha 1 theo DUNG mo ta trong DATA.md cua dong nghiep (make_mwsrc.py cua
ho khong co trong repo share, nen dung lai tu mo ta va KIEM bang cach tai lap mwsrc_jsCpy).

  1. Gop cac file src_pair_*.jsonl.
  2. Gom ham theo `pair_id` (ham khong co pair_id la nhom rieng), xao thu tu nhom bang
     random.Random(36).
  3. Lay nhom lan luot vao val cho toi khi du 10% so ham; con lai la train.
     Cap loi/va KHONG BAO GIO bi tach. test.jsonl = val.jsonl.
  4. Ngon ngu tron chung.

    python3 tools/make_mwsrc.py <ten_pool> <src_pair_a.jsonl> <src_pair_b.jsonl> ...
"""
import json, math, os, random, sys


def build(files, seed=36, val_frac=0.10):
    recs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    groups, singles = {}, []
    for i, r in enumerate(recs):
        pid = r.get("pair_id")
        if pid is None:
            singles.append([i])
        else:
            groups.setdefault(str(pid), []).append(i)
    allg = list(groups.values()) + singles
    random.Random(seed).shuffle(allg)
    need = math.ceil(len(recs) * val_frac)
    val, n = [], 0
    k = 0
    while k < len(allg) and n < need:
        val.append(allg[k]); n += len(allg[k]); k += 1
    vset = {i for g in val for i in g}
    return ([r for i, r in enumerate(recs) if i not in vset],
            [r for i, r in enumerate(recs) if i in vset])


def write(d, name, train, val):
    out = os.path.join(d, f"mwsrc_{name}", "fold1")
    os.makedirs(out, exist_ok=True)
    for fn, rows in (("train", train), ("val", val), ("test", val)):
        with open(os.path.join(out, fn + ".jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    name, files = sys.argv[1], sys.argv[2:]
    tr, va = build(files)
    p = write("data", name, tr, va)
    print(f"{p}: train {len(tr)} / val {len(va)}")
