#!/usr/bin/env python3
"""Tach diem test theo NHOM RO RI — tra loi "mo hinh dua vao dau" ma khong doi du lieu.

    python3 tools/leak_groups.py --build          # tinh nhan nhom mot lan, ghi cache
    python3 tools/leak_groups.py results/opt1_t5p results_opt1_158

Bo `sven_python_folds_norm` chia ngau nhien theo tung dong, nen moi hang test roi vao
mot trong ba tinh huong. Diem TONG tron ca ba lai va khong doc duoc gi:

  train : ban doi nghich (cung code, khac nhan) nam trong TRAIN  -> 10-18% moi fold
          Day la nhom "cam do": hai ham gan trung nhau trong khong gian embedding nen
          luc hut tu nhien la gan CUNG nhan. Doan dung o day = phan biet duoc khac biet
          nho = bang chung hoc dac trung lo hong. Nhung cung la nhom de bi hoc VET.
  test  : ban doi nghich nam trong chinh TEST                    -> 2.6-7.9%
          Day chinh la bai toan cua bo `twin`, thu nho: mo hinh chua he thay cap nao
          tuong tu. Cham theo CAP kieu PrimeVul doc duoc o day.
  none  : khong co ban doi nghich nao                            -> 70-74%
          Khai quat hoa thuong, khong dinh ro ri.

Ro ri do duoc tuong quan +0.907 voi baseline F1 qua 5 fold (RESEARCH §9), nen tach nhom
la cach duy nhat doc dung ket qua ma VAN GIU phan phoi ngau nhien theo yeu cau reviewer.

Chi doc duoc o co truong `test_probabilities` (them 07/09). O chay truoc do bi bo qua va
duoc bao ro, khong nuot im.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CACHE = Path("data/leak_groups.json")
ROOT = Path("data/sven_python_folds_norm")
THRESH = 0.75


def grams(code, n=5):
    tokens = re.sub(r"\s+", " ", code).strip().split()
    return set(tuple(tokens[i:i + n]) for i in range(max(len(tokens) - n + 1, 1)))


def build(root=ROOT, thresh=THRESH):
    """Voi moi hang test cua moi fold: ban doi nghich gan nhat nam o dau?"""
    out = {}
    for k in range(1, 6):
        splits = {}
        for name in ("train", "val", "test"):
            p = root / f"fold{k}" / f"{name}.jsonl"
            if not p.is_file():
                break
            splits[name] = [json.loads(l) for l in p.open()]
        if len(splits) < 3:
            continue
        G = {s: [(grams(r["code"]), r["label"]) for r in rows] for s, rows in splits.items()}
        labels = []
        for i, r in enumerate(splits["test"]):
            g = grams(r["code"])
            best, where = 0.0, None
            for s in ("train", "val", "test"):
                for j, (tg, lb) in enumerate(G[s]):
                    if s == "test" and j == i:
                        continue
                    if lb == r["label"]:          # chi tim NHAN NGUOC
                        continue
                    inter = len(g & tg)
                    if not inter:
                        continue
                    jc = inter / len(g | tg)
                    if jc > best:
                        best, where = jc, s
            labels.append(where if best >= thresh else "none")
        out[str(k)] = labels
        print(f"  fold{k}: " + ", ".join(
            f"{w}={labels.count(w)}" for w in ("train", "val", "test", "none")))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(CACHE)
    print(f"Ghi {CACHE}")
    return out


def macro_f1(labels, probs, thr=0.5):
    """Macro-F1 tu tay, de khong keo sklearn vao mot cong cu doc ket qua."""
    pred = [1 if p >= thr else 0 for p in probs]
    f1s = []
    for c in (0, 1):
        tp = sum(1 for y, p in zip(labels, pred) if y == c and p == c)
        fp = sum(1 for y, p in zip(labels, pred) if y != c and p == c)
        fn = sum(1 for y, p in zip(labels, pred) if y == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="tinh lai nhan nhom roi thoat")
    ap.add_argument("roots", nargs="*", default=["results/opt1_t5p"])
    args = ap.parse_args()
    if args.build:
        build()
        return
    if not CACHE.is_file():
        print("Chua co cache — chay `python3 tools/leak_groups.py --build` truoc", file=sys.stderr)
        return 1
    groups = json.loads(CACHE.read_text())

    base, arms = {}, defaultdict(dict)
    skipped = 0
    for root in args.roots:
        root = Path(root)
        for f in sorted(root.glob("baseline/seed_*/fold*.json")):
            d = json.loads(f.read_text())
            if "test_probabilities" not in d:
                skipped += 1
                continue
            base[(str(root), d["fold"])] = d
        for arm_dir in sorted(root.glob("transfer_latent_bottleneck_*")):
            name = arm_dir.name.replace("transfer_latent_bottleneck_", "")
            src, _, tag = name.partition("_l0p05_")
            if not tag:
                continue
            for f in sorted(arm_dir.glob("seed_*/fold*.json")):
                d = json.loads(f.read_text())
                if "test_probabilities" not in d:
                    skipped += 1
                    continue
                arms[tag][(str(root), d["fold"], src)] = d
    print(f"o co du doan tung mau: baseline {len(base)}, nhanh {sum(len(v) for v in arms.values())}"
          f" | BO QUA {skipped} o chay truoc 07/09 (chua ghi du doan)")
    if not base:
        print("Chua co baseline nao mang du doan — moi Delta deu quy ve baseline nen chua tach duoc.")
        return 0

    def split_scores(d, fold):
        g = groups.get(str(fold))
        if not g or len(g) != len(d["test_probabilities"]):
            return None
        out = {}
        for grp in ("train", "test", "none"):
            idx = [i for i, x in enumerate(g) if x == grp]
            if len(idx) >= 8:
                out[grp] = macro_f1([d["test_labels"][i] for i in idx],
                                    [d["test_probabilities"][i] for i in idx])
        return out

    print(f"\n{'cau hinh':<24} {'n':>3} " + " ".join(f"{g:>16}" for g in ("train", "test", "none")))
    print(f"{'':24} {'':3} " + " ".join(f"{'dF1 (n o)':>16}" for _ in range(3)))
    for tag in sorted(arms):
        acc = defaultdict(list)
        for c, d in arms[tag].items():
            b = base.get((c[0], c[1]))
            if not b:
                continue
            sa, sb = split_scores(d, c[1]), split_scores(b, c[1])
            if not sa or not sb:
                continue
            for grp in sa:
                if grp in sb:
                    acc[grp].append(sa[grp] - sb[grp])
        if not acc:
            continue
        cells = []
        for grp in ("train", "test", "none"):
            v = acc.get(grp)
            cells.append(f"{sum(v)/len(v):+.4f} ({len(v)})" if v else "        -       ")
        n = max(len(v) for v in acc.values())
        print(f"{tag:<24} {n:>3} " + " ".join(f"{c:>16}" for c in cells))
    print("\ntrain = co ban doi nghich trong TRAIN (nhom 'cam do', cung la nhom de hoc vet)")
    print("test  = ban doi nghich nam trong chinh TEST (bai toan cua `twin`, thu nho)")
    print("none  = khong co ban doi nghich nao (khai quat hoa thuong, ~70-74% du lieu)")


if __name__ == "__main__":
    sys.exit(main() or 0)
