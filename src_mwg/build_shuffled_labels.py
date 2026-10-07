#!/usr/bin/env python3
"""Dung hai ban XAO NHAN cua mot file nguon Pha 1 — doi chung "phoi nhiem hay tri thuc".

Cau hoi: loi ich Pha 1 den tu TRI THUC LO HONG hay chi tu viec backbone duoc nhin them
vai nghin ham code that? Chi co mot cach tach: giu MOI THU giong het, pha DUNG MOT thu
la moi lien he giua code va nhan.

Hai ban, pha hai thu khac nhau:

  shufall   hoan vi ngau nhien cot `label` tren toan bo file.
            giu: du lieu, ti le 50/50 toan cuc
            pha: TOAN BO thong tin nhan — va pha ca can bang trong cap (mot cap co the
                 thanh (1,1)), nen no doi HAI thu chu khong mot

  shufpair  voi moi cap day du, tung dong xu; ngua thi doi cho hai nhan cho nhau.
            giu: du lieu, ti le 50/50, VA cau truc cap (moi cap van dung mot duong mot am)
            pha: DUY NHAT mot thu — trong hai ban, ban nao la ban TRUOC khi va
            => phep mo chinh xac hon. Dong khong nam trong cap day du thi xao rieng
               voi nhau de ti le tong the khong doi.

Moi truong khac deu giu nguyen: `cwe_id`, `cwe_class`, `lang`, `pair_id`, thu tu dong.
Phep chia train/val cua Pha 1 chia THEO NHOM va khong doc nhan (train_transfer.py:292),
nen xao nhan KHONG lam doi split — hai nhanh ghep cap duoc o muc Pha 1.
"""
import argparse, json, random, collections, sys


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def complete_pairs(rows):
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("pair_id"):
            by[r["pair_id"]].append(i)
    pairs, loose = [], []
    for pid, idx in by.items():
        v = [i for i in idx if rows[i]["label"] == 1]
        f = [i for i in idx if rows[i]["label"] == 0]
        if len(v) == 1 and len(f) == 1 and len(idx) == 2:
            pairs.append((v[0], f[0]))
        else:
            loose += idx
    loose += [i for i, r in enumerate(rows) if not r.get("pair_id")]
    return pairs, sorted(set(loose))


def shuf_all(rows, rng):
    out = [dict(r) for r in rows]
    labs = [r["label"] for r in rows]
    rng.shuffle(labs)
    for r, l in zip(out, labs):
        r["label"] = l
    return out


def shuf_pair(rows, rng):
    out = [dict(r) for r in rows]
    pairs, loose = complete_pairs(rows)
    flipped = 0
    for v, f in pairs:
        if rng.random() < 0.5:
            out[v]["label"], out[f]["label"] = out[f]["label"], out[v]["label"]
            flipped += 1
    labs = [rows[i]["label"] for i in loose]
    rng.shuffle(labs)
    for i, l in zip(loose, labs):
        out[i]["label"] = l
    print(f"   doi cho {flipped}/{len(pairs)} cap | xao rieng {len(loose)} dong le")
    return out


def audit(name, orig, new):
    """Kiem CA HAI CHIEU: cai phai giu thi con nguyen, cai phai pha thi da pha."""
    n = len(orig)
    assert len(new) == n
    c0 = collections.Counter(r["label"] for r in orig)
    c1 = collections.Counter(r["label"] for r in new)
    agree = sum(1 for a, b in zip(orig, new) if a["label"] == b["label"])
    same_code = all(a["code"] == b["code"] for a, b in zip(orig, new))
    same_cwe = all(a.get("cwe_id") == b.get("cwe_id") for a, b in zip(orig, new))
    pairs, _ = complete_pairs(orig)
    bal = sum(1 for v, f in pairs if {new[v]["label"], new[f]["label"]} == {0, 1})
    print(f"   [{name}] ti le nhan: {dict(c0)} -> {dict(c1)}"
          f" {'GIU' if c0 == c1 else '!! DOI'}")
    print(f"   [{name}] trung nhan cu: {agree}/{n} = {agree/n:.3f}  (ngau nhien ~0.5)")
    print(f"   [{name}] code giu nguyen: {same_code} | cwe_id giu nguyen: {same_cwe}")
    print(f"   [{name}] cap con can bang (1 duong 1 am): {bal}/{len(pairs)} = {bal/max(1,len(pairs)):.3f}")
    ok = (c0 == c1) and same_code and same_cwe
    if not ok:
        print(f"   [{name}] !! KHONG DAT")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/phase1_common.jsonl")
    ap.add_argument("--out_all", default="data/phase1_common_shufall.jsonl")
    ap.add_argument("--out_pair", default="data/phase1_common_shufpair.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = load(a.src)
    pairs, loose = complete_pairs(rows)
    print(f"== {a.src}: {len(rows)} dong | {len(pairs)} cap day du | {len(loose)} dong le")

    print("\n== shufall (hoan vi toan cuc)")
    ra = shuf_all(rows, random.Random(a.seed))
    ok1 = audit("shufall", rows, ra)
    write(a.out_all, ra)

    print("\n== shufpair (doi cho trong cap)")
    rp = shuf_pair(rows, random.Random(a.seed + 1))
    ok2 = audit("shufpair", rows, rp)
    write(a.out_pair, rp)

    # CHIEU NGUOC LAI: chinh file goc phai KHONG DAT phep kiem "da pha"
    print("\n== chieu nguoc lai: ban GOC so voi chinh no (phai trung 1.000, cap can bang 1.000)")
    audit("goc", rows, rows)

    print(f"\nda ghi {a.out_all} va {a.out_pair}")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
