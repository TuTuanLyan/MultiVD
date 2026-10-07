#!/usr/bin/env python3
"""Ghep cac bo nguon v2 thanh pool Pha 1 theo dung quy uoc `data/mwsrc_<ten>/fold1/`.

`qg.sh` voi TGT=src<ten> doc `data/mwsrc_<ten>/fold1/{train,val,test}.jsonl` va dat
FOLDS=1 — Pha 1 chay MOT lan, ra mot checkpoint, roi Pha 2 dung lai cho ca 5 fold dich.
Cac pool cu cua du an co `test.jsonl` la BAN SAO cua `val.jsonl` (da doi chieu md5);
giu dung nhu vay.

`train_mwg.py` chi doc: code, label, lang, pair_id. Van ghi them cwe/cwe_id/cwe_class
de dung lai duoc cho muc dich khac.

Chia val: NGAU NHIEN THEO HANG, ti le --val_ratio (nguoi dung neu 0.2 cho Pha 1), seed 42.
In ra so cap bi tach giua train/val de biet ro muc ro ri song sinh trong val nguon.
"""
import argparse, collections, json, os, random

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    POOLS = {
        "v2full":   [("ccpp_primevul_from-paired_full", "ccpp"),
                     ("java_cleanvul_3-4_full", "java"),
                     ("js_cleanvul_3-4_full", "js")],
        "v2common": [("ccpp_primevul_from-paired_common", "ccpp"),
                     ("java_cleanvul_3-4_common", "java"),
                     ("js_cleanvul_3-4_common", "js")],
        "v2cwe4":   [("ccpp_primevul_from-paired_4cwe", "ccpp"),
                     ("js_cleanvul_3-4_4cwe", "js")],
        # Nguoi dung neu 22/09: muon so common CO ccpp vs KHONG ccpp.
        # Bo ccpp thi con java+js = 7008 hang, rat gan thanh phan cua `jsCjv`
        # (pool DA BIET la hoc duoc) — neu ban nay hoc duoc con ban co ccpp thi khong,
        # ccpp la thu pham.
        "v2common_nocc": [("java_cleanvul_3-4_common", "java"),
                          ("js_cleanvul_3-4_common", "js")],
    }
    print("%-10s %8s %8s %8s %9s %9s %8s %10s" %
          ("pool", "ccpp", "java", "js", "TONG", "train", "val", "cap tach"))
    print("-" * 82)
    for name, parts in POOLS.items():
        rows = []
        per = collections.Counter()
        for base, lang in parts:
            recs = load(os.path.join(a.src_dir, base + ".jsonl"))
            metas = load(os.path.join(a.src_dir, base + ".meta.jsonl"))
            for i, (r, m) in enumerate(zip(recs, metas)):
                pid = m.get("pair_id")
                if not pid:                      # PrimeVul: hai dong lien tiep la mot cap
                    pid = "primevul:%s:%d" % (m.get("split_goc", "?"), m.get("idx_goc", i) // 2)
                rows.append({"code": r["code"], "label": r["label"], "lang": lang,
                             "cwe": r["cwe"], "cwe_id": r["cwe_id"], "cwe_class": r["cwe_class"],
                             "pair_id": pid, "src_bo": base})
            per[lang] += len(recs)
        rng = random.Random(a.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        nval = int(round(len(rows) * a.val_ratio))
        vset = set(idx[:nval])
        train = [rows[i] for i in range(len(rows)) if i not in vset]
        val = [rows[i] for i in range(len(rows)) if i in vset]
        # dem cap bi tach giua train va val
        side = {}
        for i, r in enumerate(rows):
            side.setdefault(r["pair_id"], []).append(i in vset)
        tach = sum(1 for v in side.values() if len(v) == 2 and v[0] != v[1])
        d = os.path.join(a.out_root, "mwsrc_" + name, "fold1")
        os.makedirs(d, exist_ok=True)
        for fn, data in (("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", val)):
            with open(os.path.join(d, fn), "w", encoding="utf-8") as fh:
                for r in data:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("%-10s %8d %8d %8d %9d %9d %8d %10d" %
              (name, per["ccpp"], per["java"], per["js"], len(rows), len(train), len(val), tach))
    print()
    print("  test.jsonl la BAN SAO cua val.jsonl — dung quy uoc cac pool nguon cu cua du an")
    print("  (danh gia that nam o Pha 2 tren tap dich python, khong o day).")
    print("  'cap tach' = so cap ma mot nua o train, nua kia o val — ro ri song sinh trong")
    print("  val NGUON. Val nguon chi dung de CHON CHECKPOINT nen chap nhan duoc, nhung")
    print("  KHONG duoc doc nhu mot thuoc do.")

if __name__ == "__main__":
    main()
