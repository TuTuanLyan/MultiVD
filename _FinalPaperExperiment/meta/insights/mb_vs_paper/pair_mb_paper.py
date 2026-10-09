#!/usr/bin/env python3
"""Ghép cặp khối Mosbach val JS+Py (mbmix) với tab paper đầu (AdamW, val JS) theo (nguồn, seed, fold).

Chỉ đọc results/ và logs/ local 161 (sync đã chuyển kết quả máy xa về đây). In:
  1) bảng từng ô: 4 chỉ số test của paper / mbmix / e30 / Mosbach val-JS cũ (archive)
  2) Δ ghép cặp (mbmix - paper, e30 - paper, mbmix - mb val-JS) kèm +/n, min-max
  3) chẩn đoán từ log: Pha 1 (epoch chọn, train loss tại đó, val chọn), Pha 2 (train loss / val ROC ep1,
     epoch đầu train loss < 0,6, best epoch)
Chạy: python3 meta/insights/mb_vs_paper/pair_mb_paper.py  (từ _FinalPaperExperiment)
"""
import json
import os
import re
import statistics

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RES = os.path.join(BASE, "results")
LOG = os.path.join(BASE, "logs")
ARC = os.path.join(BASE, "archive_0810_mosbach_valjs")
SRCS = ["common_ext", "4cwe", "full"]
SEEDS = [("", 42), ("_s1234", 1234), ("_s7", 7)]
METRICS = [("test_roc_auc", "ROC"), ("test_pr_auc", "PR"), ("test_macro_f1_at_0.5", "F1@0.5"),
           ("test_macro_f1_at_valcal", "F1@val")]
EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+) \| train loss ([\d.]+) \| val loss ([\d.]+) \| val roc_auc ([\d.]+)"
                      r" \| val macro_f1 ([\d.]+) \| best ep (\d+)")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def arms(src, sfx):
    """Tên run Pha 2 của từng nhánh, và Pha 1 tương ứng."""
    return {
        "paper": (os.path.join(RES, f"rasam_jspy41jsval_{src}{sfx}"), os.path.join(LOG, f"rasam_jspy41jsval_{src}{sfx}"),
                  os.path.join(LOG, f"p1_jspy41jsval_{src}{sfx}"), os.path.join(RES, f"p1_jspy41jsval_{src}{sfx}")),
        "mbmix": (os.path.join(RES, f"rasam_mbmix_jspy41mixval_{src}{sfx}"),
                  os.path.join(LOG, f"rasam_mbmix_jspy41mixval_{src}{sfx}"),
                  os.path.join(LOG, f"p1_mbmix_jspy41mixval_{src}{sfx}"),
                  os.path.join(RES, f"p1_mbmix_jspy41mixval_{src}{sfx}")),
        "e30": (os.path.join(RES, f"rasam_e30_jspy41jsval_{src}{sfx}"), os.path.join(LOG, f"rasam_e30_jspy41jsval_{src}{sfx}"),
                os.path.join(LOG, f"p1_e30_jspy41jsval_{src}{sfx}"), os.path.join(RES, f"p1_e30_jspy41jsval_{src}{sfx}")),
        "mbjs": (os.path.join(ARC, "results", f"rasam_mb_jspy41jsval_{src}{sfx}"),
                 os.path.join(ARC, "logs", f"rasam_mb_jspy41jsval_{src}{sfx}"),
                 os.path.join(ARC, "logs", f"p1_mb_jspy41jsval_{src}{sfx}"),
                 os.path.join(ARC, "results", f"p1_mb_jspy41jsval_{src}{sfx}")),
    }


def epochs(path):
    """Danh sách (ep, train_loss, val_roc, best_ep) từ một log; chỉ lấy lượt train cuối cùng trong file."""
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = EPOCH_RE.search(line)
                if m:
                    ep = int(m.group(1))
                    if ep == 1:
                        out = []
                    out.append((ep, float(m.group(3)), float(m.group(5)), int(m.group(7))))
    except OSError:
        return []
    return out


def p1_diag(logdir, resdir, fold):
    ep = epochs(os.path.join(logdir, f"fold{fold}.log"))
    r = load(os.path.join(resdir, f"fold{fold}.json")) or {}
    be = r.get("best_epoch")
    if not ep:
        return None
    if be is None:
        be = ep[-1][3]
    tl = {e: t for e, t, _, _ in ep}
    vr = {e: v for e, _, v, _ in ep}
    return {"best_ep": be, "n_ep": len(ep), "tl_best": tl.get(be), "val_best": vr.get(be),
            "tl_min": min(tl.values()), "val_ep1": vr.get(1), "val_max": max(vr.values()),
            "val_max_ep": max(vr, key=vr.get), "p1_test_roc": r.get("test_roc_auc")}


def p2_diag(logdir, fold):
    ep = epochs(os.path.join(logdir, f"fold{fold}.log"))
    if not ep:
        return None
    esc = next((e for e, t, _, _ in ep if t < 0.6), None)
    return {"tl1": ep[0][1], "val1": ep[0][2], "esc": esc, "best": ep[-1][3], "n_ep": len(ep),
            "tl_max3": max(t for _, t, _, _ in ep[:3])}


def sign_summary(ds):
    ds = [d for d in ds if d is not None]
    if not ds:
        return "n=0"
    pos = sum(d > 1e-9 for d in ds)
    neg = sum(d < -1e-9 for d in ds)
    return (f"{statistics.mean(ds):+.4f} (+{pos}/-{neg}, n={len(ds)}, min {min(ds):+.4f}, max {max(ds):+.4f},"
            f" trung vị {statistics.median(ds):+.4f})")


def main():
    rows = []
    for src in SRCS:
        for sfx, seed in SEEDS:
            A = arms(src, sfx)
            for fold in range(1, 6):
                mb = load(os.path.join(A["mbmix"][0], f"fold{fold}.json"))
                if mb is None:
                    continue
                row = {"src": src, "seed": seed, "fold": fold}
                for arm, (rdir, l2, l1, r1) in A.items():
                    r = load(os.path.join(rdir, f"fold{fold}.json"))
                    row[arm] = {k: r.get(k) for k, _ in METRICS} if r else None
                    row[arm + "_p1"] = p1_diag(l1, r1, fold)
                    row[arm + "_p2"] = p2_diag(l2, fold)
                rows.append(row)

    print(f"# Ô mbmix đã xong: {len(rows)} (kiểm chứng, n={len(rows)})\n")
    print("## Từng ô - test ROC / PR / F1@0.5 / F1@val")
    for r in rows:
        def fmt(arm):
            v = r[arm]
            return "-" if v is None else "/".join(f"{v[k]:.3f}" if v[k] is not None else "?" for k, _ in METRICS)
        print(f"{r['src']:<10} s{r['seed']:<4} f{r['fold']} | paper {fmt('paper')} | mbmix {fmt('mbmix')}"
              f" | e30 {fmt('e30')} | mb-valJS {fmt('mbjs')}")

    for a, b in [("mbmix", "paper"), ("e30", "paper"), ("mbmix", "mbjs"), ("mbmix", "e30")]:
        print(f"\n## Δ ghép cặp {a} - {b}")
        for k, name in METRICS:
            ds = [r[a][k] - r[b][k] for r in rows if r[a] and r[b] and r[a][k] is not None and r[b][k] is not None]
            print(f"  {name:<7} {sign_summary(ds)}")

    print("\n## Chẩn đoán Pha 1 (epoch chọn | train loss tại đó | val chọn | val ep1 | val max@ep | train loss min)")
    for r in rows:
        for arm in ("paper", "mbjs", "mbmix", "e30"):
            d = r[arm + "_p1"]
            if d:
                print(f"{r['src']:<10} s{r['seed']:<4} f{r['fold']} {arm:<6} chọn ep{d['best_ep']:>2}/{d['n_ep']:<2}"
                      f" tl {d['tl_best']:.3f} val {d['val_best']:.3f} | val ep1 {d['val_ep1']:.3f}"
                      f" | val max {d['val_max']:.3f}@ep{d['val_max_ep']} | tl min {d['tl_min']:.3f}")

    print("\n## Chẩn đoán Pha 2 (train loss ep1 | max train loss 3 ep đầu | val ROC ep1 | ep đầu tl<0,6 | best ep)")
    for r in rows:
        for arm in ("paper", "mbjs", "mbmix", "e30"):
            d = r[arm + "_p2"]
            if d:
                print(f"{r['src']:<10} s{r['seed']:<4} f{r['fold']} {arm:<6} tl1 {d['tl1']:.3f} tlmax3 {d['tl_max3']:.3f}"
                      f" val1 {d['val1']:.3f} thoát@{d['esc']} best ep{d['best']}/{d['n_ep']}")

    # tương quan đơn giản: train loss Pha 2 ep1 với ROC test, trên mọi ô có đủ
    pts = []
    for r in rows:
        for arm in ("paper", "mbjs", "mbmix", "e30"):
            if r[arm] and r[arm + "_p2"]:
                pts.append((r[arm + "_p2"]["tl1"], r[arm]["test_roc_auc"], arm))
    print("\n## Train loss Pha 2 ep1 theo nhánh (trung vị, min-max)")
    for arm in ("paper", "mbjs", "mbmix", "e30"):
        v = [p[0] for p in pts if p[2] == arm]
        if v:
            print(f"  {arm:<6} n={len(v)} trung vị {statistics.median(v):.3f} min {min(v):.3f} max {max(v):.3f}")
    print("\n## Ô có train loss Pha 2 ep1 > 0,69 (khởi đầu ở / trên ln2)")
    for tl, roc, arm in sorted(pts):
        if tl > 0.69:
            print(f"  {arm:<6} tl1 {tl:.3f} -> test ROC {roc:.3f}")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def paper_all():
    """Mọi ô tab paper đầu (3 nguồn x 3 seed x 5 fold): chẩn đoán Pha 1 / Pha 2 và test ROC, đánh dấu ô yếu."""
    print("\n# Toàn bộ tab paper đầu (n=45): Pha 1 chọn ep / train loss tại đó / tl min ; Pha 2 tl1 / thoát@ / best ; test 4 chỉ số")
    flagged = []
    for src in SRCS:
        for sfx, seed in SEEDS:
            A = arms(src, sfx)
            rdir, l2, l1, r1 = A["paper"]
            for fold in range(1, 6):
                r = load(os.path.join(rdir, f"fold{fold}.json"))
                if not r:
                    continue
                d1, d2 = p1_diag(l1, r1, fold), p2_diag(l2, fold)
                t = "/".join(f"{r.get(k, float('nan')):.3f}" for k, _ in METRICS)
                p1s = (f"P1 ep{d1['best_ep']:>2}/{d1['n_ep']:<2} tl {d1['tl_best']:.3f} tlmin {d1['tl_min']:.3f}"
                       if d1 else "P1 ?")
                p2s = f"P2 tl1 {d2['tl1']:.3f} thoát@{d2['esc']} best {d2['best']}" if d2 else "P2 ?"
                flag = []
                if d1 and d1["tl_best"] > 0.68:
                    flag.append("P1-chọn-ckpt-chưa-học")
                if d1 and d1["tl_min"] > 0.68:
                    flag.append("P1-kẹt-ln2")
                if d2 and (d2["esc"] is None or d2["esc"] > 2):
                    flag.append(f"P2-thoát-muộn@{d2['esc']}")
                if r.get("test_roc_auc", 1) < 0.92:
                    flag.append("ROC<0,92")
                line = f"{src:<10} s{seed:<4} f{fold} | {p1s} | {p2s} | {t} {' '.join(flag)}"
                print(line)
                if flag:
                    flagged.append(line)
    print(f"\n## Ô bị đánh dấu: {len(flagged)}")
    for x in flagged:
        print(x)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--paper-all":
    paper_all()


def by_seed():
    """Δ mbmix - paper tách theo seed (để thấy phần giảm dồn vào seed nào)."""
    for seeds, name in (((1234,), "chỉ s1234"), ((42, 7), "s42 + s7")):
        print(f"\n## Δ mbmix - paper, {name}")
        for k, lab in METRICS:
            ds = []
            for src in SRCS:
                for sfx, seed in SEEDS:
                    if seed not in seeds:
                        continue
                    A = arms(src, sfx)
                    for fold in range(1, 6):
                        a = load(os.path.join(A["mbmix"][0], f"fold{fold}.json"))
                        b = load(os.path.join(A["paper"][0], f"fold{fold}.json"))
                        if a and b:
                            ds.append(a[k] - b[k])
            print(f"  {lab:<7} {sign_summary(ds)}")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "--by-seed":
    by_seed()
