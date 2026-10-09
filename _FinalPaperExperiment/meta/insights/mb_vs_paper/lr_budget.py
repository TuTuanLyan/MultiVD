#!/usr/bin/env python3
"""Ngân sách LR Pha 1 tới epoch được chọn (tích phân lr theo epoch, lịch tuyến tính có warmup như get_linear_schedule_with_warmup)
so với dấu hiệu Pha 2 (train loss ep1, epoch thoát ln2) và test ROC; kèm thời gian thật mỗi epoch theo nguồn/pha.
Chạy: python3 meta/insights/mb_vs_paper/lr_budget.py (từ _FinalPaperExperiment)"""
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_mb_paper as P  # noqa: E402

SCHED = {"paper": (16, 4.0, 5.66), "mbmix": (20, 2.0, 2.0), "mbjs": (20, 2.0, 2.0), "e30": (30, 7.5, 5.66)}


def lr_at(t, E, W, peak):
    return peak * t / W if t < W else peak * max(0.0, (E - t) / (E - W))


def lr_integral(e, E, W, peak, n=2000):
    h = e / n
    return sum(lr_at((i + 0.5) * h, E, W, peak) for i in range(n)) * h


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for k, i in enumerate(o):
        r[i] = k
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    pts = []
    print("arm    nguồn       seed  f | chọn ep | ∫lr tới ep (1e-5·epoch) | lr tại ep | tl P1 | P2 tl1 | thoát@ | test ROC")
    for src in P.SRCS:
        for sfx, seed in P.SEEDS:
            A = P.arms(src, sfx)
            for fold in range(1, 6):
                for arm, (rdir, l2, l1, r1) in A.items():
                    r = P.load(os.path.join(rdir, f"fold{fold}.json"))
                    d1, d2 = P.p1_diag(l1, r1, fold), P.p2_diag(l2, fold)
                    if not (r and d1 and d2):
                        continue
                    E, W, pk = SCHED[arm]
                    e = d1["best_ep"]
                    I, L = lr_integral(e, E, W, pk), lr_at(e, E, W, pk)
                    pts.append(dict(arm=arm, src=src, seed=seed, fold=fold, ep=e, I=I, L=L, tlp1=d1["tl_best"],
                                    tl1=d2["tl1"], esc=d2["esc"] if d2["esc"] is not None else 31, roc=r["test_roc_auc"]))
                    print(f"{arm:<6} {src:<10} s{seed:<4} f{fold} | ep{e:>2} | {I:6.1f} | {L:5.2f} | {d1['tl_best']:.3f} |"
                          f" {d2['tl1']:.3f} | {d2['esc']} | {r['test_roc_auc']:.3f}")
    print(f"\nn = {len(pts)} ô (mọi nhánh, mọi ô có đủ log + kết quả)")
    for k in ("I", "L", "tlp1"):
        print(f"Spearman({k}, P2 tl1) = {spearman([p[k] for p in pts], [p['tl1'] for p in pts]):+.3f};"
              f" ({k}, thoát@) = {spearman([p[k] for p in pts], [p['esc'] for p in pts]):+.3f};"
              f" ({k}, test ROC) = {spearman([p[k] for p in pts], [p['roc'] for p in pts]):+.3f}")
    print("\nTheo nhánh: ∫lr tới epoch chọn (trung vị, min-max) và tỉ lệ ô P2 thoát muộn (>2)")
    for arm in SCHED:
        q = [p for p in pts if p["arm"] == arm]
        if q:
            Is = [p["I"] for p in q]
            late = sum(p["esc"] > 2 for p in q)
            print(f"  {arm:<6} n={len(q):>2} ∫lr trung vị {statistics.median(Is):5.1f} [{min(Is):.1f}-{max(Is):.1f}]"
                  f" | thoát muộn {late}/{len(q)}")
    print("\nTheo seed (mọi nhánh): P2 tl1 trung vị | thoát muộn")
    for seed in (42, 1234, 7):
        for arm in ("paper", "mbmix", "e30"):
            q = [p for p in pts if p["seed"] == seed and p["arm"] == arm]
            if q:
                print(f"  s{seed:<4} {arm:<6} n={len(q):>2} tl1 trung vị {statistics.median([p['tl1'] for p in q]):.3f}"
                      f" | thoát muộn {sum(p['esc'] > 2 for p in q)}/{len(q)}")
    # ngân sách cả lượt
    for arm, (E, W, pk) in SCHED.items():
        print(f"∫lr cả lịch {arm}: {lr_integral(E, E, W, pk):.1f}")

    # thời gian thật mỗi epoch (giây) theo pha / nguồn / nhánh
    ep_re = re.compile(r"Epoch \d+/\d+ .*\| (\d+)s$")
    print("\nGiây mỗi epoch (trung vị) theo log:")
    for arm, pref1, pref2 in (("paper", "p1_jspy41jsval_", "rasam_jspy41jsval_"), ("mbmix", "p1_mbmix_jspy41mixval_",
                                                                                     "rasam_mbmix_jspy41mixval_")):
        for src in P.SRCS:
            for pref, ph in ((pref1, "P1"), (pref2, "P2")):
                secs, neps = [], []
                for sfx, _ in P.SEEDS:
                    d = os.path.join(P.LOG, pref + src + sfx)
                    for fold in range(1, 6):
                        f = os.path.join(d, f"fold{fold}.log")
                        if not os.path.exists(f):
                            continue
                        s = [int(m.group(1)) for m in (ep_re.search(x.rstrip()) for x in open(f, errors="replace")) if m]
                        if s:
                            secs += s
                            neps.append(len(s))
                if secs:
                    print(f"  {arm:<6} {ph} {src:<10} {statistics.median(secs):4.0f} s/epoch, epoch/lượt trung vị"
                          f" {statistics.median(neps):.0f} (n lượt {len(neps)})")


if __name__ == "__main__":
    main()
