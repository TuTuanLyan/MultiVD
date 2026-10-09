#!/usr/bin/env python3
"""Tính OFFLINE (từ log, 0 GPU) epoch Pha 1 sẽ được chọn nếu đổi --select_after_drop trên đúng các lượt tab paper đầu.
Hợp lệ vì: lịch LR / dữ liệu / seed không đổi => quỹ đạo train loss + val JS ROC từng epoch trùng log (tất định §86);
Pha 1 paper luôn chạy đủ 16 epoch (min_epochs 10 + patience 8 => sớm nhất dừng ep17) nên luật dừng không đổi.
KHÔNG suy ra được điểm Pha 2 (checkpoint epoch mới chưa từng lưu)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pair_mb_paper as P  # noqa: E402


def select(ep, d):
    """Mô phỏng vòng chọn của src_mwonly/train.py: trước cổng chọn theo train loss nhỏ nhất, từ epoch cổng mở chọn val ROC lớn nhất."""
    best, best_ep, gate = float("-inf"), 0, d <= 0
    tl0 = ep[0][1]
    for e, tl, vr, _ in ep:
        if not gate and tl < (1 - d) * tl0:
            gate, best = True, float("-inf")
        score = vr if gate else -tl
        if score > best:
            best, best_ep = score, e
    return best_ep


def main():
    ds = (0.02, 0.05, 0.10, 0.15)
    print("nguồn       seed  f | thực tế | " + " | ".join(f"d={d:.2f}: ep (tl)" for d in ds) + " | test ROC paper")
    changed = {d: 0 for d in ds}
    mism = 0
    for src in P.SRCS:
        for sfx, seed in P.SEEDS:
            rdir, l2, l1, r1 = P.arms(src, sfx)["paper"]
            for fold in range(1, 6):
                ep = P.epochs(os.path.join(l1, f"fold{fold}.log"))
                r = P.load(os.path.join(r1, f"fold{fold}.json")) or {}
                r2 = P.load(os.path.join(rdir, f"fold{fold}.json")) or {}
                if not ep:
                    continue
                tl = {e: t for e, t, _, _ in ep}
                actual = r.get("best_epoch")
                sel = {d: select(ep, d) for d in ds}
                if sel[0.02] != actual:
                    mism += 1
                for d in ds:
                    changed[d] += sel[d] != actual
                cells = " | ".join(f"ep{sel[d]:>2} ({tl[sel[d]]:.3f}){'*' if sel[d] != actual else ' '}" for d in ds)
                print(f"{src:<10} s{seed:<4} f{fold} | ep{actual:>2} | {cells} | {r2.get('test_roc_auc', float('nan')):.3f}")
    print(f"\nKiểm mô phỏng: d=0,02 lệch epoch thực tế ở {mism} ô (phải bằng 0)")
    for d in ds:
        print(f"d={d:.2f}: {changed[d]} ô đổi epoch chọn so với thực tế")


if __name__ == "__main__":
    main()
