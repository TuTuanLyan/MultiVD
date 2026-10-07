#!/usr/bin/env python3
"""Pha 1 THỬ LẠI KHI KẸT cho các run có "p1_retry" trong runs.json (tab refactor; người dùng 28/09: "retry khi bị sụp").

    python3 p1_retry.py <host> <run_id>          (run.sh gọi qua TRAIN_CMD do `fpe.py plan` sinh; stdout/stderr đã về log fold)

Làm đúng cơ chế scripts/run_p1.sh của nhánh refactor (GraphTransferVD@c1f1641), nhưng qua hạ tầng khối final (det_launch tất định,
cờ lấy từ runs.json):
  - lượt i dùng seed p1_retry.seeds[i] (mặc định 42 → 1234 → 7 — CLAUDE.md: seed phải là 42/1234/7), checkpoint
    <ckpt>/<run>/seed_<S>/fold1/best.pt; train.py tự kiểm ở --stuck_epoch và thoát mã 3 nếu train loss giảm < --stuck_min_drop;
  - mã 3 (kẹt): best.pt -> stuck.pt, best_<also>.pt -> stuck_best_<also>.pt (GIỮ để kiểm, như KEEP_STUCK=1), đoạn log của lượt
    đó chép ra logs/<run>/fold1_seed<S>_stuck.log, rồi thử seed kế; lượt đã kẹt ở lần chạy trước (có cả stuck log lẫn stuck.pt) được bỏ qua;
  - lượt không kẹt: test ngay trên checkpoint đó (kết quả ở results/<run>/fold1.json), ghi <ckpt>/<run>/P1_CKPT = "seed_<S>"
    (fpe.py đọc để Pha 2 nạp đúng checkpoint), thoát 0;
  - kẹt CẢ mọi lượt: KHÔNG bỏ ô (CLAUDE.md §3 — ô trống không viết được gì) — dùng lượt cuối (stuck.pt -> best.pt), test, ghi
    P1_CKPT = "seed_<S> STUCK_ALL" để artifact đánh dấu, thoát 0.
Dòng bắt đầu bằng "### P1 " là mốc cho fpe.py live (chỉ đọc lượt thử cuối) và cho artifact."""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fpe  # noqa: E402

STUCK_EXIT = 3


def say(msg):
    print("### P1 " + msg, flush=True)


def main():
    host, run_id = sys.argv[1], sys.argv[2]
    run = fpe.RUNS[run_id]
    cfg = run["p1_retry"]
    seeds = [int(s) for s in cfg.get("seeds", [42, 1234, 7])]
    h = fpe.M["hosts"][host]
    also = fpe.merged_flags(run).get("--also_select")
    root = os.path.join(h["ckpt"], run_id)
    ptr = os.path.join(root, "P1_CKPT")
    logf = fpe.paths(host, run, 1)["log"]
    if os.path.exists(ptr):
        os.remove(ptr)
    for i, s in enumerate(seeds, 1):
        train, test, p = fpe.commands(host, run, 1, seed=s)
        ck = p["ckpt"]
        d = os.path.dirname(ck)
        stuck_pt = os.path.join(d, "stuck.pt")
        stuck_log = os.path.join(os.path.dirname(logf), "fold1_seed%d_stuck.log" % s)
        if os.path.exists(stuck_log) and os.path.exists(stuck_pt) and i < len(seeds):
            say("seed %d đã kẹt ở lần chạy trước (%s) — bỏ qua" % (s, os.path.basename(stuck_log)))
            continue
        say("thử seed %d (lượt %d/%d) · checkpoint %s" % (s, i, len(seeds), ck))
        sys.stdout.flush()
        start = os.path.getsize(logf) if os.path.exists(logf) else 0
        rc = subprocess.call(train, cwd=h["root"])
        if rc == STUCK_EXIT:
            for name in ["best.pt"] + (["best_%s.pt" % also] if also else []):
                q = os.path.join(d, name)
                if os.path.exists(q):
                    os.replace(q, os.path.join(d, "stuck.pt" if name == "best.pt" else "stuck_" + name))
            try:
                with open(logf, "rb") as fh:
                    fh.seek(start)
                    chunk = fh.read()
                with open(stuck_log, "wb") as fh:
                    fh.write(chunk)
            except OSError as e:
                say("không chép được log lượt kẹt: %s" % e)
            if i < len(seeds):
                say("KẸT seed %d (mã thoát 3) → checkpoint giữ ở stuck.pt, log ở %s → thử seed kế" % (s, os.path.basename(stuck_log)))
                continue
            # kẹt cả mọi lượt: dùng lượt cuối (CLAUDE.md §3)
            say("KẸT CẢ %d lượt (seed %s) → vẫn dùng lượt cuối seed %d để Pha 2 chạy và ghi nhận (CLAUDE.md §3)"
                % (len(seeds), ", ".join(map(str, seeds)), s))
            os.replace(stuck_pt, ck)
            tag = "seed_%d STUCK_ALL" % s
        elif rc != 0:
            say("seed %d LỖI (mã thoát %d) — dừng, không thử seed khác" % (s, rc))
            sys.exit(rc)
        else:
            tag = "seed_%d" % s
            say("seed %d KHÔNG kẹt → dùng checkpoint này" % s)
        sys.stdout.flush()
        rc = subprocess.call(test, cwd=h["root"])
        if rc != 0:
            say("test seed %d LỖI (mã thoát %d)" % (s, rc))
            sys.exit(rc)
        os.makedirs(root, exist_ok=True)
        with open(ptr + ".tmp", "w", encoding="utf-8") as fh:
            fh.write(tag + "\n")
        os.replace(ptr + ".tmp", ptr)
        say("xong · P1_CKPT = %s" % tag)
        return
    say("không còn seed nào để thử — dừng")
    sys.exit(1)


if __name__ == "__main__":
    main()
