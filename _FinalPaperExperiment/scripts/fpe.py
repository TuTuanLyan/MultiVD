#!/usr/bin/env python3
"""Khối thí nghiệm final (`_FinalPaperExperiment`): sinh lệnh từ scripts/runs.json — nguồn sự thật DUY NHẤT cho script và artifact.

Bố trí: scripts/ (mã + runs.json, thư mục DUY NHẤT để commit) · meta/ (bảng sinh tự động) · state/ (lock, log driver)
· logs/<run>/ · results/<run>/ (mỗi run MỘT bản, chỉ trên 161).

  fpe.py plan <host> <run> <fold>    in các biến bash (lệnh train/test, đường log/kết quả/checkpoint) cho run.sh `eval`
  fpe.py resolve                     ghi meta/runs_resolved.json: cờ đã gộp của từng run + mẫu lệnh (cho artifact)
  fpe.py collect                     gom results/<run>/fold*.json -> results/summary.json (cho artifact)
  fpe.py order                       thứ tự chạy: Pha 1 trước, rồi các nhánh đích (in id, mỗi dòng một run)
  fpe.py effective                   ghi meta/runs_effective.json: MỌI tham số hiệu lực (kể cả mặc định) do chính trainer parse
  fpe.py dbrows [host]               ghi meta/dbrows/<run>__f<F>.json đúng khuôn collection `folds` của artifact (kèm kiểm tham số)
"""
import glob, json, os, re, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
M = json.load(open(os.path.join(HERE, "runs.json"), encoding="utf-8"))
RUNS = {r["id"]: r for r in M["runs"]}
SEED = M["block"]["seed"]
FLAG_ORDER_HEAD = ("--phase", "--run_name", "--fold")


def merged_flags(run):
    out = {}
    for name in run.get("flag_sets", []):
        out.update(M["flag_sets"][name])
    out.update(run.get("flags", {}))
    return out


def _argv(flags):
    argv = []
    for k, v in flags.items():
        argv += [k, str(v)]
    return argv


def p1_seed_dir(host, run_id):
    """Thư mục seed của checkpoint Pha 1 DÙNG ĐƯỢC. Run có p1_retry (tab refactor): scripts/p1_retry.py ghi tên thư mục (vd. seed_1234)
    vào <ckpt>/<run>/P1_CKPT khi một lượt không kẹt — ghi TƯƠNG ĐỐI để chép sang máy khác vẫn dùng được. Còn lại: seed của khối."""
    ptr = os.path.join(M["hosts"][host]["ckpt"], run_id, "P1_CKPT")
    if RUNS[run_id].get("p1_retry"):
        # chưa có P1_CKPT (Pha 1 chưa xong / lỗi thật) ⇒ trỏ tới thư mục KHÔNG tồn tại: run.sh in "!! thiếu checkpoint Pha 1 — bỏ qua fold"
        # thay vì nạp một best.pt dở dang của lượt vừa lỗi (thử mô phỏng 28/09, ca "seed 42 lỗi thật")
        return open(ptr, encoding="utf-8").read().split()[0] if os.path.exists(ptr) else "seed_ghi_trong_P1_CKPT"
    return "seed_%d" % run_seed(RUNS[run_id])


def run_seed(run):
    """Seed của MỘT run: khoá "seed" trong runs.json (tab Kết quả paper, bậc 3: 42 / 1234 / 7 - người dùng 07/10), mặc định seed của khối."""
    return run.get("seed", SEED)


def paths(host, run, fold, seed=None):
    h = M["hosts"][host]
    ck = os.path.join(h["ckpt"], run["id"], "seed_%d" % (run_seed(run) if seed is None else seed), "fold%d" % fold, "best.pt")
    init = None
    if run.get("init_from"):
        # init_same_fold (06/10, Pha 1 trộn Python theo fold): Pha 2 fold k nạp checkpoint Pha 1 CỦA fold k; mặc định mọi fold dùng fold1
        init_fold = fold if run.get("init_same_fold") else 1
        init = os.path.join(h["ckpt"], run["init_from"], p1_seed_dir(host, run["init_from"]), "fold%d" % init_fold, "best.pt")
    return {"log": os.path.join(h["out"], "logs", run["id"], "fold%d.log" % fold),
            "result": os.path.join(h["out"], "results", run["id"], "fold%d.json" % fold),
            "ckpt": ck, "init_ckpt": init}


def commands(host, run, fold, seed=None):
    """seed: chỉ p1_retry.py dùng (lượt thử lại của Pha 1); mặc định là seed của khối."""
    h = M["hosts"][host]
    p = paths(host, run, fold, seed)
    flags = merged_flags(run)
    if seed is not None:
        flags["--seed"] = seed
    elif "seed" in run:                                   # run có seed riêng (tab paper): đè --seed của bộ cờ chung
        flags["--seed"] = run["seed"]
    head = [h["python"], os.path.join(h["out"], "scripts", "det_launch.py"), run.get("src_dir", h["src"]), run["trainer"]]
    common = ["--run_name", "final_" + run["id"], "--fold", str(fold), "--model_name", h["model"], "--checkpoint_path", p["ckpt"]]
    train_flags = dict(flags)
    if p["init_ckpt"]:
        train_flags["--init_ckpt"] = p["init_ckpt"]
    test_flags = {k: v for k, v in flags.items() if k not in ("--init", "--init_ckpt")}
    train = head + ["--phase", "train"] + common + _argv(train_flags)
    test = head + ["--phase", "test"] + common + _argv(test_flags) + ["--result_path", p["result"]]
    return train, test, p


def cmd_plan(host, run_id, fold):
    run = RUNS[run_id]
    fold = int(fold)
    folds = run.get("folds", M["block"]["target_folds"])
    if fold not in folds:
        sys.exit("fold %d không thuộc run %s (%s)" % (fold, run_id, folds))
    skip = os.path.join(M["hosts"][host]["out"], "state", "skip.txt")      # chia việc giữa hai máy mà không dừng driver đang chạy
    if os.path.exists(skip):
        want = {ln.split("#")[0].strip() for ln in open(skip, encoding="utf-8")}
        if "%s:%d" % (run_id, fold) in want or run_id in want:
            # KHÔNG thoát lỗi: run.sh (kể cả bản đang chạy) làm `eval "$(plan)" || continue`, mà eval chuỗi rỗng trả 0 ⇒ nó sẽ dùng lại
            # lệnh của fold TRƯỚC (xoá rồi chạy lại fold đó). Thay vào đó trỏ INIT_CKPT tới file không tồn tại: nhánh "thiếu checkpoint
            # Pha 1 — bỏ qua fold này" của run.sh chạy TRƯỚC mọi lệnh xoá.
            skip_marker = os.path.join(M["hosts"][host]["out"], "state", "SKIP__%s_f%d__chay_o_may_khac" % (run_id, fold))
            print("bỏ qua %s f%d theo %s" % (run_id, fold, skip), file=sys.stderr)
        else:
            skip_marker = None
    else:
        skip_marker = None
    train, test, p = commands(host, run, fold)
    if skip_marker:
        p = dict(p, init_ckpt=skip_marker)
    if run.get("p1_retry") and not skip_marker:
        # Pha 1 có thử lại khi kẹt (tab refactor): train + test do p1_retry.py làm (seed và checkpoint chỉ biết sau khi chạy)
        train = ["python3", os.path.join(HERE, "p1_retry.py"), host, run_id]
        test = ["true"]
    env = {"PYTHONHASHSEED": "42", "TOKENIZERS_PARALLELISM": "false"}
    env.update(run.get("env", {}))
    print("RUN_ENV=%s" % shlex.quote(" ".join("%s=%s" % kv for kv in env.items())))
    print("TRAIN_CMD=%s" % shlex.quote(shlex.join(train)))
    print("TEST_CMD=%s" % shlex.quote(shlex.join(test)))
    for k in ("log", "result", "ckpt"):
        print("%s=%s" % (k.upper(), shlex.quote(p[k])))
    print("INIT_CKPT=%s" % shlex.quote(p["init_ckpt"] or ""))
    print("STAGE=%s" % run["stage"])


def cmd_resolve():
    out = []
    for run in M["runs"]:
        train, test, _ = commands("161", run, run.get("folds", M["block"]["target_folds"])[0])
        out.append({"id": run["id"], "flags": merged_flags(run), "env": run.get("env", {}), "init_from": run.get("init_from"),
                    "folds": run.get("folds", M["block"]["target_folds"]),
                    "train_cmd_161": shlex.join(train), "test_cmd_161": shlex.join(test)})
    json.dump(out, open(os.path.join(BASE, "meta", "runs_resolved.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ghi meta/runs_resolved.json: %d run" % len(out))


def cmd_collect():
    summary = {}
    for run in M["runs"]:
        rows = {}
        for f in sorted(glob.glob(os.path.join(BASE, "results", run["id"], "fold*.json"))):
            fold = int(os.path.basename(f)[4:-5])
            r = json.load(open(f, encoding="utf-8"))
            rows[fold] = {k: r.get(k) for k in M["block"]["metrics"] + ["best_epoch"]}
            rows[fold]["hyperparameters"] = r.get("hyperparameters")
        if rows:
            summary[run["id"]] = rows
    json.dump(summary, open(os.path.join(BASE, "results", "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("gom %d run có kết quả" % len(summary))


def cmd_effective():
    """Chạy effective_args.py (CPU) cho lệnh train fold đầu của từng run trên 161; kiểm cờ trong runs.json khớp giá trị parse."""
    out = {}
    for run in M["runs"]:
        train, _, _ = commands("161", run, run.get("folds", M["block"]["target_folds"])[0])
        py, _, src, trainer = train[:4]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="42")
        res = subprocess.run([py, os.path.join(HERE, "effective_args.py"), src, trainer] + train[4:], cwd=M["hosts"]["161"]["root"],
                             env=env, capture_output=True, text=True)
        line = [l for l in res.stdout.splitlines() if l.startswith("EFFECTIVE_ARGS ")]
        if not line:
            sys.exit("không parse được %s:\n%s" % (run["id"], res.stderr[-2000:]))
        eff = json.loads(line[0][len("EFFECTIVE_ARGS "):])
        for k, v in merged_flags(run).items():                     # cờ ta ghi phải đúng giá trị trainer nhận
            got = eff[k.lstrip("-")]
            assert str(got) == str(v) or (isinstance(got, float) and float(v) == got), (run["id"], k, v, got)
        out[run["id"]] = eff
        print("%-28s %d tham số, khớp runs.json" % (run["id"], len(eff)))
    json.dump(out, open(os.path.join(BASE, "meta", "runs_effective.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


PATHLIKE = {"checkpoint_path", "init_ckpt", "model_name", "run_name", "fold", "phase", "result_path", "data_root"}


def hparam_check(run, hp):
    """So hyperparameters ghi trong kết quả với tham số hiệu lực của runs.json (bỏ các khoá đường dẫn/tên chạy)."""
    if not hp:
        return "không có hyperparameters trong kết quả"
    ref = json.load(open(os.path.join(BASE, "meta", "runs_effective.json"), encoding="utf-8")).get(run["id"])
    if ref is None:                  # run mới chưa sinh lại runs_effective.json — đừng làm hỏng cả lượt dbrows
        return "chưa có tham số hiệu lực (chạy fpe.py effective)"
    keys = [k for k in ref if k not in PATHLIKE and k in hp]
    bad = ["%s=%s≠%s" % (k, hp[k], ref[k]) for k in keys if str(hp[k]) != str(ref[k])
           and not (isinstance(hp[k], (int, float)) and isinstance(ref[k], (int, float)) and float(hp[k]) == float(ref[k]))]
    miss = [k for k in ref if k not in PATHLIKE and k not in hp]
    return ("lệch: " + ", ".join(bad)) if bad else "khớp %d tham số" % len(keys) + (" (thiếu %d khoá)" % len(miss) if miss else "")


# tên máy trong dòng đầu log -> khoá host; old_hostnames: máy cũ cùng nhãn đã destroy (06/10: paper_mw3 thuê lại, ô cũ vẫn phải ra "paper_mw3")
HOSTNAMES = {hn: k for k, h in M["hosts"].items() for hn in [h.get("hostname")] + list(h.get("old_hostnames", [])) if hn}


def log_head(path):
    """Dòng đầu log fold do run.sh ghi: '# 2026-09-24 23:19:49 flink-tm | ...' -> (giờ bắt đầu, máy)."""
    try:
        m = re.match(r"# (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\S+)", open(path, encoding="utf-8").readline())
    except OSError:
        return None, None
    return (m.group(1), HOSTNAMES.get(m.group(2), m.group(2))) if m else (None, None)


def _secs(a, b):
    if not a or not b:
        return None
    t = lambda s: time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))
    return int(t(b) - t(a))


def _roc_auc(y, s):
    """ROC-AUC theo hạng (Mann–Whitney, hạng trung bình cho giá trị hoà); None nếu thiếu một lớp."""
    pos = sum(y); neg = len(y) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i]); ranks = [0.0] * len(s); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return (sum(r for r, t in zip(ranks, y) if t) - pos * (pos + 1) / 2) / (pos * neg)


def _macro_f1(y, pred):
    f = []
    for c in (0, 1):
        tp = sum(1 for a, b in zip(y, pred) if a == c and b == c); fp = sum(1 for a, b in zip(y, pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, pred) if a == c and b != c)
        f.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f) / 2


def per_cwe(run, fold, npz_path):
    """ROC và macro-F1@0,5 theo CWE trên tập test SVEN của fold, từ xác suất từng mẫu (.probs.npz, cùng thứ tự test.jsonl — kiểm theo
    nhãn). Cần numpy (chạy dbrows bằng python của vdenv). Trả None nếu không có file hoặc thứ tự lệch."""
    try:
        import numpy as np
    except ImportError:
        return None
    if not os.path.exists(npz_path):
        return None
    flags = merged_flags(run)
    root = M["hosts"]["161"]["root"]
    test = os.path.join(root, flags.get("--data_root", ""), "fold%d" % fold, "test.jsonl")
    if not os.path.exists(test):
        return None
    rows = [json.loads(l) for l in open(test, encoding="utf-8")]
    z = np.load(npz_path)
    prob, lab = [float(x) for x in z["probabilities"]], [int(x) for x in z["labels"]]
    if len(rows) != len(lab) or any(r["label"] != t for r, t in zip(rows, lab)):
        print("!! %s f%d: thứ tự .probs.npz lệch test.jsonl — bỏ per_cwe" % (run["id"], fold))
        return None
    out = {"_all": {"roc": _roc_auc(lab, prob), "f1": _macro_f1(lab, [int(p >= 0.5) for p in prob]), "n": len(lab)}}
    for c in sorted({r["cwe"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["cwe"] == c]
        y = [lab[i] for i in idx]; s = [prob[i] for i in idx]
        out[c] = {"roc": _roc_auc(y, s), "f1": _macro_f1(y, [int(p >= 0.5) for p in s]), "n": len(idx), "pos": sum(y)}
    return out


def cmd_dbrows(host="161"):
    out = os.path.join(BASE, "meta", "dbrows")
    os.makedirs(out, exist_ok=True)
    n = 0
    for run in M["runs"]:
        for f in sorted(glob.glob(os.path.join(BASE, "results", run["id"], "fold*.json"))):
            if f.endswith(".probs.npz"):
                continue
            fold = int(os.path.basename(f)[4:-5])
            r = json.load(open(f, encoding="utf-8"))
            started, h = log_head(os.path.join(BASE, "logs", run["id"], "fold%d.log" % fold))
            finished = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(f)))   # rsync -a giữ mtime
            row = {"run": run["id"], "fold": fold, "status": "done", "host": h or host, "started": started, "finished": finished,
                   "seconds": _secs(started, finished),
                   "roc": r.get("test_roc_auc"), "f1_05": r.get("test_macro_f1_at_0.5"), "f1_val": r.get("test_macro_f1_at_valcal"),
                   "pr": r.get("test_pr_auc"), "best_epoch": r.get("best_epoch"),
                   "hparam_check": hparam_check(run, r.get("hyperparameters")), "updated": time.strftime("%Y-%m-%d %H:%M")}
            if run["stage"] == "target":                             # chỉ số theo CWE của tập đích SVEN
                pc = per_cwe(run, fold, f[:-5] + ".probs.npz")
                if pc:
                    row["cwe_check"] = {"roc": pc["_all"]["roc"], "f1": pc["_all"]["f1"]}   # tính lại toàn tập để đối chiếu với trainer
                    row["cwe"] = {k: v for k, v in pc.items() if k != "_all"}
            lp = os.path.join(BASE, "logs", run["id"], "fold%d.log" % fold)
            if os.path.exists(lp):                                   # lịch sử epoch cho tab Kết quả
                cmd_live(run["id"], fold, lp, row["host"])
                live = json.load(open(os.path.join(out, "%s__f%d.json" % (run["id"], fold)), encoding="utf-8"))
                row.update({k: live[k] for k in ("history", "epochs", "best_val_roc")})
                row["phase_note"] = live["phase_note"]
                row.update({k: live[k] for k in ("p1_tries", "stuck_check", "selection") if k in live})
            if run.get("p1_retry"):                                  # seed Pha 1 thực sự dùng (P1_CKPT do p1_retry.py ghi)
                ptr = os.path.join(M["hosts"][host]["ckpt"], run["id"], "P1_CKPT")
                if os.path.exists(ptr):
                    row["p1_ckpt"] = open(ptr, encoding="utf-8").read().strip()
            json.dump(row, open(os.path.join(out, "%s__f%d.json" % (run["id"], fold)), "w", encoding="utf-8"), ensure_ascii=False)
            n += 1
    print("ghi %d bản ghi vào meta/dbrows/" % n)


EPOCH_RE = re.compile(r"^(\S+ \S+) - INFO - Epoch (\d+)/(\d+) \| train loss ([\d.naif-]+) \| val loss ([\d.naif-]+) \| val roc_auc ([\d.naif-]+) "
                      r"\| val macro_f1 ([\d.naif-]+) \| best ep (\d+) \| patience (\d+)/(\d+) \| (\d+)s")


BASE_EPOCH_RE = re.compile(r"^(\S+ \S+) - INFO - Epoch (\d+)/(\d+) \| Train loss: ([\d.]+) \| Val loss: ([\d.]+) \| LR: \S+ \| Best epoch: (\d+) "
                           r"\| Patience: \d+ \| .*\| Epoch: ([\d.]+)s")


def stall_check(run_id, hist, pair):
    """Cảnh báo "train loss không giảm / không học", theo loại run (người dùng 27/09: "lưu ý cẩn thận các trường hợp loss train không giảm").
    Pha 1: sau warmup (ep ≥ ceil(warmup×epochs)+1) mà train loss > bình nguyên − 0,05 (bình nguyên = ln2 + λ·softplus(margin)·tỉ lệ hàng
    trong cặp, FACTS §84). mixsrc: train loss tập trộn tự nhiên đứng ~0,70 ⇒ chỉ xét val SVEN: ep ≥ 6 mà best val ROC < 0,6.
    Pha 2 / baseline: ep ≥ 13 mà train loss CHƯA TỪNG xuống dưới 0,6; train_mwg thêm: ep ≥ 12 mà best val ROC < 0,7.
    Hiệu chỉnh 27/09 trên 35 log Pha 2/baseline đã xong: RecAdam+ASAM đi ngang 4–8 epoch sau warmup là BÌNH THƯỜNG — luật cũ
    ("không giảm 3 epoch" từ ep5; "best val < 0,7" từ ep6) báo nhầm 8/35 và 13/35 log, mọi log đó test ROC 0,89–0,96. Chậm nhất
    trong số đã xong: train loss lần đầu < 0,6 ở ep11, best val ≥ 0,7 ở ep9 ⇒ ngưỡng đặt sau hai mốc đó. Trả chuỗi cảnh báo hoặc None."""
    import math
    if not hist:
        return None
    run = RUNS[run_id]; e = hist[-1]["ep"]; tl = [h["train_loss"] for h in hist]
    rocs = [h["val_roc"] for h in hist if h.get("val_roc") is not None]; best = max(rocs) if rocs else None
    if run["stage"] == "source":
        # Khối mwonly5 (04/10): Pha 1 16 epoch, warmup 0,25 (lr đỉnh cuối ep4), KHÔNG pair loss. Luật kẹt của refactor (ep5 giảm < 2 %) hiệu
        # chỉnh trên lượt CÓ pair loss (loss ~1,9) và BÁO NHẦM khi không pair loss: 21:27 p1_4cwe_jsonly ep5 giảm 0,8 % rồi ep6-7 xuống 0,695 ->
        # 0,674, val ROC 0,41 -> 0,55. Thay bằng mốc bình nguyên CE (ln2 = 0,693): ep >= 8 mà train loss CHƯA TỪNG < 0,68 = nghi kẹt.
        # Val ROC Pha 1 trên nguồn cặp chia ngẫu nhiên vốn quanh 0,4-0,6 (nửa kia của cặp nằm ở train, nhãn ngược) nên KHÔNG dùng để báo.
        if e >= 8 and min(tl) >= 0.68:
            return "nghi KẸT Pha 1: ep%d train loss chưa từng < 0,68 (min %.4f; bình nguyên CE ln2 = 0,693)" % (e, min(tl))
        return None
    if run_id.startswith("mwg_mixsrc"):
        if e >= 6 and best is not None and best < 0.6:
            return "mixsrc KHÔNG HỌC: ep%d best val ROC %.3f < 0,6" % (e, best)
        return None
    if e >= 13 and min(tl) > 0.6:
        return "train loss CHƯA XUỐNG dưới 0,6 sau %d epoch (min %.3f; chậm nhất bình thường: ep11)" % (e, min(tl))
    # 05/10 18:3x: mốc val ROC 0,7 hiệu chỉnh trên đích SVEN; đích khác (tên chứa "_tojs", vd JS full: trong miền chỉ ~0,59) thì bỏ luật này,
    # vẫn giữ luật train loss ở trên (bình nguyên ln2 không phụ thuộc đích). Báo nhầm 17:4x / 18:3x ở rasam_sven_python_tojsfull f1 / f2.
    if e >= 12 and best is not None and best < 0.7 and "_tojs" not in run_id:
        return "Pha 2 học chậm: ep%d best val ROC %.3f < 0,7 (chậm nhất bình thường đạt 0,7 ở ep9)" % (e, best)
    return None


def cmd_stall(run_id, logfile):
    """In cảnh báo train loss của một log (dùng cho monitor); không có gì thì in rỗng."""
    import io, contextlib, tempfile
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_live(run_id, 0, logfile, "?", write=False)
    print(_LAST_ALERT or "")


_LAST_ALERT = None


def cmd_live(run_id, fold, logfile, host="158", write=True):
    """Bản ghi TIẾN ĐỘ của một fold đang chạy, dựng từ log trainer (dòng 'Epoch k/N | ...') -> meta/dbrows/<run>__f<F>.json
    với status 'running'. Khi fold xong thì cmd_dbrows ghi đè bằng kết quả test (vẫn giữ lịch sử epoch)."""
    fold = int(fold)
    started, h = log_head(logfile)
    hist, best_roc, stop, prep, pair = [], None, None, None, None
    lines = open(logfile, encoding="utf-8", errors="replace").readlines()
    tries = [(i, l.strip()) for i, l in enumerate(lines) if l.startswith("### P1 ")]   # p1_retry.py: chỉ đọc lượt thử CUỐI
    last_try = max([i for i, l in tries if "thử seed" in l], default=-1)
    stuck_note, sel_metric = None, None
    for line in lines[last_try + 1:]:
        m = EPOCH_RE.search(line)
        if m:
            f = lambda i: (lambda v: None if v != v or v in (float("inf"), float("-inf")) else v)(float(m.group(i)))   # NaN -> null (JSON hợp lệ)
            hist.append({"ep": int(m.group(2)), "train_loss": f(4), "val_loss": f(5), "val_roc": f(6), "val_f1": f(7),
                         "best_ep": int(m.group(8)), "sec": int(m.group(11)), "at": m.group(1)[:16]})
            epochs = int(m.group(3))
        m = BASE_EPOCH_RE.search(line)                      # train_baseline.py: không in val ROC từng epoch
        if m:
            hist.append({"ep": int(m.group(2)), "train_loss": float(m.group(4)), "val_loss": float(m.group(5)), "val_roc": None,
                         "val_f1": None, "best_ep": int(m.group(6)), "sec": int(float(m.group(7))), "at": m.group(1)[:16]})
            epochs = int(m.group(3))
            best_roc = (int(m.group(6)), None)
        m = re.search(r"Best checkpoint saved \| Epoch: (\d+) \| Val (\w+): (-?[\d.]+)", line)
        if m:
            sel_metric = m.group(2)
            best_roc = (int(m.group(1)), float(m.group(3)) if m.group(2) == "roc_auc" else None)
        m = re.search(r"Kiểm kẹt \| (.*)$", line)                   # train.py nhánh refactor, cuối --stuck_epoch
        if m:
            stuck_note = m.group(1).strip()
        m = re.search(r"(?:PAIR LOSS|Pair loss) lambda=([\d.]+) margin=([\d.]+) \| (\d+) (?:cap du hai nhan|cặp đủ hai nhãn), (\d+) (?:mau le|mẫu lẻ)", line)
        if m:
            np_, ns_ = int(m.group(3)), int(m.group(4))
            pair = (float(m.group(1)), float(m.group(2)), 2 * np_ / (2 * np_ + ns_))
        m = re.search(r"(?:tien xu ly|tiền xử lý) (\d+)s", line)
        if m:
            prep = int(m.group(1))
        if "Early stopping" in line or "Train xong" in line:
            stop = line.split(" - INFO - ")[-1].strip()
    if best_roc and best_roc[1] is None:          # chọn theo metric khác val ROC (vd. train_loss): hiện val ROC của epoch được chọn
        best_roc = (best_roc[0], next((h["val_roc"] for h in hist if h["ep"] == best_roc[0]), None))
    eta_min = eta_max = None
    if hist:                                  # ETA: nhịp TB 3 epoch gần nhất; sớm nhất = dừng sớm theo patience, muộn nhất = đủ số epoch
        import datetime as _dt
        eff = json.load(open(os.path.join(BASE, "meta", "runs_effective.json"), encoding="utf-8")).get(run_id, {})
        pat, mine = int(eff.get("patience", 8) or 8), int(eff.get("min_epochs", 3) or 3)
        per = sum(e["sec"] for e in hist[-3:]) / len(hist[-3:])
        e, n_ep, b = hist[-1]["ep"], epochs, hist[-1]["best_ep"]
        test = 360 if RUNS[run_id]["stage"] == "source" else 90          # pha test: Pha 1 chấm 1 806 hàng val, Pha 2 152 hàng
        last = _dt.datetime.strptime(hist[-1]["at"], "%Y-%m-%d %H:%M")
        # patience chỉ đếm các epoch >= min_epochs (train.py: `elif ep >= args.min_epochs: wait += 1`) ⇒ dừng sớm nhất ở max(b, mine-1) + pat
        # (04/10: công thức cũ max(b + pat, mine) cho ETA sớm sai khi min_epochs lớn - Pha 1 16 epoch min 10 patience 8 không thể dừng sớm)
        rem_max = max(0, n_ep - e); rem_min = max(0, min(n_ep, max(b, mine - 1) + pat) - e)
        f = lambda r: (last + _dt.timedelta(seconds=r * per + test)).strftime("%Y-%m-%d %H:%M")
        eta_min, eta_max = f(rem_min), f(rem_max)
    row = {"run": run_id, "fold": fold, "status": "running", "host": h or host, "started": started, "eta_min": eta_min, "eta_max": eta_max,
           "epoch": hist[-1]["ep"] if hist else 0, "epochs": epochs if hist else None, "history": hist,
           "best_epoch": best_roc[0] if best_roc else None, "best_val_roc": best_roc[1] if best_roc else None,
           "phase_note": stop or ("đang train" if hist else "tiền xử lý / epoch 1"), "updated": time.strftime("%Y-%m-%d %H:%M"),
           "note": "batch %s" % merged_flags(RUNS[run_id]).get("--batch_size", "?") + (" · tiền xử lý %d s" % prep if prep else ""),
           "alert": stall_check(run_id, hist, pair)}
    if sel_metric and sel_metric != "roc_auc":
        row["selection"] = sel_metric
    if tries:                                     # p1_retry.py: các dòng '### P1 …' (lượt thử, kẹt, seed được dùng)
        row["p1_tries"] = [l[len("### P1 "):] for _, l in tries]
    if stuck_note:
        row["stuck_check"] = stuck_note
    global _LAST_ALERT
    _LAST_ALERT = row["alert"]
    if not write:
        return
    out = os.path.join(BASE, "meta", "dbrows"); os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "%s__f%d.json" % (run_id, fold))
    json.dump(row, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print(p, "| epoch %s/%s | best %s" % (row["epoch"], row["epochs"], row["best_val_roc"]))


def cmd_preflight(host, *entries):
    """Kiểm TRÊN MÁY ĐÍCH trước khi phóng/xếp hàng: python, mã (<root>/<src_dir>/<trainer>), dữ liệu (--data_root), checkpoint Pha 1.
        python3 scripts/fpe.py preflight <host> <run[:fold,fold]> ...     — in từng mục THIẾU, thoát 1 nếu có.
    29/09: refactor full bỏ Java f3 được xếp lên 158 khi 158 CHƯA có src_refactor ⇒ chết ngay (FileNotFoundError); lúc đó chỉ kiểm checkpoint."""
    h = M["hosts"][host]; miss = []; n = 0
    for e in entries:
        run_id, _, fs = e.partition(":")
        run = RUNS[run_id]
        folds = [int(x) for x in fs.split(",")] if fs else run.get("folds", M["block"]["target_folds"])
        flags = merged_flags(run)
        for f in folds:
            train, _test, p = commands(host, run, f); n += 1
            checks = [("python", train[0]), ("mã", os.path.join(h["root"], train[2], train[3]))]
            if flags.get("--data_root"):
                checks.append(("dữ liệu", os.path.join(h["root"], flags["--data_root"])))
            if p["init_ckpt"]:
                checks.append(("checkpoint Pha 1", p["init_ckpt"]))
            miss += ["%s f%d: THIẾU %s %s" % (run_id, f, what, q) for what, q in checks if not os.path.exists(q)]
    print("\n".join(miss) if miss else "preflight %s: đủ mã / dữ liệu / checkpoint cho %d mục" % (host, n))
    sys.exit(1 if miss else 0)


def cmd_order():
    for run in sorted(M["runs"], key=lambda r: (r["stage"] != "source", M["runs"].index(r))):
        print(run["id"])


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"live": cmd_live, "stall": cmd_stall, "plan": cmd_plan, "resolve": cmd_resolve, "collect": cmd_collect, "order": cmd_order, "effective": cmd_effective, "dbrows": cmd_dbrows,
     "preflight": cmd_preflight}[cmd](*args)
