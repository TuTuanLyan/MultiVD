#!/usr/bin/env python3
"""Đóng góp của RecAdam trong Pha 2 (khối mwonly5): số, bảng và hình cho bài.

Chỉ ĐỌC kết quả đã có (results/, meta/dbrows/, logs/, dữ liệu fold đích); không huấn luyện, không GPU,
không sửa gì ngoài thư mục meta/insights/recadam/. Chạy lại bất kỳ lúc nào; nếu 3 ô chạy lại với
--min_epochs 30 (`<run>_me30`) đã có kết quả thì tự đưa vào phép nhạy "thay ô sập bằng bản me30".

    /home/ntat/miniconda3/envs/vdenv/bin/python make_recadam_figures.py

Thiết kế 2x2 (cùng checkpoint Pha 1, cùng fold đích, chỉ khác cờ --recadam / --sam_rho):
    noras     = Pha 1 rồi AdamW thường          (RecAdam tắt, ASAM tắt)
    raonly    = chỉ RecAdam                     (RecAdam bật, ASAM tắt)
    asamonly  = chỉ ASAM                        (RecAdam tắt, ASAM bật)
    rasam     = RecAdam + ASAM (cột chính)      (RecAdam bật, ASAM bật)
trên 6 nguồn {4cwe, common, full} x {jsonly, jscpp}, 5 fold, seed 42.

Mọi hiệu đều GHÉP CẶP theo (nguồn, fold); không lấy hiệu hai trung bình. Khoảng tin cậy: bootstrap
theo CỤM FOLD (6 nguồn dùng chung 5 fold đích nên cặp không độc lập; lấy mẫu lại fold, giữ mọi nguồn
của fold đó). Với 5 cụm, khoảng bootstrap percentile hẹp hơn thực tế - ghi rõ trong báo cáo; kèm
khoảng t trên trung bình theo fold (df = 4) làm bản thận trọng.
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = "/drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment"
RES = os.path.join(ROOT, "results")
DBROWS = os.path.join(ROOT, "meta/dbrows")
LOGS = os.path.join(ROOT, "logs")
DATA = "/drive1/cuongtm/ntat/MultiVD/data/final_experiment_data/sven_python_folds_nocomment"
OUT = os.path.join(ROOT, "meta/insights/recadam")

SRC6 = ["4cwe_jsonly", "common_jsonly", "full_jsonly", "4cwe_jscpp", "common_jscpp", "full_jscpp"]
SRC_CCPP = ["4cwe_ccpp", "common_ccpp", "full_ccpp"]
FAMILIES = {"jsonly": [s for s in SRC6 if s.endswith("jsonly")], "jscpp": [s for s in SRC6 if s.endswith("jscpp")]}
FOLDS = [1, 2, 3, 4, 5]
VARIANTS = ["noras", "raonly", "asamonly", "rasam"]
VFLAGS = {"noras": (0, 0), "raonly": (1, 0), "asamonly": (0, 1), "rasam": (1, 1)}   # (RecAdam, ASAM)
METRICS = [("f1_05", "test_macro_f1_at_0.5", "F1@0.5"), ("f1_val", "test_macro_f1_at_valcal", "F1@val"),
           ("roc", "test_roc_auc", "ROC-AUC"), ("pr", "test_pr_auc", "PR-AUC")]
MKEYS = [m[0] for m in METRICS]
MLABEL = {m[0]: m[2] for m in METRICS}
COLLAPSE_ROC = 0.75          # 'sập' = test ROC < 0,75 (cùng định nghĩa asam_collapse.md)
PLATEAU_THR = 0.65           # bình nguyên = epoch đầu có train loss < 0,65, trừ 1
TIE_EPS = 1e-6               # |hiệu| < eps tính là hoà (chỉ số thứ hạng có thể trùng tới 1e-16)
NOISE_FLOOR = 0.010          # sàn chạy lại khác máy (CLAUDE.md §2)
B_BOOT = 10000
RNG_SEED = 20261005

SRC_EN = {"4cwe_jsonly": "4CWE, JS", "common_jsonly": "Common, JS", "full_jsonly": "Full, JS",
          "4cwe_jscpp": "4CWE, JS+C/C++", "common_jscpp": "Common, JS+C/C++", "full_jscpp": "Full, JS+C/C++",
          "4cwe_ccpp": "4CWE, C/C++", "common_ccpp": "Common, C/C++", "full_ccpp": "Full, C/C++"}
SRC_VI = {"4cwe_jsonly": "4CWE chỉ JS", "common_jsonly": "common chỉ JS", "full_jsonly": "full chỉ JS",
          "4cwe_jscpp": "4CWE JS + C/C++", "common_jscpp": "common JS + C/C++", "full_jscpp": "full JS + C/C++"}
VAR_EN = {"noras": "AdamW", "raonly": "RecAdam", "asamonly": "ASAM", "rasam": "RecAdam+ASAM"}
VAR_VI = {"noras": "noRAS (AdamW)", "raonly": "chỉ RecAdam", "asamonly": "chỉ ASAM", "rasam": "RecAdam + ASAM"}

# Các phép so: hiệu = tổ hợp tuyến tính của các biến thể trong cùng (nguồn, fold)
CONTRASTS = {
    "ra_main": ({"rasam": 0.5, "asamonly": -0.5, "raonly": 0.5, "noras": -0.5}, "RecAdam main effect",
                "tác dụng chính của RecAdam (TB hai hiệu đơn)"),
    "ra_with_asam": ({"rasam": 1, "asamonly": -1}, "RecAdam, ASAM on", "RecAdam khi có ASAM (cột chính - chỉ ASAM)"),
    "ra_without_asam": ({"raonly": 1, "noras": -1}, "RecAdam, ASAM off", "RecAdam khi không ASAM (chỉ RecAdam - noRAS)"),
    "interaction": ({"rasam": 1, "asamonly": -1, "raonly": -1, "noras": 1}, "RecAdam x ASAM interaction",
                    "tương tác RecAdam x ASAM"),
    "asam_with_ra": ({"rasam": 1, "raonly": -1}, "ASAM, RecAdam on", "ASAM khi có RecAdam (cột chính - chỉ RecAdam)"),
    "asam_without_ra": ({"asamonly": 1, "noras": -1}, "ASAM, RecAdam off", "ASAM khi không RecAdam (chỉ ASAM - noRAS)"),
    "full_vs_noras": ({"rasam": 1, "noras": -1}, "RecAdam+ASAM vs AdamW", "cột chính - noRAS"),
}
GROUPS = {"all": (SRC6, "All 6 sources", "cả 6 nguồn"), "jsonly": (FAMILIES["jsonly"], "JS only", "nguồn chỉ JS"),
          "jscpp": (FAMILIES["jscpp"], "JS + C/C++", "nguồn JS + C/C++")}
for _s in SRC6:
    GROUPS[_s] = ([_s], SRC_EN[_s], SRC_VI[_s])

# tham số được phép khác nhau giữa hai ô ghép cặp (đường dẫn, tên run, cờ của chính phép so,
# và các tham số RecAdam / ASAM không có tác dụng khi cờ tương ứng tắt)
HP_IGNORE = {"checkpoint_path", "result_path", "run_name", "model_name", "experiment_name", "recadam", "sam_rho",
             "sam_variant", "anneal_t0_ratio", "pretrain_cof", "anneal_k", "anneal_fun", "data_root"}


# ============================================================================ nạp dữ liệu
EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+) \| train loss ([\d.]+) \| val loss ([\d.na]+) \| val roc_auc ([\d.na]+) \| "
                      r"val macro_f1 ([\d.na]+) \| best ep (\d+)")


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def parse_log_history(path):
    """Dự phòng khi meta/dbrows chưa có dòng của run (ví dụ ô me30 vừa xong): đọc dòng 'Epoch k/N' trong log."""
    if not os.path.exists(path):
        return []
    hist = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m:
                hist.append({"ep": int(m.group(1)), "train_loss": _num(m.group(3)), "val_loss": _num(m.group(4)),
                             "val_roc": _num(m.group(5)), "val_f1": _num(m.group(6)), "best_ep": int(m.group(7))})
    return hist


def load_cell(run, fold):
    rp = os.path.join(RES, run, f"fold{fold}.json")
    if not os.path.exists(rp):
        return None
    res = json.load(open(rp))
    c = {"run": run, "fold": fold, "hp": res.get("hyperparameters", {}),
         "best_epoch": res.get("best_epoch"), "best_val": res.get("best_val_score"),
         "thr": res.get("val_calibrated_threshold"), "val_f1_valcal": res.get("val_macro_f1_at_valcal")}
    for k, jk, _ in METRICS:
        c[k] = res.get(jk)
    db = os.path.join(DBROWS, f"{run}__f{fold}.json")
    hist = None
    if os.path.exists(db):
        d = json.load(open(db))
        if isinstance(d.get("history"), list) and d["history"]:
            hist = d["history"]
    if hist is None:
        hist = parse_log_history(os.path.join(LOGS, run, f"fold{fold}.log"))
    c["hist"] = hist
    npz = os.path.join(RES, run, f"fold{fold}.probs.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        c["p"] = z["probabilities"].astype(float)
        c["y"] = z["labels"].astype(int)
        c["pv"] = z["val_probabilities"].astype(float)
        c["yv"] = z["val_labels"].astype(int)
    return c


def load_all():
    cells = {}
    for s in SRC6:
        for v in VARIANTS:
            for f in FOLDS:
                c = load_cell(f"{v}_{s}", f)
                if c is None:
                    sys.exit(f"THIẾU ô {v}_{s} f{f} - dừng (khối mwonly5 phải đủ 120 ô 2x2)")
                c["src"], c["var"] = s, v
                cells[(v, s, f)] = c
    extra = {}
    for run in ["baseline", "nop1_adamw", "nop1_asamonly"] + [f"rasam_{s}" for s in SRC_CCPP]:
        for f in FOLDS:
            c = load_cell(run, f)
            if c is not None:
                extra[(run, f)] = c
    me30 = {}
    for d in sorted(glob.glob(os.path.join(RES, "*_me30"))):
        run = os.path.basename(d)
        for f in FOLDS:
            c = load_cell(run, f)
            if c is not None and c.get("roc") is not None:
                me30[(run[:-5], f)] = c
    return cells, extra, me30


def load_test_cwe():
    out = {}
    for f in FOLDS:
        rows = [json.loads(l) for l in open(os.path.join(DATA, f"fold{f}", "test.jsonl"))]
        out[f] = (np.array([r["cwe"] for r in rows]), np.array([int(r["label"]) for r in rows]))
    return out


# ============================================================================ kiểm cấu hình hiệu lực
def _norm_path(v):
    if isinstance(v, str) and "mwonly5/" in v:
        return v.split("mwonly5/", 1)[1]
    return v


def hp_diff(a, b):
    keys = set(a) | set(b)
    return sorted(k for k in keys if k not in HP_IGNORE and _norm_path(a.get(k)) != _norm_path(b.get(k)))


def check_pairs(cells):
    """Mọi cặp ghép trong một (nguồn, fold) phải khác nhau CHỈ ở cờ RecAdam / ASAM (đọc từ hyperparameters)."""
    bad, n = [], 0
    for s in SRC6:
        for f in FOLDS:
            ref = cells[("noras", s, f)]["hp"]
            for v in VARIANTS:
                hp = cells[(v, s, f)]["hp"]
                n += 1
                d = hp_diff(ref, hp)
                ra, asam = VFLAGS[v]
                ok_flags = (int(hp.get("recadam", 0)) == ra) and ((float(hp.get("sam_rho", 0)) > 0) == bool(asam))
                if d or not ok_flags:
                    bad.append({"cell": f"{v}_{s} f{f}", "diff": d, "flags_ok": ok_flags})
            # cùng nhãn test ⇒ cùng thứ tự mẫu (cần cho phép so dự đoán từng mẫu)
            y0 = cells[("noras", s, f)]["y"]
            for v in VARIANTS:
                if not np.array_equal(cells[(v, s, f)]["y"], y0):
                    bad.append({"cell": f"{v}_{s} f{f}", "diff": ["test label order"], "flags_ok": True})
    return {"cells_checked": n, "mismatch": bad}


# ============================================================================ đại lượng mỗi ô
def ece_binary(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def cell_extras(c, cwe_by_fold, ref_probs):
    """Các đại lượng dẫn xuất (động học, hiệu chỉnh, theo CWE, giống mô hình không Pha 1)."""
    h = c["hist"]
    tl = [e["train_loss"] for e in h]
    vr = [e.get("val_roc") for e in h]
    last = h[-1]["ep"] if h else None
    c["stop_epoch"] = last
    ex = next((e["ep"] for e in h if e["train_loss"] is not None and e["train_loss"] < PLATEAU_THR), None)
    c["plateau"] = (ex if ex is not None else (last or 0) + 1) - 1
    c["plateau_censored"] = ex is None
    c["stuck10"] = c["plateau"] > 10
    c["collapse"] = c["roc"] is not None and c["roc"] < COLLAPSE_ROC
    for k in (1, 2, 3):
        c[f"vr{k}"] = vr[k - 1] if len(vr) >= k else None
        c[f"tl{k}"] = tl[k - 1] if len(tl) >= k else None
    seg = [x for x in vr[:10] if x is not None]
    c["vr_mean_1_10"] = float(np.mean(seg)) if seg else None
    for thr in (0.80, 0.85, 0.90):
        hit = next((e["ep"] for e in h if e.get("val_roc") is not None and e["val_roc"] >= thr), None)
        c[f"ep_to_val{int(thr * 100)}"] = hit if hit is not None else (last or 0) + 1
        c[f"ep_to_val{int(thr * 100)}_cens"] = hit is None
    if "p" in c:
        p, y = np.clip(c["p"], 1e-7, 1 - 1e-7), c["y"]
        c["ece"] = ece_binary(p, y)
        c["brier"] = float(np.mean((p - y) ** 2))
        c["nll"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        c["conf"] = float(np.mean(np.abs(p - 0.5) * 2))
        c["thr_dev"] = abs(c["thr"] - 0.5) if c.get("thr") is not None else None
        c["val_test_gap"] = (c["best_val"] - c["roc"]) if c.get("best_val") is not None else None
        cw, yy = cwe_by_fold[c["fold"]]
        assert np.array_equal(yy, y), f"thứ tự test lệch {c['run']} f{c['fold']}"
        for name, sel in (("cwe_hard", ("CWE-022", "CWE-079")), ("cwe_easy", ("CWE-078", "CWE-089")),
                          ("cwe022", ("CWE-022",)), ("cwe079", ("CWE-079",)), ("cwe078", ("CWE-078",)),
                          ("cwe089", ("CWE-089",))):
            m = np.isin(cw, sel)
            c[name] = float(roc_auc_score(y[m], p[m])) if len(set(y[m])) == 2 else None
        if ref_probs is not None:
            c["rho_nop1"] = float(stats.spearmanr(c["p"], ref_probs).statistic)
    return c


# ============================================================================ thống kê ghép cặp
_rng = np.random.default_rng(RNG_SEED)
_BOOT_IDX = _rng.integers(0, len(FOLDS), size=(B_BOOT, len(FOLDS)))


def contrast_values(cells, contrast, key, sources, mode="all", me30=None):
    """[(nguồn, fold, hiệu)] cho một phép so. mode: all | excl (bỏ cặp có ô sập) | me30 (thay ô sập bằng bản me30)."""
    w = CONTRASTS[contrast][0]
    out = []
    for s in sources:
        for f in FOLDS:
            cs = {v: cells[(v, s, f)] for v in w}
            if mode == "excl" and any(c["collapse"] for c in cs.values()):
                continue
            if mode == "me30":
                for v in w:
                    if cs[v]["collapse"] and me30 and (f"{v}_{s}", f) in me30:
                        cs[v] = me30[(f"{v}_{s}", f)]
            vals = [cs[v].get(key) for v in w]
            if any(x is None for x in vals):
                continue
            out.append((s, f, float(sum(w[v] * cs[v][key] for v in w))))
    return out


def summarize(diffs):
    if not diffs:
        return {"n": 0}
    x = np.array([d for _, _, d in diffs])
    n = len(x)
    pos, neg = int((x > TIE_EPS).sum()), int((x < -TIE_EPS).sum())
    out = {"n": n, "pos": pos, "neg": neg, "tie": n - pos - neg, "mean": float(x.mean()),
           "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}
    # theo fold (trung bình qua các nguồn trong nhóm): đơn vị độc lập thật sự là fold
    S = np.zeros(len(FOLDS))
    C = np.zeros(len(FOLDS))
    for s, f, d in diffs:
        S[f - 1] += d
        C[f - 1] += 1
    have = C > 0
    fm = S[have] / C[have]
    out["fold_means"] = [float(v) for v in fm]
    out["fold_pos"] = int((fm > TIE_EPS).sum())
    out["fold_n"] = int(have.sum())
    try:
        out["fold_wilcoxon_p"] = float(stats.wilcoxon(fm, zero_method="pratt").pvalue) if len(fm) >= 2 and np.any(fm != 0) else None
    except ValueError:
        out["fold_wilcoxon_p"] = None
    if len(fm) >= 2:
        se = fm.std(ddof=1) / math.sqrt(len(fm))
        tq = stats.t.ppf(0.975, len(fm) - 1)
        out["ci_t"] = [float(fm.mean() - tq * se), float(fm.mean() + tq * se)]
    # theo nguồn
    by_src = defaultdict(list)
    for s, f, d in diffs:
        by_src[s].append(d)
    sm = {s: float(np.mean(v)) for s, v in by_src.items()}
    out["src_means"] = sm
    out["src_pos"] = int(sum(v > TIE_EPS for v in sm.values()))
    out["src_n"] = len(sm)
    # bootstrap theo cụm fold (fold thiếu ô trong mode excl vẫn được lấy mẫu với số ô thật của nó)
    idx = _BOOT_IDX
    if have.all():
        num, den = S[idx].sum(1), C[idx].sum(1)
        ok = den > 0
        bs = num[ok] / den[ok]
        out["ci_boot"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    else:
        out["ci_boot"] = None
    if n >= 2 and np.any(np.abs(x) > TIE_EPS):
        try:
            out["cell_wilcoxon_p"] = float(stats.wilcoxon(x, zero_method="pratt").pvalue)
        except ValueError:
            out["cell_wilcoxon_p"] = None
    return out


def cell_mean_ci(vals_by_fold):
    """TB các ô của một biến thể trong một nhóm nguồn, kèm CI bootstrap theo cụm fold. vals_by_fold: {fold: [giá trị]}."""
    S = np.array([np.sum(vals_by_fold.get(f, [])) for f in FOLDS])
    C = np.array([len(vals_by_fold.get(f, [])) for f in FOLDS])
    mean = S.sum() / C.sum()
    num, den = S[_BOOT_IDX].sum(1), C[_BOOT_IDX].sum(1)
    ok = den > 0
    bs = num[ok] / den[ok]
    return float(mean), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ============================================================================ λ(t) của RecAdam
def recadam_schedule(hp, steps_per_epoch):
    epochs = int(hp["epochs"])
    total = steps_per_epoch * epochs
    t0 = max(1, int(float(hp["anneal_t0_ratio"]) * total))
    k = float(hp["anneal_k"])
    cof = float(hp["pretrain_cof"])
    lr0 = float(hp["learning_rate"])
    warm = int(total * float(hp.get("warmup_ratio", 0.0)))
    t = np.arange(1, total + 1)
    lam = 1.0 / (1.0 + np.exp(-k * (t - t0)))
    # get_linear_schedule_with_warmup: bước tối ưu thứ t dùng hệ số của lần gọi sched thứ t-1
    s = t - 1
    lrf = np.where(s < warm, s / max(1, warm), np.maximum(0.0, (total - s) / max(1, total - warm)))
    lr = lr0 * lrf
    pull = lr * (1 - lam) * cof
    ep = t / steps_per_epoch
    per_epoch = []
    for e in range(1, 8):
        m = (t > (e - 1) * steps_per_epoch) & (t <= e * steps_per_epoch)
        per_epoch.append({"epoch": e, "lambda_mean": float(lam[m].mean()), "pull_mean": float(pull[m].mean())})
    return {"t": t, "ep": ep, "lam": lam, "lr": lr, "lr0": lr0, "pull": pull, "t0": t0, "k": k, "cof": cof,
            "total": total, "steps_per_epoch": steps_per_epoch, "warmup_steps": warm,
            "lam_step1": float(lam[0]), "step_lam90": int(t[np.argmax(lam >= 0.9)]), "step_lam99": int(t[np.argmax(lam >= 0.99)]),
            "cum_eff_lr_ep1_3_adamw": float(lr[t <= 3 * steps_per_epoch].sum()),
            "cum_eff_lr_ep1_3_recadam": float((lam * lr)[t <= 3 * steps_per_epoch].sum()),
            "cum_pull": float(pull.sum()), "per_epoch": per_epoch}


# ============================================================================ phân tích
def analyse():
    cells, extra, me30 = load_all()
    cwe_by_fold = load_test_cwe()
    hpcheck = check_pairs(cells)
    nop1_probs = {f: extra[("nop1_adamw", f)]["p"] for f in FOLDS if ("nop1_adamw", f) in extra}
    for c in list(cells.values()) + list(extra.values()) + list(me30.values()):
        cell_extras(c, cwe_by_fold, nop1_probs.get(c["fold"]) if not c["run"].startswith("nop1_adamw") else None)

    num = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "hp_check": hpcheck,
           "me30_available": sorted(f"{r} f{f}" for r, f in me30)}

    # ---- (1) bảng chính: TB ± sd mỗi biến thể × nguồn
    main = {}
    for s in SRC6:
        for v in VARIANTS:
            row = {}
            for k in MKEYS:
                x = np.array([cells[(v, s, f)][k] for f in FOLDS])
                row[k] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "min": float(x.min()), "max": float(x.max()),
                          "folds": [float(t) for t in x]}
            row["collapse"] = int(sum(cells[(v, s, f)]["collapse"] for f in FOLDS))
            main[f"{v}|{s}"] = row
    for run in ["baseline", "nop1_adamw", "nop1_asamonly"] + [f"rasam_{s}" for s in SRC_CCPP]:
        cs = [extra[(run, f)] for f in FOLDS if (run, f) in extra]
        if len(cs) == 5:
            row = {}
            for k in MKEYS:
                x = np.array([c[k] for c in cs])
                row[k] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "min": float(x.min()), "max": float(x.max()),
                          "folds": [float(t) for t in x]}
            row["collapse"] = int(sum(c["collapse"] for c in cs))
            main[f"extra|{run}"] = row
    num["main_table"] = main

    # ---- (2) hiệu ghép cặp: phép so × nhóm × chỉ số × mode
    deltas = {}
    for con in CONTRASTS:
        for g, (srcs, _, _) in GROUPS.items():
            for mode in ("all", "excl", "me30"):
                if mode == "me30" and not me30:
                    continue
                for k in MKEYS + ["cwe_hard", "cwe_easy"]:
                    deltas[f"{con}|{g}|{mode}|{k}"] = summarize(contrast_values(cells, con, k, srcs, mode, me30))
    num["deltas"] = deltas
    num["collapsed_cells"] = sorted(f"{v}_{s} f{f}" for (v, s, f), c in cells.items() if c["collapse"])

    # ---- (3) độ ổn định
    stab = {}
    fold_med = {f: float(np.median([cells[(v, s, f)]["roc"] for v in VARIANTS for s in SRC6])) for f in FOLDS}
    fold_med_f1 = {f: float(np.median([cells[(v, s, f)]["f1_val"] for v in VARIANTS for s in SRC6])) for f in FOLDS}
    num["fold_median_roc"] = fold_med
    for v in VARIANTS:
        cs = [cells[(v, s, f)] for s in SRC6 for f in FOLDS]
        sd_src = {s: float(np.std([cells[(v, s, f)]["roc"] for f in FOLDS], ddof=1)) for s in SRC6}
        sdf_src = {s: float(np.std([cells[(v, s, f)]["f1_val"] for f in FOLDS], ddof=1)) for s in SRC6}
        adj_src = {s: float(np.std([cells[(v, s, f)]["roc"] - fold_med[f] for f in FOLDS], ddof=1)) for s in SRC6}
        adjf_src = {s: float(np.std([cells[(v, s, f)]["f1_val"] - fold_med_f1[f] for f in FOLDS], ddof=1)) for s in SRC6}
        worst = {s: float(min(cells[(v, s, f)]["roc"] for f in FOLDS)) for s in SRC6}
        worstf = {s: float(min(cells[(v, s, f)]["f1_val"] for f in FOLDS)) for s in SRC6}
        rng = {s: float(max(cells[(v, s, f)]["roc"] for f in FOLDS) - min(cells[(v, s, f)]["roc"] for f in FOLDS)) for s in SRC6}
        plats = [c["plateau"] for c in cs]
        # độ nhạy theo nguồn: trong cùng fold, sd ROC giữa 3 nguồn cùng họ
        src_sens = {fam: [float(np.std([cells[(v, s, f)]["roc"] for s in FAMILIES[fam]], ddof=1)) for f in FOLDS]
                    for fam in FAMILIES}
        stab[v] = {"n": len(cs), "collapse": int(sum(c["collapse"] for c in cs)), "stuck10": int(sum(c["stuck10"] for c in cs)),
                   "plateau_median": float(np.median(plats)), "plateau_min": int(min(plats)), "plateau_max": int(max(plats)),
                   "plateau_censored": int(sum(c["plateau_censored"] for c in cs)),
                   "sd_roc_by_src": sd_src, "sd_f1val_by_src": sdf_src, "sd_roc_adj_by_src": adj_src, "sd_f1val_adj_by_src": adjf_src,
                   "worst_roc_by_src": worst, "worst_f1val_by_src": worstf, "range_roc_by_src": rng,
                   "sd_roc_mean": float(np.mean(list(sd_src.values()))), "sd_f1val_mean": float(np.mean(list(sdf_src.values()))),
                   "sd_roc_adj_mean": float(np.mean(list(adj_src.values()))),
                   "sd_f1val_adj_mean": float(np.mean(list(adjf_src.values()))),
                   "worst_roc_mean": float(np.mean(list(worst.values()))), "worst_f1val_mean": float(np.mean(list(worstf.values()))),
                   "range_roc_mean": float(np.mean(list(rng.values()))),
                   "min_roc": float(min(c["roc"] for c in cs)), "min_f1val": float(min(c["f1_val"] for c in cs)),
                   "src_sensitivity_sd": src_sens,
                   "src_sensitivity_sd_mean": {fam: float(np.mean(x)) for fam, x in src_sens.items()}}
    # RecAdam có làm sd theo fold nhỏ hơn không: ghép theo nguồn (6 cặp) cho từng bối cảnh ASAM
    sd_pairs = {}
    for ctx, (on, off) in {"with_asam": ("rasam", "asamonly"), "without_asam": ("raonly", "noras")}.items():
        for q in ("sd_roc_by_src", "sd_f1val_by_src", "sd_roc_adj_by_src", "sd_f1val_adj_by_src", "worst_roc_by_src",
                  "worst_f1val_by_src"):
            d = {s: stab[on][q][s] - stab[off][q][s] for s in SRC6}
            sd_pairs[f"{ctx}|{q}"] = {"by_src": d, "mean": float(np.mean(list(d.values()))),
                                      "n_lower": int(sum(x < -TIE_EPS for x in d.values())),
                                      "n_higher": int(sum(x > TIE_EPS for x in d.values())),
                                      "mean_jsonly": float(np.mean([d[s] for s in FAMILIES["jsonly"]])),
                                      "mean_jscpp": float(np.mean([d[s] for s in FAMILIES["jscpp"]]))}
        # bản bỏ ô sập: sd tính trên 4 fold còn lại của nguồn có ô sập (cả hai vế cùng bỏ fold đó)
        d = {}
        for s in SRC6:
            keep = [f for f in FOLDS if not (cells[(on, s, f)]["collapse"] or cells[(off, s, f)]["collapse"])]
            d[s] = float(np.std([cells[(on, s, f)]["roc"] for f in keep], ddof=1) -
                         np.std([cells[(off, s, f)]["roc"] for f in keep], ddof=1))
        sd_pairs[f"{ctx}|sd_roc_by_src_excl"] = {"by_src": d, "mean": float(np.mean(list(d.values()))),
                                                 "n_lower": int(sum(x < -TIE_EPS for x in d.values())),
                                                 "n_higher": int(sum(x > TIE_EPS for x in d.values()))}
        # độ nhạy theo nguồn trong fold: ghép theo (họ, fold)
        dd = [stab[on]["src_sensitivity_sd"][fam][i] - stab[off]["src_sensitivity_sd"][fam][i]
              for fam in FAMILIES for i in range(len(FOLDS))]
        sd_pairs[f"{ctx}|src_sensitivity"] = {"diffs": dd, "mean": float(np.mean(dd)),
                                              "n_lower": int(sum(x < -TIE_EPS for x in dd)), "n": len(dd)}
    stab["paired"] = sd_pairs
    num["stability"] = stab

    # ---- (4) nhớ nguồn (gián tiếp) + hiệu chỉnh + động học: các đại lượng mỗi ô, ghép cặp như chỉ số
    derived = ["ece", "brier", "nll", "conf", "thr_dev", "val_test_gap", "rho_nop1", "cwe022", "cwe079", "cwe078", "cwe089",
               "vr1", "vr2", "vr3", "vr_mean_1_10", "tl1", "tl2", "tl3", "ep_to_val80", "ep_to_val85", "ep_to_val90",
               "best_epoch", "stop_epoch", "plateau", "best_val"]
    dd = {}
    for con in ("ra_main", "ra_with_asam", "ra_without_asam", "interaction"):
        for g in ("all", "jsonly", "jscpp"):
            for mode in ("all", "excl"):
                for k in derived:
                    dd[f"{con}|{g}|{mode}|{k}"] = summarize(contrast_values(cells, con, k, GROUPS[g][0], mode))
    num["derived_deltas"] = dd
    per_var = {}
    for v in VARIANTS:
        cs = [cells[(v, s, f)] for s in SRC6 for f in FOLDS]
        per_var[v] = {}
        for k in derived + ["cwe_hard", "cwe_easy"]:
            vals = [c[k] for c in cs if c.get(k) is not None]
            per_var[v][k] = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
            for fam in FAMILIES:
                vv = [cells[(v, s, f)][k] for s in FAMILIES[fam] for f in FOLDS if cells[(v, s, f)].get(k) is not None]
                per_var[v][f"{k}|{fam}"] = {"mean": float(np.mean(vv)), "median": float(np.median(vv))}
    num["per_variant"] = per_var
    # tương đồng dự đoán giữa các nguồn trong cùng fold (Spearman xác suất test), ghép (fold, cặp nguồn)
    inter = {}
    for v in VARIANTS:
        for f in FOLDS:
            for i, s1 in enumerate(SRC6):
                for s2 in SRC6[i + 1:]:
                    inter[(v, f, s1, s2)] = float(stats.spearmanr(cells[(v, s1, f)]["p"], cells[(v, s2, f)]["p"]).statistic)
    isim = {}
    for ctx, (on, off) in {"with_asam": ("rasam", "asamonly"), "without_asam": ("raonly", "noras")}.items():
        for scope in ("within_family", "all_pairs"):
            diffs = []
            for (v, f, s1, s2), r in inter.items():
                if v != on:
                    continue
                if scope == "within_family" and s1.split("_")[1] != s2.split("_")[1]:
                    continue
                excl = any(cells[(w, s, f)]["collapse"] for w in (on, off) for s in (s1, s2))
                diffs.append((f"{s1}~{s2}", f, r - inter[(off, f, s1, s2)], excl))
            full = summarize([(a, b, c) for a, b, c, _ in diffs])
            ex = summarize([(a, b, c) for a, b, c, e in diffs if not e])
            isim[f"{ctx}|{scope}"] = {"all": full, "excl": ex}
    for v in VARIANTS:
        isim[f"mean|{v}"] = float(np.mean([r for (vv, f, s1, s2), r in inter.items() if vv == v]))
        isim[f"mean_within|{v}"] = float(np.mean([r for (vv, f, s1, s2), r in inter.items()
                                                  if vv == v and s1.split("_")[1] == s2.split("_")[1]]))
    num["inter_source_similarity"] = isim
    # độ nhất trí giữa hai vế của cùng phép so RecAdam (cùng nguồn, cùng fold): RecAdam đổi dự đoán bao nhiêu
    agree = {}
    for ctx, (on, off) in {"with_asam": ("rasam", "asamonly"), "without_asam": ("raonly", "noras"),
                           "asam_without_ra": ("asamonly", "noras")}.items():
        rs, flips = [], []
        for s in SRC6:
            for f in FOLDS:
                a, b = cells[(on, s, f)], cells[(off, s, f)]
                if a["collapse"] or b["collapse"]:
                    continue
                rs.append(float(stats.spearmanr(a["p"], b["p"]).statistic))
                flips.append(float(np.mean((a["p"] >= 0.5) != (b["p"] >= 0.5))))
        absd = {k: [abs(cells[(on, s, f)][k] - cells[(off, s, f)][k]) for s in SRC6 for f in FOLDS
                    if not (cells[(on, s, f)]["collapse"] or cells[(off, s, f)]["collapse"])] for k in ("roc", "f1_val")}
        agree[ctx] = {"n": len(rs), "spearman_median": float(np.median(rs)), "spearman_min": float(min(rs)),
                      "flip_rate_mean": float(np.mean(flips)), "flip_rate_median": float(np.median(flips)),
                      "abs_droc_mean": float(np.mean(absd["roc"])), "abs_df1val_mean": float(np.mean(absd["f1_val"]))}
    num["pairwise_agreement"] = agree

    # tham chiếu nhiễu HẠT GIỐNG (khối cũ archive_20261004, batch 8: cùng cấu hình chỉ ASAM, nguồn common chỉ JS,
    # chỉ khác seed của cả hai pha 42 / 1234 / 7) - để đặt độ lớn thay đổi do RecAdam cạnh độ lớn do đổi seed
    arch = os.path.join(ROOT, "archive_20261004", "results")
    seedref = {"flip": [], "spearman": [], "abs_droc": [], "abs_df1val": [], "configs": []}
    for base in ("mw_assemble_asamonly_jsonly", "mwg_assemble_asamonly_jsonly"):
        runs = [base, base + "_s1234", base + "_s7"]
        if not all(os.path.exists(os.path.join(arch, r, "fold5.probs.npz")) for r in runs):
            continue
        seedref["configs"].append(base)
        for f in FOLDS:
            P = [np.load(os.path.join(arch, r, f"fold{f}.probs.npz")) for r in runs]
            J = [json.load(open(os.path.join(arch, r, f"fold{f}.json"))) for r in runs]
            for i in range(3):
                for j in range(i + 1, 3):
                    if not np.array_equal(P[i]["labels"], P[j]["labels"]):
                        continue
                    pi, pj = P[i]["probabilities"].astype(float), P[j]["probabilities"].astype(float)
                    if min(J[i]["test_roc_auc"], J[j]["test_roc_auc"]) < COLLAPSE_ROC:
                        continue
                    seedref["flip"].append(float(np.mean((pi >= 0.5) != (pj >= 0.5))))
                    seedref["spearman"].append(float(stats.spearmanr(pi, pj).statistic))
                    seedref["abs_droc"].append(abs(J[i]["test_roc_auc"] - J[j]["test_roc_auc"]))
                    seedref["abs_df1val"].append(abs(J[i]["test_macro_f1_at_valcal"] - J[j]["test_macro_f1_at_valcal"]))
    if seedref["flip"]:
        num["seed_reference"] = {"n": len(seedref["flip"]), "configs": seedref["configs"],
                                 "flip_rate_mean": float(np.mean(seedref["flip"])),
                                 "spearman_median": float(np.median(seedref["spearman"])),
                                 "abs_droc_mean": float(np.mean(seedref["abs_droc"])),
                                 "abs_df1val_mean": float(np.mean(seedref["abs_df1val"]))}

    # RecAdam × họ nguồn: theo fold, (TB hiệu trên 3 nguồn chỉ JS) - (TB hiệu trên 3 nguồn JS + C/C++); thăm dò
    famint = {}
    for con in ("ra_with_asam", "ra_without_asam"):
        for mode in ("all", "excl"):
            for k in ("roc", "f1_val", "f1_05", "pr", "cwe_hard", "cwe_easy"):
                byf = {}
                for fam in FAMILIES:
                    for s, f, d in contrast_values(cells, con, k, FAMILIES[fam], mode):
                        byf.setdefault(f, {}).setdefault(fam, []).append(d)
                vals = [float(np.mean(v["jsonly"]) - np.mean(v["jscpp"])) for f, v in sorted(byf.items())
                        if "jsonly" in v and "jscpp" in v]
                famint[f"{con}|{mode}|{k}"] = {"fold_diffs": vals, "mean": float(np.mean(vals)),
                                               "fold_pos": int(sum(x > TIE_EPS for x in vals)), "fold_n": len(vals)}
    num["family_interaction"] = famint

    # ---- (5) lịch λ(t)
    hp = cells[("rasam", "common_jsonly", 1)]["hp"]
    n_train = sum(1 for _ in open(os.path.join(DATA, "fold1", "train.jsonl")))
    spe = math.ceil(n_train / int(hp["batch_size"]))
    sch = recadam_schedule(hp, spe)
    num["schedule"] = {k: v for k, v in sch.items() if not isinstance(v, np.ndarray)}
    num["schedule"]["n_train"] = n_train

    # ---- (6) ô me30 (chạy lại 3 ô sập với --min_epochs 30): kết quả nếu đã có, và tiến độ đọc từ log
    prog = {}
    for d in sorted(glob.glob(os.path.join(LOGS, "*_me30"))):
        run = os.path.basename(d)
        for lp in sorted(glob.glob(os.path.join(d, "fold*.log"))):
            f = int(re.search(r"fold(\d+)", lp).group(1))
            h = parse_log_history(lp)
            prog[f"{run} f{f}"] = {"last_epoch": h[-1]["ep"] if h else 0, "train_loss": h[-1]["train_loss"] if h else None,
                                   "val_roc": h[-1]["val_roc"] if h else None,
                                   "min_train_loss": min((e["train_loss"] for e in h), default=None),
                                   "done": os.path.exists(os.path.join(RES, run, f"fold{f}.json"))}
    num["me30_progress"] = prog
    num["me30"] = {f"{r} f{f}": {"roc": c["roc"], "f1_05": c["f1_05"], "f1_val": c["f1_val"], "pr": c["pr"],
                                  "best_epoch": c["best_epoch"], "stop_epoch": c["stop_epoch"], "plateau": c["plateau"],
                                  "plateau_censored": c["plateau_censored"]} for (r, f), c in me30.items()}
    return cells, extra, me30, sch, num


# ============================================================================ định dạng
def fmt(x, nd=3, sign=False, comma=False):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if s in ("+0." + "0" * nd, "-0." + "0" * nd):
        s = "0." + "0" * nd if not sign else "±0." + "0" * nd
    return s.replace(".", ",") if comma else s


def kn(st, fold=False):
    if fold:
        return f"{st['fold_pos']}/{st['fold_n']}"
    return f"{st['pos']}/{st['n']}"


# ============================================================================ bảng
def write_tables(num):
    main, deltas, stab = num["main_table"], num["deltas"], num["stability"]
    files = []

    # --- Bảng 1: TB ± sd, 4 biến thể × 6 nguồn (+ đối chứng)
    head = ["Phase-1 source"] + [f"{VAR_EN[v]} ROC" for v in VARIANTS] + [f"{VAR_EN[v]} F1@val" for v in VARIANTS]
    rows = []
    for s in SRC6:
        r = [SRC_EN[s]]
        for k in ("roc", "f1_val"):
            for v in VARIANTS:
                m = main[f"{v}|{s}"][k]
                r.append(f"{m['mean']:.3f} ± {m['sd']:.3f}" + ("*" if main[f"{v}|{s}"]["collapse"] and k == "roc" else ""))
        rows.append(r)
    for keyset, label in (("roc", "Mean (6 sources)"),):
        r = [label]
        for k in ("roc", "f1_val"):
            for v in VARIANTS:
                x = [main[f"{v}|{s}"][k]["mean"] for s in SRC6]
                r.append(f"{np.mean(x):.3f}")
        rows.append(r)
    extra_rows = [("extra|baseline", "No Phase 1: CodeBERT baseline"), ("extra|nop1_adamw", "No Phase 1: MW, AdamW"),
                  ("extra|nop1_asamonly", "No Phase 1: MW, ASAM")] + \
                 [(f"extra|rasam_{s}", f"{SRC_EN[s]} (RecAdam+ASAM only)") for s in SRC_CCPP]
    for key, label in extra_rows:
        if key not in main:
            continue
        r = [label]
        for k in ("roc", "f1_val"):
            m = main[key][k]
            cell = f"{m['mean']:.3f} ± {m['sd']:.3f}" + ("*" if main[key]["collapse"] and k == "roc" else "")
            if key.startswith("extra|rasam_"):
                r += ["", "", "", cell]
            elif key == "extra|nop1_asamonly":
                r += ["", "", cell, ""]
            else:
                r += [cell, "", "", ""]
        rows.append(r)
    csv_rows = [["source", "variant", "metric", "mean", "sd", "min", "max", "n_collapsed", "fold1", "fold2", "fold3", "fold4", "fold5"]]
    for key, row in main.items():
        a, b = key.split("|")
        for k in MKEYS:
            m = row[k]
            csv_rows.append([b, a, k, f"{m['mean']:.4f}", f"{m['sd']:.4f}", f"{m['min']:.4f}", f"{m['max']:.4f}", row["collapse"]]
                            + [f"{t:.4f}" for t in m["folds"]])
    files += _emit("table_main", head, rows, csv_rows,
                   caption="Test performance of the four Phase-2 optimiser variants (mean ± SD over 5 target folds, seed 42). "
                           "Asterisk: at least one fold collapsed (test ROC-AUC < 0.75). AdamW = Phase 1 then plain AdamW; "
                           "RecAdam+ASAM is the main configuration.", label="tab:recadam-main",
                   colspec="l" + "c" * 8, group_head=[("", 1), ("ROC-AUC", 4), ("Macro-F1 at validation threshold", 4)],
                   sub_head=["Phase-1 source"] + [VAR_EN[v] for v in VARIANTS] * 2)

    # --- Bảng 2: hiệu ghép cặp của RecAdam (4 chỉ số, k/n, CI)
    head = ["Contrast", "Sources", "Cells"] + [MLABEL[k] for k in MKEYS]
    rows, csv_rows = [], [["contrast", "group", "mode", "metric", "n", "mean", "median", "ci_boot_lo", "ci_boot_hi", "ci_t_lo",
                           "ci_t_hi", "pos", "neg", "tie", "fold_pos", "fold_n", "fold_wilcoxon_p", "src_pos", "src_n", "min", "max"]]
    order = [("ra_main", "all"), ("ra_main", "jsonly"), ("ra_main", "jscpp"),
             ("ra_with_asam", "all"), ("ra_with_asam", "jsonly"), ("ra_with_asam", "jscpp"),
             ("ra_without_asam", "all"), ("ra_without_asam", "jsonly"), ("ra_without_asam", "jscpp"),
             ("interaction", "all"), ("interaction", "jsonly"), ("interaction", "jscpp")]
    for con, g in order:
        for mode in ("all", "excl", "me30"):
            if f"{con}|{g}|{mode}|roc" not in deltas:
                continue
            if mode != "all" and all(deltas[f"{con}|{g}|{mode}|{k}"]["n"] == deltas[f"{con}|{g}|all|{k}"]["n"] for k in MKEYS):
                continue        # không có ô sập trong nhóm: bản nhạy trùng bản chính
            if mode == "me30" and all(abs(deltas[f"{con}|{g}|me30|{k}"]["mean"] - deltas[f"{con}|{g}|all|{k}"]["mean"]) < 1e-12
                                      for k in MKEYS):
                continue
            r = [CONTRASTS[con][1], GROUPS[g][1], {"all": "all", "excl": "excl. collapsed", "me30": "collapsed rerun (me30)"}[mode]]
            for k in MKEYS:
                st = deltas[f"{con}|{g}|{mode}|{k}"]
                ci = st.get("ci_boot")
                r.append(f"{st['mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {st['pos']}/{st['n']}" if ci else
                         f"{st['mean']:+.3f} {st['pos']}/{st['n']}")
            rows.append(r)
    for key, st in deltas.items():
        con, g, mode, k = key.split("|")
        if st["n"] == 0:
            continue
        ci, ct = st.get("ci_boot") or [None, None], st.get("ci_t") or [None, None]
        csv_rows.append([con, g, mode, k, st["n"], f"{st['mean']:.4f}", f"{st['median']:.4f}"] +
                        [("" if x is None else f"{x:.4f}") for x in ci + ct] +
                        [st["pos"], st["neg"], st["tie"], st["fold_pos"], st["fold_n"],
                         "" if st.get("fold_wilcoxon_p") is None else f"{st['fold_wilcoxon_p']:.4f}", st["src_pos"], st["src_n"],
                         f"{st['min']:.4f}", f"{st['max']:.4f}"])
    files += _emit("table_recadam_delta", head, rows, csv_rows,
                   caption="Paired effect of RecAdam on test metrics (difference within the same Phase-1 source and target fold). "
                           "Mean difference, 95% fold-cluster bootstrap CI (5 clusters; anti-conservative), and number of positive "
                           "pairs / pairs. Main effect = average of the two simple effects. 'excl. collapsed' drops pairs containing "
                           "a collapsed cell (test ROC-AUC < 0.75). Rerun noise floor: 0.010.",
                   label="tab:recadam-delta", colspec="lll" + "r" * 4, small=True)

    # --- Bảng 3: độ ổn định
    head = ["Variant", "Cells", "Collapsed", "Plateau >10 ep", "Plateau median (range)", "SD ROC (fold)", "SD ROC (fold-adj.)",
            "SD F1@val (fold)", "Worst-fold ROC", "Min ROC"]
    rows, csv_rows = [], [["variant", "n", "collapse", "stuck10", "plateau_median", "plateau_min", "plateau_max", "sd_roc_mean",
                           "sd_roc_adj_mean", "sd_f1val_mean", "sd_f1val_adj_mean", "worst_roc_mean", "worst_f1val_mean",
                           "min_roc", "min_f1val", "src_sens_sd_jsonly", "src_sens_sd_jscpp"]]
    for v in VARIANTS:
        st = stab[v]
        rows.append([VAR_EN[v], st["n"], st["collapse"], st["stuck10"],
                     f"{st['plateau_median']:.0f} ({st['plateau_min']}-{st['plateau_max']})",
                     f"{st['sd_roc_mean']:.3f}", f"{st['sd_roc_adj_mean']:.3f}", f"{st['sd_f1val_mean']:.3f}",
                     f"{st['worst_roc_mean']:.3f}", f"{st['min_roc']:.3f}"])
        csv_rows.append([v, st["n"], st["collapse"], st["stuck10"], st["plateau_median"], st["plateau_min"], st["plateau_max"],
                         f"{st['sd_roc_mean']:.4f}", f"{st['sd_roc_adj_mean']:.4f}", f"{st['sd_f1val_mean']:.4f}",
                         f"{st['sd_f1val_adj_mean']:.4f}", f"{st['worst_roc_mean']:.4f}", f"{st['worst_f1val_mean']:.4f}",
                         f"{st['min_roc']:.4f}", f"{st['min_f1val']:.4f}", f"{st['src_sensitivity_sd_mean']['jsonly']:.4f}",
                         f"{st['src_sensitivity_sd_mean']['jscpp']:.4f}"])
    files += _emit("table_stability", head, rows, csv_rows,
                   caption="Stability of the four variants over 6 Phase-1 sources x 5 folds. Collapsed: test ROC-AUC < 0.75. "
                           "Plateau: epochs before the training loss first drops below 0.65. SD: standard deviation over the 5 "
                           "folds, averaged over sources; fold-adj. subtracts the per-fold median of all 2x2 cells. "
                           "Worst-fold ROC: lowest fold per source, averaged over sources.",
                   label="tab:recadam-stability", colspec="l" + "r" * 9, small=True)

    # --- Bảng 4: chỉ báo gián tiếp (nhớ nguồn, hiệu chỉnh, động học)
    dd = num["derived_deltas"]
    items = [("rho_nop1", "Spearman with no-Phase-1 model", 3), ("cwe_hard", "ROC-AUC on CWE-022+079", 3),
             ("cwe_easy", "ROC-AUC on CWE-078+089", 3), ("ece", "ECE (10 bins)", 3), ("brier", "Brier score", 3),
             ("nll", "Log loss", 3), ("val_test_gap", "Val ROC - test ROC", 3),
             ("vr1", "Val ROC, epoch 1", 3), ("vr_mean_1_10", "Mean val ROC, epochs 1-10", 3),
             ("ep_to_val85", "Epochs to val ROC >= 0.85", 2), ("best_epoch", "Selected epoch", 2), ("stop_epoch", "Stop epoch", 2)]
    head = ["Quantity", "RecAdam, ASAM on", "RecAdam, ASAM on (excl. collapsed)", "RecAdam, ASAM off"]
    rows, csv_rows = [], [["quantity", "contrast", "mode", "n", "mean", "median", "ci_boot_lo", "ci_boot_hi", "pos", "neg", "tie",
                           "fold_pos", "fold_n"]]
    for k, lab, nd in items:
        r = [lab]
        for con, mode in (("ra_with_asam", "all"), ("ra_with_asam", "excl"), ("ra_without_asam", "all")):
            st = deltas.get(f"{con}|all|{mode}|{k}") or dd.get(f"{con}|all|{mode}|{k}")
            ci = st.get("ci_boot")
            r.append(f"{st['mean']:+.{nd}f} [{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}] {st['pos']}/{st['n']}" if ci else
                     f"{st['mean']:+.{nd}f} {st['pos']}/{st['n']}")
        rows.append(r)
        for con in ("ra_with_asam", "ra_without_asam", "ra_main", "interaction"):
            for mode in ("all", "excl"):
                st = deltas.get(f"{con}|all|{mode}|{k}") or dd.get(f"{con}|all|{mode}|{k}")
                if not st or st["n"] == 0:
                    continue
                ci = st.get("ci_boot") or [None, None]
                csv_rows.append([k, con, mode, st["n"], f"{st['mean']:.4f}", f"{st['median']:.4f}"] +
                                [("" if x is None else f"{x:.4f}") for x in ci] +
                                [st["pos"], st["neg"], st["tie"], st["fold_pos"], st["fold_n"]])
    isim = num["inter_source_similarity"]
    r = ["Spearman between Phase-1 sources (15 pairs x 5 folds)"]
    for ctx, mode in (("with_asam", "all"), ("with_asam", "excl"), ("without_asam", "all")):
        st = isim[f"{ctx}|all_pairs"][mode]
        ci = st["ci_boot"]
        r.append(f"{st['mean']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {st['pos']}/{st['n']}")
        csv_rows.append(["inter_source_rho", f"ra_{ctx}", mode, st["n"], f"{st['mean']:.4f}", f"{st['median']:.4f}",
                         f"{ci[0]:.4f}", f"{ci[1]:.4f}", st["pos"], st["neg"], st["tie"], st["fold_pos"], st["fold_n"]])
    rows.insert(1, r)
    files += _emit("table_indirect", head, rows, csv_rows,
                   caption="Indirect indicators of what RecAdam changes (paired difference RecAdam on minus off, all 6 sources, "
                           "30 pairs or fewer). Mean [95% fold-cluster bootstrap CI] and positive pairs / pairs. A source-memory "
                           "effect would show as lower similarity to the no-Phase-1 model and higher ROC on the CWEs that need "
                           "transfer (CWE-022, CWE-079).", label="tab:recadam-indirect", colspec="lrrr", small=True)
    return files


def _emit(name, head, rows, csv_rows, caption, label, colspec, group_head=None, sub_head=None, small=False):
    import csv
    paths = []
    p = os.path.join(OUT, f"{name}.csv")
    with open(p, "w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)
    paths.append(p)
    # Markdown (tiếng Anh, cho bài)
    p = os.path.join(OUT, f"{name}.md")
    with open(p, "w") as fh:
        fh.write(f"<!-- {caption} -->\n\n")
        fh.write("| " + " | ".join(str(h) for h in head) + " |\n|" + "---|" * len(head) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(c) for c in r) + " |\n")
    paths.append(p)
    # LaTeX booktabs
    def tex(s):
        s = str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$")
        s = s.replace(">=", r"$\geq$").replace(">10", r"$>$10").replace("|", r"$\mid$").replace(" < ", r" $<$ ")
        return s
    p = os.path.join(OUT, f"{name}.tex")
    with open(p, "w") as fh:
        fh.write("% sinh tự động bởi make_recadam_figures.py - cần \\usepackage{booktabs}\n")
        fh.write("\\begin{table*}[t]\n\\centering\n" + ("\\scriptsize\n" if small else "\\footnotesize\n"))
        fh.write(f"\\caption{{{tex(caption)}}}\n\\label{{{label}}}\n")
        fh.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        if group_head:
            fh.write(" & ".join(f"\\multicolumn{{{n}}}{{c}}{{{tex(t)}}}" if n > 1 else tex(t) for t, n in group_head) + " \\\\\n")
            start = 1
            for t, n in group_head:
                if n > 1:
                    fh.write(f"\\cmidrule(lr){{{start}-{start + n - 1}}}")
                start += n
            fh.write("\n" + " & ".join(tex(h) for h in sub_head) + " \\\\\n\\midrule\n")
        else:
            fh.write(" & ".join(tex(h) for h in head) + " \\\\\n\\midrule\n")
        for r in rows:
            fh.write(" & ".join(tex(c) for c in r) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    paths.append(p)
    return paths


# ============================================================================ hình
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
VCOLOR = {"noras": ORANGE, "asamonly": ORANGE, "raonly": BLUE, "rasam": BLUE}       # màu = RecAdam bật / tắt
VFILLED = {"noras": False, "raonly": False, "asamonly": True, "rasam": True}           # đặc = ASAM bật
VLS = {"noras": (0, (4, 2)), "raonly": (0, (4, 2)), "asamonly": "-", "rasam": "-"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "lines.solid_capstyle": "round",
    })
    return plt


def _save(fig, name):
    out = []
    for ext in ("svg", "pdf"):
        p = os.path.join(OUT, f"{name}.{ext}")
        fig.savefig(p, metadata={"Date": None} if ext == "svg" else {"CreationDate": None})
        out.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def _vmarker(ax, x, y, v, size=5.5, zorder=3, **kw):
    ax.plot(x, y, marker="o", ms=size, ls="none", mec=VCOLOR[v], mew=1.2,
            mfc=VCOLOR[v] if VFILLED[v] else "white", zorder=zorder, **kw)


def fig_forest(num, plt):
    deltas = num["deltas"]
    rows = [("all", "All 6 sources"), ("jsonly", "JS only (3)"), ("jscpp", "JS+C/C++ (3)")] + [(s, SRC_EN[s]) for s in SRC6]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 3.3), sharey=True)
    ys = {g: -i - (0.6 if i >= 3 else 0) for i, (g, _) in enumerate(rows)}
    for ax, k in zip(axes, MKEYS):
        ax.axvspan(-NOISE_FLOOR, NOISE_FLOOR, color="#f0efec", zorder=0, lw=0)
        ax.axvline(0, color=MUTED, lw=0.7, zorder=1)
        lo_all, hi_all = [], []
        for g, _ in rows:
            for off, con, mode, mfc, mk in ((0.17, "ra_with_asam", "all", BLUE, "o"), (-0.17, "ra_without_asam", "all", "white", "o"),
                                            (0.17, "ra_with_asam", "excl", None, "D")):
                st = deltas[f"{con}|{g}|{mode}|{k}"]
                if mode == "excl" and st["n"] == deltas[f"{con}|{g}|all|{k}"]["n"]:
                    continue
                y = ys[g] + off + (0.0 if mode == "all" else -0.0)
                ci = st["ci_boot"]
                if mode == "excl":
                    ax.plot(st["mean"], y, marker="D", ms=3.6, mfc=MUTED, mec="white", mew=0.6, ls="none", zorder=5)
                    continue
                ax.plot(ci, [y, y], color=BLUE, lw=1.1 if mfc == BLUE else 0.9, zorder=2,
                        ls="-" if mfc == BLUE else (0, (3, 1.5)))
                ax.plot(st["mean"], y, marker=mk, ms=4.6, mfc=mfc, mec=BLUE, mew=1.1, ls="none", zorder=4)
                lo_all.append(ci[0])
                hi_all.append(ci[1])
        lim = 0.10
        ax.set_xlim(-lim, lim)
        ax.set_xticks([-0.05, 0, 0.05])
        ax.set_xticklabels(["-0.05", "0", "+0.05"])
        # đánh dấu CI bị cắt
        for g, _ in rows:
            for off, con in ((0.17, "ra_with_asam"), (-0.17, "ra_without_asam")):
                ci = deltas[f"{con}|{g}|all|{k}"]["ci_boot"]
                if ci[1] > lim:
                    ax.annotate("", xy=(lim, ys[g] + off), xytext=(lim * 0.86, ys[g] + off),
                                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.8, mutation_scale=6))
        ax.set_title(MLABEL[k], loc="center", fontweight="bold")
        ax.set_xlabel("Δ (RecAdam on - off)")
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks([ys[g] for g, _ in rows])
    axes[0].set_yticklabels([lab for _, lab in rows])
    for t in axes[0].get_yticklabels()[:3]:
        t.set_fontweight("bold")
    axes[0].set_ylim(min(ys.values()) - 0.6, 0.6)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], marker="o", ms=4.6, mfc=BLUE, mec=BLUE, color=BLUE, lw=1.1, label="with ASAM: RecAdam+ASAM - ASAM"),
         Line2D([], [], marker="o", ms=4.6, mfc="white", mec=BLUE, color=BLUE, lw=0.9, ls=(0, (3, 1.5)),
                label="without ASAM: RecAdam - AdamW"),
         Line2D([], [], marker="D", ms=3.6, mfc=MUTED, mec="white", ls="none", label="with ASAM, collapsed pairs removed"),
         Patch(fc="#f0efec", ec="none", label="rerun noise floor (±0.010)")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06), handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.12)
    return _save(fig, "fig_recadam_forest")


def fig_interaction(cells, num, plt):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.25))
    panels = [("roc", "jsonly"), ("roc", "jscpp"), ("f1_val", "jsonly"), ("f1_val", "jscpp")]
    store = {}
    for ax, (k, fam) in zip(axes, panels):
        for asam, ls in ((0, (0, (4, 2))), (1, "-")):
            pts = []
            for ra in (0, 1):
                v = [vv for vv, fl in VFLAGS.items() if fl == (ra, asam)][0]
                byf = {f: [cells[(v, s, f)][k] for s in FAMILIES[fam]] for f in FOLDS}
                m, ci = cell_mean_ci(byf)
                pts.append((ra, m, ci, v))
                store[f"{k}|{fam}|{v}"] = {"mean": m, "ci": ci}
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=MUTED, lw=1.0, ls=ls, zorder=1)
            for ra, m, ci, v in pts:
                ax.plot([ra, ra], ci, color=VCOLOR[v], lw=1.0, zorder=2)
                _vmarker(ax, ra, m, v, size=5.5)
        # ASAM only, bỏ ô sập
        byf = {f: [cells[("asamonly", s, f)][k] for s in FAMILIES[fam] if not cells[("asamonly", s, f)]["collapse"]] for f in FOLDS}
        if sum(len(x) for x in byf.values()) < 15:
            m, ci = cell_mean_ci(byf)
            store[f"{k}|{fam}|asamonly_excl"] = {"mean": m, "ci": ci}
            ax.plot(-0.16, m, marker="D", ms=3.8, mfc=MUTED, mec="white", mew=0.6, ls="none", zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["off", "on"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_xlabel("RecAdam")
        ax.set_title(f"{MLABEL[k]} - {'JS only' if fam == 'jsonly' else 'JS+C/C++'}", loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    # cùng thang cho hai họ nguồn của một chỉ số
    for k, pair in (("roc", axes[:2]), ("f1_val", axes[2:])):
        vals = [x for key, d in store.items() if key.startswith(k + "|") for x in d["ci"] + [d["mean"]]]
        lo, hi = min(vals), max(vals)
        pad = 0.06 * (hi - lo)
        for ax in pair:
            ax.set_ylim(lo - pad, hi + pad)
    axes[0].set_ylabel("Test score (mean of 15 cells)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=MUTED, lw=1.0, ls="-", marker="o", mfc=INK2, mec=INK2, ms=4.5, label="ASAM on"),
         Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 2)), marker="o", mfc="white", mec=INK2, ms=4.5, label="ASAM off"),
         Line2D([], [], color=ORANGE, marker="o", ls="none", ms=4.5, mfc=ORANGE, label="RecAdam off"),
         Line2D([], [], color=BLUE, marker="o", ls="none", ms=4.5, mfc=BLUE, label="RecAdam on"),
         Line2D([], [], marker="D", ls="none", ms=3.8, mfc=MUTED, mec="white", label="ASAM only, collapsed folds removed")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.2), columnspacing=1.0)
    fig.subplots_adjust(wspace=0.38)
    num["interaction_means"] = store
    return _save(fig, "fig_recadam_interaction")


def fig_paired(cells, plt):
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6))
    panels = [("roc", "without"), ("roc", "with"), ("f1_val", "without"), ("f1_val", "with")]
    # thang zoom vào khối ô bình thường; ô sập vẽ ở mép dưới kèm giá trị thật
    ylims = {}
    for k in ("roc", "f1_val"):
        ok = [cells[(v, s, f)][k] for v in VARIANTS for s in SRC6 for f in FOLDS if not cells[(v, s, f)]["collapse"]]
        ylims[k] = (min(ok) - 0.03, max(ok) + 0.01)
    for ax, (k, ctx) in zip(axes, panels):
        off, on = ("noras", "raonly") if ctx == "without" else ("asamonly", "rasam")
        lo, hi = ylims[k]
        floor = lo + 0.012
        up = down = 0
        offscale = []
        for s in SRC6:
            for f in FOLDS:
                a, b = cells[(off, s, f)][k], cells[(on, s, f)][k]
                col = cells[(off, s, f)]["collapse"] or cells[(on, s, f)]["collapse"]
                jit = (SRC6.index(s) - 2.5) * 0.018
                aa, bb = max(a, floor), max(b, floor)
                ax.plot([0 + jit, 1 + jit], [aa, bb], color=RED if col else ("#9fbfe6" if b > a else "#c9c8c3"),
                        lw=1.3 if col else 0.7, zorder=3 if col else 1)
                fam_m = "o" if s.endswith("jsonly") else "^"
                for xx, vv, val, shown in ((0 + jit, off, a, aa), (1 + jit, on, b, bb)):
                    if shown != val:          # ngoài thang: mũi tên xuống, giá trị thật ghi chung một dòng
                        ax.plot(xx, shown, marker="v", ms=4.2, ls="none", mec=RED, mfc=RED, zorder=5)
                        offscale.append(val)
                        continue
                    ax.plot(xx, val, marker=fam_m, ms=3.2, ls="none", mec=VCOLOR[vv], mew=0.8,
                            mfc=VCOLOR[vv] if VFILLED[vv] else "white", zorder=4)
                up += b > a + TIE_EPS
                down += b < a - TIE_EPS
        if offscale:
            ax.text(0.1, floor, "off-scale: " + ", ".join(f"{v:.2f}" for v in sorted(offscale)), fontsize=6, color=RED,
                    va="center", ha="left")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([VAR_EN[off], VAR_EN[on]])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(lo, hi)
        ttl = f"{MLABEL[k]} - {'without' if ctx == 'without' else 'with'} ASAM"
        ax.set_title(ttl, loc="left")
        ax.text(0.98, 0.985, f"up {up} / down {down}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test score per (source, fold)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", ms=3.6, mfc=INK2, mec=INK2, label="JS-only source"),
         Line2D([], [], marker="^", ls="none", ms=3.6, mfc=INK2, mec=INK2, label="JS+C/C++ source"),
         Line2D([], [], color="#9fbfe6", lw=1.0, label="RecAdam higher"), Line2D([], [], color="#c9c8c3", lw=1.0, label="RecAdam lower"),
         Line2D([], [], color=RED, lw=1.3, label="pair with a collapsed cell")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(wspace=0.32)
    return _save(fig, "fig_recadam_paired_folds")


def fig_sd(num, plt):
    stab = num["stability"]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.55))
    from matplotlib.ticker import FixedLocator, NullLocator
    for ax, (q, lab) in zip(axes, (("sd_roc_by_src", "ROC-AUC"), ("sd_f1val_by_src", "F1@val"))):
        mn, mx = 1, 0
        for ctx, (on, off), filled in (("with", ("rasam", "asamonly"), True), ("without", ("raonly", "noras"), False)):
            for s in SRC6:
                x, y = stab[off][q][s], stab[on][q][s]
                mx, mn = max(mx, x, y), min(mn, x, y)
                ax.plot(x, y, marker="o" if s.endswith("jsonly") else "^", ms=5, ls="none", mec=BLUE, mew=1.1,
                        mfc=BLUE if filled else "white", zorder=3)
        lo, hi = mn / 1.4, mx * 1.4
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ticks = [t for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2) if lo <= t <= hi]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_aspect("equal")
        ax.set_xlabel("SD over 5 folds, RecAdam off")
        ax.set_ylabel("SD over 5 folds, RecAdam on")
        ax.set_title(lab, loc="left")
        ax.text(0.97, 0.04, "below diagonal:\nRecAdam less variable", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6, color=INK2)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", ms=4.6, mfc=BLUE, mec=BLUE, label="with ASAM"),
         Line2D([], [], marker="o", ls="none", ms=4.6, mfc="white", mec=BLUE, label="without ASAM"),
         Line2D([], [], marker="o", ls="none", ms=4.6, mfc=INK2, mec=INK2, label="JS-only source"),
         Line2D([], [], marker="^", ls="none", ms=4.6, mfc=INK2, mec=INK2, label="JS+C/C++ source")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(wspace=0.45)
    return _save(fig, "fig_recadam_fold_sd")


def fig_dynamics(cells, sch, plt):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.1))
    E = 17                         # min_epochs 10 + patience 8 ⇒ không ô nào dừng trước epoch 17
    ep = np.arange(1, E + 1)
    for j, fam in enumerate(("jsonly", "jscpp")):
        for i, (key, lab) in enumerate((("val_roc", "Validation ROC-AUC"), ("train_loss", "Training loss"))):
            ax = axes[i, j]
            for v in VARIANTS:
                M = np.array([[cells[(v, s, f)]["hist"][e - 1][key] for e in ep] for s in FAMILIES[fam] for f in FOLDS], dtype=float)
                med = np.nanmedian(M, 0)
                q1, q3 = np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0)
                ax.fill_between(ep, q1, q3, color=VCOLOR[v], alpha=0.10 if VFILLED[v] else 0.07, lw=0)
                ax.plot(ep, med, color=VCOLOR[v], lw=1.5, ls=VLS[v], label=VAR_EN[v])
            if key == "train_loss":
                ax.axhline(math.log(2), color=MUTED, lw=0.6)
                ax.text(E, math.log(2) + 0.012, "ln 2", ha="right", va="bottom", fontsize=6, color=INK2)
                ax.set_ylim(0, 0.8)
            else:
                ax.set_ylim(0.45, 1.0)
            ax.set_xlim(1, E)
            ax.set_xticks([1, 5, 10, 15])
            ax.set_xlabel("Epoch")
            ax.set_ylabel(lab)
            ax.set_title(f"{'JS only' if fam == 'jsonly' else 'JS+C/C++'} (median, IQR)", loc="left")
            ax.yaxis.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
    # λ(t) và lr hiệu dụng
    ax = axes[0, 2]
    e = sch["ep"]
    m = e <= 10
    ax.plot(e[m], sch["lam"][m], color=BLUE, lw=1.5, label="λ(t)")
    ax.plot(e[m], (sch["lr"] / sch["lr0"])[m], color=ORANGE, lw=1.5, ls=(0, (4, 2)), label="AdamW step: lr(t)/lr0")
    ax.plot(e[m], (sch["lam"] * sch["lr"] / sch["lr0"])[m], color=BLUE, lw=1.5, ls=(0, (1, 1.2)), label="RecAdam step: λ·lr(t)/lr0")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Epoch (15 steps each)")
    ax.set_ylabel("Multiplier")
    ax.set_title("RecAdam annealing", loc="left")
    ax.legend(loc="lower right", handlelength=2.0)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax = axes[1, 2]
    ax.semilogy(e[m], sch["pull"][m], color=BLUE, lw=1.5)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Epoch (15 steps each)")
    ax.set_ylabel("Pull to anchor per step")
    ax.set_title("lr(t)·(1-λ)·γ, γ = 500", loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.5, which="major")
    ax.set_axisbelow(True)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.36, -0.045), handlelength=2.6)
    fig.subplots_adjust(hspace=0.55, wspace=0.42)
    return _save(fig, "fig_recadam_dynamics")


def inter_source_diffs(cells, con):
    """Hiệu (RecAdam bật - tắt) của tương quan Spearman giữa dự đoán test của hai nguồn khác nhau trong cùng fold."""
    on, off = {"ra_with_asam": ("rasam", "asamonly"), "ra_without_asam": ("raonly", "noras")}[con]
    out = []
    for f in FOLDS:
        for i, s1 in enumerate(SRC6):
            for s2 in SRC6[i + 1:]:
                r_on = stats.spearmanr(cells[(on, s1, f)]["p"], cells[(on, s2, f)]["p"]).statistic
                r_off = stats.spearmanr(cells[(off, s1, f)]["p"], cells[(off, s2, f)]["p"]).statistic
                col = any(cells[(v, s, f)]["collapse"] for v in (on, off) for s in (s1, s2))
                out.append((f"{s1}~{s2}", f, float(r_on - r_off), col))
    return out


def fig_memory(cells, num, plt):
    items = [("rho_nop1", "Spearman ρ with the\nno-Phase-1 model"), ("inter_src", "Spearman ρ between\nPhase-1 sources"),
             ("cwe_hard", "ROC-AUC on\nCWE-022 + CWE-079"), ("ece", "Expected calibration\nerror (10 bins)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.45))
    rng = np.random.default_rng(7)
    store = {}
    for ax, (k, lab) in zip(axes, items):
        ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
        shown = []
        for xi, (con, filled) in enumerate((("ra_without_asam", False), ("ra_with_asam", True))):
            if k == "inter_src":
                rows = inter_source_diffs(cells, con)
                dif = [(a, b, c) for a, b, c, _ in rows]
                colset = {(a, b) for a, b, _, e in rows if e}
                fam = {a: a.split("~")[0].endswith("jsonly") and a.split("~")[1].endswith("jsonly") for a, _, _, _ in rows}
            else:
                dif = contrast_values(cells, con, k, SRC6, "all")
                ok = {(s, f) for s, f, _ in contrast_values(cells, con, k, SRC6, "excl")}
                colset = {(s, f) for s, f, _ in dif if (s, f) not in ok}
                fam = {s: s.endswith("jsonly") for s in SRC6}
            xs = xi + rng.uniform(-0.17, 0.17, len(dif))
            for x, (s, f, d) in zip(xs, dif):
                col = (s, f) in colset
                c = RED if col else BLUE
                ax.plot(x, d, marker="o" if fam[s] else "^", ms=3.0 if k == "inter_src" else 3.4, ls="none", mec=c,
                        mfc=c if filled else "white", mew=0.7, alpha=0.85, zorder=2)
            st = summarize([(s, f, d) for s, f, d in dif if (s, f) not in colset])
            store[f"{k}|{con}"] = {kk: st[kk] for kk in ("n", "mean", "ci_boot", "pos", "neg", "fold_pos", "fold_n")}
            ax.plot([xi + 0.29, xi + 0.29], st["ci_boot"], color=INK, lw=1.2, zorder=3)
            ax.plot(xi + 0.29, st["mean"], marker="_", ms=9, mew=1.6, color=INK, zorder=4)
            shown += [d for s, f, d in dif if (s, f) not in colset]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ASAM off", "ASAM on"])
        ax.set_xlim(-0.45, 1.5)
        ax.set_title(lab, loc="left", fontsize=7.5)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        lo, hi = min(shown), max(shown)
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        n_out = 0
        for ln in ax.lines:
            yd = ln.get_ydata()
            if len(yd) == 1 and ln.get_color() != INK and (yd[0] < lo - pad or yd[0] > hi + pad):
                n_out += 1
        if n_out:
            ax.text(0.98, 0.98, f"{n_out} collapsed pair(s)\noff-scale", transform=ax.transAxes, ha="right", va="top",
                    fontsize=5.8, color=RED)
    axes[0].set_ylabel("Δ (RecAdam on - off)")
    num["indirect_figure"] = store
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", ms=4, mfc=BLUE, mec=BLUE, label="with ASAM"),
         Line2D([], [], marker="o", ls="none", ms=4, mfc="white", mec=BLUE, label="without ASAM"),
         Line2D([], [], marker="^", ls="none", ms=4, mfc="white", mec=INK2, label="JS+C/C++ source involved"),
         Line2D([], [], marker="o", ls="none", ms=4, mfc=RED, mec=RED, label="pair with a collapsed cell"),
         Line2D([], [], color=INK, marker="_", ms=9, mew=1.6, lw=1.2, label="mean, 95% fold-cluster CI (collapsed excluded)")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(wspace=0.5)
    return _save(fig, "fig_recadam_indirect")



# ============================================================================ main
def main():
    os.makedirs(OUT, exist_ok=True)
    cells, extra, me30, sch, num = analyse()
    plt = setup_mpl()
    figs = []
    figs += fig_forest(num, plt)
    figs += fig_interaction(cells, num, plt)
    figs += fig_paired(cells, plt)
    figs += fig_sd(num, plt)
    figs += fig_dynamics(cells, sch, plt)
    figs += fig_memory(cells, num, plt)
    tabs = write_tables(num)
    num["outputs"] = [os.path.basename(p) for p in figs + tabs]
    with open(os.path.join(OUT, "numbers.json"), "w") as fh:
        json.dump(num, fh, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    # văn bản báo cáo (json + md) dựng lại từ đúng numbers.json vừa ghi
    sys.path.insert(0, OUT)
    sys.dont_write_bytecode = True
    import recadam_report
    for p in recadam_report.write(json.load(open(os.path.join(OUT, "numbers.json")))):
        print("ghi", os.path.relpath(p, OUT))
    print("hp_check: %d ô, %d lệch" % (num["hp_check"]["cells_checked"], len(num["hp_check"]["mismatch"])))
    print("me30 có:", num["me30_available"] or "chưa")
    print("ô sập:", num["collapsed_cells"])
    for p in figs + tabs:
        print("ghi", os.path.relpath(p, OUT))


if __name__ == "__main__":
    main()
