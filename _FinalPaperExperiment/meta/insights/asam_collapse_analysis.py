#!/usr/bin/env python3
"""Phân tích: ASAM có gây bình nguyên ln2 / sập ở Pha 2 không, RecAdam có đỡ không.

Chỉ ĐỌC kết quả đã có (meta/dbrows, results, logs), không huấn luyện, không dùng GPU.
Ghi số ra meta/insights/asam_collapse_numbers.json để viết báo cáo.

Định nghĩa (nêu rõ trong báo cáo):
  - exit_epoch(thr): epoch ĐẦU TIÊN có train loss < thr. Không có epoch nào thì bị KIỂM DUYỆT:
    gán last_epoch + 1 và cờ censored. Ngưỡng chính 0,65; độ nhạy 0,60 / 0,67 / 0,68.
  - plateau_len = exit_epoch - 1 (số epoch ở trên ngưỡng trước khi thoát).
  - stuck10: plateau_len > 10 (11 epoch đầu đều >= ngưỡng), tức đứng ở ln2 > 10 epoch.
  - collapse: test ROC < 0,75.
Mọi so sánh ghép cặp theo (nguồn/checkpoint Pha 1, fold); không lấy hiệu hai trung bình.
"""
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = "/drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment"
ARCH = os.path.join(ROOT, "archive_20261004")
OUT = os.path.join(ROOT, "meta/insights/asam_collapse_numbers.json")
LN2 = math.log(2)
THRS = (0.60, 0.65, 0.67, 0.68)
MAIN_THR = 0.65


# ----------------------------------------------------------------------------- đo từng ô
def cell_metrics(hist, test_roc):
    """Các đại lượng của một ô Pha 2 từ history từng epoch."""
    tl = [h["train_loss"] for h in hist]
    vr = [h.get("val_roc") for h in hist]
    last = hist[-1]["ep"]
    m = {"last_ep": last, "test_roc": test_roc, "collapse": test_roc is not None and test_roc < 0.75}
    for thr in THRS:
        ex = next((h["ep"] for h in hist if h["train_loss"] < thr), None)
        key = f"{thr:.2f}"
        m[f"exit_{key}"] = ex if ex is not None else last + 1
        m[f"cens_{key}"] = ex is None
        m[f"plat_{key}"] = (ex if ex is not None else last + 1) - 1
        m[f"stuck10_{key}"] = (m[f"plat_{key}"] > 10)
    m["tl1"], m["tl2"], m["tl3"] = (tl + [None] * 3)[:3]
    m["vr1"], m["vr2"], m["vr3"] = (vr + [None] * 3)[:3]
    m["max_tl_1_3"] = max(tl[:3])
    m["min_tl_1_3"] = min(tl[:3])
    # "học rồi sụp": đã xuống dưới 0,65 rồi về lại >= 0,69 ở epoch sau
    ex = next((i for i, x in enumerate(tl) if x < MAIN_THR), None)
    m["relapse"] = ex is not None and any(x >= 0.69 for x in tl[ex + 1:])
    # train loss có tăng ở ep2 hoặc ep3 so với ep1 không (vọt sau bước đầu)
    m["tl_up_ep2_3"] = len(tl) >= 3 and max(tl[1:3]) > tl[0]
    # độ trội trên ln2 của trung bình ep2-10 (đứng QUANH hay TẠI ln2)
    seg = tl[1:10]
    m["mean_tl_2_10"] = float(np.mean(seg)) if seg else None
    m["best_ep"] = None
    return m


def load_new_block():
    cells = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "meta/dbrows/*.json"))):
        d = json.load(open(f))
        run = d["run"]
        if run.startswith(("p1_", "baseline")):
            continue
        res = json.load(open(os.path.join(ROOT, "results", run, f"fold{d['fold']}.json")))
        m = cell_metrics(d["history"], res["test_roc_auc"])
        m["best_ep"] = res["best_epoch"]
        m["run"] = run
        cells[(run, d["fold"])] = m
    return cells


# ----------------------------------------------------------------------------- khối cũ (archive)
def tf32_of(log_path):
    try:
        with open(log_path, errors="replace") as fh:
            head = fh.read(4000)
    except OSError:
        return None
    if "FPE_TF32=1" in head or "tf32(matmul/cudnn)=True" in head:
        return True
    return False


def load_old_block():
    cells = {}
    for f in sorted(glob.glob(os.path.join(ARCH, "meta/dbrows/*.json"))):
        d = json.load(open(f))
        run, fold = d["run"], d["fold"]
        if run.startswith(("p1_", "baseline")) or d.get("status") != "done" or not d.get("history"):
            continue
        rp = os.path.join(ARCH, "results", run, f"fold{fold}.json")
        if not os.path.exists(rp):
            continue
        res = json.load(open(rp))
        hp = res.get("hyperparameters", {})
        if hp.get("phase") not in (None, "train", "test") or hp.get("max_train_samples"):
            continue
        m = cell_metrics(d["history"], res["test_roc_auc"])
        m["best_ep"] = res.get("best_epoch")
        m["run"] = run
        m["tf32"] = tf32_of(os.path.join(ARCH, "logs", run, f"fold{fold}.log"))
        m["hp"] = {k: hp.get(k) for k in ("batch_size", "learning_rate", "warmup_ratio", "sam_rho", "recadam", "init",
                                          "init_ckpt", "seed", "min_epochs", "fusion", "graph", "anneal_t0_ratio",
                                          "anneal_k", "pretrain_cof", "sam_start_loss", "patience_start_loss",
                                          "data_root", "epochs")}
        cells[(run, fold)] = m
    return cells


# ----------------------------------------------------------------------------- thống kê ghép cặp
def paired(diffs, label=""):
    """diffs: list các (khoá, hiệu). Trả n, dương/âm/hoà, min-max, TB, trung vị, sign test, Wilcoxon."""
    x = np.array([v for _, v in diffs], dtype=float)
    n = len(x)
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    tie = n - pos - neg
    out = {"label": label, "n": n, "pos": pos, "neg": neg, "tie": tie}
    if n == 0:
        return out
    out.update(mean=float(x.mean()), median=float(np.median(x)), min=float(x.min()), max=float(x.max()))
    if pos + neg > 0:
        out["sign_p"] = float(stats.binomtest(pos, pos + neg, 0.5).pvalue)
    nz = x[x != 0]
    if len(nz) >= 1:
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(x, zero_method="pratt").pvalue)
        except ValueError:
            out["wilcoxon_p"] = None
    return out


def cluster_summary(diffs, idx):
    """Gộp hiệu theo cụm (idx=0: nguồn, idx=1: fold) rồi đếm dấu trên trung bình cụm."""
    g = defaultdict(list)
    for k, v in diffs:
        g[k[idx]].append(v)
    means = {c: float(np.mean(v)) for c, v in g.items()}
    vals = np.array(list(means.values()))
    pos, neg = int((vals > 0).sum()), int((vals < 0).sum())
    p = float(stats.binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else None
    return {"n_clusters": len(vals), "pos": pos, "neg": neg, "sign_p": p, "means": means}


def mcnemar(pairs_bin):
    """pairs_bin: list (a, b) bool. b01 = a không, b có; b10 = a có, b không. Exact McNemar = binomtest trên cặp lệch."""
    b10 = sum(1 for a, b in pairs_bin if a and not b)
    b01 = sum(1 for a, b in pairs_bin if b and not a)
    p = float(stats.binomtest(b10, b10 + b01, 0.5).pvalue) if b10 + b01 else 1.0
    return {"a_only": b10, "b_only": b01, "both": sum(1 for a, b in pairs_bin if a and b),
            "neither": sum(1 for a, b in pairs_bin if not a and not b), "p_exact": p}


# ----------------------------------------------------------------------------- lịch RecAdam
def recadam_schedule(n_train, batch, epochs, lr, warmup_ratio, t0_ratio, k, cof):
    steps_ep = math.ceil(n_train / batch)
    total = steps_ep * epochs
    t0 = max(1, int(t0_ratio * total))
    wu = int(total * warmup_ratio)
    rows = []
    cum_pull = 0.0
    lam_list, lr_list = [], []
    for s in range(1, total + 1):
        # get_linear_schedule_with_warmup: lr ở bước s dùng hệ số của bước s-1 (scheduler bước SAU optimizer)
        c = s - 1
        if wu > 0 and c < wu:
            f = c / max(1, wu)
        else:
            f = max(0.0, (total - c) / max(1, total - wu))
        lr_s = lr * f
        lam = 1.0 / (1.0 + math.exp(-k * (s - t0)))
        cum_pull += lr_s * (1 - lam) * cof
        lam_list.append(lam)
        lr_list.append(lr_s)
    lam_a, lr_a = np.array(lam_list), np.array(lr_list)
    per_ep = []
    for e in range(epochs):
        sl = slice(e * steps_ep, (e + 1) * steps_ep)
        per_ep.append({"ep": e + 1, "lam_mean": float(lam_a[sl].mean()), "lam_end": float(lam_a[sl][-1]),
                       "lr_adamw_mean": float(lr_a[sl].mean()), "lr_recadam_mean": float((lr_a[sl] * lam_a[sl]).mean()),
                       "pull_coef_mean": float((lr_a[sl] * (1 - lam_a[sl]) * cof).mean())})

    def first_step(th):
        i = int(np.argmax(lam_a >= th)) + 1
        return {"step": i, "epoch": math.ceil(i / steps_ep)}

    cum = np.cumsum(lr_a * (1 - lam_a) * cof)
    return {"steps_per_epoch": steps_ep, "total_steps": total, "t0": t0, "warmup_steps": wu,
            "lam_step1": float(lam_a[0]), "lam_ge": {str(th): first_step(th) for th in (0.9, 0.95, 0.99)},
            "cum_pull_total": float(cum_pull), "cum_pull_ep3": float(cum[3 * steps_ep - 1]),
            "sum_lr_eff_ep1_3_adamw": float(lr_a[:3 * steps_ep].sum()),
            "sum_lr_eff_ep1_3_recadam": float((lr_a * lam_a)[:3 * steps_ep].sum()),
            "first_step_lr_adamw": float(lr_a[0]), "first_step_lr_recadam": float(lr_a[0] * lam_a[0]),
            "per_epoch": per_ep[:8]}


# ----------------------------------------------------------------------------- 2×2 khối mới
SOURCES6 = [f"{a}_{b}" for a in ("4cwe", "common", "full") for b in ("jsonly", "jscpp")]
VAR = {"rasam": (1, 1), "asamonly": (0, 1), "raonly": (1, 0), "noras": (0, 0)}   # (RecAdam, ASAM)


def grid_new(cells):
    g = {}
    for s in SOURCES6:
        for v in VAR:
            for f in range(1, 6):
                g[(s, v, f)] = cells[(f"{v}_{s}", f)]
    # không Pha 1: chỉ có AdamW và chỉ ASAM
    for f in range(1, 6):
        g[("nop1", "noras", f)] = cells[("nop1_adamw", f)]
        g[("nop1", "asamonly", f)] = cells[("nop1_asamonly", f)]
    # 3 nguồn chỉ C/C++ chỉ có cột chính
    for s in ("4cwe_ccpp", "common_ccpp", "full_ccpp"):
        for f in range(1, 6):
            g[(s, "rasam", f)] = cells[(f"rasam_{s}", f)]
    return g


def diffs_of(g, va, vb, metric, sources):
    out = []
    for s in sources:
        for f in range(1, 6):
            a, b = g.get((s, va, f)), g.get((s, vb, f))
            if a is None or b is None:
                continue
            out.append(((s, f), a[metric] - b[metric]))
    return out


def summarize_variant(g, v, sources):
    rows = [g[(s, v, f)] for s in sources for f in range(1, 6) if (s, v, f) in g]
    if not rows:
        return None
    pl = np.array([r[f"plat_{MAIN_THR:.2f}"] for r in rows])
    return {"n": len(rows), "collapse": sum(r["collapse"] for r in rows),
            **{f"stuck10_{t:.2f}": sum(r[f"stuck10_{t:.2f}"] for r in rows) for t in THRS},
            "censored_0.65": sum(r["cens_0.65"] for r in rows),
            "plat_median": float(np.median(pl)), "plat_mean": float(pl.mean()), "plat_min": int(pl.min()), "plat_max": int(pl.max()),
            "relapse": sum(r["relapse"] for r in rows), "tl_up_ep2_3": sum(r["tl_up_ep2_3"] for r in rows),
            "tl1_median": float(np.median([r["tl1"] for r in rows])),
            "vr1_median": float(np.median([r["vr1"] for r in rows])),
            "max_tl_1_3_gt_0.72": sum(r["max_tl_1_3"] > 0.72 for r in rows),
            "test_roc_median": float(np.median([r["test_roc"] for r in rows]))}


def main():
    res = {}
    new = load_new_block()
    g = grid_new(new)
    with_p1 = SOURCES6
    res["new_variant_summary"] = {
        **{v: summarize_variant(g, v, with_p1) for v in VAR},
        "nop1_adamw": summarize_variant(g, "noras", ["nop1"]),
        "nop1_asamonly": summarize_variant(g, "asamonly", ["nop1"]),
        "rasam_ccpp3": summarize_variant(g, "rasam", ["4cwe_ccpp", "common_ccpp", "full_ccpp"]),
    }
    # bảng từng ô (đọc tay)
    res["new_cells"] = {f"{s}|{v}|f{f}": {k: g[(s, v, f)][k] for k in ("plat_0.65", "cens_0.65", "plat_0.60", "plat_0.68", "test_roc",
                                                                         "tl1", "tl2", "tl3", "vr1", "vr2", "vr3", "best_ep", "last_ep",
                                                                         "relapse", "mean_tl_2_10")}
                        for (s, v, f) in sorted(g)}

    comps = {
        "ASAM_noRA (asamonly - noras)": ("asamonly", "noras", with_p1 + ["nop1"]),
        "ASAM_noRA_withP1 (asamonly - noras)": ("asamonly", "noras", with_p1),
        "ASAM_withRA (rasam - raonly)": ("rasam", "raonly", with_p1),
        "RA_withASAM (rasam - asamonly)": ("rasam", "asamonly", with_p1),
        "RA_noASAM (raonly - noras)": ("raonly", "noras", with_p1),
        "rasam - noras": ("rasam", "noras", with_p1),
    }
    res["new_paired"] = {}
    for name, (va, vb, srcs) in comps.items():
        block = {}
        for thr in THRS:
            dd = diffs_of(g, va, vb, f"plat_{thr:.2f}", srcs)
            block[f"plat_{thr:.2f}"] = paired(dd, name)
            if thr == MAIN_THR:
                block["by_source"] = cluster_summary(dd, 0)
                block["by_fold"] = cluster_summary(dd, 1)
                block["diffs"] = {f"{k[0]}|f{k[1]}": v for k, v in dd}
        for met in ("tl1", "tl2", "tl3", "vr1", "test_roc"):
            block[met] = paired(diffs_of(g, va, vb, met, srcs), name)
        # nhị phân: kẹt > 10 epoch và sập
        for flag in ("stuck10_0.65", "stuck10_0.68", "collapse"):
            pb = [(g[(s, va, f)][flag], g[(s, vb, f)][flag]) for s in srcs for f in range(1, 6)]
            block[f"mcnemar_{flag}"] = mcnemar(pb)
        res["new_paired"][name] = block

    # tương tác 2×2: (rasam - raonly) - (asamonly - noras) theo từng ô
    inter = []
    for s in with_p1:
        for f in range(1, 6):
            p = lambda v: g[(s, v, f)][f"plat_{MAIN_THR:.2f}"]
            inter.append(((s, f), (p("rasam") - p("raonly")) - (p("asamonly") - p("noras"))))
    res["new_interaction_plat_0.65"] = paired(inter, "(rasam-raonly)-(asamonly-noras)")
    res["new_interaction_plat_0.65"]["by_source"] = cluster_summary(inter, 0)
    res["new_interaction_plat_0.65"]["by_fold"] = cluster_summary(inter, 1)

    # Fisher không ghép cặp cho số ô kẹt / sập giữa các cột (để đối chiếu)
    def fisher(v1, v2, flag, srcs1, srcs2):
        a = [g[(s, v1, f)][flag] for s in srcs1 for f in range(1, 6) if (s, v1, f) in g]
        b = [g[(s, v2, f)][flag] for s in srcs2 for f in range(1, 6) if (s, v2, f) in g]
        tab = [[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]
        return {"table": tab, "p": float(stats.fisher_exact(tab)[1])}
    allp1_rasam = with_p1 + ["4cwe_ccpp", "common_ccpp", "full_ccpp"]
    res["new_fisher"] = {
        "collapse rasam(45) vs asamonly+nop1(35)": fisher("rasam", "asamonly", "collapse", allp1_rasam, with_p1 + ["nop1"]),
        "stuck10 rasam(45) vs asamonly+nop1(35)": fisher("rasam", "asamonly", "stuck10_0.65", allp1_rasam, with_p1 + ["nop1"]),
        "stuck10 asamonly(30) vs noras(30)": fisher("asamonly", "noras", "stuck10_0.65", with_p1, with_p1),
    }

    # tương quan: độ dài bình nguyên của chỉ ASAM so với của noRAS cùng ô (checkpoint chậm dưới AdamW thì ASAM kéo dài?)
    a = [g[(s, "asamonly", f)]["plat_0.65"] for s in with_p1 + ["nop1"] for f in range(1, 6)]
    b = [g[(s, "noras", f)]["plat_0.65"] for s in with_p1 + ["nop1"] for f in range(1, 6)]
    c = [g[(s, "rasam", f)]["plat_0.65"] for s in with_p1 for f in range(1, 6)]
    d = [g[(s, "raonly", f)]["plat_0.65"] for s in with_p1 for f in range(1, 6)]
    e = [g[(s, "asamonly", f)]["plat_0.65"] for s in with_p1 for f in range(1, 6)]
    res["new_spearman"] = {
        "asamonly~noras (35)": list(map(float, stats.spearmanr(a, b))),
        "rasam~raonly (30)": list(map(float, stats.spearmanr(c, d))),
        "rasam~asamonly (30)": list(map(float, stats.spearmanr(c, e))),
    }

    # ----------------------------------------------------------------- khối cũ
    old = load_old_block()
    res["old_n_cells"] = len(old)
    # khoá cấu hình: mọi thứ trừ optimizer
    def okey(m):
        hp = m["hp"]
        ck = hp.get("init_ckpt") or "NONE"
        ck = re.sub(r"^.*/model/[^/]+/", "", ck)
        return (hp.get("fusion"), hp.get("graph"), ck, hp.get("seed"), hp.get("batch_size"), hp.get("learning_rate"),
                hp.get("warmup_ratio"), hp.get("min_epochs"), bool(m.get("tf32")), hp.get("data_root"),
                hp.get("sam_start_loss") or 0, hp.get("anneal_t0_ratio") if hp.get("recadam") else None,
                hp.get("anneal_k") if hp.get("recadam") else None)

    def ovar(m):
        hp = m["hp"]
        return (int(bool(hp.get("recadam"))), int((hp.get("sam_rho") or 0) > 0))

    og = defaultdict(dict)        # (key_without_ra_params, fold) -> {var: cell}
    dup = []
    for (run, fold), m in old.items():
        if run.startswith(("chk_", "fix_")):
            continue
        k = okey(m)
        k = k[:-2]                # bỏ t0/k khỏi khoá để RA và không RA ghép được (khối cũ dùng t0 0,01, k 0,05)
        v = ovar(m)
        if v in og[(k, fold)]:
            dup.append((run, fold, og[(k, fold)][v]["run"]))
            continue
        og[(k, fold)][v] = m
    res["old_duplicates"] = dup[:20]
    names = {(1, 1): "rasam", (0, 1): "asamonly", (1, 0): "raonly", (0, 0): "noras"}

    def old_diffs(va, vb, metric, filt=None):
        out = []
        for (k, fold), d in og.items():
            if filt and not filt(k):
                continue
            if va in d and vb in d:
                ra = d[va]["run"]
                out.append(((ra, fold), d[va][metric] - d[vb][metric]))
        return out

    res["old_paired"] = {}
    for name, (va, vb) in {"ASAM_noRA (asamonly - noras)": ((0, 1), (0, 0)),
                           "ASAM_withRA (rasam - raonly)": ((1, 1), (1, 0)),
                           "RA_withASAM (rasam - asamonly)": ((1, 1), (0, 1)),
                           "RA_noASAM (raonly - noras)": ((1, 0), (0, 0))}.items():
        dd = old_diffs(va, vb, "plat_0.65")
        blk = {"plat_0.65": paired(dd, name), "plat_0.60": paired(old_diffs(va, vb, "plat_0.60"), name),
               "plat_0.68": paired(old_diffs(va, vb, "plat_0.68"), name),
               "tl1": paired(old_diffs(va, vb, "tl1"), name), "vr1": paired(old_diffs(va, vb, "vr1"), name),
               "test_roc": paired(old_diffs(va, vb, "test_roc"), name)}
        pb = []
        for (k, fold), d in og.items():
            if va in d and vb in d:
                pb.append((d[va]["stuck10_0.65"], d[vb]["stuck10_0.65"]))
        blk["mcnemar_stuck10_0.65"] = mcnemar(pb)
        pb = [(d[va]["collapse"], d[vb]["collapse"]) for (k, fold), d in og.items() if va in d and vb in d]
        blk["mcnemar_collapse"] = mcnemar(pb)
        blk["runs_a"] = sorted({r for (r, _), _ in dd})
        res["old_paired"][name] = blk
    # tóm tắt theo biến thể (khối cũ, mọi ô train_mwg cấu hình chuẩn batch 8)
    osum = defaultdict(list)
    for (k, fold), d in og.items():
        for v, m in d.items():
            osum[names[v]].append(m)
    res["old_variant_summary"] = {v: {"n": len(ms), "collapse": sum(m["collapse"] for m in ms),
                                      "stuck10_0.65": sum(m["stuck10_0.65"] for m in ms),
                                      "plat_median": float(np.median([m["plat_0.65"] for m in ms])),
                                      "tl1_median": float(np.median([m["tl1"] for m in ms])),
                                      "tl_up_ep2_3": sum(m["tl_up_ep2_3"] for m in ms),
                                      "relapse": sum(m["relapse"] for m in ms)} for v, ms in osum.items()}
    # tương tác khối cũ: các cụm có đủ 4 biến thể
    inter_o = []
    for (k, fold), d in og.items():
        if all(v in d for v in names):
            p = lambda v: d[v]["plat_0.65"]
            inter_o.append(((d[(1, 1)]["run"], fold), (p((1, 1)) - p((1, 0))) - (p((0, 1)) - p((0, 0)))))
    res["old_interaction_plat_0.65"] = paired(inter_o, "old interaction")
    res["old_interaction_plat_0.65"]["runs"] = sorted({r for (r, _), _ in inter_o})

    # ASAM muộn (K15): fix_lateasam vs RA+ASAM gốc và chỉ RecAdam cùng checkpoint, cùng fold
    late = {}
    for (run, fold), m in old.items():
        if run.startswith("fix_lateasam"):
            late[(run, fold)] = m
    pairs_late = []
    for (run, fold), m in late.items():
        k = okey(m)
        base_key = k[:10] + (0,) + k[11:]          # cùng cấu hình nhưng sam_start_loss = 0
        base_key = base_key[:-2]
        d = og.get((base_key, fold), {})
        pairs_late.append({"run": run, "fold": fold, "late_plat": m["plat_0.65"], "late_roc": m["test_roc"],
                           "rasam_plat": d.get((1, 1), {}).get("plat_0.65"), "rasam_roc": d.get((1, 1), {}).get("test_roc"),
                           "raonly_plat": d.get((1, 0), {}).get("plat_0.65"), "noras_plat": d.get((0, 0), {}).get("plat_0.65"),
                           "rasam_run": d.get((1, 1), {}).get("run")})
    res["old_lateasam"] = pairs_late
    ok = [p for p in pairs_late if p["rasam_plat"] is not None]
    res["old_lateasam_paired"] = paired([((p["run"], p["fold"]), p["late_plat"] - p["rasam_plat"]) for p in ok], "late - rasam")
    ok2 = [p for p in pairs_late if p["raonly_plat"] is not None]
    res["old_lateasam_vs_raonly"] = paired([((p["run"], p["fold"]), p["late_plat"] - p["raonly_plat"]) for p in ok2], "late - raonly")

    # RecAdam t0/k quét (khối cũ, chỉ RecAdam, MWG common chỉ JS f1-f3)
    sweep = {}
    for (run, fold), m in old.items():
        if run.startswith(("chk_ra_", "chk_ep_recadamonly", "chk_ep_rasam")):
            sweep[f"{run}|f{fold}"] = {"plat_0.65": m["plat_0.65"], "tl1": m["tl1"], "tl2": m["tl2"], "test_roc": m["test_roc"],
                                       "t0": m["hp"]["anneal_t0_ratio"], "k": m["hp"]["anneal_k"], "sam_rho": m["hp"]["sam_rho"],
                                       "batch": m["hp"]["batch_size"], "warmup": m["hp"]["warmup_ratio"]}
    res["old_recadam_sweep"] = sweep

    # ----------------------------------------------------------------- lịch RecAdam theo bước thật
    res["schedule_new"] = recadam_schedule(456, 32, 30, 5.66e-5, 0.0, 0.01, 0.05, 500)
    res["schedule_old"] = recadam_schedule(456, 8, 30, 2e-5, 0.10, 0.01, 0.05, 500)

    with open(OUT, "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1, default=lambda o: o if not isinstance(o, (np.integer, np.floating, np.bool_)) else o.item())
    print("ghi", OUT)


if __name__ == "__main__":
    main()
