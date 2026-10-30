import argparse
import json
import math
import os
from fractions import Fraction

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT, "experiments.json"), encoding="utf-8") as handle:
    EXPERIMENTS = json.load(handle)
SEEDS = (42, 1234, 7)
FOLDS = (1, 2, 3, 4, 5)
T4 = stats.t.ppf(0.975, 4)
TARGET_CWES = ("CWE-022", "CWE-078", "CWE-079", "CWE-089")
METRICS = {"roc": "test_roc_auc", "pr": "test_pr_auc", "f1": "test_macro_f1_at_0.5"}
CACHE = {}
ROWS = {}


def test_rows(name, fold):
    path = os.path.join(ROOT, EXPERIMENTS[name]["flags"]["--data_root"], "fold%d" % fold, "test.jsonl")
    if path not in ROWS:
        with open(path, encoding="utf-8") as handle:
            ROWS[path] = [json.loads(line) for line in handle]
    return ROWS[path]


def load(results, name, seed, fold):
    key = (results, name, seed, fold)
    if key in CACHE:
        return CACHE[key]
    base = os.path.join(results, name, "seed_%d" % seed, "fold%d" % fold)
    cell = None
    if os.path.exists(base + ".json"):
        with open(base + ".json", encoding="utf-8") as handle:
            d = json.load(handle)
        cell = {m: float(d[k]) for m, k in METRICS.items()}
        cell["json"] = d
        cell["p"] = cell["y"] = cell["cwe"] = None
        if os.path.exists(base + ".probs.npz"):
            z = np.load(base + ".probs.npz")
            cell["p"], cell["y"] = z["probabilities"].astype(float), z["labels"].astype(int)
            rows = test_rows(name, fold)
            if len(rows) == len(cell["y"]) and all(int(r["label"]) == int(t) for r, t in zip(rows, cell["y"])):
                cell["cwe"] = np.array([str(r.get("cwe") or "") for r in rows])
    CACHE[key] = cell
    return cell


def exact_auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    greater = int((pos[:, None] > neg[None, :]).sum())
    equal = int((pos[:, None] == neg[None, :]).sum())
    return Fraction(2 * greater + equal, 2 * len(pos) * len(neg))


def values(results, name, metric, mask=None):
    out = {}
    for seed in SEEDS:
        for fold in FOLDS:
            cell = load(results, name, seed, fold)
            if cell is None:
                continue
            if mask is None:
                out[(seed, fold)] = (cell[metric], None)
                if metric == "roc" and cell["p"] is not None:
                    out[(seed, fold)] = (cell[metric], exact_auc(cell["p"], cell["y"]))
            elif cell["cwe"] is not None:
                m = mask(cell["cwe"])
                y, p = cell["y"][m], cell["p"][m]
                if y.min() != y.max():
                    out[(seed, fold)] = (float(roc_auc_score(y, p)), exact_auc(p, y))
    return out


def paired(a, b):
    keys = [k for k in a if k in b]
    if not keys:
        return None
    d = {k: a[k][0] - b[k][0] for k in keys}
    pos = 0
    for k in keys:
        if a[k][1] is not None and b[k][1] is not None:
            pos += a[k][1] - b[k][1] > 0
        else:
            pos += d[k] > 0
    fold_means = [np.mean([d[(s, f)] for s in SEEDS if (s, f) in d]) for f in FOLDS if any((s, f) in d for s in SEEDS)]
    mean = float(np.mean(list(d.values())))
    ci = None
    if len(fold_means) == 5:
        h = T4 * float(np.std(fold_means, ddof=1)) / math.sqrt(5)
        ci = (float(np.mean(fold_means)) - h, float(np.mean(fold_means)) + h)
    return {"delta": mean, "ci": ci, "pos": pos, "n": len(keys), "folds_pos": sum(v > 0 for v in fold_means),
            "rel": mean / float(np.mean([b[k][0] for k in keys]))}


def signed(x):
    return ("+" if x >= 0 else "-") + "%.3f" % abs(x)


def delta_text(r):
    if r is None:
        return "n/a"
    ci = "[%s, %s]" % (signed(r["ci"][0]), signed(r["ci"][1])) if r["ci"] else "[n/a]"
    return "%s %s %d/%d (%.1f%%)" % (signed(r["delta"]), ci, r["pos"], r["n"], 100 * r["rel"])


def mean_text(v):
    return "%.3f" % np.mean([x[0] for x in v.values()]) if v else "n/a"


def table(results, title, groups, metrics=("roc", "pr", "f1")):
    print("\n## %s" % title)
    print("| Method | n | " + " | ".join(m.upper() for m in metrics) + " | SD ROC | dROC [95% CI] Pos (rel) |")
    print("|---" * (len(metrics) + 4) + "|")
    for group, ref, rows in groups:
        print("| *%s* |" % group + " |" * (len(metrics) + 3))
        ref_roc = values(results, ref, "roc")
        for label, name in rows:
            v = {m: values(results, name, m) for m in metrics}
            roc = [x[0] for x in values(results, name, "roc").values()]
            sd = "%.3f" % np.std(roc, ddof=1) if len(roc) > 1 else "n/a"
            dl = "reference" if name == ref else delta_text(paired(values(results, name, "roc"), ref_roc))
            print("| %s | %d | %s | %s | %s |" % (label, len(roc), " | ".join(mean_text(v[m]) for m in metrics), sd, dl))


def pair_lines(results, title, items):
    print("\n## %s" % title)
    for label, a, b, metrics in items:
        for m in metrics:
            print("- %s, %s: %s" % (label, m.upper(), delta_text(paired(values(results, a, m), values(results, b, m)))))


def cwe_table(results, title, groups, masks):
    print("\n## %s" % title)
    print("| CWE | " + " | ".join(label for label, _ in groups[0][1]) + " | dROC (GuideVul - Baseline) [95% CI] Pos |")
    print("|---" * (len(groups[0][1]) + 2) + "|")
    for group, rows in groups:
        print("| *%s* |" % group + " |" * (len(rows) + 1))
        for cwe_label, mask in masks:
            v = [values(results, name, "roc", mask) for _, name in rows]
            print("| %s | %s | %s |" % (cwe_label, " | ".join(mean_text(x) for x in v), delta_text(paired(v[-1], v[0]))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(ROOT, "results"))
    r = parser.parse_args().results
    cb = [("Baseline", "baseline_codebert"), ("Multi-window, target only", "multiwindow_codebert"), ("MVD-joint", "mvd_joint"),
          ("MVD-inc", "mvd_inc"), ("BABEL, joint", "babel_joint"), ("BABEL, sequential", "babel_sequential"),
          ("GuideVul (Applicable)", "guidevul_codebert_applicable")]
    t5 = [("Baseline", "baseline_codet5"), ("Multi-window, target only", "multiwindow_codet5"), ("MulVuln", "mulvuln"),
          ("GuideVul (Applicable)", "guidevul_codet5_applicable")]
    table(r, "RQ1: detection performance on the Python target", [("CodeBERT", "baseline_codebert", cb), ("CodeT5", "baseline_codet5", t5)])
    pair_lines(r, "RQ1: GuideVul against MVD-joint", [("GuideVul - MVD-joint", "guidevul_codebert_applicable", "mvd_joint", ("roc", "pr", "f1"))])
    levels = lambda plm: [("Full", "guidevul_%s_full" % plm), ("Applicable", "guidevul_%s_applicable" % plm),
                          ("Target-CWE", "guidevul_%s_targetcwe" % plm)]
    table(r, "RQ2: screening levels of the JavaScript source",
          [("CodeBERT", "guidevul_codebert_full", levels("codebert")), ("CodeT5", "guidevul_codet5_full", levels("codet5"))])
    pair_lines(r, "RQ2: screening levels against the baseline", [
        ("CodeT5 Full - Baseline", "guidevul_codet5_full", "baseline_codet5", ("roc",)),
        ("CodeT5 Target-CWE - Baseline", "guidevul_codet5_targetcwe", "baseline_codet5", ("roc",))])
    table(r, "RQ2: JavaScript and C/C++ sources (CodeBERT)", [("CodeBERT", "baseline_codebert", [
        ("Baseline (no source)", "baseline_codebert"), ("JavaScript", "guidevul_codebert_applicable"), ("C/C++", "guidevul_codebert_cpp")])])
    pair_lines(r, "RQ2: JavaScript against C/C++", [
        ("JavaScript - C/C++", "guidevul_codebert_applicable", "guidevul_codebert_cpp", ("roc", "f1")),
        ("C/C++ - Baseline", "guidevul_codebert_cpp", "baseline_codebert", ("f1",))])
    table(r, "RQ3: comparison with ordinary two-stage training (CodeBERT)", [("CodeBERT", "twostage_codebert", [
        ("Baseline", "baseline_codebert"), ("Ordinary two-stage", "twostage_codebert"), ("GuideVul (Applicable)", "guidevul_codebert_applicable")])])
    pair_lines(r, "RQ3: paired gains", [
        ("GuideVul - Ordinary two-stage", "guidevul_codebert_applicable", "twostage_codebert", ("pr", "f1")),
        ("Ordinary two-stage - Baseline", "twostage_codebert", "baseline_codebert", ("roc",))])
    one = lambda c: (lambda w: w == c)
    cwe_table(r, "RQ4: ROC-AUC on the test samples of each target CWE", [
        ("CodeBERT", [("Baseline", "baseline_codebert"), ("Multi-window", "multiwindow_codebert"), ("GuideVul", "guidevul_codebert_applicable")]),
        ("CodeT5", [("Baseline", "baseline_codet5"), ("Multi-window", "multiwindow_codet5"), ("GuideVul", "guidevul_codet5_applicable")])],
        [(c, one(c)) for c in TARGET_CWES])
    table(r, "RQ5: JavaScript target with the Python source (CodeBERT)", [("CodeBERT", "baseline_codebert_javascript", [
        ("Baseline", "baseline_codebert_javascript"), ("Multi-window, target only", "multiwindow_codebert_javascript"),
        ("GuideVul", "guidevul_codebert_javascript")])])
    pair_lines(r, "RQ5: paired gains", [
        ("GuideVul - Baseline", "guidevul_codebert_javascript", "baseline_codebert_javascript", ("pr", "f1")),
        ("GuideVul - Multi-window", "guidevul_codebert_javascript", "multiwindow_codebert_javascript", ("roc", "pr"))])
    present = lambda w: np.isin(w, TARGET_CWES)
    absent = lambda w: (~np.isin(w, TARGET_CWES)) & (w != "")
    cwe_table(r, "RQ5: ROC-AUC grouped by source CWE coverage", [
        ("CodeBERT", [("Baseline", "baseline_codebert_javascript"), ("GuideVul", "guidevul_codebert_javascript")])],
        [("Present", present), ("Absent", absent)])
    table(r, "Threats: BABEL learning rate", [
        ("joint", "babel_joint_lr5e4", [("lr 5e-4 (original)", "babel_joint_lr5e4"), ("lr 1e-4", "babel_joint")]),
        ("sequential", "babel_sequential_lr5e4", [("lr 5e-4 (original)", "babel_sequential_lr5e4"), ("lr 1e-4", "babel_sequential")])])
    print("\n## Threats: MulVuln lambda, mean validation ROC-AUC on folds 1-3 (seed 42)")
    for label, name in (("0.01", "mulvuln"), ("0.03", "mulvuln_lambda_0p03"), ("0.1", "mulvuln_lambda_0p1"), ("0.3", "mulvuln_lambda_0p3")):
        cells = [load(r, name, 42, f) for f in (1, 2, 3)]
        score = "n/a" if None in cells else "%.4f" % np.mean([c["json"]["best_val_selection_score"] for c in cells])
        print("- lambda %s: %s" % (label, score))


if __name__ == "__main__":
    main()
