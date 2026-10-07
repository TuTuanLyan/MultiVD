#!/usr/bin/env python3
"""ROC của checkpoint Pha 1 trộn JS:Py 4:1 tách theo ngôn ngữ trên val Pha 1 (JS val + SVEN val fold k) - chấm dự đoán J1/J2.
Đọc results/p1_jspy41_<mức>/fold<k>.probs.npz (thứ tự = test.jsonl = bản sao val) và data/mwonly5_sources/js_py41_<mức>/fold<k>/test.jsonl."""
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(BASE, "..", "data", "mwonly5_sources")
for lvl in ("common", "full"):
    for k in range(1, 6):
        p = os.path.join(BASE, "results", "p1_jspy41_%s" % lvl, "fold%d.probs.npz" % k)
        if not os.path.exists(p):
            continue
        z = np.load(p)
        rows = [json.loads(l) for l in open(os.path.join(DATA, "js_py41_%s" % lvl, "fold%d" % k, "test.jsonl"), encoding="utf-8")]
        y = np.array([r["label"] for r in rows]); lang = np.array([r["lang"] for r in rows])
        assert (z["labels"] == y).all(), "thứ tự nhãn lệch ở %s f%d" % (lvl, k)
        pr = z["probabilities"]
        print("%s f%d | val trộn %.4f | JS (%d) %.4f | Python SVEN val (%d) %.4f" % (
            lvl, k, roc_auc_score(y, pr), (lang == "js").sum(), roc_auc_score(y[lang == "js"], pr[lang == "js"]),
            (lang == "python").sum(), roc_auc_score(y[lang == "python"], pr[lang == "python"])))
