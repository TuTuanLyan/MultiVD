#!/usr/bin/env python3
"""Bang ket qua cho khoi chay o 161 — BON chi so + dem dau fold + per-CWE.

Ghep cap theo FOLD (CLAUDE.md muc 2: khong bao gio lay hieu hai trung binh).
CHI ghep cac o chay tren CUNG MOT MAY — doc `hardware.gpu_name` tu file resource va
TU CHOI ghep neu khac, vi san nhieu cross-GPU la 0,028 so voi 0,010 cung loai.

    python3 tools/report161.py --a p2_v3common_2sven --b baseline161_codebert_py
    python3 tools/report161.py --list
"""
import argparse, glob, json, os, re, statistics, sys

M4 = [("test_macro_f1_at_0.5", "F1@0.5"), ("test_macro_f1_at_valcal", "F1@val"),
      ("test_roc_auc", "ROC-AUC"), ("test_pr_auc", "PR-AUC")]
ROOT = "clean_local/results"

def cells(tag):
    out = {}
    for p in glob.glob(os.path.join(ROOT, tag, "**", "fold*.json"), recursive=True):
        if not re.search(r"fold(\d+)\.json$", p):
            continue
        f = int(re.search(r"fold(\d+)\.json$", p).group(1))
        try:
            out[f] = json.load(open(p))
        except Exception:
            pass
    return out

def gpu_of(tag):
    for p in glob.glob(os.path.join(ROOT, tag, "**", "*resource*.json"), recursive=True):
        try:
            return (json.load(open(p)).get("hardware") or {}).get("gpu_name")
        except Exception:
            pass
    return None

def table(tag):
    c = cells(tag)
    if not c:
        return None
    print("  %-34s n=%d  GPU=%s" % (tag, len(c), gpu_of(tag) or "?"))
    for k, lab in M4:
        v = [c[f][k] for f in sorted(c) if k in c[f]]
        if v:
            print("      %-8s %.4f   (min %.4f  max %.4f)" % (lab, statistics.mean(v), min(v), max(v)))
    return c

def paired(A, B, na, nb):
    ks = sorted(set(A) & set(B))
    ga, gb = gpu_of(na), gpu_of(nb)
    print("\n=== %s  −  %s   (ghep cap theo fold, n=%d) ===" % (na, nb, len(ks)))
    if ga and gb and ga != gb:
        print("  !! TU CHOI GHEP: hai o chay tren GPU KHAC LOAI (%s vs %s)." % (ga, gb))
        print("     San nhieu cross-GPU la 0,028 — moi hieu so duoi nguong do khong doc duoc.")
        print("     Phai chay lai mot trong hai o tren cung may truoc khi so.")
        return
    for k, lab in M4:
        d = [A[f][k] - B[f][k] for f in ks if k in A[f] and k in B[f]]
        if not d:
            continue
        pos = sum(1 for x in d if x > 0)
        flag = "" if abs(statistics.mean(d)) >= 0.010 else "   <- DUOI san nhieu 0,010"
        print("  %-8s %+.4f  %d/%d fold duong | min %+.4f max %+.4f%s"
              % (lab, statistics.mean(d), pos, len(d), min(d), max(d), flag))

def percwe(A, B, na, nb):
    ks = sorted(set(A) & set(B))
    cw = sorted({c for f in ks for c in (A[f].get("per_cwe") or {})})
    if not cw:
        return
    print("\n=== PER-CWE, ghep cap theo fold (n=%d) ===" % len(ks))
    print("  %-10s %5s | %-22s | %-22s" % ("cwe", "n TB", "ROC-AUC", "F1@0.5"))
    for c in cw:
        row = []
        for key in ("roc_auc", "macro_f1_at_0.5"):
            d = [A[f]["per_cwe"][c][key] - B[f]["per_cwe"][c][key] for f in ks
                 if c in (A[f].get("per_cwe") or {}) and c in (B[f].get("per_cwe") or {})
                 and A[f]["per_cwe"][c].get(key) is not None and B[f]["per_cwe"][c].get(key) is not None]
            row.append(("%+.4f  %d/%d" % (statistics.mean(d), sum(1 for x in d if x > 0), len(d))) if d else "khong du lop")
        n = [A[f]["per_cwe"][c]["n"] for f in ks if c in (A[f].get("per_cwe") or {})]
        print("  %-10s %5.0f | %-22s | %-22s" % (c, statistics.mean(n) if n else 0, row[0], row[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--list", action="store_true")
    x = ap.parse_args()
    if x.list or not x.a:
        print("Cac o co trong %s:" % ROOT)
        for d in sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []:
            if cells(d):
                table(d)
        return
    A = table(x.a)
    if x.b:
        B = table(x.b)
        if A and B:
            paired(A, B, x.a, x.b); percwe(A, B, x.a, x.b)

if __name__ == "__main__":
    main()
