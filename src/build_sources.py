#!/usr/bin/env python3
"""Dung lai ba nguon Phase 1 tu BO GOC, co kiem tra cat 512 token.

Ly do phai lam lai chu khong vá file cu
-------------------------------------
RevisitVD (arXiv:2507.16887) do duoc 27% cap cua PrimeVul tro thanh MAU THUAN
NHAN duoi cua so 512 token: ham loi va ham da va bi cat thanh hai chuoi token
GIONG HET nhau nhung mang nhan nguoc nhau. Do tren chinh du lieu cua repo nay
(tokenizer codebert-base) con te hon: 35.5% tren toan bo PrimeVul C/C++ ghep cap,
va 32.9% tren dung file `data/train_ccpp_filtered.jsonl` ma Phase 1 dang doc.
Tuc gan mot phan ba tin hieu Phase 1 phia C/C++ la nhieu nhan thuan tuy — lon
hon han moi hieu ung lambda / head phu ma du an dang do (~0.028 Macro-F1).

Hai loi nua phat hien khi ra soat:

1. `data/train_js_filtered.jsonl` ghep cap theo THU TU DONG, va 37/569 cap ke
   nhau khong phai cap that (khac CWE, do tuong dong trung vi 0.41 so voi 0.945
   cua cap that). Script nay ghep lai bang noi dung.
2. Schema chuan hoa cu (`code, cwe, cwe_class, cwe_id, label, lang`) KHONG co
   khoa ghep cap, nen sau khi chuan hoa la mat thong tin cap. Script nay giu
   `pair_id` de `build_folds.py` chia fold theo nhom duoc.

Nguyen tac: khong sua file dau vao, luon ghi ra duong dan moi.

Chinh sach cat
--------------
Moi hang duoc gan `n_tok_max` = do dai token lon nhat qua CAC tokenizer that su
dung trong thi nghiem (ho RoBERTa: codebert-base; ho T5: codet5-base,
codet5p-110m-embedding). Mot cap bi danh dau `collapse=True` neu hai ve tro
thanh chuoi token giong het nhau sau khi cat o 512 duoi BAT KY tokenizer nao.

`--drop collapse` (mac dinh) bo cac cap sap — giu duoc nhieu du lieu nhat ma van
xoa het mau thuan nhan.
`--drop truncated` chat hon: chi giu cap ma ca hai ve deu lot tron trong 512
token, tuc mo hinh khong bao gio nhin thay ham bi cat.
`--drop none` giu tat ca, chi gan co — de con ablate duoc.
"""

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

PRIMEVUL_RAW = Path("/home/ntat/workspace/Archive/PrimeVulRaw")
FOUR_CWE = {"CWE-22": 0, "CWE-78": 1, "CWE-79": 2, "CWE-89": 3}
TOKENIZERS = [
    "microsoft/codebert-base",          # ho RoBERTa BPE — dung cho codebert + unixcoder
    "Salesforce/codet5-base",
    "Salesforce/codet5p-110m-embedding",
]


def norm_cwe(raw):
    """'[\'CWE-119\']' hay 'CWE-078' -> 'CWE-119' / 'CWE-78'. Nhieu CWE -> lay cai dau."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.startswith("["):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    m = re.match(r"CWE-0*(\d+)$", str(raw).strip().upper()) if raw else None
    return "CWE-%s" % m.group(1) if m else None


def load_cwe_list(path):
    out = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(norm_cwe(line) or line.upper())
    return out


def read_primevul_pairs():
    """Doc 3 file *_paired, ghep cap bang commit_id chu KHONG tin thu tu dong.

    Bo nho ghi ro: ~0.4% cap trong PrimeVul v0.1 co hai dong ke nhau doi nhan
    nhung KHONG chung commit_id — la loi dung file, khong duoc tin adjacency.
    """
    pairs, skipped = [], Counter()
    for split in ("train", "valid", "test"):
        rows = []
        with open(PRIMEVUL_RAW / ("primevul_%s_paired.jsonl" % split)) as fh:
            for line in fh:
                rows.append(json.loads(line))
        i = 0
        while i < len(rows) - 1:
            a, b = rows[i], rows[i + 1]
            same_commit = a.get("commit_id") == b.get("commit_id")
            alt_label = str(a.get("target")) != str(b.get("target"))
            if same_commit and alt_label:
                vul, safe = (a, b) if str(a["target"]) == "1" else (b, a)
                pairs.append((vul, safe, split))
                i += 2
            else:
                skipped["khong chung commit_id" if not same_commit else "khong doi nhan"] += 1
                i += 1
    return pairs, skipped


def read_cleanvul_js(csv_paths, ext="js"):
    """Doc CleanVul CSV (ban GitHub) -> cap (func_before=loi, func_after=da va).

    Moi DONG cua CleanVul da la mot cap san, nen khong phai doan ghep cap gi ca —
    day la ly do dung thang CSV goc tot hon moi ban dan xuat dang co tren may.
    Bo dong khong co cwe_id, va dong co hai ve giong het nhau.
    """
    import csv as _csv
    _csv.field_size_limit(10 ** 9)
    pairs, drop = [], Counter()
    for p in csv_paths:
        score = Path(p).stem.split("_")[-1]
        for r in _csv.DictReader(open(p, newline="", encoding="utf-8")):
            if str(r.get("extension", "")).lower() != ext:
                continue
            if str(r.get("is_test", "")).strip().lower() == "true":
                drop["is_test"] += 1
                continue
            cwe = norm_cwe(r.get("cwe_id"))
            if cwe is None:
                drop["khong co cwe_id"] += 1
                continue
            before, after = r.get("func_before") or "", r.get("func_after") or ""
            if not before.strip() or not after.strip():
                drop["thieu mot ve"] += 1
                continue
            if before.strip() == after.strip():
                drop["hai ve giong het"] += 1
                continue
            key = r.get("commit_url") or r.get("cve_id") or ""
            pairs.append((
                {"code": before, "cwe": cwe, "commit_id": "%s|%s" % (key, hashlib.md5(before.encode()).hexdigest()[:8])},
                {"code": after, "cwe": cwe},
                "score%s" % score,
            ))
    return pairs, drop


def read_js_rows(path):
    return [json.loads(l) for l in open(path)]


def repair_js_pairs(rows, min_sim=0.6):
    """Dung lai cap JS bang noi dung trong cung CWE, thay vi tin thu tu dong."""
    codes = [r["code"] for r in rows]
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(norm_cwe(r.get("cwe")), str(r.get("label")))].append(i)
    used, pairs, orphans = set(), [], []
    for (cwe, lab) in list(by):
        if lab != "1":
            continue
        cands = by.get((cwe, "0"), [])
        for i in by[(cwe, "1")]:
            if i in used:
                continue
            best, best_s = None, 0.0
            for j in cands:
                if j in used:
                    continue
                s = difflib.SequenceMatcher(None, codes[i], codes[j]).quick_ratio()
                if s > best_s:
                    best_s, best = s, j
            if best is not None and best_s >= min_sim:
                used.add(i)
                used.add(best)
                pairs.append((rows[i], rows[best], "cleanvul"))
    orphans = [rows[i] for i in range(len(rows)) if i not in used]
    return pairs, orphans


def token_profile(codes):
    """-> (n_tok_max theo tung code, prefix512 theo tung tokenizer)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from transformers import AutoTokenizer

    n = len(codes)
    n_tok_max = [0] * n
    prefixes = []
    for name in TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(name)
        pre = [None] * n
        for s in range(0, n, 256):
            enc = tok(codes[s:s + 256], truncation=False, add_special_tokens=True)["input_ids"]
            for k, ids in enumerate(enc):
                idx = s + k
                n_tok_max[idx] = max(n_tok_max[idx], len(ids))
                pre[idx] = hashlib.md5(str(ids[:512]).encode()).hexdigest()
        prefixes.append(pre)
        print("    tokenizer %-38s xong" % name, flush=True)
    return n_tok_max, prefixes


def build(pairs, lang, cwe_filter, name, drop, out_dir):
    """pairs = [(vul_row, safe_row, split)]. cwe_filter = None | set."""
    sel = []
    for vul, safe, split in pairs:
        cwe = norm_cwe(vul.get("cwe"))
        if cwe is None:
            continue
        if cwe_filter is not None and cwe not in cwe_filter:
            continue
        sel.append((vul, safe, split, cwe))

    codes = []
    for vul, safe, _, _ in sel:
        codes.append(vul.get("func") or vul.get("code"))
        codes.append(safe.get("func") or safe.get("code"))
    if not codes:
        print("!! %s: 0 cap sau khi loc CWE" % name)
        return
    print("  %s: %d cap -> do token..." % (name, len(sel)), flush=True)
    n_tok_max, prefixes = token_profile(codes)

    out_rows, stat = [], Counter()
    for p, (vul, safe, split, cwe) in enumerate(sel):
        ia, ib = 2 * p, 2 * p + 1
        collapse = any(pre[ia] == pre[ib] for pre in prefixes)
        truncated = n_tok_max[ia] > 512 or n_tok_max[ib] > 512
        stat["cap"] += 1
        stat["cap_sap"] += collapse
        stat["cap_bi_cat"] += truncated
        if drop == "collapse" and collapse:
            stat["bo_vi_sap"] += 1
            continue
        if drop == "truncated" and truncated:
            stat["bo_vi_cat"] += 1
            continue
        pid = "%s:%s" % (name, vul.get("commit_id") or hashlib.md5(codes[ia].encode()).hexdigest()[:12])
        for row, idx, label in ((vul, ia, 1), (safe, ib, 0)):
            out_rows.append({
                "code": codes[idx],
                "label": label,
                "cwe": cwe,
                "cwe_id": cwe.split("-")[1],
                "cwe_class": FOUR_CWE.get(cwe) if cwe_filter == set(FOUR_CWE) else None,
                "lang": lang,
                "pair_id": pid,
                "split_goc": split,
                "n_tok_max": n_tok_max[idx],
                "bi_cat": n_tok_max[idx] > 512,
            })
        stat["giu"] += 1

    out = Path(out_dir) / ("src_%s.jsonl" % name)
    with open(out, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    cw = Counter(r["cwe"] for r in out_rows)
    print("  -> %-28s %4d hang / %3d cap giu | sap %d (%.1f%%) | bi cat %d | %d CWE"
          % (out.name, len(out_rows), stat["giu"], stat["cap_sap"],
             100 * stat["cap_sap"] / max(1, stat["cap"]), stat["cap_bi_cat"], len(cw)))
    print("     top CWE:", dict(cw.most_common(6)), flush=True)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=("ccpp", "js", "cleanvul", "all"), default="all")
    ap.add_argument("--cleanvul_csv", nargs="*", default=[],
                    help="CSV CleanVul ban GitHub, thuong la vulnerability_score_3.csv va _4.csv")
    ap.add_argument("--drop", choices=("collapse", "truncated", "none"), default="collapse")
    ap.add_argument("--js_path", default="data/train_js_filtered.jsonl")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    if args.which in ("ccpp", "all"):
        print("== PrimeVul C/C++ tu %s ==" % PRIMEVUL_RAW)
        pairs, skipped = read_primevul_pairs()
        print("  %d cap hop le | bo qua: %s" % (len(pairs), dict(skipped)))
        ccpp_py = load_cwe_list("cwe_ccpp_py.txt")
        build(pairs, "ccpp", set(FOUR_CWE), "ccpp_4cwe", args.drop, args.out_dir)
        build(pairs, "ccpp", ccpp_py, "ccpp_common", args.drop, args.out_dir)
        build(pairs, "ccpp", None, "ccpp_full", args.drop, args.out_dir)

    if args.which == "cleanvul":
        print("== CleanVul JS tu CSV goc: %s ==" % ", ".join(args.cleanvul_csv))
        pairs, drop = read_cleanvul_js(args.cleanvul_csv)
        print("  %d cap JS | bo qua: %s" % (len(pairs), dict(drop)))
        js_py = load_cwe_list("cwe_js_py.txt")
        build(pairs, "js", None, "js_full", args.drop, args.out_dir)
        build(pairs, "js", js_py, "js_common", args.drop, args.out_dir)
        build(pairs, "js", set(FOUR_CWE), "js_4cwe", args.drop, args.out_dir)

    if args.which in ("js", "all"):
        print("\n== CleanVul JS tu %s ==" % args.js_path)
        rows = read_js_rows(args.js_path)
        pairs, orphans = repair_js_pairs(rows)
        print("  %d hang -> %d cap dung lai duoc, %d hang mo coi (bi loai)"
              % (len(rows), len(pairs), len(orphans)))
        build(pairs, "js", set(FOUR_CWE), "js_4cwe", args.drop, args.out_dir)
        js_py = load_cwe_list("cwe_js_py.txt")
        build(pairs, "js", js_py, "js_common", args.drop, args.out_dir)


if __name__ == "__main__":
    main()
