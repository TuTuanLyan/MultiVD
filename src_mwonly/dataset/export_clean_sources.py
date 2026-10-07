#!/usr/bin/env python3
"""Xuất thư mục NGUỒN CHUẨN từ data/sources_v4: chỉ dữ liệu đã giữ, mỗi dòng đủ trường, kèm README mô tả chính dữ liệu.

    <out>/by_source/<nguồn>/{full,common,4cwe}.jsonl
    <out>/merged/{full,common,4cwe}.jsonl               = primevul_paired + cleanvul_java + cleanvul_js
    <out>/merged_undersampled/{full,common,4cwe}.jsonl  = primevul_unpaired (undersample nhãn 0, seed 42) + java + js
    <out>/README.md                                     mọi con số đo trên chính các file vừa ghi

    python dataset/export_clean_sources.py [data/clean_sources_v4] [--src=data/sources_v4]
"""
import collections
import hashlib
import json
import os
import random
import re
import statistics
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import filters as FL                              # noqa: E402
from mwg.comments import remove_comments         # noqa: E402

ARGS = [x for x in sys.argv[1:] if not x.startswith("--src=")]
SRC = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--src=")), "data/sources_v4")
OUT = ARGS[0] if ARGS else "data/clean_sources_v4"
SOURCES = [("primevul_paired", "ccpp_primevul_paired", "C/C++", "PrimeVul v0.1 — bộ paired", "pair"),
         ("primevul_unpaired", "ccpp_primevul_unpaired", "C/C++", "PrimeVul v0.1 — bộ đầy đủ (unpaired)", "row"),
         ("cleanvul_java", "java_cleanvul_3-4", "Java", "CleanVul mức 3–4 (GitHub yikun-li/CleanVul @ cbad711)", "pair"),
         ("cleanvul_js", "js_cleanvul_3-4", "JavaScript", "CleanVul mức 3–4 (GitHub yikun-li/CleanVul @ cbad711)", "pair")]
MERGED = ["primevul_paired", "cleanvul_java", "cleanvul_js"]
UNDERSAMPLE_ABOVE = 10000     # phần C/C++ của merged_undersampled lớn hơn ngưỡng này thì undersample nhãn 0
LX = {"ccpp": "c", "java": "java", "js": "js"}
nows = lambda s: re.sub(r"\s+", "", s)
fmt = lambda n: f"{n:,}".replace(",", " ")


def load_set(base, tag):
    pj, pm = os.path.join(SRC, "%s_%s.jsonl" % (base, tag)), os.path.join(SRC, "%s_%s.meta.jsonl" % (base, tag))
    if not os.path.exists(pj):
        return None
    out = []
    for l, m in zip(open(pj, encoding="utf-8"), open(pm, encoding="utf-8")):
        r, m = json.loads(l), json.loads(m)
        d = {"id": m.get("row_id"), "pair_id": m.get("pair_id")}
        d.update({k: r[k] for k in ("code", "label", "lang", "cwe", "cwe_id", "cwe_class")})
        d.update({k: m.get(k) for k in ("cve", "project", "file_name", "commit_url")})
        if d.get("file_name") in ("None", ""):
            d["file_name"] = None
        out.append(d)
    return out


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_comment_free(args):
    code, lang = args
    return remove_comments(code, LX[lang]) == code


def compute_stats(rows, pool):
    labels = collections.Counter(r["label"] for r in rows)
    lang = collections.Counter(r["lang"] for r in rows)
    cwe = collections.Counter(r["cwe"] for r in rows if r["cwe"])
    pairs = collections.defaultdict(list)
    for r in rows:
        if r["pair_id"]:
            pairs[r["pair_id"]].append(r["label"])
    complete_pairs = sum(1 for v in pairs.values() if sorted(v) == [0, 1])
    lines = [r["code"].count("\n") + 1 for r in rows]
    # trùng lặp / mâu thuẫn trong file
    exact_labels = collections.defaultdict(set)
    for r in rows:
        exact_labels[nows(r["code"])].add(r["label"])
    label_conflicts = sum(1 for v in exact_labels.values() if len(v) > 1)
    exact_keys = collections.Counter((nows(r["code"]), r["label"]) for r in rows)
    exact_dups = sum(v - 1 for v in exact_keys.values() if v > 1)
    abstract_keys = collections.Counter((FL.alpha_norm(r["code"], LX[r["lang"]]), r["label"]) for r in rows)
    abstract_dups = sum(v - 1 for v in abstract_keys.values() if v > 1)
    comment_free = sum(pool.map(_is_comment_free, [(r["code"], r["lang"]) for r in rows], chunksize=500))
    return {"n": len(rows), "pos": labels.get(1, 0), "neg": labels.get(0, 0), "lang": dict(lang), "n_cwe": len(cwe),
            "top_cwe": cwe.most_common(6), "no_cwe": sum(1 for r in rows if not r["cwe"]), "pairs": len(pairs), "complete_pairs": complete_pairs,
            "median_lines": statistics.median(lines) if lines else 0, "max_lines": max(lines) if lines else 0,
            "label_conflicts": label_conflicts, "exact_dups": exact_dups, "abstract_dups": abstract_dups, "comment_free": comment_free,
            "with_file_name": sum(1 for r in rows if r.get("file_name")), "with_cve": sum(1 for r in rows if r.get("cve"))}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        sys.exit("đã tồn tại: %s — xoá trước khi xuất lại" % OUT)
    stats, files = {}, []
    data = {}
    with Pool(4) as pool:
        for name, base, *_ in SOURCES:
            for tag in ("full", "common", "4cwe"):
                rows = load_set(base, tag)
                if not rows:
                    continue
                data[(name, tag)] = rows
                p = os.path.join(OUT, "by_source", name, tag + ".jsonl"); write_jsonl(p, rows); files.append(p)
                stats[(name, tag)] = compute_stats(rows, pool)
                print(name, tag, stats[(name, tag)]["n"], flush=True)
        # merged_undersampled/<loại>: C/C++ từ PrimeVul unpaired. Phần C/C++ nhiều hơn UNDERSAMPLE_ABOVE hàng thì giữ MỌI
        # nhãn 1 và chọn NGẪU NHIÊN (seed 42) đúng bằng số đó nhãn 0, giữ thứ tự gốc; `4cwe` luôn giữ cả (người dùng chốt 23/09).
        # Cộng Java và JS cùng loại.
        for tag in ("full", "common", "4cwe"):
            unp = data[("primevul_unpaired", tag)]
            if tag != "4cwe" and len(unp) > UNDERSAMPLE_ABOVE:
                pos_idx = [i for i, r in enumerate(unp) if r["label"] == 1]
                neg_idx = [i for i, r in enumerate(unp) if r["label"] == 0]
                keep = set(pos_idx) | set(random.Random(42).sample(neg_idx, len(pos_idx)))
                unp = [r for i, r in enumerate(unp) if i in keep]
            rows = unp + data.get(("cleanvul_java", tag), []) + data.get(("cleanvul_js", tag), [])
            p = os.path.join(OUT, "merged_undersampled", tag + ".jsonl"); write_jsonl(p, rows); files.append(p)
            stats[("merged_undersampled", tag)] = compute_stats(rows, pool)
            print("merged_undersampled", tag, stats[("merged_undersampled", tag)]["n"], flush=True)
        for tag in ("full", "common", "4cwe"):
            rows = [r for name in MERGED for r in data.get((name, tag), [])]
            p = os.path.join(OUT, "merged", tag + ".jsonl"); write_jsonl(p, rows); files.append(p)
            stats[("merged", tag)] = compute_stats(rows, pool)
            print("merged", tag, stats[("merged", tag)]["n"], flush=True)
    write_readme(stats, files)


def write_readme(stats, files):
    L = []
    L.append("""# Nguồn dữ liệu chuẩn v4 — phát hiện lỗ hổng mức hàm (C/C++, Java, JavaScript)

Mỗi dòng là **một hàm**, nhãn `1` = hàm có lỗ hổng, `0` = hàm sạch. Mã đã được **xoá comment**, đã **khử trùng**
và **loại hàm lỗi nhãn / mã sinh tự động / hàm rỗng**. Thư mục chỉ chứa dữ liệu dùng được — không có bản bị loại.

## Dùng file nào

| mục đích | file |
|---|---|
| nguồn gộp C/C++ + Java + JS, **đủ mọi CWE** | `merged/full.jsonl` |
| nguồn gộp, chỉ các CWE **dùng chung được với Python** (đích Python) | `merged/common.jsonl` |
| nguồn gộp, chỉ 4 CWE **022, 078, 079, 089** | `merged/4cwe.jsonl` |
| như `merged/` nhưng C/C++ lấy từ **PrimeVul bộ đầy đủ** (unpaired), undersample cho cân bằng nhãn | `merged_undersampled/{full,common,4cwe}.jsonl` |
| một nguồn riêng | `by_source/<nguồn>/{full,common,4cwe}.jsonl` |

`merged/` = `by_source/primevul_paired` + `by_source/cleanvul_java` + `by_source/cleanvul_js` nối lại (đúng thứ tự đó).
`primevul_unpaired` **không** nằm trong `merged/` vì lệch nhãn mạnh (~3 % nhãn 1) — dùng riêng khi cần.
`merged_undersampled/<loại>.jsonl` = phần C/C++ lấy từ `by_source/primevul_unpaired/<loại>`, rồi nối `by_source/cleanvul_java/<loại>`
và `by_source/cleanvul_js/<loại>`. Phần C/C++ có hơn 10 000 hàm thì giữ MỌI hàm nhãn 1 và chọn NGẪU NHIÊN (seed 42) đúng bằng số đó
hàm nhãn 0, giữ thứ tự gốc — áp cho `full` và `common`; `4cwe` giữ cả phần C/C++ (không undersample, nên lệch nhãn).
Phần C/C++ ở đây là hàm đơn lẻ, không có cặp (`pair_id` = null); phần Java/JS vẫn đủ cặp.

- `full` ⊇ `common`; `4cwe` ⊆ `full`. Các bộ theo cặp luôn đủ cặp: mỗi `pair_id` có đúng một hàm nhãn 1 (bản lỗi) và một hàm nhãn 0 (bản đã vá cùng hàm).
- `common`: giữ hàm mà MỌI nhãn CWE của nó được luật MITRE CWE v4.20 xác định là áp dụng được cho Python.
  Java vào `common` nguyên vẹn vì CleanVul không gán CWE cho Java.

## Các nguồn
""")
    L.append("| nguồn (`by_source/…`) | ngôn ngữ | lấy từ | đơn vị |\n|---|---|---|---|")
    for name, base, language, origin, unit in SOURCES:
        L.append("| `%s` | %s | %s | %s |" % (name, language, origin, "cặp (bản lỗi + bản vá)" if unit == "pair" else "hàm đơn lẻ"))
    L.append("\n## Số mẫu\n")
    L.append("| file | số hàm | nhãn 1 / nhãn 0 | số cặp | ngôn ngữ | số CWE | hàm không có CWE | số dòng (trung vị / lớn nhất) |\n|---|---|---|---|---|---|---|---|")
    order = ([(t, g) for t, *_ in SOURCES for g in ("full", "common", "4cwe")] + [("merged", g) for g in ("full", "common", "4cwe")]
             + [("merged_undersampled", g) for g in ("full", "common", "4cwe")])
    for k in order:
        if k not in stats:
            continue
        s = stats[k]; name = (("%s/%%s" % k[0]) if k[0].startswith("merged") else "by_source/%s/%%s" % k[0]) % k[1]
        L.append("| `%s.jsonl` | %s | %s / %s | %s | %s | %s | %s | %s / %s |" % (
            name, fmt(s["n"]), fmt(s["pos"]), fmt(s["neg"]), fmt(s["complete_pairs"]) if s["pairs"] else "—",
            ", ".join("%s %s" % (a, fmt(b)) for a, b in sorted(s["lang"].items())), s["n_cwe"], fmt(s["no_cwe"]),
            int(s["median_lines"]), fmt(s["max_lines"])))
    L.append("\nCWE nhiều nhất (theo số hàm):\n")
    for k in order:
        if k in stats:
            name = (("%s/%%s" % k[0]) if k[0].startswith("merged") else "by_source/%s/%%s" % k[0]) % k[1]
            L.append("- `%s`: %s" % (name, ", ".join("%s %s" % (a, fmt(b)) for a, b in stats[k]["top_cwe"]) or "—"))
    L.append("""
## Trường của mỗi dòng

| trường | kiểu | ý nghĩa |
|---|---|---|
| `code` | chuỗi | mã nguồn của hàm, **đã xoá comment** |
| `label` | 0 / 1 | 1 = có lỗ hổng, 0 = sạch |
| `lang` | chuỗi | `ccpp` (C/C++), `java`, `js` |
| `cwe` | chuỗi | CWE chính dạng `CWE-079`; rỗng nếu nguồn không gán |
| `cwe_id` | số | số CWE chính; `-1` nếu không có |
| `cwe_class` | số | lớp pillar CWE-1000 của CWE chính (284, 435, 664, 682, 691, 693, 697, 703, 707, 710); `-100` nếu không tra được. Trong `4cwe`: chỉ số 0–3 theo thứ tự CWE-022, 078, 079, 089 |
| `id` | chuỗi | mã định danh hàm trong nguồn gốc (`primevul:<idx>`, `cleanvul:<mức>:<dòng>:<1/0>`) |
| `pair_id` | chuỗi / null | mã cặp; hai hàm cùng `pair_id` là bản lỗi và bản vá của cùng một hàm. `null` ở `primevul_unpaired` |
| `cve` | chuỗi / null | mã CVE nếu nguồn có |
| `project` | chuỗi / null | tên dự án (chỉ PrimeVul) |
| `file_name` | chuỗi / null | tên file chứa hàm (PrimeVul: một phần có, CleanVul: đường dẫn trong repo) |
| `commit_url` | chuỗi / null | commit sửa lỗi |

## Tình trạng dữ liệu
""")
    L.append("| file | comment còn lại | lỗi nhãn (mã giống hệt mang cả 0 và 1) | trùng nguyên văn cùng nhãn | trùng dạng abstract cùng nhãn |\n|---|---|---|---|---|")
    for k in order:
        if k in stats:
            s = stats[k]; name = (("%s/%%s" % k[0]) if k[0].startswith("merged") else "by_source/%s/%%s" % k[0]) % k[1]
            L.append("| `%s.jsonl` | %s | %s | %s | %s |" % (name, "0" if s["comment_free"] == s["n"] else "%s hàm" % fmt(s["n"] - s["comment_free"]),
                                                          fmt(s["label_conflicts"]), fmt(s["exact_dups"]), fmt(s["abstract_dups"])))
    L.append("""
- **Comment**: đã xoá hết comment `//`, `/* */` (C/C++, Java, JS) — chạy lại bộ xoá comment trên từng hàm không đổi ký tự nào
  (cột “comment còn lại”). Chuỗi chứa `//` như URL `"http://…"`, regex, `"*/"` được giữ nguyên vì là mã.
  **Chỉ thị tiền xử lý C** (`#define`, `#if`, `#include`…) được **giữ** — là một phần logic của hàm.
  Hàm dạng bản vá chỉ sửa comment (bản lỗi và bản vá giống hệt sau khi xoá comment) đã bị loại.
- **Trùng lặp**: so sau khi bỏ mọi khoảng trắng (“nguyên văn”) và sau khi đổi định danh → `ID1, ID2…`, chuỗi → `S`, số → `N`
  (“abstract”). Trong mỗi file không còn hai hàm cùng nhãn trùng nhau ở cả hai mức. Hai hàm **ngược nhãn** giống nhau ở dạng abstract
  được **giữ**: đó thường là bản vá thật chỉ đổi tên hàm/biến.
- **Đã loại**: mã giống hệt nhau mà mang cả nhãn 0 và 1 (lỗi nhãn); mã JavaScript sinh tự động / bị nén (webpack, babel, bundle,
  `.min.js` có nội dung nén, tên hàm 1–2 ký tự, mã một dòng); hàm không có thân (khai báo, thân rỗng, constructor chỉ có danh sách
  khởi tạo). Hàm bị cắt cụt thiếu `}` cuối (có trong PrimeVul) được **giữ**.
- **Gộp (`merged/`)**: `id`/`pair_id` không trùng giữa các nguồn (tiền tố `primevul:` / `cleanvul:`). Các cột tình trạng ở trên đo lại
  trên chính file gộp.
""")
    L.append("## File, số dòng, md5\n\n| file | số dòng | md5 |\n|---|---|---|")
    for p in files:
        L.append("| `%s` | %s | `%s` |" % (os.path.relpath(p, OUT), fmt(sum(1 for _ in open(p, encoding="utf-8"))), md5(p)))
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
