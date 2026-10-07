"""Xuất thư mục NGUỒN CHUẨN: chỉ dữ liệu đã giữ, mỗi dòng đủ trường, kèm README mô tả chính dữ liệu.

    <out>/by_source/<nguồn>/{full,common,4cwe}.jsonl
    <out>/merged/{full,common,4cwe}.jsonl          = primevul_paired + cleanvul_java + cleanvul_js
    <out>/by_source/primevul_merged_undersampled/  = PrimeVul paired + unpaired gộp, lọc trùng, undersample full trước
    <out>/merged_undersampled/{full,common,4cwe}.jsonl = primevul_merged_undersampled + cleanvul_java + cleanvul_js
    <out>/README.md
Mọi con số trong README đo trên chính các file vừa ghi.
"""
import collections, hashlib, json, os, random, re, statistics, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from strip_comments import remove_comments       # noqa: E402
import clean_v3 as C3                             # noqa: E402

SRC = "data/sources_v4"
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/clean_sources_v4"
SOURCES = [("primevul_paired", "ccpp_primevul_paired", "C/C++", "PrimeVul v0.1 — bộ paired", "pair"),
         ("primevul_unpaired", "ccpp_primevul_unpaired", "C/C++", "PrimeVul v0.1 — bộ đầy đủ (unpaired)", "row"),
         ("cleanvul_java", "java_cleanvul_3-4", "Java", "CleanVul mức 3–4 (GitHub yikun-li/CleanVul @ cbad711)", "pair"),
         ("cleanvul_js", "js_cleanvul_3-4", "JavaScript", "CleanVul mức 3–4 (GitHub yikun-li/CleanVul @ cbad711)", "pair")]
MERGED = ["primevul_paired", "cleanvul_java", "cleanvul_js"]
MERGED_C = "ccpp_primevul_merged"     # sources_v4: PrimeVul paired + unpaired gộp, lọc T0–T4 trên toàn bộ
UNDERSAMPLE_ABOVE = 10000     # `common` phần C/C++ (sau khi lọc từ `full` đã undersample) còn hơn ngưỡng này thì undersample tiếp
IMBALANCE_ABOVE = 3           # `4cwe` phần C/C++ lệch nhãn hơn 3:1 thì undersample tiếp về 1:1
WORKERS = int(os.environ.get("EXPORT_WORKERS", "3"))
KEEP_META = ("pair_id", "row_id", "cve", "project", "file_name", "commit_url")
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


_FEAT = {}                    # md5(lang, code) -> (không còn comment, md5 dạng abstract): mỗi đoạn mã chỉ tính MỘT lần


def _features(args):
    code, lang = args
    return remove_comments(code, LX[lang]) == code, hashlib.md5(C3.alpha_norm(code, LX[lang]).encode("utf-8")).hexdigest()


def features(rows, pool):
    keys = [hashlib.md5((r["lang"] + "\0" + r["code"]).encode("utf-8")).digest() for r in rows]
    todo = {}
    for k, r in zip(keys, rows):
        if k not in _FEAT and k not in todo:
            todo[k] = (r["code"], r["lang"])
    for k, v in zip(todo, pool.map(_features, list(todo.values()), chunksize=200)):
        _FEAT[k] = v
    return [_FEAT[k] for k in keys]


def undersample(rows, seed=42):
    """Về 1:1: giữ mọi hàm của nhãn thiểu số và mọi hàm thuộc một cặp; bốc ngẫu nhiên (seed) hàm LẺ của nhãn đa số,
    giữ thứ tự gốc. Cặp luôn đủ hai nửa nên không cặp nào bị cắt."""
    lab = collections.Counter(r["label"] for r in rows)
    big = 0 if lab[0] >= lab[1] else 1
    fixed = [i for i, r in enumerate(rows) if r["label"] != big or r["pair_id"]]
    loose = [i for i, r in enumerate(rows) if r["label"] == big and not r["pair_id"]]
    need = lab[1 - big] - sum(1 for i in fixed if rows[i]["label"] == big)
    if not 0 <= need <= len(loose):
        raise ValueError("không undersample được về 1:1 (cần %d hàm lẻ, có %d)" % (need, len(loose)))
    keep = set(fixed) | set(random.Random(seed).sample(loose, need))
    return [r for i, r in enumerate(rows) if i in keep]


def assert_complete_pairs(rows):
    labs = collections.defaultdict(list)
    for r in rows:
        if r["pair_id"]:
            labs[r["pair_id"]].append(r["label"])
    bad = [p for p, v in labs.items() if sorted(v) != [0, 1]]
    assert not bad, "cặp không đủ hai nửa: %s" % bad[:5]


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
    feats = features(rows, pool)
    abstract_keys = collections.Counter((f[1], r["label"]) for f, r in zip(feats, rows))
    abstract_dups = sum(v - 1 for v in abstract_keys.values() if v > 1)
    comment_free = sum(f[0] for f in feats)
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
    with Pool(WORKERS) as pool:
        for name, base, *_ in SOURCES:
            for tag in ("full", "common", "4cwe"):
                rows = load_set(base, tag)
                if not rows:
                    continue
                data[(name, tag)] = rows
                p = os.path.join(OUT, "by_source", name, tag + ".jsonl"); write_jsonl(p, rows); files.append(p)
                stats[(name, tag)] = compute_stats(rows, pool)
                print(name, tag, stats[(name, tag)]["n"], flush=True)
        # by_source/primevul_merged_undersampled/<loại> (người dùng chốt 24/09): PrimeVul paired + unpaired GỘP, lọc T0–T4 trên
        # toàn bộ (sources_v4/ccpp_primevul_merged_*), rồi undersample `full` TRƯỚC khi lọc: giữ mọi hàm nhãn 1 và mọi cặp đủ,
        # bốc ngẫu nhiên (seed 42) hàm nhãn 0 LẺ cho tới 1:1. `common` / `4cwe` = hàng của bộ gộp cùng loại nằm trong `full` đã
        # undersample; `common` còn hơn UNDERSAMPLE_ABOVE hàm, hoặc `4cwe` lệch nhãn hơn IMBALANCE_ABOVE:1, thì undersample tiếp.
        cfull = undersample(load_set(MERGED_C, "full"))
        kept_ids = {r["id"] for r in cfull}
        assert len(kept_ids) == len(cfull), "id trùng trong bộ gộp"
        csets = {"full": cfull}
        for tag in ("common", "4cwe"):
            rows = [r for r in load_set(MERGED_C, tag) if r["id"] in kept_ids]
            lab = collections.Counter(r["label"] for r in rows)
            hi, lo = max(lab[0], lab[1]), min(lab[0], lab[1])
            if (tag == "common" and len(rows) > UNDERSAMPLE_ABOVE) or (tag == "4cwe" and lo and hi > IMBALANCE_ABOVE * lo):
                rows = undersample(rows)
            csets[tag] = rows
        for tag, rows in csets.items():
            assert_complete_pairs(rows)
            data[("primevul_merged_undersampled", tag)] = rows
            p = os.path.join(OUT, "by_source", "primevul_merged_undersampled", tag + ".jsonl"); write_jsonl(p, rows); files.append(p)
            stats[("primevul_merged_undersampled", tag)] = compute_stats(rows, pool)
            print("primevul_merged_undersampled", tag, stats[("primevul_merged_undersampled", tag)]["n"], flush=True)
        # merged_undersampled/<loại> = primevul_merged_undersampled + Java + JS cùng loại
        for tag in ("full", "common", "4cwe"):
            rows = csets[tag] + data.get(("cleanvul_java", tag), []) + data.get(("cleanvul_js", tag), [])
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
| như `merged/` nhưng C/C++ lấy từ **PrimeVul paired + unpaired gộp**, đã lọc trùng trên toàn bộ và undersample về 1:1 | `merged_undersampled/{full,common,4cwe}.jsonl` |
| một nguồn riêng | `by_source/<nguồn>/{full,common,4cwe}.jsonl` |

`merged/` = `by_source/primevul_paired` + `by_source/cleanvul_java` + `by_source/cleanvul_js` nối lại (đúng thứ tự đó).
`primevul_unpaired` **không** nằm trong `merged/` vì lệch nhãn mạnh (~3 % nhãn 1) — dùng riêng khi cần.
`merged_undersampled/<loại>.jsonl` = `by_source/primevul_merged_undersampled/<loại>` nối `by_source/cleanvul_java/<loại>` và
`by_source/cleanvul_js/<loại>`. Phần C/C++ (`primevul_merged_undersampled`) dựng như sau:
1. **Gộp** PrimeVul paired + unpaired (paired nằm trọn trong unpaired: cùng hàm, cùng nhãn) rồi **lọc lại trên toàn bộ** với cùng luật
   (lỗi nhãn, hàm không thân, trùng nguyên văn, trùng dạng abstract); khi trùng thì giữ bản thuộc cặp.
2. **Undersample `full` trước**: giữ mọi hàm nhãn 1 và mọi cặp (bản lỗi + bản vá), chọn NGẪU NHIÊN (seed 42) hàm nhãn 0 lẻ cho tới
   đúng 1:1, giữ thứ tự gốc.
3. **Rồi mới lọc** `common` và `4cwe` từ `full` đã undersample (cùng luật CWE như mọi bộ). `common` còn hơn 10 000 hàm thì undersample
   tiếp về 1:1; `4cwe` lệch nhãn hơn 3:1 thì undersample tiếp về 1:1 — cả hai đều không xảy ra ở dữ liệu hiện tại.
Phần C/C++ gồm các cặp đủ (`pair_id` có giá trị) và các hàm lẻ (`pair_id` = null); phần Java/JS vẫn đủ cặp.

- `full` ⊇ `common`; `4cwe` ⊆ `full`. Các bộ theo cặp luôn đủ cặp: mỗi `pair_id` có đúng một hàm nhãn 1 (bản lỗi) và một hàm nhãn 0 (bản đã vá cùng hàm).
- `common`: giữ hàm mà MỌI nhãn CWE của nó được luật MITRE CWE v4.20 xác định là áp dụng được cho Python.
  Java vào `common` nguyên vẹn vì CleanVul không gán CWE cho Java.

## Các nguồn
""")
    L.append("| nguồn (`by_source/…`) | ngôn ngữ | lấy từ | đơn vị |\n|---|---|---|---|")
    for name, base, language, origin, unit in SOURCES:
        L.append("| `%s` | %s | %s | %s |" % (name, language, origin, "cặp (bản lỗi + bản vá)" if unit == "pair" else "hàm đơn lẻ"))
    L.append("| `primevul_merged_undersampled` | C/C++ | PrimeVul paired + unpaired gộp, lọc trùng trên toàn bộ, undersample `full` về 1:1 "
             "rồi mới lọc `common`/`4cwe` | cặp + hàm đơn lẻ |")
    L.append("\n## Số mẫu\n")
    L.append("| file | số hàm | nhãn 1 / nhãn 0 | số cặp | ngôn ngữ | số CWE | hàm không có CWE | số dòng (trung vị / lớn nhất) |\n|---|---|---|---|---|---|---|---|")
    order = ([(t, g) for t, *_ in SOURCES for g in ("full", "common", "4cwe")]
             + [("primevul_merged_undersampled", g) for g in ("full", "common", "4cwe")]
             + [("merged", g) for g in ("full", "common", "4cwe")]
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
| `pair_id` | chuỗi / null | mã cặp; hai hàm cùng `pair_id` là bản lỗi và bản vá của cùng một hàm. `null` ở `primevul_unpaired` và ở hàm lẻ của `primevul_merged_undersampled` |
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
