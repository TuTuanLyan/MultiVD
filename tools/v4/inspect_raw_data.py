"""Đo: (a) đuôi comment dính đầu hàm, (b) hàm cụt thiếu `}`, (c) hàm rỗng THẬT sau khi sửa luật."""
import json, re, csv, collections, sys, os
sys.path.insert(0, "tools"); from strip_comments_multi import strip
csv.field_size_limit(10**9)
INIT = re.compile(r"\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override\s*)?:(?!:)")

def duoi_comment(raw):
    """Tiền tố kết thúc bằng `*/` mà KHÔNG có `/*` mở trước nó, và không chứa ; { } " ' — tức là
    phần cuối của một comment thuộc đoạn mã đứng trước hàm."""
    i = raw.find("*/")
    if i < 0: return None
    j = raw.find("/*")
    if 0 <= j < i: return None
    pre = raw[:i]
    if re.search(r"[;{}\"'`]", pre): return None
    return raw[:i + 2]

def than(code):
    """Thân hàm: từ `{` đầu tiên tới `}` KHỚP với nó (bỏ qua chuỗi/ký tự); cụt thì lấy tới hết."""
    i = code.find("{")
    if i < 0: return None, False
    d, k, n, q = 0, i, len(code), None
    while k < n:
        ch = code[k]
        if q:
            if ch == "\\": k += 2; continue
            if ch == q: q = None
        elif ch in "\"'`": q = ch
        elif ch == "{": d += 1
        elif ch == "}":
            d -= 1
            if d == 0: return code[i + 1:k], True
        k += 1
    return code[i + 1:], False

def main():
    src = []
    for f in ("primevul_train_paired", "primevul_valid_paired", "primevul_test_paired", "primevul_train", "primevul_valid", "primevul_test"):
        tag = "pv_paired" if "paired" in f else "pv_unpaired"
        for l in open("/drive1/cuongtm/ntat/Archive/PrimeVulRaw/%s.jsonl" % f):
            x = json.loads(l); src.append((tag, "c", int(x["target"]), x["func"]))
    for s in (3, 4):
        for r in csv.DictReader(open("/drive1/cuongtm/dgmoe/MAML/data/js_java_cleanvul/vulnerability_score_%d.csv" % s, newline="", encoding="utf-8")):
            e = (r.get("extension") or "").strip().lower()
            if e in ("java", "js"):
                src.append((e, e, 1, r.get("func_before") or "")); src.append((e, e, 0, r.get("func_after") or ""))
    c = collections.Counter(); ex = collections.defaultdict(list)
    for tag, lx, lab, raw in src:
        c[(tag, "tong")] += 1
        t = duoi_comment(raw)
        if t is not None:
            c[(tag, "a_duoi_comment_dau_ham")] += 1
            if len(ex[tag + "a"]) < 3: ex[tag + "a"].append(repr(raw[:len(t) + 40]))
        code, _ = strip(raw.replace("\r\n", "\n"), lx)
        b, closed = than(code)
        if b is None:
            c[(tag, "c_khong_co_{_(khai_bao)", lab)] += 1; continue
        if not closed:
            c[(tag, "b_cut_thieu_}")] += 1
        if not b.strip():
            head = code[:code.find("{")]
            k = "c_than_rong_CO_khoi_tao" if INIT.search(head) else "c_than_rong_that"
            c[(tag, k, lab)] += 1
    for k, v in sorted(c.items(), key=lambda x: str(x[0])): print(k, v)
    for k, v in ex.items(): print("vd", k, v)

main()
