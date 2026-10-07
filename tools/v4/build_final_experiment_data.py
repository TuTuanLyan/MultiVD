#!/usr/bin/env python3
"""Dựng thư mục dữ liệu CHUẨN cho thí nghiệm final: `data/final_experiment_data/` (người dùng chốt 24/09/2026).

Phần C/C++ lấy từ PrimeVul paired + unpaired GỘP đã lọc T0–T4 trên toàn bộ (`sources_v4/ccpp_primevul_merged_*`):
    full    mọi hàm nhãn 1 (5 888) + 6 000 hàm nhãn 0 = MỌI hàm nhãn 0 thuộc cặp (4 602 bản vá) + hàm nhãn 0 lẻ bốc ngẫu nhiên
            (seed 42) cho đủ 6 000.
    common  mọi hàm nhãn 1 của bộ gộp `common` (CWE dùng chung được với Python) + 3 000 hàm nhãn 0 bốc ngẫu nhiên (seed 42)
            TỪ 6 000 hàm nhãn 0 của full — KHÔNG xét CWE của hàm nhãn 0 (hàm sạch; CWE của nó chỉ là CWE của commit).
    4cwe    mọi hàm nhãn 1 của bộ gộp `4cwe` (CWE-022/078/079/089) + 200 hàm nhãn 0 bốc ngẫu nhiên (seed 42) TỪ 3 000 hàm nhãn 0
            của common. Hàm nhãn 0 có CWE ngoài bốn loại: `cwe_class` = -100 (head phụ 4 lớp bỏ qua).
Java, JS: lấy nguyên `clean_sources_v4/by_source/cleanvul_{java,js}/<loại>` như mọi bộ khác.
Mỗi file: `pair_id` chỉ giữ khi CẢ HAI nửa của cặp cùng nằm trong file; nửa lẻ -> null.
Pha 1: `phase1_<loại>.jsonl` (mọi hàng) và `phase1_<loại>/fold1/{train,val,test}.jsonl` — val 15 % ngẫu nhiên theo hàng (seed 42),
`test.jsonl` = bản sao `val.jsonl` (cách chia như build_mwsrc_v4.py, tỉ lệ 15 % theo người dùng chốt 24/09).
Đích: chép nguyên `sven_python_folds_nocomment/`.
Một pha trộn nguồn (nhánh mwg_mixsrc): `mixsrc_common/fold{1..5}/` — train = mọi hàng `phase1_common.jsonl` + SVEN train của fold,
val/test = SVEN val/test của fold (người dùng chốt 24/09). `--mixsrc_only`: chỉ dựng phần này vào thư mục đã có.
Common bỏ C/C++ (nguồn Pha 1 thứ hai, người dùng chốt 24/09): `phase1_common_noccpp/` = đúng các hàng Java + JS của `phase1_common`
(cả file gộp lẫn từng tập fold1, giữ thứ tự và cách chia) — Java/JS giữ nguyên tập train/val như trong common nên hai nguồn so được
với nhau. `--noccpp_only`: chỉ dựng phần này vào thư mục đã có.
Full bỏ C/C++ (người dùng 27/09): `phase1_full_noccpp/` = đúng các hàng Java + JS của `phase1_full`, cùng cách lọc/không chia lại.
`--full_noccpp_only`: chỉ dựng phần này vào thư mục đã có.
Common/full bỏ Java (người dùng 28/09: Java không có nhãn CWE nên vào cả common lẫn full mà không xét CWE): `phase1_common_nojava/`,
`phase1_full_nojava/` = đúng các hàng C/C++ + JS của `phase1_common` / `phase1_full`, cùng cách lọc/không chia lại.
`--nojava_only`: chỉ dựng hai phần này vào thư mục đã có.
Chỉ JS (người dùng 29/09): `phase1_common_jsonly/`, `phase1_full_jsonly/` = đúng các hàng JS, không chia lại. `--jsonly_only`.
Uncommon C/C++ + JS chỉ hàm có CWE (người dùng 01/10): `phase1_uncommon_nojava_knowncwe/` = hàng của `phase1_full_nojava` không có
trong `phase1_common_nojava` (so nguyên văn code), bỏ hàng `cwe` rỗng, không chia lại, gỡ `pair_id` lẻ. `--uncommon_nojava_knowncwe_only`.

Chạy từ thư mục gốc MultiVD:  python tools/v4/build_final_experiment_data.py [data/final_experiment_data]
"""
import collections, hashlib, json, os, random, shutil, sys

SRC = "data/sources_v4"
CLEAN = "data/clean_sources_v4/by_source"
SVEN = "data/sven_python_folds_nocomment"
OUT = next((a for a in sys.argv[1:] if not a.startswith("--")), "data/final_experiment_data")
MERGED_C = "ccpp_primevul_merged"
SEED = 42
N_NEG = {"full": 6000, "common": 3000, "4cwe": 200}
VAL_RATIO = 0.15             # người dùng chốt 24/09: val 15 %, ở giữa 10 % (multibabel) và 20 % (pool v4 cũ)
FIELDS = ("id", "pair_id", "code", "label", "lang", "cwe", "cwe_id", "cwe_class", "cve", "project", "file_name", "commit_url")
fmt = lambda n: f"{n:,}".replace(",", " ")


def load_merged(tag):
    """Hàng của bộ gộp C/C++ theo đúng định dạng nguồn chuẩn (như export_clean_sources.load_set)."""
    out = []
    with open(os.path.join(SRC, "%s_%s.jsonl" % (MERGED_C, tag)), encoding="utf-8") as f1, \
         open(os.path.join(SRC, "%s_%s.meta.jsonl" % (MERGED_C, tag)), encoding="utf-8") as f2:
        for l, m in zip(f1, f2):
            r, m = json.loads(l), json.loads(m)
            d = {"id": m.get("row_id"), "pair_id": m.get("pair_id")}
            d.update({k: r[k] for k in ("code", "label", "lang", "cwe", "cwe_id", "cwe_class")})
            d.update({k: m.get(k) for k in ("cve", "project", "file_name", "commit_url")})
            if d.get("file_name") in ("None", ""):
                d["file_name"] = None
            out.append(d)
    return out


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in FIELDS}, ensure_ascii=False) + "\n")


def sample_in_order(rows, k, seed=SEED):
    """Bốc k hàng ngẫu nhiên (seed), trả về theo THỨ TỰ GỐC."""
    keep = set(random.Random(seed).sample(range(len(rows)), k))
    return [r for i, r in enumerate(rows) if i in keep]


def only_complete_pairs(rows):
    """`pair_id` giữ khi đủ hai nửa (một nhãn 1, một nhãn 0) trong cùng file; nửa lẻ -> null. Trả (hàng, số nửa bị gỡ)."""
    labs = collections.defaultdict(list)
    for r in rows:
        if r["pair_id"]:
            labs[r["pair_id"]].append(r["label"])
    ok = {p for p, v in labs.items() if sorted(v) == [0, 1]}
    out, n = [], 0
    for r in rows:
        if r["pair_id"] and r["pair_id"] not in ok:
            r = dict(r, pair_id=None); n += 1
        out.append(r)
    return out, n


def build_ccpp():
    full, common, cwe4 = load_merged("full"), load_merged("common"), load_merged("4cwe")
    by_id = {r["id"]: r for r in full}
    assert len(by_id) == len(full), "id trùng trong bộ gộp"
    # full: mọi nhãn 1 + mọi nhãn 0 thuộc cặp + nhãn 0 lẻ bốc ngẫu nhiên cho đủ N_NEG["full"]
    neg_pair = [r for r in full if r["label"] == 0 and r["pair_id"]]
    neg_loose = [r for r in full if r["label"] == 0 and not r["pair_id"]]
    picked = {r["id"] for r in neg_pair} | {r["id"] for r in sample_in_order(neg_loose, N_NEG["full"] - len(neg_pair))}
    neg_full = [r for r in full if r["id"] in picked]
    sets = {"full": [r for r in full if r["label"] == 1 or r["id"] in picked]}
    # common: mọi nhãn 1 của bộ gộp common + N_NEG["common"] nhãn 0 bốc từ nhãn 0 của full (không xét CWE)
    neg_common = sample_in_order(neg_full, N_NEG["common"])
    nc = {r["id"] for r in neg_common}
    pos_common = {r["id"] for r in common if r["label"] == 1}
    sets["common"] = [r for r in full if r["id"] in pos_common or r["id"] in nc]
    # 4cwe: mọi nhãn 1 của bộ gộp 4cwe + N_NEG["4cwe"] nhãn 0 bốc từ nhãn 0 của common. Trường CWE lấy theo bộ 4cwe
    # (cwe_class 0–3); nhãn 0 có CWE ngoài bốn loại giữ cwe/cwe_id gốc, cwe_class = -100.
    c4_rows = {r["id"]: r for r in cwe4}
    n4 = {r["id"] for r in sample_in_order(neg_common, N_NEG["4cwe"])}
    pos4 = {r["id"] for r in cwe4 if r["label"] == 1}
    rows4 = []
    for r in full:
        if r["id"] in pos4 or r["id"] in n4:
            rows4.append(c4_rows[r["id"]] if r["id"] in c4_rows else dict(r, cwe_class=-100))
    sets["4cwe"] = rows4
    return sets


def split_fold1(rows, seed=SEED):
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    val_idx = set(idx[:int(round(len(rows) * VAL_RATIO))])
    return [r for i, r in enumerate(rows) if i not in val_idx], [r for i, r in enumerate(rows) if i in val_idx]


def build_mixsrc(out):
    """mixsrc_common/fold{1..5}: nguồn common (mọi hàng) + SVEN train của fold; val/test = SVEN. Giữ nguyên các trường của
    từng nguồn (hàng SVEN không có id/pair_id — train_mwg coi là hàng lẻ)."""
    src = load_jsonl(os.path.join(out, "phase1_common.jsonl"))
    sven = os.path.join(out, os.path.basename(SVEN))
    stats = {}
    for f in range(1, 6):
        d = os.path.join(out, "mixsrc_common", "fold%d" % f)
        os.makedirs(d, exist_ok=True)
        parts = {"train": src + load_jsonl(os.path.join(sven, "fold%d" % f, "train.jsonl"))}
        for sp in ("val", "test"):
            parts[sp] = load_jsonl(os.path.join(sven, "fold%d" % f, sp + ".jsonl"))
        for sp, rows in parts.items():
            with open(os.path.join(d, sp + ".jsonl"), "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[f] = {sp: describe_any(rows) for sp, rows in parts.items()}
    return stats


LANG_LABEL = {"ccpp": "C/C++", "java": "Java", "js": "JS"}


def build_noccpp(out, base="phase1_common", drop="ccpp"):
    """<base>_no<drop>: lọc bỏ hàng `lang = drop` khỏi <base> (file gộp + fold1/{train,val,test}); không chia lại."""
    stats = {}
    dst_name = base + "_no" + drop
    src = {"all": os.path.join(out, base + ".jsonl")}
    src.update({sp: os.path.join(out, base, "fold1", sp + ".jsonl") for sp in ("train", "val", "test")})
    for sp, q in src.items():
        rows = [r for r in load_jsonl(q) if r["lang"] != drop]
        dst = os.path.join(out, dst_name + ".jsonl") if sp == "all" else os.path.join(out, dst_name, "fold1", sp + ".jsonl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[sp] = describe_any(rows)
    pids = collections.defaultdict(set)
    vids = {r["id"] for r in load_jsonl(os.path.join(out, dst_name, "fold1", "val.jsonl"))}
    for r in load_jsonl(os.path.join(out, dst_name + ".jsonl")):
        if r.get("pair_id"):
            pids[r["pair_id"]].add(r["id"] in vids)
    stats["pairs"] = len(pids)
    stats["pairs_split"] = sum(1 for v in pids.values() if len(v) == 2)
    return stats


def build_lang_subset(out, base, keep, suffix):
    """<base>_<suffix>: CHỈ giữ hàng có lang thuộc `keep` (file gộp + fold1/{train,val,test}); không chia lại."""
    stats = {}
    dst_name = base + "_" + suffix
    src = {"all": os.path.join(out, base + ".jsonl")}
    src.update({sp: os.path.join(out, base, "fold1", sp + ".jsonl") for sp in ("train", "val", "test")})
    for sp, q in src.items():
        rows = [r for r in load_jsonl(q) if r["lang"] in keep]
        dst = os.path.join(out, dst_name + ".jsonl") if sp == "all" else os.path.join(out, dst_name, "fold1", sp + ".jsonl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[sp] = describe_any(rows)
    pids = collections.defaultdict(set)
    vids = {r["id"] for r in load_jsonl(os.path.join(out, dst_name, "fold1", "val.jsonl"))}
    for r in load_jsonl(os.path.join(out, dst_name + ".jsonl")):
        if r.get("pair_id"):
            pids[r["pair_id"]].add(r["id"] in vids)
    stats["pairs"] = len(pids)
    stats["pairs_split"] = sum(1 for v in pids.values() if len(v) == 2)
    return stats


def build_minus(out, base, minus, name):
    """<name>: các hàng của <base> KHÔNG có trong <minus> (so theo nguyên văn `code`, trên mọi tập của <minus>), file gộp +
    fold1/{train,val,test}; không chia lại — hàng giữ đúng tập train/val như trong <base> (người dùng 30/09: full bỏ common)."""
    drop = {r["code"] for r in load_jsonl(os.path.join(out, minus + ".jsonl"))}
    stats = {}
    src = {"all": os.path.join(out, base + ".jsonl")}
    src.update({sp: os.path.join(out, base, "fold1", sp + ".jsonl") for sp in ("train", "val", "test")})
    for sp, q in src.items():
        rows = [r for r in load_jsonl(q) if r["code"] not in drop]
        dst = os.path.join(out, name + ".jsonl") if sp == "all" else os.path.join(out, name, "fold1", sp + ".jsonl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[sp] = describe_any(rows)
    pids = collections.defaultdict(set)
    vids = {r["id"] for r in load_jsonl(os.path.join(out, name, "fold1", "val.jsonl"))}
    for r in load_jsonl(os.path.join(out, name + ".jsonl")):
        if r.get("pair_id"):
            pids[r["pair_id"]].add(r["id"] in vids)
    stats["pairs"] = len(pids)
    stats["pairs_split"] = sum(1 for v in pids.values() if len(v) == 2)
    stats["dropped"] = sum(1 for r in load_jsonl(os.path.join(out, base + ".jsonl")) if r["code"] in drop)
    stats["cwe"] = collections.Counter(str(r.get("cwe") or "không có") for r in load_jsonl(os.path.join(out, name + ".jsonl"))).most_common()
    return stats


def build_minus_knowncwe(out, base, minus, name):
    """<name>: các hàng của <base> KHÔNG có trong <minus> (so nguyên văn `code`, mọi tập của <minus>) VÀ có CWE (bỏ hàng `cwe` rỗng),
    file gộp + fold1/{train,val,test}; không chia lại. `pair_id` chỉ giữ khi đủ hai nửa trong file gộp MỚI (người dùng 01/10: uncommon
    C/C++ + JS, bỏ mẫu không rõ CWE) — ở C/C++ phép trừ cắt cặp vì nhãn 0 của common bốc ngẫu nhiên, không theo cặp."""
    rows_minus = load_jsonl(os.path.join(out, minus + ".jsonl"))
    drop = {r["code"] for r in rows_minus}
    base_all = load_jsonl(os.path.join(out, base + ".jsonl"))
    keep = lambda r: r["code"] not in drop and bool(r.get("cwe"))
    kept, n_unpaired = only_complete_pairs([r for r in base_all if keep(r)])
    pid = {r["id"]: r["pair_id"] for r in kept}
    stats = {}
    src = {"all": None}
    src.update({sp: os.path.join(out, base, "fold1", sp + ".jsonl") for sp in ("train", "val", "test")})
    for sp, q in src.items():
        rows = kept if sp == "all" else [dict(r, pair_id=pid[r["id"]]) for r in load_jsonl(q) if keep(r)]
        dst = os.path.join(out, name + ".jsonl") if sp == "all" else os.path.join(out, name, "fold1", sp + ".jsonl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[sp] = describe_any(rows)
        lab = collections.Counter((r["lang"], r["label"]) for r in rows)
        stats[sp]["by_lang"] = {lg: [lab[(lg, 1)], lab[(lg, 0)]] for lg in ("ccpp", "java", "js")}
    pids = collections.defaultdict(set)
    vids = {r["id"] for r in load_jsonl(os.path.join(out, name, "fold1", "val.jsonl"))}
    for r in kept:
        if r.get("pair_id"):
            pids[r["pair_id"]].add(r["id"] in vids)
    stats["pairs"] = len(pids)
    stats["pairs_split"] = sum(1 for v in pids.values() if len(v) == 2)
    stats["unpaired_halves"] = n_unpaired
    stats["dropped_in_minus"] = sum(1 for r in base_all if r["code"] in drop)
    stats["dropped_unknown_cwe"] = collections.Counter("%s nhãn %d" % (r["lang"], r["label"]) for r in base_all
                                                       if r["code"] not in drop and not r.get("cwe"))
    stats["cwe"] = collections.Counter("%s/%s" % (r["lang"], r["cwe"]) for r in kept if r["label"] == 1).most_common()
    return stats


def minus_readme_section(st, base, minus, name, title, note=""):
    L = ["## %s (`%s/`)\n" % (title, name),
         "Các hàng của `%s` KHÔNG có trong `%s` (so nguyên văn `code` trên mọi tập của `%s`; bỏ %s hàng), **không chia lại** — hàng giữ đúng "
         "tập train/val như trong `%s`. `test.jsonl` = bản sao `val.jsonl` như mọi nguồn Pha 1.%s\n"
         % (base, minus, minus, fmt(st["dropped"]), base, (" " + note) if note else ""),
         "| tập | số hàm (nhãn 1 / nhãn 0) |", "|---|---|"]
    for sp in ("all", "train", "val", "test"):
        t = st[sp]
        L.append("| %s | %s (%s / %s) |" % (name + ".jsonl" if sp == "all" else "fold1/" + sp, fmt(t["n"]), fmt(t["pos"]), fmt(t["neg"])))
    L.append("\nCặp đủ: %s; cặp bị tách giữa train/val: %s. CWE (số hàm): %s.\n"
             % (fmt(st["pairs"]), fmt(st["pairs_split"]), ", ".join("%s %s" % (c, fmt(n)) for c, n in st["cwe"])))
    return "\n".join(L) + "\n"


def load_cwe_labels():
    """row_id -> danh sách nhãn CWE ĐẦY ĐỦ (trường cwe_labels của meta sources_v4; với hàm nhãn 0 C/C++ là CWE của commit)."""
    out = {}
    for f in ("ccpp_primevul_merged_full.meta.jsonl", "js_cleanvul_3-4_full.meta.jsonl"):
        for line in open(os.path.join(SRC, f), encoding="utf-8"):
            m = json.loads(line)
            if m.get("row_id"):
                out[m["row_id"]] = m.get("cwe_labels") or [None]
    return out


def build_minus_strict(out, base, minus, name):
    """<name>: hàng của <base> KHÔNG có trong <minus> (so nguyên văn code) mà MỌI nhãn CWE đều là mã CWE thật và đều KHÔNG áp dụng cho Python
    theo đúng luật dựng common (tools/cwe_rules.py R1-R6, cwec_v4.20.xml) - người dùng 01/10: "Nếu 1 mẫu có nhiều hơn 1 CWE và 1 trong số đó nằm
    trong common thì phải bỏ"; nhãn không phải mã CWE (NVD-CWE-*, rỗng) cũng bỏ; hàm nhãn 0 C/C++ xét theo CWE của commit (người dùng: bỏ, không bù).
    Không chia lại, gỡ pair_id lẻ."""
    sys.path.insert(0, "tools")
    from build_sources_v2 import get_rules
    rules = get_rules("data/cwe_spec/cwec_v4.20.xml", "Python")
    labels = load_cwe_labels()
    drop = {r["code"] for r in load_jsonl(os.path.join(out, minus + ".jsonl"))}
    base_all = load_jsonl(os.path.join(out, base + ".jsonl"))
    why = collections.Counter()

    def verdict(r):
        if r["code"] in drop:
            return "in_minus"
        lab = labels.get(r["id"])
        if lab is None:
            return "no_meta"
        d = [rules.decide(x if x is not None else "") for x in lab]
        if any(k for k, _, _ in d):
            return "has_common_cwe"
        if any(rule == "R1" for _, rule, _ in d):
            return "unknown_cwe"
        return "keep"
    # người dùng 01/10: common / uncommon chỉ xét NHÃN 1. JS: nhãn 0 đi theo nửa nhãn 1 cùng cặp. C/C++: nhãn 0 là hàm sạch bốc từ bộ đã
    # undersample cho bằng nhãn 1, KHÔNG lọc theo CWE ⇒ giữ mọi hàm nhãn 0 của base không có trong minus (phần bù của nhãn 0 common).
    v = {r["id"]: verdict(r) for r in base_all if r["label"] == 1}
    by_pair = {r["pair_id"]: v[r["id"]] for r in base_all if r["label"] == 1 and r.get("pair_id")}
    for r in base_all:
        if r["label"] == 1:
            continue
        if r["code"] in drop:
            v[r["id"]] = "in_minus"
        elif r["lang"] == "ccpp":
            v[r["id"]] = "keep"                                    # hàm sạch C/C++: không lọc CWE
        elif r.get("pair_id") in by_pair:
            v[r["id"]] = by_pair[r["pair_id"]]                     # JS: theo nửa nhãn 1 cùng cặp
        else:
            v[r["id"]] = verdict(r)                                # JS lẻ cặp (không có ở các bộ hiện tại): xét nhãn của chính nó
    for r in base_all:
        why["%s nhãn %d: %s" % (r["lang"], r["label"], v[r["id"]])] += 1
    assert not any(x == "no_meta" for x in v.values()), "có hàng không tra được nhãn CWE gốc"
    kept, n_unpaired = only_complete_pairs([r for r in base_all if v[r["id"]] == "keep"])
    pid = {r["id"]: r["pair_id"] for r in kept}
    stats = {}
    src = {"all": None}
    src.update({sp: os.path.join(out, base, "fold1", sp + ".jsonl") for sp in ("train", "val", "test")})
    for sp, q in src.items():
        rows = kept if sp == "all" else [dict(r, pair_id=pid[r["id"]]) for r in load_jsonl(q) if v[r["id"]] == "keep"]
        dst = os.path.join(out, name + ".jsonl") if sp == "all" else os.path.join(out, name, "fold1", sp + ".jsonl")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[sp] = describe_any(rows)
        lab = collections.Counter((r["lang"], r["label"]) for r in rows)
        stats[sp]["by_lang"] = {lg: [lab[(lg, 1)], lab[(lg, 0)]] for lg in ("ccpp", "java", "js")}
    pids = collections.defaultdict(set)
    vids = {r["id"] for r in load_jsonl(os.path.join(out, name, "fold1", "val.jsonl"))}
    for r in kept:
        if r.get("pair_id"):
            pids[r["pair_id"]].add(r["id"] in vids)
    stats["pairs"] = len(pids)
    stats["pairs_split"] = sum(1 for x in pids.values() if len(x) == 2)
    stats["unpaired_halves"] = n_unpaired
    stats["reasons"] = dict(sorted(why.items()))
    stats["cwe"] = collections.Counter("%s/%s" % (r["lang"], "+".join(sorted(str(x) for x in labels[r["id"]]))) for r in kept if r["label"] == 1).most_common()
    return stats


def pair_pattern(rows_by_split):
    """pair_id -> kiểu chia của cặp theo tập chứa nửa nhãn 1 và nửa nhãn 0: 'tt', 'vv', '1t0v' (nhãn 1 ở train, nhãn 0 ở val), '0t1v'."""
    where = collections.defaultdict(dict)
    for sp, rows in rows_by_split.items():
        for r in rows:
            if r.get("pair_id"):
                where[r["pair_id"]][r["label"]] = sp
    pat = {}
    for p, w in where.items():
        if set(w) != {0, 1}:
            continue
        pat[p] = {("train", "train"): "tt", ("val", "val"): "vv", ("train", "val"): "1t0v", ("val", "train"): "0t1v"}[(w[1], w[0])]
    return pat


def build_pair_subsample(out, base, like, name, seed=SEED):
    """<name>: rút ngẫu nhiên (seed) các CẶP của <base> sao cho số cặp theo từng kiểu chia (pair_pattern) TRÙNG với <like> ⇒ cùng số hàm
    train / val và cùng số nhãn 1 / nhãn 0 trong mỗi tập như <like>. Không chia lại: mỗi hàng giữ nguyên tập của nó trong <base>.
    Người dùng 02/10 ("chủ động kiểm theo hướng của tôi"): tách CỠ nguồn khỏi GIỐNG NHÃN khi so common chỉ JS với uncommon chỉ JS chặt."""
    sp_base = {sp: load_jsonl(os.path.join(out, base, "fold1", sp + ".jsonl")) for sp in ("train", "val")}
    sp_like = {sp: load_jsonl(os.path.join(out, like, "fold1", sp + ".jsonl")) for sp in ("train", "val")}
    pat_base, pat_like = pair_pattern(sp_base), pair_pattern(sp_like)
    need = collections.Counter(pat_like.values())
    assert sum(need.values()) * 2 == sum(len(v) for v in sp_like.values()), "<like> có hàng không thuộc cặp đủ hai nửa"
    rng, keep = random.Random(seed), set()
    for k in sorted(need):
        pool = sorted(p for p, v in pat_base.items() if v == k)
        assert len(pool) >= need[k], "không đủ cặp kiểu %s trong %s" % (k, base)
        keep |= set(rng.sample(pool, need[k]))
    all_rows = [r for r in load_jsonl(os.path.join(out, base + ".jsonl")) if r.get("pair_id") in keep]
    files = {"all": (os.path.join(out, name + ".jsonl"), all_rows)}
    for sp in ("train", "val"):
        files[sp] = (os.path.join(out, name, "fold1", sp + ".jsonl"), [r for r in sp_base[sp] if r.get("pair_id") in keep])
    files["test"] = (os.path.join(out, name, "fold1", "test.jsonl"), files["val"][1])     # như base: test.jsonl = bản sao val.jsonl
    stats, labels = {}, load_cwe_labels()
    for sp, (dst, rows) in files.items():
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[sp] = describe_any(rows)
        lab = collections.Counter((r["lang"], r["label"]) for r in rows)
        stats[sp]["by_lang"] = {lg: [lab[(lg, 1)], lab[(lg, 0)]] for lg in ("ccpp", "java", "js")}
    for sp in ("train", "val"):
        a = collections.Counter(r["label"] for r in files[sp][1]); b = collections.Counter(r["label"] for r in sp_like[sp])
        assert a == b, "số nhãn tập %s lệch <like>: %s / %s" % (sp, dict(a), dict(b))
    sv = {"CWE-22", "CWE-78", "CWE-79", "CWE-89"}
    pos = [r for r in all_rows if r["label"] == 1]
    norm = lambda c: "CWE-%d" % int(str(c).split("-")[1]) if str(c).startswith("CWE-") and str(c).split("-")[1].isdigit() else str(c)
    stats["pairs"] = len(keep)
    stats["patterns"] = dict(sorted(need.items()))
    stats["sven_share"] = sum(1 for r in pos if {norm(x) for x in labels.get(r["id"], []) if x} & sv) / len(pos)
    stats["cwe"] = collections.Counter("+".join(sorted(norm(x) for x in labels.get(r["id"], []) if x)) for r in pos).most_common(12)
    return stats


def pair_subsample_readme_section(st, base, like, name, title):
    L = ["## %s (`%s/`)\n" % (title, name),
         "Rút ngẫu nhiên (seed %d) %s cặp của `%s` sao cho số cặp theo từng kiểu chia (cả hai nửa ở train; cả hai ở val; nhãn 1 ở train còn nhãn 0 ở val; "
         "ngược lại) TRÙNG với `%s` ⇒ cùng số hàm train / val và cùng số nhãn 1 / nhãn 0 trong mỗi tập. Không chia lại: mỗi hàng giữ nguyên tập của nó "
         "trong `%s`. `test.jsonl` = bản sao `val.jsonl`. Mục đích: tách CỠ nguồn khỏi GIỐNG NHÃN khi so `%s` với `%s` (người dùng 02/10). "
         "Tỉ lệ hàm nhãn 1 có ít nhất một nhãn thuộc 4 CWE của SVEN: %.1f %%.\n"
         % (SEED, fmt(st["pairs"]), base, like, base, base, like, 100 * st["sven_share"]),
         "| tập | số hàm (nhãn 1 / nhãn 0) |", "|---|---|"]
    for sp in ("all", "train", "val", "test"):
        t = st[sp]
        L.append("| %s | %s (%s / %s) |" % (name + ".jsonl" if sp == "all" else "fold1/" + sp, fmt(t["n"]), fmt(t["pos"]), fmt(t["neg"])))
    L.append("\nKiểu chia cặp: " + "; ".join("%s %s" % (k, fmt(v)) for k, v in st["patterns"].items())
             + ". CWE của hàm nhãn 1 (nhãn đầy đủ): " + "; ".join("%s %s" % (k or "(rỗng)", fmt(v)) for k, v in st["cwe"]) + ".\n")
    return "\n".join(L) + "\n"


def minus_strict_readme_section(st, base, minus, name, title, note=""):
    L = ["## %s (`%s/`)\n" % (title, name),
         "Các hàng của `%s` KHÔNG có trong `%s` (so nguyên văn `code`) mà MỌI nhãn CWE (trường `cwe_labels` của meta sources_v4, không chỉ CWE chính) "
         "đều là mã CWE thật và đều KHÔNG áp dụng cho Python theo đúng luật dựng common (`tools/cwe_rules.py` R1-R6, `data/cwe_spec/cwec_v4.20.xml`). "
         "**Không chia lại**, `pair_id` lẻ gỡ thành null (%s nửa). `test.jsonl` = bản sao `val.jsonl`.%s\n"
         % (base, minus, fmt(st["unpaired_halves"]), (" " + note) if note else ""),
         "| tập | số hàm (nhãn 1 / nhãn 0) | C/C++ (nhãn 1 / nhãn 0) | JS (nhãn 1 / nhãn 0) |", "|---|---|---|---|"]
    for sp in ("all", "train", "val", "test"):
        t = st[sp]
        L.append("| %s | %s (%s / %s) | %s / %s | %s / %s |" % (name + ".jsonl" if sp == "all" else "fold1/" + sp, fmt(t["n"]), fmt(t["pos"]),
                                                             fmt(t["neg"]), *map(fmt, t["by_lang"]["ccpp"]), *map(fmt, t["by_lang"]["js"])))
    L.append("\nLý do loại theo hàng: " + "; ".join("%s %s" % (k, fmt(n)) for k, n in st["reasons"].items()) + ".")
    L.append("\nCặp đủ: %s; cặp bị tách giữa train/val: %s. Bộ nhãn CWE của hàm nhãn 1 (10 loại nhiều nhất): %s.\n"
             % (fmt(st["pairs"]), fmt(st["pairs_split"]), ", ".join("%s %s" % (c, fmt(n)) for c, n in st["cwe"][:10])))
    return "\n".join(L) + "\n"


def minus_knowncwe_readme_section(st, base, minus, name, title, note=""):
    L = ["## %s (`%s/`)\n" % (title, name),
         "Các hàng của `%s` KHÔNG có trong `%s` (so nguyên văn `code` trên mọi tập của `%s`; bỏ %s hàng) VÀ có CWE (bỏ thêm %s hàng "
         "`cwe` rỗng: %s), **không chia lại** — hàng giữ đúng tập train/val như trong `%s`; `pair_id` lẻ (nửa kia không còn trong file) "
         "gỡ thành null (%s nửa). `test.jsonl` = bản sao `val.jsonl` như mọi nguồn Pha 1.%s\n"
         % (base, minus, minus, fmt(st["dropped_in_minus"]), fmt(sum(st["dropped_unknown_cwe"].values())),
            ", ".join("%s %s" % (k, fmt(v)) for k, v in sorted(st["dropped_unknown_cwe"].items())), base, fmt(st["unpaired_halves"]),
            (" " + note) if note else ""),
         "| tập | số hàm (nhãn 1 / nhãn 0) | C/C++ (nhãn 1 / nhãn 0) | JS (nhãn 1 / nhãn 0) |", "|---|---|---|---|"]
    for sp in ("all", "train", "val", "test"):
        t = st[sp]
        L.append("| %s | %s (%s / %s) | %s / %s | %s / %s |" % (name + ".jsonl" if sp == "all" else "fold1/" + sp, fmt(t["n"]), fmt(t["pos"]),
                                                             fmt(t["neg"]), *map(fmt, t["by_lang"]["ccpp"]), *map(fmt, t["by_lang"]["js"])))
    top = ", ".join("%s %s" % (c, fmt(n)) for c, n in st["cwe"][:15])
    L.append("\nCặp đủ: %s; cặp bị tách giữa train/val: %s. CWE của hàm nhãn 1 (15 loại nhiều nhất, trên %s loại): %s.\n"
             % (fmt(st["pairs"]), fmt(st["pairs_split"]), fmt(len(st["cwe"])), top))
    return "\n".join(L) + "\n"


def jsonly_readme_section(st, base):
    name, label = base + "_jsonly", {"phase1_common": "Common", "phase1_full": "Full"}[base]
    L = ["## %s chỉ JS (`%s/`)\n" % (label, name),
         "Đúng các hàng JS của `%s` (bỏ C/C++ và Java), **không chia lại** — hàng JS nằm ở cùng tập train/val như trong `%s`. " % (base, base) +
         "`test.jsonl` = bản sao `val.jsonl` như mọi nguồn Pha 1 (người dùng 29/09: \"chạy thêm nguồn js only full và common\").\n",
         "| tập | số hàm (nhãn 1 / nhãn 0) | JS |", "|---|---|---|"]
    for sp in ("all", "train", "val", "test"):
        t = st[sp]
        L.append("| %s | %s (%s / %s) | %s |" % (name + ".jsonl" if sp == "all" else "fold1/" + sp, fmt(t["n"]), fmt(t["pos"]), fmt(t["neg"]),
                                             fmt(t["lang"].get("js", 0))))
    L.append("\nCặp đủ: %s; cặp bị tách giữa train/val: %s.\n" % (fmt(st["pairs"]), fmt(st["pairs_split"])))
    return "\n".join(L) + "\n"


def langonly_readme_section(st, base, lang, note=""):
    """Như jsonly_readme_section nhưng cho một ngôn ngữ bất kỳ (chỉ Java, chỉ C/C++ — người dùng 29/09)."""
    name, label = base + "_" + lang + "only", {"phase1_common": "Common", "phase1_full": "Full", "phase1_4cwe": "4cwe"}[base]
    others = " và ".join(LANG_LABEL[g] for g in ("ccpp", "java", "js") if g != lang)
    L = ["## %s chỉ %s (`%s/`)\n" % (label, LANG_LABEL[lang], name),
         "Đúng các hàng %s của `%s` (bỏ %s), **không chia lại** — hàng %s nằm ở cùng tập train/val như trong `%s`. "
         % (LANG_LABEL[lang], base, others, LANG_LABEL[lang], base) +
         "`test.jsonl` = bản sao `val.jsonl` như mọi nguồn Pha 1 (người dùng 29/09: \"thêm bản riêng cho java và ccpp (full/common)\").%s\n"
         % ((" " + note) if note else ""),
         "| tập | số hàm (nhãn 1 / nhãn 0) | %s |" % LANG_LABEL[lang], "|---|---|---|"]
    for sp in ("all", "train", "val", "test"):
        t = st[sp]
        L.append("| %s | %s (%s / %s) | %s |" % (name + ".jsonl" if sp == "all" else "fold1/" + sp, fmt(t["n"]), fmt(t["pos"]), fmt(t["neg"]),
                                             fmt(t["lang"].get(lang, 0))))
    L.append("\nCặp đủ: %s; cặp bị tách giữa train/val: %s.\n" % (fmt(st["pairs"]), fmt(st["pairs_split"])))
    return "\n".join(L) + "\n"


def noccpp_readme_section(st, base="phase1_common", drop="ccpp", note=""):
    name, label = base + "_no" + drop, {"phase1_common": "Common", "phase1_full": "Full"}[base]
    keep = [g for g in ("ccpp", "java", "js") if g != drop]
    kept = "/".join(LANG_LABEL[g] for g in keep)
    L = ["## %s bỏ %s (`%s/`)\n" % (label, LANG_LABEL[drop], name),
         "Đúng các hàng %s của `%s`: lọc bỏ `lang = %s` khỏi file gộp và khỏi từng tập `fold1/{train,val,test}`, "
         % (" + ".join(LANG_LABEL[g] for g in keep), base, drop) +
         "**không chia lại** — hàng %s nằm ở cùng tập train/val như trong `%s`, nên hai nguồn so được với nhau. " % (kept, base) +
         "`test.jsonl` = bản sao `val.jsonl` như mọi nguồn Pha 1.%s\n" % ((" " + note) if note else ""),
         "| tập | số hàm (nhãn 1 / nhãn 0) | %s | %s |" % tuple(LANG_LABEL[g] for g in keep), "|---|---|---|---|"]
    for sp in ("all", "train", "val", "test"):
        t = st[sp]
        L.append("| %s | %s (%s / %s) | %s | %s |" % ((name + ".jsonl" if sp == "all" else "fold1/" + sp, fmt(t["n"]), fmt(t["pos"]), fmt(t["neg"]))
                                                     + tuple(fmt(t["lang"].get(g, 0)) for g in keep)))
    L.append("\nCặp đủ: %s; cặp bị tách giữa train/val: %s (giống hệt các cặp %s bị tách trong `%s`).\n" % (fmt(st["pairs"]), fmt(st["pairs_split"]), kept, base))
    return "\n".join(L) + "\n"


def describe_any(rows):
    lab = collections.Counter(r["label"] for r in rows)
    return {"n": len(rows), "pos": lab[1], "neg": lab[0], "lang": dict(collections.Counter(r["lang"] for r in rows))}


def mixsrc_readme_section(stats):
    L = ["## Một pha trộn nguồn (`mixsrc_common/`, nhánh mwg_mixsrc)\n",
         "Mỗi fold: `train.jsonl` = mọi hàng `phase1_common.jsonl` (C/C++ + Java + JS) + SVEN `train` của fold; `val.jsonl`, "
         "`test.jsonl` = SVEN `val`, `test` của fold (chọn checkpoint và đánh giá hoàn toàn trên đích Python).\n",
         "| fold | train (nhãn 1 / nhãn 0) | trong đó Python | val | test |", "|---|---|---|---|---|"]
    for f, st in stats.items():
        t = st["train"]
        L.append("| %d | %s (%s / %s) | %s | %s | %s |" % (f, fmt(t["n"]), fmt(t["pos"]), fmt(t["neg"]), fmt(t["lang"].get("python", 0)),
                                                       fmt(st["val"]["n"]), fmt(st["test"]["n"])))
    return "\n".join(L) + "\n\n"


def rewrite_md5_table(out, extra_section):
    """Chèn mục mixsrc (nếu chưa có) trước bảng md5 và dựng lại bảng md5 từ MỌI file đang có trong thư mục."""
    p = os.path.join(out, "README.md")
    s = open(p, encoding="utf-8").read()
    head, _, tail = s.partition("## File, số dòng, md5")
    if extra_section.split("\n", 1)[0] not in head:
        head = head + extra_section
    rebuild = tail[tail.find("\nDựng lại:"):] if "\nDựng lại:" in tail else ""
    L = ["## File, số dòng, md5\n\n| file | số dòng | md5 |\n|---|---|---|"]
    for root, dirs, fs in sorted(os.walk(out)):
        dirs.sort()
        for f in sorted(fs):
            q = os.path.join(root, f)
            if os.path.relpath(q, out) == "README.md":
                continue
            L.append("| `%s` | %s | `%s` |" % (os.path.relpath(q, out), fmt(sum(1 for _ in open(q, encoding="utf-8"))), md5(q)))
    open(p, "w", encoding="utf-8").write(head + "\n".join(L) + "\n" + rebuild)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def describe(rows):
    lab = collections.Counter(r["label"] for r in rows)
    lang = collections.Counter(r["lang"] for r in rows)
    pairs = collections.defaultdict(list)
    for r in rows:
        if r["pair_id"]:
            pairs[r["pair_id"]].append(r["label"])
    return {"n": len(rows), "pos": lab[1], "neg": lab[0], "lang": dict(lang),
            "pairs": sum(1 for v in pairs.values() if sorted(v) == [0, 1])}


def main():
    if "--noccpp_only" in sys.argv:
        if not os.path.exists(os.path.join(OUT, "phase1_common", "fold1", "train.jsonl")):
            sys.exit("chưa có %s/phase1_common — dựng đầy đủ trước" % OUT)
        if os.path.exists(os.path.join(OUT, "phase1_common_noccpp")) or os.path.exists(os.path.join(OUT, "phase1_common_noccpp.jsonl")):
            sys.exit("đã có %s/phase1_common_noccpp — xoá trước khi dựng lại" % OUT)
        st = build_noccpp(OUT)
        rewrite_md5_table(OUT, noccpp_readme_section(st))
        print(json.dumps(st, ensure_ascii=False))
        return
    if "--full_noccpp_only" in sys.argv:                  # người dùng 27/09: "thử full no ccpp"
        if not os.path.exists(os.path.join(OUT, "phase1_full", "fold1", "train.jsonl")):
            sys.exit("chưa có %s/phase1_full — dựng đầy đủ trước" % OUT)
        if os.path.exists(os.path.join(OUT, "phase1_full_noccpp")) or os.path.exists(os.path.join(OUT, "phase1_full_noccpp.jsonl")):
            sys.exit("đã có %s/phase1_full_noccpp — xoá trước khi dựng lại" % OUT)
        st = build_noccpp(OUT, "phase1_full")
        rewrite_md5_table(OUT, noccpp_readme_section(st, "phase1_full"))
        print(json.dumps(st, ensure_ascii=False))
        return
    if "--nojava_only" in sys.argv:                       # người dùng 28/09: Java không có CWE mà vào cả common lẫn full
        note = ("Java (CleanVul) không có nhãn CWE nên vào nguyên vẹn cả common lẫn full. Bỏ Java thì mọi hàng nhãn 1 của common "
                "đều mang CWE dùng chung với Python; full vẫn lấy mọi hàng C/C++ + JS như cũ (kể cả 29 hàm JS nhãn 1 không ghi CWE).")
        for base in ("phase1_common", "phase1_full"):
            if not os.path.exists(os.path.join(OUT, base, "fold1", "train.jsonl")):
                sys.exit("chưa có %s/%s — dựng đầy đủ trước" % (OUT, base))
            if os.path.exists(os.path.join(OUT, base + "_nojava")) or os.path.exists(os.path.join(OUT, base + "_nojava.jsonl")):
                sys.exit("đã có %s/%s_nojava — xoá trước khi dựng lại" % (OUT, base))
        for base in ("phase1_common", "phase1_full"):
            st = build_noccpp(OUT, base, "java")
            rewrite_md5_table(OUT, noccpp_readme_section(st, base, "java", note))
            print(base, json.dumps(st, ensure_ascii=False))
        return
    if "--jsonly_only" in sys.argv:                       # người dùng 29/09: "chạy thêm nguồn js only full và common"
        for base in ("phase1_common", "phase1_full"):
            if not os.path.exists(os.path.join(OUT, base, "fold1", "train.jsonl")):
                sys.exit("chưa có %s/%s — dựng đầy đủ trước" % (OUT, base))
            if os.path.exists(os.path.join(OUT, base + "_jsonly")) or os.path.exists(os.path.join(OUT, base + "_jsonly.jsonl")):
                sys.exit("đã có %s/%s_jsonly — xoá trước khi dựng lại" % (OUT, base))
        for base in ("phase1_common", "phase1_full"):
            st = build_lang_subset(OUT, base, ("js",), "jsonly")
            rewrite_md5_table(OUT, jsonly_readme_section(st, base))
            print(base, json.dumps(st, ensure_ascii=False))
        return
    # người dùng 29/09: "thêm bản riêng cho java và ccpp (full/common)" — cùng cách với chỉ JS, không chia lại
    lang_flags = [(lang, flag) for lang, flag in (("java", "--javaonly_only"), ("ccpp", "--ccpponly_only")) if flag in sys.argv]
    if lang_flags:
        notes = {"java": "Java (CleanVul) không có nhãn CWE nên common và full chứa CÙNG 5 184 hàm Java; hai nguồn chỉ Java chỉ khác "
                         "nhau ở cách chia train/val (mỗi nguồn gốc chia 15 % riêng)."}
        for lang, _ in lang_flags:
            for base in ("phase1_common", "phase1_full"):
                if not os.path.exists(os.path.join(OUT, base, "fold1", "train.jsonl")):
                    sys.exit("chưa có %s/%s — dựng đầy đủ trước" % (OUT, base))
                name = base + "_" + lang + "only"
                if os.path.exists(os.path.join(OUT, name)) or os.path.exists(os.path.join(OUT, name + ".jsonl")):
                    sys.exit("đã có %s/%s — xoá trước khi dựng lại" % (OUT, name))
        for lang, _ in lang_flags:
            for base in ("phase1_common", "phase1_full"):
                st = build_lang_subset(OUT, base, (lang,), lang + "only")
                rewrite_md5_table(OUT, langonly_readme_section(st, base, lang, notes.get(lang, "")))
                print(base, lang, json.dumps(st, ensure_ascii=False))
        return
    # người dùng 30/09: "chạy thêm 4cwe chỉ js", "lên lịch 4 cwe js + ccpp" — tách phase1_4cwe theo ngôn ngữ, không chia lại
    if "--cwe4_langonly_only" in sys.argv:
        base = "phase1_4cwe"
        notes = {"ccpp": "Phần C/C++ của 4cwe = nhãn 1 thuộc bốn CWE + 200 nhãn 0 bốc ngẫu nhiên từ common (hàm sạch, không phải bản vá "
                         "của chính hàm lỗi) ⇒ lệch nhãn khoảng 1:2 và hầu như không có cặp đủ hai nhãn."}
        if not os.path.exists(os.path.join(OUT, base, "fold1", "train.jsonl")):
            sys.exit("chưa có %s/%s — dựng đầy đủ trước" % (OUT, base))
        for lang in ("js", "ccpp"):
            name = base + "_" + lang + "only"
            if os.path.exists(os.path.join(OUT, name)) or os.path.exists(os.path.join(OUT, name + ".jsonl")):
                sys.exit("đã có %s/%s — xoá trước khi dựng lại" % (OUT, name))
        for lang in ("js", "ccpp"):
            st = build_lang_subset(OUT, base, (lang,), lang + "only")
            rewrite_md5_table(OUT, langonly_readme_section(st, base, lang, notes.get(lang, "")))
            print(base, lang, json.dumps(st, ensure_ascii=False))
        return
    # người dùng 30/09: "full setting nhưng dùng bộ full bỏ common để đối chứng" → "chỉ thử với ngôn ngữ JS thôi" → đặt tên "uncommon"
    if "--uncommon_jsonly_only" in sys.argv:
        base, minus, name = "phase1_full_jsonly", "phase1_common_jsonly", "phase1_uncommon_jsonly"
        note = ("Luật common (clean_sources_v4): giữ hàm mà MỌI nhãn CWE của nó áp dụng được cho Python (MITRE CWE v4.20) ⇒ phần còn lại là "
                "các cặp JS có ít nhất một nhãn CWE không áp dụng cho Python, hoặc không có CWE; common JS nằm trọn trong full JS nên phép trừ "
                "không cắt cặp nào.")
        for b in (base, minus):
            if not os.path.exists(os.path.join(OUT, b, "fold1", "train.jsonl")):
                sys.exit("chưa có %s/%s — dựng trước" % (OUT, b))
        if os.path.exists(os.path.join(OUT, name)) or os.path.exists(os.path.join(OUT, name + ".jsonl")):
            sys.exit("đã có %s/%s — xoá trước khi dựng lại" % (OUT, name))
        st = build_minus(OUT, base, minus, name)
        rewrite_md5_table(OUT, minus_readme_section(st, base, minus, name, "Uncommon chỉ JS", note))
        print(json.dumps(st, ensure_ascii=False))
        return
    # người dùng 01/10: "chạy thêm bản uncommon của CCPP kèm với uncommon của js (nhưng không bao gồm các mẫu unknown cwe)"
    if "--uncommon_nojava_knowncwe_only" in sys.argv:
        base, minus, name = "phase1_full_nojava", "phase1_common_nojava", "phase1_uncommon_nojava_knowncwe"
        note = ("C/C++: hàm nhãn 1 ngoài common là hàm có ít nhất một CWE không áp dụng cho Python (luật MITRE CWE v4.20 của clean_sources_v4); "
                "hàm nhãn 0 KHÔNG lọc theo CWE (hàm sạch, CWE ghi kèm chỉ là CWE của commit) nên nhãn 0 ở đây là nửa còn lại của 6 000 hàm "
                "nhãn 0 của full sau khi common bốc ngẫu nhiên 3 000 (seed 42). JS: như `phase1_uncommon_jsonly` nhưng bỏ các cặp không có CWE. "
                "Java không có (không có nhãn CWE).")
        for b in (base, minus):
            if not os.path.exists(os.path.join(OUT, b, "fold1", "train.jsonl")):
                sys.exit("chưa có %s/%s — dựng trước" % (OUT, b))
        if os.path.exists(os.path.join(OUT, name)) or os.path.exists(os.path.join(OUT, name + ".jsonl")):
            sys.exit("đã có %s/%s — xoá trước khi dựng lại" % (OUT, name))
        st = build_minus_knowncwe(OUT, base, minus, name)
        rewrite_md5_table(OUT, minus_knowncwe_readme_section(st, base, minus, name, "Uncommon C/C++ + JS, chỉ hàm có CWE", note))
        print(json.dumps({k: v for k, v in st.items() if k != "cwe"}, ensure_ascii=False))
        return
    # người dùng 01/10: uncommon CHẶT - bỏ mẫu có bất kỳ nhãn CWE nào thuộc common (và nhãn không phải mã CWE); nhãn 0 C/C++ theo CWE commit: bỏ, không bù
    if "--uncommon_strict_only" in sys.argv:
        jobs = [("phase1_full_jsonly", "phase1_common_jsonly", "phase1_uncommon_jsonly_strict", "Uncommon chỉ JS, CHẶT",
                 "Thay cho `phase1_uncommon_jsonly` (bộ đó còn cặp có nhãn common, vd CWE-79 + CWE-1321, và cặp không có CWE)."),
                ("phase1_full_nojava", "phase1_common_nojava", "phase1_uncommon_nojava_strict", "Uncommon C/C++ + JS, CHẶT",
                 "Thay cho `phase1_uncommon_nojava_knowncwe`. Chỉ xét CWE ở NHÃN 1 (người dùng 01/10); JS: nhãn 0 đi theo nửa nhãn 1 cùng cặp; C/C++: nhãn 0 "
                 "là hàm sạch, giữ nguyên phần bù trong 6 000 hàm nhãn 0 đã undersample của full (không lọc theo CWE của commit).")]
        # `--only=nojava`: chỉ dựng lại bộ có tên chứa chuỗi đó (đối số không có `--` sẽ bị hiểu là thư mục OUT)
        want = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
        jobs = [j for j in jobs if not want or any(w in j[2] for w in want)]
        for base, minus, name, title, note in jobs:
            for b in (base, minus):
                if not os.path.exists(os.path.join(OUT, b, "fold1", "train.jsonl")):
                    sys.exit("chưa có %s/%s — dựng trước" % (OUT, b))
            if os.path.exists(os.path.join(OUT, name)) or os.path.exists(os.path.join(OUT, name + ".jsonl")):
                sys.exit("đã có %s/%s — xoá trước khi dựng lại" % (OUT, name))
        for base, minus, name, title, note in jobs:
            st = build_minus_strict(OUT, base, minus, name)
            # dựng lại: gỡ mục README cũ cùng tiêu đề trước (rewrite_md5_table chỉ CHÈN mục khi chưa có tiêu đề đó)
            rp = os.path.join(OUT, "README.md"); s = open(rp, encoding="utf-8").read(); hdr = "## %s (`%s/`)" % (title, name)
            if hdr in s:
                a = s.index(hdr); b = s.find("\n## ", a + len(hdr))
                open(rp, "w", encoding="utf-8").write(s[:a] + (s[b + 1:] if b >= 0 else ""))
            rewrite_md5_table(OUT, minus_strict_readme_section(st, base, minus, name, title, note))
            print(name, json.dumps({k: v for k, v in st.items() if k != "cwe"}, ensure_ascii=False))
        return
    if "--pair_subsample_only" in sys.argv:                # người dùng 02/10: "chủ động kiểm theo hướng của tôi"
        base, like, name = "phase1_common_jsonly", "phase1_uncommon_jsonly_strict", "phase1_common_jsonly_n60"
        for b in (base, like):
            if not os.path.exists(os.path.join(OUT, b, "fold1", "train.jsonl")):
                sys.exit("chưa có %s/%s — dựng trước" % (OUT, b))
        if os.path.exists(os.path.join(OUT, name)) or os.path.exists(os.path.join(OUT, name + ".jsonl")):
            sys.exit("đã có %s/%s — xoá trước khi dựng lại" % (OUT, name))
        st = build_pair_subsample(OUT, base, like, name)
        rewrite_md5_table(OUT, pair_subsample_readme_section(st, base, like, name, "Common chỉ JS rút còn 60 cặp (cùng cỡ uncommon chỉ JS chặt)"))
        print(name, json.dumps(st, ensure_ascii=False))
        return
    if "--mixsrc_only" in sys.argv:
        if not os.path.exists(os.path.join(OUT, "phase1_common.jsonl")):
            sys.exit("chưa có %s — dựng đầy đủ trước" % OUT)
        if os.path.exists(os.path.join(OUT, "mixsrc_common")):
            sys.exit("đã có %s/mixsrc_common — xoá trước khi dựng lại" % OUT)
        st = build_mixsrc(OUT)
        rewrite_md5_table(OUT, mixsrc_readme_section(st))
        print(json.dumps(st, ensure_ascii=False))
        return
    if os.path.exists(OUT):
        sys.exit("đã tồn tại: %s — xoá trước khi dựng lại" % OUT)
    ccpp = build_ccpp()
    java = {t: load_jsonl(os.path.join(CLEAN, "cleanvul_java", t + ".jsonl")) for t in N_NEG}
    js = {t: load_jsonl(os.path.join(CLEAN, "cleanvul_js", t + ".jsonl")) for t in N_NEG}
    stats, files, split_stats, unpaired_halves = {}, [], {}, {}
    for tag in N_NEG:
        c_rows, unpaired_halves[tag] = only_complete_pairs(ccpp[tag])
        rows = c_rows + java[tag] + js[tag]
        assert len({r["id"] for r in rows}) == len(rows), "id trùng trong %s" % tag
        p = os.path.join(OUT, "phase1_%s.jsonl" % tag); write_jsonl(p, rows); files.append(p)
        stats[tag] = describe(rows); stats[tag]["ccpp"] = describe(c_rows)
        train, val = split_fold1(rows)
        for name, part in (("train", train), ("val", val), ("test", val)):
            p = os.path.join(OUT, "phase1_%s" % tag, "fold1", name + ".jsonl"); write_jsonl(p, part); files.append(p)
        side = collections.defaultdict(set)
        vids = {r["id"] for r in val}
        for r in rows:
            if r["pair_id"]:
                side[r["pair_id"]].add(r["id"] in vids)
        split_stats[tag] = {"train": describe(train), "val": describe(val), "pairs_split": sum(1 for v in side.values() if len(v) == 2)}
        print(tag, json.dumps(stats[tag], ensure_ascii=False), flush=True)
    # chỉ chép dữ liệu: thư mục nguồn có thể mang .git (24/09 có người `git init` + remote sven_python_folds ở đó)
    shutil.copytree(SVEN, os.path.join(OUT, os.path.basename(SVEN)), ignore=shutil.ignore_patterns(".*"))
    for root, _, fs in sorted(os.walk(os.path.join(OUT, os.path.basename(SVEN)))):
        for f in sorted(fs):
            files.append(os.path.join(root, f))
    write_readme(stats, split_stats, unpaired_halves, files)
    rewrite_md5_table(OUT, mixsrc_readme_section(build_mixsrc(OUT)))
    rewrite_md5_table(OUT, noccpp_readme_section(build_noccpp(OUT)))


def write_readme(stats, split_stats, unpaired_halves, files):
    s4 = stats["4cwe"]
    L = ["""# Dữ liệu thí nghiệm final (dựng 24/09/2026)

Thư mục DUY NHẤT dùng cho mọi thí nghiệm final: nguồn Pha 1 (C/C++ + Java + JS) và đích Pha 2 (SVEN Python).
Mã trong mọi file đã **xoá comment**, đã **khử trùng** (lỗi nhãn, trùng nguyên văn, trùng dạng abstract) và loại hàm không thân.

## Dùng file nào

| mục đích | đường dẫn |
|---|---|
| Pha 1, đủ mọi CWE | `phase1_full/` (chia sẵn `fold1/{train,val,test}.jsonl`) — hoặc `phase1_full.jsonl` (mọi hàng) |
| Pha 1, CWE dùng chung được với Python | `phase1_common/` — hoặc `phase1_common.jsonl` |
| Pha 1, 4 CWE: 022, 078, 079, 089 | `phase1_4cwe/` — hoặc `phase1_4cwe.jsonl` |
| Pha 2 / baseline: đích Python, 5 fold | `sven_python_folds_nocomment/fold{1..5}/{train,val,test}.jsonl` |

Pha 1 chạy MỘT lần trên `phase1_<loại>/fold1`: `val.jsonl` = 15 % hàng ngẫu nhiên (seed 42) để chọn checkpoint,
`test.jsonl` = bản sao `val.jsonl` (đánh giá thật nằm ở Pha 2).

## Số mẫu Pha 1
"""]
    L.append("| file | số hàm | nhãn 1 / nhãn 0 | C/C++ (nhãn 1 / nhãn 0) | Java | JS | cặp đủ |\n|---|---|---|---|---|---|---|")
    for tag in N_NEG:
        s, c = stats[tag], stats[tag]["ccpp"]
        L.append("| `phase1_%s` | %s | %s / %s | %s (%s / %s) | %s | %s | %s |" % (
            tag, fmt(s["n"]), fmt(s["pos"]), fmt(s["neg"]), fmt(c["n"]), fmt(c["pos"]), fmt(c["neg"]),
            fmt(s["lang"].get("java", 0)) if s["lang"].get("java") else "—", fmt(s["lang"].get("js", 0)), fmt(s["pairs"])))
    L.append("\n| thư mục | train (nhãn 1 / nhãn 0) | val = test (nhãn 1 / nhãn 0) | cặp bị tách giữa train/val |\n|---|---|---|---|")
    for tag in N_NEG:
        t, v = split_stats[tag]["train"], split_stats[tag]["val"]
        L.append("| `phase1_%s/fold1` | %s (%s / %s) | %s (%s / %s) | %s |" % (
            tag, fmt(t["n"]), fmt(t["pos"]), fmt(t["neg"]), fmt(v["n"]), fmt(v["pos"]), fmt(v["neg"]), fmt(split_stats[tag]["pairs_split"])))
    L.append("""
## Cách dựng phần C/C++

Nguồn: PrimeVul v0.1 **paired + unpaired gộp** (paired nằm trọn trong unpaired), lọc lại trên toàn bộ: bỏ mọi bản của mã giống hệt
mang cả nhãn 0 và 1, bỏ hàm không thân, giữ một bản cho mỗi nhóm trùng nguyên văn (bỏ khoảng trắng) hoặc trùng dạng abstract
(định danh → ID, chuỗi → S, số → N) cùng nhãn; khi trùng thì giữ bản thuộc cặp. Kết quả: 197 063 hàm (5 888 nhãn 1 / 191 175 nhãn 0),
trong đó 4 602 cặp (bản lỗi + bản vá).

1. **full**: mọi hàm nhãn 1 (5 888) + **6 000** hàm nhãn 0 = mọi hàm nhãn 0 thuộc cặp (4 602 bản vá) + 1 398 hàm nhãn 0 lẻ bốc ngẫu
   nhiên (seed 42).
2. **common**: mọi hàm nhãn 1 có CWE dùng chung được với Python (luật MITRE CWE v4.20) + **3 000** hàm nhãn 0 bốc ngẫu nhiên (seed 42)
   **từ 6 000 hàm nhãn 0 của full**.
3. **4cwe**: mọi hàm nhãn 1 có CWE-022/078/079/089 + **200** hàm nhãn 0 bốc ngẫu nhiên (seed 42) **từ 3 000 hàm nhãn 0 của common**.

Nhãn 0 là hàm sạch nên **không lọc theo CWE** (CWE ghi kèm của nó chỉ là CWE của commit); hàm nhãn 0 của `4cwe` ⊂ của `common` ⊂
của `full`. Ở `4cwe`, `cwe_class` là chỉ số 0–3 (theo thứ tự CWE-022, 078, 079, 089); hàm nhãn 0 có CWE ngoài bốn loại giữ `cwe`/`cwe_id`
gốc và có `cwe_class` = **-100** (head phụ bỏ qua) — %s/%s hàm nhãn 0 của `4cwe`.
`pair_id` chỉ có giá trị khi cả hai nửa của cặp cùng nằm trong file; nửa kia không được bốc thì `pair_id` = null
(C/C++: full %s, common %s, 4cwe %s nửa).

Java (CleanVul mức 3–4, GitHub yikun-li/CleanVul @ cbad711; không có nhãn CWE nên vào `common` nguyên vẹn, không có trong `4cwe`) và
JS (CleanVul mức 3–4) lấy nguyên như nguồn chuẩn `clean_sources_v4`, đủ cặp.

## Đích SVEN (`sven_python_folds_nocomment/`)

Chép nguyên `data/sven_python_folds_nocomment`: 760 hàm Python (380 nhãn 1 / 380 nhãn 0), 5 fold, mỗi fold train/val/test =
456/152/152, cùng fold và thứ tự với `sven_python_folds_norm`; chỉ khác là đã xoá comment `#` và docstring (không còn comment nào;
AST trùng bản gốc bỏ docstring ở 748/748 hàm parse được). `data.jsonl` = cả 760 hàm.

## Trường của mỗi dòng (Pha 1)

| trường | ý nghĩa |
|---|---|
| `code` | mã hàm, đã xoá comment |
| `label` | 1 = có lỗ hổng, 0 = sạch |
| `lang` | `ccpp`, `java`, `js` |
| `cwe` / `cwe_id` | CWE chính (`CWE-079` / 79); rỗng / -1 nếu nguồn không gán |
| `cwe_class` | full/common: pillar CWE-1000 của CWE chính (-100 nếu không tra được); 4cwe: 0–3, hoặc -100 |
| `id` / `pair_id` | mã hàm trong nguồn gốc / mã cặp (null nếu không đủ cặp trong file) |
| `cve`, `project`, `file_name`, `commit_url` | thông tin nguồn nếu có |

Trường SVEN: `code, label, cwe, cwe_id, cwe_class, lang`.

## File, số dòng, md5

| file | số dòng | md5 |
|---|---|---|""" % (fmt(sum(1 for r in load_jsonl(os.path.join(OUT, "phase1_4cwe.jsonl")) if r["label"] == 0 and r["cwe_class"] == -100)),
                  fmt(s4["ccpp"]["neg"]), unpaired_halves["full"], unpaired_halves["common"], unpaired_halves["4cwe"]))
    for p in files:
        L.append("| `%s` | %s | `%s` |" % (os.path.relpath(p, OUT), fmt(sum(1 for _ in open(p, encoding="utf-8"))), md5(p)))
    L.append("\nDựng lại: `python tools/v4/build_final_experiment_data.py data/final_experiment_data` (từ thư mục gốc MultiVD; "
             "cần `data/sources_v4`, `data/clean_sources_v4`, `data/sven_python_folds_nocomment`).")
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
