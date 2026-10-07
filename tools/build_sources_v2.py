#!/usr/bin/env python3
"""Dung lai bo nguon v2 tu DU LIEU GOC, xoa comment dung theo tung ngon ngu.

Nguon:
  ccpp  <- PrimeVul  /drive1/cuongtm/ntat/Archive/PrimeVul/PrimeVul/*_paired_norm.jsonl
           (train+val+test GOP LAM MOT; da doi chieu tung dong == PrimeVulRaw/*.jsonl)
  java  <- CleanVul  vulnerability_score_{3,4}.csv, loc extension == "java"
  js    <- CleanVul  vulnerability_score_{3,4}.csv, loc extension == "js"

Schema ra — DUNG Y HET `data/sven_python_folds_norm/data.jsonl`:
    code (str) | label (int) | cwe (str) | cwe_id (int) | cwe_class (int) | lang (str)

  * `cwe`       chuan hoa "CWE-%03d"; khong co nhan -> ""
  * `cwe_id`    so nguyen; khong co nhan -> -1
  * `cwe_class` bo *_4cwe : 0..3 theo thu tu CWE-022/078/079/089 — DUNG NHU sven
                bo *_full : chi so pillar CWE-1000 (10 lop, bang data/cwe_root_parents.txt),
                            khong tra cuu duoc -> -100 (ignore_index, theo src/build_parent_labels.py)
  * `lang`      "ccpp" | "java" | "js"

Moi bo ghi ra ba file:
    <ten>.json        mang JSON (dung de chia se)
    <ten>.jsonl       moi dong mot ban ghi (dung cho pipeline cua repo)
    <ten>.meta.jsonl  xuat xu + ma GOC chua xoa comment, cung thu tu dong
"""
import argparse, ast, collections, csv, hashlib, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strip_comments_multi import strip

csv.field_size_limit(10 ** 9)

ARCH = "/drive1/cuongtm/ntat/Archive/PrimeVul/PrimeVul"
FOUR = [22, 78, 79, 89]                      # dung thu tu cua sven: 022,078,079,089 -> 0,1,2,3
IGNORE = -100
PILLAR_LINE = re.compile(r"^CWE-\s*(\d+):\s*(?:CWE-\s*(\d+)|\(ROOT)")


def load_pillars(path):
    par = collections.defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            m = PILLAR_LINE.match(ln.strip())
            if m and m.group(2):
                par[int(m.group(1))].add(int(m.group(2)))
    pillars = sorted({p for s in par.values() for p in s})
    idx = {p: i for i, p in enumerate(pillars)}
    # CWE nhieu pillar -> lay pillar ID NHO NHAT (tat dinh, theo src/build_parent_labels.py)
    out = {c: idx[min(s)] for c, s in par.items()}
    # BAT BUOC: chinh muoi pillar phai tra cuu duoc CHINH NO. Trong bang chung ghi la
    # "CWE- 703: (ROOT - No parent)" nen khong co dong cha -> vong tren bo qua -> mot hang
    # co cwe_id DUNG BANG mot pillar se bi gan -100. Da do: 563 hang bi sai vi loi nay
    # (CWE-703 419 hang, CWE-284 128, CWE-682/697/664/707 16).
    for pl in pillars:
        out[pl] = idx[pl]
    return out, pillars


def norm_cwe_label(x):
    r"""Chuan hoa MOT nhan -> ("num", 79) hoac ("text", "nvd-cwe-noinfo") hoac None.

    Theo dung `apply_dataset.py:norm_label` cua kho cwe-common (nhanh `fix`):
    fullmatch cua `(?:cwe[-_ ]?)?0*(\d+)` khong phan biet hoa thuong. Nho vay
    `CWE-022`, `cwe_22`, `022`, `22` deu ra CUNG mot khoa.
    Them mot nhanh `search` cho yeu cau "neu chua so thi map theo so" — `CWE-79 (XSS)`
    van ra 79. Nhan khong co so nao thi so theo VAN BAN DAY DU, ha chu, gom khoang trang.
    """
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none", "null", "n/a", "na"):
        return None
    m = re.fullmatch(r"(?:cwe[-_ ]?)?0*(\d+)", s, re.I)
    if m:
        return ("num", int(m.group(1)))
    m = re.search(r"cwe[-_ ]?0*(\d+)", s, re.I)
    if m:
        return ("num", int(m.group(1)))
    return ("text", re.sub(r"\s+", " ", s).lower())


def cwe_labels(raw):
    """Tach o nguon thanh DANH SACH NHAN THO, giu nguyen van.

    PrimeVul: mot chuoi ("cwe-310"). CleanVul: chuoi cua mot list Python
    ("['CWE-74', 'CWE-79']"). Phai giu nguyen van vi `unknown` / `NVD-CWE-*`
    khong co so nao — chung chi so duoc theo van ban.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                items = [str(x) for x in v] if isinstance(v, (list, tuple)) else [s]
            except Exception:
                items = [m.group(2) for m in re.finditer(r"(['\"])(.*?)\1", s)]
        else:
            items = [s]
    return [str(x).strip() for x in items if str(x).strip()]


def load_drops(path):
    """Doc drop_*.txt -> (tap so, tap van ban). Bo dong trong va dong # chu thich."""
    nums, txts = set(), set()
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            k = norm_cwe_label(ln)
            if k is None:
                continue
            (nums if k[0] == "num" else txts).add(k[1])
    return nums, txts


def dropped_by_common(labels, nums, txts):
    """True = BO hang. Luat: cwe rong -> bo; dinh MOT nhan trong danh sach -> bo ca hang."""
    if not labels:
        return True, "cwe rong"
    for l in labels:
        k = norm_cwe_label(l)
        if k is None:
            return True, "nhan rong"
        if k[0] == "num" and k[1] in nums:
            return True, "CWE-%d" % k[1]
        if k[0] == "text" and k[1] in txts:
            return True, str(l)
    return False, ""


def cwe_all(raw):
    """Tra ve TAT CA so CWE trong o nguon, theo thu tu.

    CleanVul ghi cot `cwe_id` duoi dang chuoi cua mot danh sach Python, vi du
    "['CWE-74', 'CWE-79']" — 132 hang js co tu hai CWE tro len. Lay moi so DAU TIEN
    lam 8 hang mang CWE-079 o vi tri thu hai bi loai oan khoi bo 4cwe.
    Bo qua cac nhan khong phai so nhu 'NVD-CWE-noinfo', 'NVD-CWE-Other'.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none", "null", "na"):
            return []
        items = [s]
    out = []
    for it in items:
        for m in re.finditer(r"CWE-\s*(\d+)|(?<![\w-])(\d{1,4})(?![\w-])", str(it), re.I):
            v = int(m.group(1) or m.group(2))
            if v not in out:
                out.append(v)
    return out


def cwe_int(raw):
    """CWE chinh = cai dau tien. Giu lai de cac cho khac dung nhu cu."""
    a = cwe_all(raw)
    return a[0] if a else None


def make_row(code, label, cid, lang, pil):
    return {
        "code": code,
        "label": int(label),
        "cwe": ("CWE-%03d" % cid) if cid is not None else "",
        "cwe_id": int(cid) if cid is not None else -1,
        "cwe_class": pil.get(cid, IGNORE) if cid is not None else IGNORE,
        "lang": lang,
    }


# --------------------------------------------------------------------- nguon
def read_primevul(paired=True):
    """train+val+test gop lam mot. Tra ve list dict tho."""
    names = (["train_paired_norm.jsonl", "val_paired_norm.jsonl", "test_paired_norm.jsonl"]
             if paired else ["train_norm.jsonl", "val_norm.jsonl", "test_norm.jsonl"])
    out = []
    for nm in names:
        split = nm.split("_")[0]
        with open(os.path.join(ARCH, nm), encoding="utf-8") as fh:
            for i, ln in enumerate(fh):
                d = json.loads(ln)
                out.append({"code": d["code"], "label": int(d["label"]),
                            "cwe_raw": d.get("CWE_ID"), "split_goc": split, "idx_goc": i})
    return out


def read_cleanvul(data_dir, ext):
    """score_3 + score_4, loc theo extension. Moi cap -> hai hang (1 = truoc va, 0 = sau va)."""
    rows, seen, dup = [], set(), 0
    for s in (3, 4):
        path = os.path.join(data_dir, "vulnerability_score_%d.csv" % s)
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if (r.get("extension") or "").strip().lower() != ext:
                    continue
                before, after = r.get("func_before") or "", r.get("func_after") or ""
                key = (before, after)
                if key in seen:            # dedup y het src/dataset_converter.py cua CleanVul
                    dup += 1
                    continue
                seen.add(key)
                base = {"cwe_raw": r.get("cwe_id"), "cve": r.get("cve_id"),
                        "commit_url": r.get("commit_url"), "file_name": r.get("file_name"),
                        "score": s, "date": r.get("date"),
                        "pair_id": "cleanvul:%s" % hashlib.md5((before + "\x00" + after).encode()).hexdigest()[:16]}
                rows.append(dict(base, code=before, label=1, half="before"))
                rows.append(dict(base, code=after, label=0, half="after"))
    return rows, dup


# ---------------------------------------------------------------------- build
def build(raw_rows, lang, pil, tag):
    recs, metas = [], []
    warn = collections.Counter()
    empties = 0
    for r in raw_rows:
        code0 = r["code"]
        code, st = strip(code0, "c" if lang == "ccpp" else lang)
        for k, v in st.items():
            if k in ("n_spans", "chars_removed", "lines_in", "lines_out"):
                continue
            if v:
                warn[k] += v
        if not code.strip():
            empties += 1
            continue
        cids = cwe_all(r.get("cwe_raw"))
        cid = cids[0] if cids else None
        recs.append(make_row(code, r["label"], cid, lang, pil))
        m = {k: v for k, v in r.items() if k != "code"}
        m.update({"cwe_all": cids, "cwe_labels": cwe_labels(r.get("cwe_raw")),
                  "code_raw": code0, "chars_removed": st["chars_removed"],
                  "n_comment_spans": st["n_spans"],
                  "lines_in": st["lines_in"], "lines_out": st["lines_out"],
                  "sha1_raw": hashlib.sha1(code0.encode("utf-8", "replace")).hexdigest()})
        metas.append(m)
    return recs, metas, warn, empties


def mark_conflicts(recs, metas):
    """Danh dau hang co `code` TRUNG HET nhau nhung NHAN NGUOC nhau.

    Xuat hien khi ban va chi sua COMMENT: hai nua cua cap thanh mot chuoi giong het
    sau khi xoa comment, mang hai nhan nguoc nhau => nhieu nhan khong hoc duoc.
    Tren `code_raw` KHONG co cap nao nhu vay — day hoan toan la he qua cua buoc xoa.
    KHONG xoa am tham: chi danh dau de nguoi dung tu quyet loc hay giu.
    """
    by = collections.defaultdict(list)
    for i, r in enumerate(recs):
        by[r["code"]].append(i)
    n_conf = n_pair = 0
    for _, idxs in by.items():
        if len(idxs) < 2:
            continue
        labs = {recs[i]["label"] for i in idxs}
        dup_raw = len({metas[i].get("code_raw") for i in idxs}) == 1
        for i in idxs:
            metas[i]["dup_group_size"] = len(idxs)
            metas[i]["dup_same_raw"] = dup_raw
        if len(labs) > 1:                       # nhan nguoc nhau
            n_conf += len(idxs)
            for i in idxs:
                metas[i]["dup_label_conflict"] = True
            # cung mot cap va (pair_id hoac hai chi so lien tiep cua PrimeVul)
            pids = {metas[i].get("pair_id") for i in idxs}
            if len(idxs) == 2 and (len(pids) == 1 and None not in pids
                                   or abs(idxs[0] - idxs[1]) == 1):
                n_pair += 1
                for i in idxs:
                    metas[i]["patch_changed_only_comment"] = True
    return n_conf, n_pair


def write_set(outdir, name, recs, metas):
    os.makedirs(outdir, exist_ok=True)
    pj = os.path.join(outdir, name + ".json")
    pl = os.path.join(outdir, name + ".jsonl")
    pm = os.path.join(outdir, name + ".meta.jsonl")
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(recs, fh, ensure_ascii=False, indent=None)
    with open(pl, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(pm, "w", encoding="utf-8") as fh:
        for m in metas:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    return pj, pl, pm


# Chi ccpp (PrimeVul) va js (CleanVul) co bo common — nguoi dung neu 21/09.
DROP_FILE = {"ccpp": "drop_primevul_ccpp_py.txt", "js": "drop_cleanvul_js_py.txt"}


_RULES_CACHE = {}


def get_rules(xml_path, target="Python"):
    """Bo luat MITRE R1-R6, nap mot lan (XML 18 MB, parse ~5s)."""
    key = (xml_path, target)
    if key not in _RULES_CACHE:
        import cwe_rules
        _RULES_CACHE[key] = cwe_rules.Rules(cwe_rules.load_catalog(xml_path), target)
    return _RULES_CACHE[key]


def subset_common(recs, metas, lang, xml_path, target="Python"):
    """Bo `common`: giu hang ma MOI nhan CWE cua no deu duoc luat MITRE GIU.

    Luat R1-R6 doc thang tu `cwec_v4.20.xml`, cai dat o `tools/cwe_rules.py`
    (vendored tu maytinhdibo/GraphTransferVD nhanh `fix` commit 79f0308) — khong con
    danh sach tay nao, va hai ben ra cung con so.

      R1  nhan khong phai ma CWE (unknown / NVD-CWE-* / rong)   -> BO
      R2  Category|View: >=50% thanh vien giu -> GIU; Deprecated/Obsolete -> BO
      R3  khai `Not Language-Specific`                          -> GIU
      R4  khai thang ngon ngu dich, hoac lop chua no            -> GIU
      R5  chi khai ngon ngu khac                                -> BO
      R6  khong khai: ke thua to tien gan nhat; khong co -> GIU (yeu)

    Mot hang dinh MOT nhan bi bo la BO CA HANG (nguoi dung neu 21/09).

    JAVA la ngoai le do nguoi dung chot: CleanVul khong gan nhan CWE cho java
    (0/2754 cap) nen R1 se xoa 100% java — nguoc han y "common se co java".
    Vay java vao common NGUYEN VEN, va ghi ro ly do trong thong ke.
    """
    if lang == "java":
        return list(recs), list(metas), {"(java vao nguyen ven: CleanVul khong co nhan CWE cho java)": 0}
    if not xml_path or not os.path.exists(xml_path):
        return None, None, {}
    rules = get_rules(xml_path, target)
    keep_flag, why = [], collections.Counter()
    for r, m in zip(recs, metas):
        lbls = m.get("cwe_labels") or [None]
        drop, reason = False, ""
        for l in lbls:
            k, rule, _ = rules.decide(l if l is not None else "")
            if not k:
                drop, reason = True, "%s [%s]" % (l if l else "(cwe rong)", rule)
                break
        keep_flag.append(not drop)
        if drop:
            why[reason] += 1
    n_orphan = 0
    groups = collections.defaultdict(list)
    for i, m in enumerate(metas):
        groups[m.get("pair_id") if m.get("pair_id") else ("seq", i // 2)].append(i)
    for _, idxs in groups.items():
        if len(idxs) == 2 and keep_flag[idxs[0]] != keep_flag[idxs[1]]:
            for i in idxs:
                if keep_flag[i]:
                    metas[i]["pair_broken_by_common"] = True
                    n_orphan += 1
    R = [r for r, k in zip(recs, keep_flag) if k]
    M = [m for m, k in zip(metas, keep_flag) if k]
    if n_orphan:
        why["(ghi chu) hang con lai mo coi vi nua kia bi bo"] = n_orphan
    return R, M, dict(why.most_common())


def subset_4cwe(recs, metas):
    """Lay moi hang co IT NHAT MOT CWE thuoc bon CWE dich.

    Xet ca danh sach chu khong chi CWE dau tien: mot hang gan ['CWE-74','CWE-79']
    van la ca CWE-079. Khi co nhieu hon mot CWE dich thi lay cai co thu tu nho nhat
    trong FOUR cho tat dinh, va GHI DE `cwe`/`cwe_id` theo dung CWE duoc chon de ba
    truong cwe/cwe_id/cwe_class luon nhat quan.
    """
    keep = {c: i for i, c in enumerate(FOUR)}
    R, M = [], []
    for r, m in zip(recs, metas):
        cands = [c for c in (m.get("cwe_all") or ([r["cwe_id"]] if r["cwe_id"] >= 0 else []))
                 if c in keep]
        if not cands:
            continue
        pick = min(cands, key=lambda c: FOUR.index(c))
        r2 = dict(r)
        r2["cwe_id"] = pick
        r2["cwe"] = "CWE-%03d" % pick
        r2["cwe_class"] = keep[pick]
        R.append(r2); M.append(m)
    return R, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanvul_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pillars", default="/drive1/cuongtm/ntat/MultiVD/data/cwe_root_parents.txt")
    ap.add_argument("--cwe_xml", default="/drive1/cuongtm/ntat/MultiVD/data/cwe_spec/cwec_v4.20.xml",
                    help="dac ta MITRE CWE; bo `common` dung LUAT R1-R6 doc tu day")
    ap.add_argument("--cwe_target", default="Python", help="ngon ngu dich cua tieu chi common")
    ap.add_argument("--with_unpaired", action="store_true",
                    help="them ca PrimeVul KHONG paired (224 533 hang, lech nhan manh)")
    a = ap.parse_args()

    pil, pillars = load_pillars(a.pillars)
    print("pillar CWE-1000:", pillars, "| %d CWE tra cuu duoc" % len(pil))

    jobs = [("ccpp_primevul_from-paired", "ccpp", lambda: (read_primevul(True), 0))]
    if a.with_unpaired:
        jobs.append(("ccpp_primevul_from-unpaired", "ccpp", lambda: (read_primevul(False), 0)))
    jobs.append(("java_cleanvul_3-4", "java", lambda: read_cleanvul(a.cleanvul_dir, "java")))
    jobs.append(("js_cleanvul_3-4", "js", lambda: read_cleanvul(a.cleanvul_dir, "js")))

    summary = []
    for base, lang, fn in jobs:
        raw, dup = fn()
        recs, metas, warn, empties = build(raw, lang, pil, base)
        n_conf, n_pair = mark_conflicts(recs, metas)
        pj, pl, pm = write_set(a.out, base + "_full", recs, metas)
        r4, m4 = subset_4cwe(recs, metas)
        if r4:
            write_set(a.out, base + "_4cwe", r4, m4)
        rc, mc, why = subset_common(recs, metas, lang, a.cwe_xml)
        if rc:
            write_set(a.out, base + "_common", rc, mc)
        lab = collections.Counter(r["label"] for r in recs)
        cwes = collections.Counter(r["cwe"] for r in recs)
        summary.append({
            "bo": base, "lang": lang, "n_full": len(recs), "n_4cwe": len(r4),
            "n_common": len(rc) if rc else 0, "common_bo_vi": why,
            "label": dict(lab), "cap_trung_bo": dup, "rong_sau_xoa": empties,
            "so_cwe_khac_nhau": len([c for c in cwes if c]), "khong_co_cwe": cwes.get("", 0),
            "hang_nhan_xung_dot_sau_xoa": n_conf, "cap_va_chi_sua_comment": n_pair,
            "canh_bao": dict(warn),
        })
        print("\n=== %s (%s) ===" % (base, lang))
        print("  full: %d hang  | nhan %s" % (len(recs), dict(lab)))
        print("  4cwe: %d hang" % len(r4))
        if rc is not None:
            lab_c = collections.Counter(r["label"] for r in rc)
            print("  common: %d hang (bo %d) | nhan %s" % (len(rc), len(recs) - len(rc), dict(lab_c)))
            top = list(why.items())[:8]
            print("     bo nhieu nhat:", ", ".join("%s=%d" % (k, v) for k, v in top))
        print("  cap trung bi bo: %d | rong sau khi xoa comment: %d" % (dup, empties))
        print("  CWE khac nhau: %d | khong co nhan CWE: %d" % (len([c for c in cwes if c]), cwes.get("", 0)))
        print("  nhan XUNG DOT sau khi xoa comment: %d hang (%d cap ma ban va chi sua comment)"
              % (n_conf, n_pair))
        print("  canh bao quet:", dict(warn))

    with open(os.path.join(a.out, "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\nDa ghi vao", a.out)


if __name__ == "__main__":
    main()
