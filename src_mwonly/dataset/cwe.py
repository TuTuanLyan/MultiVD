"""Nhãn CWE: chuẩn hoá, lớp pillar CWE-1000, bộ `common` (luật MITRE R1–R6) và bộ `4cwe`.

Bộ `common` cho đích Python: giữ hàm mà MỌI nhãn CWE của nó được luật đọc từ đặc tả MITRE (cwec_v4.20.xml) giữ:
  R1  nhãn không phải mã CWE (unknown, NVD-CWE-*, rỗng), hoặc không có trong đặc tả       -> BỎ
  R2  Deprecated / Obsolete -> BỎ; Category / View: >= 50 % thành viên (Has_Member) giữ -> GIỮ
  R3  Weakness khai `Not Language-Specific`                                                 -> GIỮ
  R4  khai thẳng ngôn ngữ đích, hoặc lớp chứa nó (Python ⊂ Interpreted)                    -> GIỮ
  R5  chỉ khai các ngôn ngữ khác                                                             -> BỎ
  R6  không khai: kế thừa tổ tiên gần nhất có khai (ChildOf, view 1000); không có -> GIỮ
Java vào `common` nguyên vẹn: CleanVul không gán CWE cho Java (luật R1 sẽ xoá hết).
"""
import ast
import collections
import os
import re
import xml.etree.ElementTree as ET

FOUR = [22, 78, 79, 89]                      # thứ tự lớp của bộ 4cwe: 022, 078, 079, 089 -> 0..3
IGNORE = -100
PILLAR_LINE = re.compile(r"^CWE-\s*(\d+):\s*(?:CWE-\s*(\d+)|\(ROOT)")
LANGUAGE_CLASS = {"Python": {"Interpreted"}, "JavaScript": {"Interpreted"}, "Java": {"Compiled"},
                  "C": {"Compiled", "Memory-Unsafe"}, "C++": {"Compiled", "Memory-Unsafe"}}
NLS = "Not Language-Specific"


# ----------------------------------------------------------------------------- nhãn
def load_pillars(path):
    """cwe_root_parents.txt -> {CWE: chỉ số pillar}. CWE nhiều pillar lấy pillar có ID nhỏ nhất; mỗi pillar tra ra chính nó."""
    par = collections.defaultdict(set)
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            m = PILLAR_LINE.match(ln.strip())
            if m and m.group(2):
                par[int(m.group(1))].add(int(m.group(2)))
    pillars = sorted({p for s in par.values() for p in s})
    idx = {p: i for i, p in enumerate(pillars)}
    out = {c: idx[min(s)] for c, s in par.items()}
    for pl in pillars:
        out[pl] = idx[pl]
    return out, pillars


def cwe_labels(raw):
    """Ô CWE của nguồn -> danh sách nhãn THÔ giữ nguyên văn (PrimeVul: "cwe-310"; CleanVul: "['CWE-74', 'CWE-79']")."""
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


def cwe_all(raw):
    """Mọi số CWE trong ô nguồn, theo thứ tự, không lặp (bỏ nhãn không có số như NVD-CWE-noinfo)."""
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


def make_row(code, label, cid, lang, pil):
    return {"code": code, "label": int(label),
            "cwe": ("CWE-%03d" % cid) if cid is not None else "",
            "cwe_id": int(cid) if cid is not None else -1,
            "cwe_class": pil.get(cid, IGNORE) if cid is not None else IGNORE,
            "lang": lang}


# ----------------------------------------------------------------------------- luật MITRE
def load_catalog(xml_path):
    root = ET.parse(xml_path).getroot()
    ns = root.tag.split("}")[0] + "}"
    cat = {}
    for w in root.iter(ns + "Weakness"):
        ap = w.find(ns + "Applicable_Platforms")
        langs = [l.get("Name") or l.get("Class") for l in (ap.findall(ns + "Language") if ap is not None else [])]
        parents = [int(r.get("CWE_ID")) for r in w.iter(ns + "Related_Weakness")
                   if r.get("Nature") == "ChildOf" and r.get("View_ID") == "1000"]
        cat[int(w.get("ID"))] = dict(kind="Weakness", name=w.get("Name"), status=w.get("Status"), langs=langs, parents=parents)
    for kind in ("Category", "View"):
        for c in root.iter(ns + kind):
            members = [int(m.get("CWE_ID")) for m in c.iter(ns + "Has_Member")]
            cat[int(c.get("ID"))] = dict(kind=kind, name=c.get("Name"), status=c.get("Status"), langs=[], parents=[], members=members)
    return cat


def cwe_id(label):
    m = re.fullmatch(r"\s*(?:cwe[-_ ]?)?0*(\d+)\s*", str(label), flags=re.I)
    return int(m.group(1)) if m else None


class Rules:
    def __init__(self, cat, target):
        self.cat, self.target = cat, target
        self.classes = LANGUAGE_CLASS.get(target, set())
        self._memo = {}

    def decide(self, label):
        """-> (giữ, luật, lý do)"""
        i = cwe_id(label)
        if i is None:
            return False, "R1", f"nhãn `{label}` không phải mã CWE"
        return self._decide_id(i)

    def _decide_id(self, i):
        if i in self._memo:
            return self._memo[i]
        e = self.cat.get(i)
        if e is None:
            r = (False, "R1", f"CWE-{i} không có trong đặc tả")
        elif e["status"] in ("Deprecated", "Obsolete"):
            r = (False, "R2", f"{e['kind']} có Status={e['status']}")
        elif e["kind"] in ("Category", "View"):
            votes = [self._decide_id(m)[0] for m in e["members"] if m in self.cat and self.cat[m]["kind"] == "Weakness"]
            keep = bool(votes) and sum(votes) * 2 >= len(votes)
            r = (keep, "R2", f"{e['kind']} không khai nền tảng; Has_Member {sum(votes)}/{len(votes)} thành viên giữ")
        else:
            langs = e["langs"]
            if NLS in langs:
                r = (True, "R3", f"Applicable_Platforms có `{NLS}`")
            elif self.target in langs or (set(langs) & self.classes):
                hit = self.target if self.target in langs else "/".join(sorted(set(langs) & self.classes))
                r = (True, "R4", f"Applicable_Platforms khai `{hit}`")
            elif langs:
                r = (False, "R5", f"Applicable_Platforms chỉ khai `{'; '.join(langs)}`, không có {self.target}")
            else:
                anc = self._nearest_declared_ancestor(i)
                if anc is None:
                    r = (True, "R6", "không khai nền tảng, không tổ tiên nào khai → giữ yếu")
                else:
                    k, rule, why = self._decide_id(anc)
                    r = (k, "R6", f"không khai nền tảng; kế thừa CWE-{anc} ({rule}: {why})")
        self._memo[i] = r
        return r

    def _nearest_declared_ancestor(self, i, seen=None):
        seen = seen or set()
        for p in self.cat.get(i, {}).get("parents", []):
            if p in seen:
                continue
            seen.add(p)
            pe = self.cat.get(p)
            if pe and pe["kind"] == "Weakness" and pe["langs"]:
                return p
            a = self._nearest_declared_ancestor(p, seen)
            if a is not None:
                return a
        return None


_RULES_CACHE = {}


def get_rules(xml_path, target="Python"):
    key = (xml_path, target)
    if key not in _RULES_CACHE:
        _RULES_CACHE[key] = Rules(load_catalog(xml_path), target)
    return _RULES_CACHE[key]


# ----------------------------------------------------------------------------- tập con
def subset_common(recs, metas, lang, xml_path, target="Python"):
    """-> (hàng giữ, meta giữ, {lý do bỏ: số hàng}). Một nhãn bị bỏ là bỏ cả hàng.
    Đánh dấu `pair_broken_by_common` trên nửa còn lại của cặp bị tách (ghi vào meta)."""
    if lang == "java":
        return list(recs), list(metas), {"(java vao nguyen ven: CleanVul khong co nhan CWE cho java)": 0}
    if not xml_path or not os.path.exists(xml_path):
        return None, None, {}
    rules = get_rules(xml_path, target)
    keep_flag, why = [], collections.Counter()
    for m in metas:
        drop, reason = False, ""
        for l in m.get("cwe_labels") or [None]:
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
    for idxs in groups.values():
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
    """Hàng có ít nhất một CWE thuộc FOUR; nhiều CWE đích thì lấy cái đứng trước trong FOUR và ghi đè cwe/cwe_id/cwe_class."""
    keep = {c: i for i, c in enumerate(FOUR)}
    R, M = [], []
    for r, m in zip(recs, metas):
        cands = [c for c in (m.get("cwe_all") or ([r["cwe_id"]] if r["cwe_id"] >= 0 else [])) if c in keep]
        if not cands:
            continue
        pick = min(cands, key=lambda c: FOUR.index(c))
        r2 = dict(r)
        r2["cwe_id"], r2["cwe"], r2["cwe_class"] = pick, "CWE-%03d" % pick, keep[pick]
        R.append(r2)
        M.append(m)
    return R, M
