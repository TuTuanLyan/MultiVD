# Vendored tu maytinhdibo/GraphTransferVD nhanh 'fix' commit 79f0308 (tools/cwe_rules.py)
# Giu NGUYEN VEN de hai ben ra cung ket qua. Tac gia goc: Cuong Tran.
"""cwe_rules.py — dựng danh sách CWE "common" cho một ngôn ngữ đích bằng LUẬT đọc từ đặc tả MITRE CWE (cwec_vX.xml).

Không có danh sách tay. Mọi quyết định đều truy được về một luật và một trường trong XML:
  Weakness/Applicable_Platforms/Language[@Name | @Class]   (nền tảng khai)
  Weakness/Related_Weaknesses/Related_Weakness[@Nature='ChildOf']  (kế thừa khi không khai)
  Category/Relationships/Has_Member                        (giải Category theo thành viên)
  @Status = Deprecated | Obsolete

Luật, xét theo thứ tự, dừng ở luật đầu tiên khớp:
  R1  nhãn không phải mã CWE (unknown, NVD-CWE-Other, rỗng)            -> BỎ
  R2  Category / View: giải qua Has_Member — >= 50 % thành viên GIỮ     -> GIỮ, ngược lại BỎ;
      Deprecated/Obsolete                                               -> BỎ
  R3  Weakness khai `Not Language-Specific`                             -> GIỮ
  R4  Weakness khai thẳng ngôn ngữ đích, hoặc lớp chứa nó (Python ⊂ Interpreted) -> GIỮ
  R5  Weakness khai danh sách ngôn ngữ cụ thể KHÔNG có đích              -> BỎ
  R6  Weakness không khai nền tảng: kế thừa quyết định của tổ tiên gần nhất có khai (ChildOf, view 1000);
      không có tổ tiên nào khai                                        -> GIỮ (yếu, ghi rõ)

Dùng:
  python cwe_rules.py --xml cwec_v4.20.xml --target Python --pool src_pair_jsc.jsonl \
        --name js_cleanvul --outdir audits/cwe_common [--write-kept out.jsonl]
  Pool là jsonl, trường `cwe` là chuỗi ("cwe-79") hoặc danh sách (["CWE-119", ...]); trường `pair_id` (tuỳ chọn) để đếm cặp.
"""
import argparse, collections, json, re, sys
import xml.etree.ElementTree as ET

# lớp nền tảng trong XML chứa ngôn ngữ đích (đọc từ đặc tả: Python là ngôn ngữ thông dịch)
LANGUAGE_CLASS = {"Python": {"Interpreted"}, "JavaScript": {"Interpreted"}, "Java": {"Compiled"},
                  "C": {"Compiled", "Memory-Unsafe"}, "C++": {"Compiled", "Memory-Unsafe"}}
NLS = "Not Language-Specific"


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
    for c in root.iter(ns + "Category"):
        members = [int(m.get("CWE_ID")) for m in c.iter(ns + "Has_Member")]
        cat[int(c.get("ID"))] = dict(kind="Category", name=c.get("Name"), status=c.get("Status"), langs=[], parents=[], members=members)
    for v in root.iter(ns + "View"):
        members = [int(m.get("CWE_ID")) for m in v.iter(ns + "Has_Member")]
        cat[int(v.get("ID"))] = dict(kind="View", name=v.get("Name"), status=v.get("Status"), langs=[], parents=[], members=members)
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
        """-> (giữ: bool, luật, lý do đọc từ đặc tả)"""
        i = cwe_id(label)
        if i is None:
            return False, "R1", f"nhãn `{label}` không phải mã CWE"
        return self._decide_id(i, depth=0)

    def _decide_id(self, i, depth):
        if i in self._memo:
            return self._memo[i]
        e = self.cat.get(i)
        if e is None:
            r = (False, "R1", f"CWE-{i} không có trong đặc tả")
        elif e["status"] in ("Deprecated", "Obsolete"):
            r = (False, "R2", f"{e['kind']} có Status={e['status']}")
        elif e["kind"] in ("Category", "View"):
            votes = [self._decide_id(m, depth + 1)[0] for m in e["members"] if m in self.cat and self.cat[m]["kind"] == "Weakness"]
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
                    k, rule, why = self._decide_id(anc, depth + 1)
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


def read_pool(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for r in rows:
        c = r.get("cwe")
        r["_labels"] = [c] if isinstance(c, str) or c is None else list(c)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True); ap.add_argument("--target", default="Python")
    ap.add_argument("--pool", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--outdir", default="."); ap.add_argument("--write-kept", default="")
    a = ap.parse_args()
    cat = load_catalog(a.xml); rules = Rules(cat, a.target); rows = read_pool(a.pool)

    per = collections.OrderedDict()   # label -> thống kê
    kept_rows = []
    for r in rows:
        labels = r["_labels"] or [None]
        decisions = [(lb,) + rules.decide(lb if lb is not None else "") for lb in labels]
        # một hàm giữ khi MỌI nhãn của nó đều giữ (hàm nhiều nhãn có một nhãn memory-unsafe thì bỏ)
        keep_row = all(d[1] for d in decisions)
        if keep_row:
            kept_rows.append(r)
        for lb, keep, rule, why in decisions:
            key = str(lb)
            s = per.setdefault(key, dict(keep=keep, rule=rule, why=why, n=0, pairs=set(), name=""))
            s["n"] += 1
            if r.get("pair_id") is not None:
                s["pairs"].add(r["pair_id"])
            i = cwe_id(lb) if lb is not None else None
            s["name"] = cat.get(i, {}).get("name", "") if i else "(không có mã CWE)"
            s["langs"] = "; ".join(cat.get(i, {}).get("langs", [])) if i else ""

    drop = sorted([(k, s) for k, s in per.items() if not s["keep"]], key=lambda x: -x[1]["n"])
    keep = sorted([(k, s) for k, s in per.items() if s["keep"]], key=lambda x: -x[1]["n"])
    n_drop_rows = len(rows) - len(kept_rows)
    pairs_all = {r.get("pair_id") for r in rows if r.get("pair_id") is not None}
    pairs_kept = {r.get("pair_id") for r in kept_rows if r.get("pair_id") is not None}

    L = [f"# {a.name} — CWE BỎ khỏi common cho đích {a.target} (luật đọc từ `{a.xml.split('/')[-1]}`)\n",
         f"Pool `{a.pool.split('/')[-1]}`: {len(rows)} hàm, {len(pairs_all)} cặp → giữ **{len(kept_rows)} hàm / {len(pairs_kept)} cặp**, bỏ {n_drop_rows} hàm ({100*n_drop_rows/len(rows):.1f} %). "
         "Một hàm bị bỏ khi bất kỳ nhãn nào của nó bị bỏ. Luật R1–R6 xem đầu `tools/cwe_rules.py`.\n",
         "## Bỏ\n", "| nhãn | tên | nền tảng khai | luật | lý do (từ đặc tả) | hàm | cặp |", "|---|---|---|---|---|---:|---:|"]
    for k, s in drop:
        L.append(f"| {k} | {s['name']} | {s.get('langs','') or '—'} | {s['rule']} | {s['why']} | {s['n']} | {len(s['pairs'])} |")
    L += ["\n## Giữ\n", "| nhãn | tên | nền tảng khai | luật | hàm |", "|---|---|---|---|---:|"]
    for k, s in keep:
        L.append(f"| {k} | {s['name']} | {s.get('langs','') or '—'} | {s['rule']} | {s['n']} |")
    md = f"{a.outdir}/{a.name}_drop.md"; js = f"{a.outdir}/{a.name}_drop.json"
    open(md, "w").write("\n".join(L) + "\n")
    json.dump({"target": a.target, "xml": a.xml.split("/")[-1], "pool": a.pool.split("/")[-1],
               "drop": [dict(label=k, name=s["name"], langs=s.get("langs", ""), rule=s["rule"], why=s["why"], n=s["n"], pairs=len(s["pairs"])) for k, s in drop],
               "keep": [dict(label=k, name=s["name"], rule=s["rule"], n=s["n"]) for k, s in keep],
               "rows": len(rows), "rows_kept": len(kept_rows), "pairs": len(pairs_all), "pairs_kept": len(pairs_kept)},
              open(js, "w"), indent=1, ensure_ascii=False)
    if a.write_kept:
        with open(a.write_kept, "w") as o:
            for r in kept_rows:
                r = dict(r); r.pop("_labels", None); o.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.name}: {len(rows)} hàm → giữ {len(kept_rows)} ({len(pairs_kept)} cặp), bỏ {n_drop_rows} | {len(drop)} nhãn bỏ, {len(keep)} nhãn giữ | {md}")
    for k, s in drop:
        print(f"  BỎ {k:14s} {s['n']:5d}  {s['rule']}  {s['why'][:90]}")


if __name__ == "__main__":
    main()
