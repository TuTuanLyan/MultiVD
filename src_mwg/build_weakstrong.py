#!/usr/bin/env python3
"""build_weakstrong.py — hai pool Phase 1 chi khac NHOM CWE, moi thu khac ghim chat.

CAU HOI (tu FACTS §23): loi ich cua transfer don vao dung hai lop dich YEU
(CWE-022 baseline macro-F1 0.303, CWE-079 0.578) chu khong vao hai lop MANH
(CWE-078 0.778, CWE-089 0.922). Bien du bao tot nhat la do yeu cua baseline
(Pearson -0.886), khong phai so dong nguon (+0.838, va CWE-022 con it dong
nguon hon dong dich ma van tang nhieu nhat).

Neu dung, thi CHON nguon theo cho yeu cua dich phai thang chon theo cho manh.
Day la phep kiem truc tiep, va la mot PHUONG PHAP chu khong phai sieu tham so.

  wk146 = chi CWE-022 + CWE-079   (hai lop dich YEU)
  st146 = chi CWE-078 + CWE-089   (hai lop dich MANH)

Ghim cho bang nhau, neu khong thi Delta doi nhieu bien cung luc (bai hoc RESEARCH §B.1
noi luoi `pur*` keo ti le js sap theo va thanh vo dung):
  n           = 146 ca hai (rang buoc: nhom MANH chi co 146 dong trong toan nguon)
  nhan        = 73/73 ca hai
  ti le js    = 0.712 ca hai (ti le tu nhien cua nhom MANH; nhom YEU lay mau khop)
  do tinh khiet = 100% ca hai (moi dong deu thuoc 4 CWE cua dich)

PHAI CHAY VOI NHANH `none`, KHONG phai `latent_bottleneck`. Ly do (do 09/09):
`com`/`full` gan `cwe_class` theo PILLAR cua CWE chu khong theo tung CWE —
  class 2 = CWE-664 Improper Control of a Resource  <- CWE-022 nam day
  class 8 = CWE-707 Improper Neutralization         <- CWE-078/079/089 nam day
Nen nhom YEU {22,79} trai HAI pillar con nhom MANH {78,89} chi mot. Head phu se
co 2 lop o mot ben va 1 lop o ben kia -> phep so doi hai bien. Nhanh `none` khong
co head nen bo han cai lech do, va FACTS §23 da cho thay hieu ung ro ri qua `none`
(+0.1492 CWE-022, +0.1799 CWE-079) nen khong mat gi.

Chay: python3 src/build_weakstrong.py
"""
import json, random, re, sys
from collections import defaultdict
from pathlib import Path

SRC = Path("data/phase1_full.jsonl")
OUT = Path("data/pool")
WEAK, STRONG = {22, 79}, {78, 89}
SEED = 42


def cwe_num(r):
    m = re.search(r"(\d+)", str(r.get("cwe") or r.get("cwe_id") or ""))
    return int(m.group(1)) if m else None


def remap_cwe_class(rows):
    """Danh lai `cwe_class` thanh 0..N-1 LIEN TUC — xem src/build_purity_grid.py."""
    present = sorted({r["cwe_class"] for r in rows if r.get("cwe_class", -100) >= 0})
    m = {c: i for i, c in enumerate(present)}
    for r in rows:
        c = r.get("cwe_class", -100)
        if c >= 0:
            r["cwe_class"] = m[c]
    return rows


def pick(rows, n_js, n_cc, rng):
    """Lay dung n_js dong js va n_cc dong ccpp, moi ben can bang nhan 50/50."""
    out = []
    for lang, need in (("js", n_js), ("ccpp", n_cc)):
        pool = defaultdict(list)
        for r in rows:
            if r.get("lang") == lang:
                pool[r["label"]].append(r)
        half = need // 2
        if len(pool[0]) < half or len(pool[1]) < half:
            sys.exit(f"!! khong du dong {lang}: can {half}/nhan, co {len(pool[0])}/{len(pool[1])}")
        for lb in (0, 1):
            out += rng.sample(pool[lb], half)
    return out


def main():
    rng = random.Random(SEED)
    recs = [json.loads(l) for l in SRC.open()]
    grp = {"wk": [r for r in recs if cwe_num(r) in WEAK],
           "st": [r for r in recs if cwe_num(r) in STRONG]}

    # Nhom MANH la rang buoc: lay TRON no, roi khop nhom YEU theo dung so js/ccpp do.
    st = grp["st"]
    n_js = sum(1 for r in st if r.get("lang") == "js")
    n_cc = len(st) - n_js
    if n_js % 2 or n_cc % 2:                       # can chan de can bang nhan duoc
        n_js -= n_js % 2
        n_cc -= n_cc % 2
    st_sel = pick(st, n_js, n_cc, rng)
    wk_sel = pick(grp["wk"], n_js, n_cc, rng)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, sel in (("wk", wk_sel), ("st", st_sel)):
        rng.shuffle(sel)
        sel = remap_cwe_class([dict(r) for r in sel])
        p = OUT / f"{name}{len(sel)}.jsonl"
        with p.open("w") as f:
            for r in sel:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        js = sum(1 for r in sel if r.get("lang") == "js")
        pos = sum(1 for r in sel if r["label"] == 1)
        cw = defaultdict(int)
        for r in sel:
            cw[cwe_num(r)] += 1
        cls = sorted({r["cwe_class"] for r in sel if r.get("cwe_class", -100) >= 0})
        print(f"{p}: n={len(sel)} js={js/len(sel):.3f} nhan1={pos/len(sel):.3f} "
              f"CWE={dict(sorted(cw.items()))} cwe_class={cls}")


if __name__ == "__main__":
    main()
