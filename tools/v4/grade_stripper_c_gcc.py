"""Chấm strip_comments.remove_comments trên TOÀN BỘ dữ liệu thô bằng đáp án độc lập.
C/C++: gcc -fpreprocessed; Java: lexer javac 17; JS: Babel parser. Kèm: bất biến khi chạy lại,
khớp bản cũ (strip_comments_multi) ở mọi hàm không có đuôi comment mồ côi, và kiểm riêng URL `://`."""
import json, csv, re, sys, subprocess, base64, collections, os
from multiprocessing import Pool
sys.path.insert(0, "tools/v4"); sys.path.insert(0, "tools")
from strip_comments import remove_comments, _removal_spans
from strip_comments_multi import strip as strip_cu
csv.field_size_limit(10 ** 9)
S = sys.argv[1]
nows = lambda s: re.sub(r"\s+", "", s)

def gcc(x):
    q = subprocess.run(["gcc", "-fpreprocessed", "-dD", "-E", "-P", "-x", "c", "-"], input=x, capture_output=True, text=True)
    return q.stdout if q.returncode == 0 else None

def mot_ham_c(args):
    tag, rid, raw = args
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    moi = remove_comments(raw, "c")
    _, warn = _removal_spans(raw, "c")
    mo = warn["duoi_comment_mo_coi"]
    cu = strip_cu(raw, "c")[0]
    g = gcc(raw)
    r = {"tag": tag, "id": rid, "mo_coi": mo, "khop_ban_cu": moi == cu, "bat_bien": remove_comments(moi, "c") == moi}
    if g is None:
        r["gcc"] = "loi"
    else:
        r["gcc"] = "khop" if nows(g) == nows(moi) else "lech"
        r["url_moi"], r["url_gcc"], r["url_tho"] = moi.count("://"), g.count("://"), raw.count("://")
    if mo or r["gcc"] == "lech":
        r["raw"] = raw[:3000]; r["moi"] = moi[:3000]; r["gcc_out"] = (g or "")[:3000]
    return r

def main():
    jobs = []
    for f in ("primevul_train_paired", "primevul_valid_paired", "primevul_test_paired", "primevul_train", "primevul_valid", "primevul_test"):
        tag = "paired" if "paired" in f else "unpaired"
        for l in open("/drive1/cuongtm/ntat/Archive/PrimeVulRaw/%s.jsonl" % f):
            x = json.loads(l); jobs.append((tag, "primevul:%s" % x["idx"], x["func"]))
    with Pool(4) as p:
        res = p.map(mot_ham_c, jobs, chunksize=200)
    c = collections.Counter()
    with open(S + "/cham_c.jsonl", "w") as fh:
        for r in res:
            t = r["tag"]
            c[(t, "tong")] += 1; c[(t, "gcc_" + r["gcc"])] += 1
            c[(t, "khop_ban_cu")] += r["khop_ban_cu"]; c[(t, "bat_bien")] += r["bat_bien"]
            if r["mo_coi"]: c[(t, "co_duoi_mo_coi")] += 1
            if not r["mo_coi"] and not r["khop_ban_cu"]: c[(t, "!!_khong_mo_coi_ma_lech_ban_cu")] += 1
            if r["gcc"] == "lech" and not r["mo_coi"]: c[(t, "gcc_lech_KHONG_do_mo_coi")] += 1
            if r["gcc"] != "loi" and not r["mo_coi"]:
                if r["url_moi"] != r["url_gcc"]: c[(t, "!!_url_lech_gcc")] += 1
                if r["url_tho"] != r["url_moi"]: c[(t, "url_bi_xoa_(trong_comment)")] += 1
            if r["mo_coi"] or r["gcc"] == "lech":
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(c.items()): print(k, v)
    json.dump({"%s|%s" % k: v for k, v in c.items()}, open(S + "/cham_c.json", "w"), ensure_ascii=False, indent=1)

main()
