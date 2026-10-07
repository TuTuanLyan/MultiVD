#!/usr/bin/env python3
"""Đẩy dòng `folds` lên artifact khối mwonly5 theo NHỊP (người dùng 04/10 22:4x: "30 phút cập nhật artifact một lần nếu có kết quả mới").

    python3 scripts/artifact_push.py init '<json {doc: version}>' [mặc định]   sổ phiên bản từ một lần list (mọi doc khác = mặc định)
    python3 scripts/artifact_push.py plan        so meta/dbrows/*.json với lần đẩy trước -> meta/push/writes.json (mảng writes cho ArtifactData batch)
    python3 scripts/artifact_push.py commit      gọi SAU khi batch thành công: version += 1, ghi nhận nội dung đã đẩy

Chỉ phiên điều phối ghi `folds` (trang chỉ ghi `alerts`), nên mỗi lần update thành công tăng version đúng 1 - sổ tự giữ được. Batch bị
từ chối vì lệch version thì chạy lại `init` từ một lần list mới. Dòng đẩy đi được CHÉP sang meta/push/rows/ để lần dbrows sau không đổi
nội dung đã gửi. Bỏ trường null (epochs/eta/best...) như live_all.sh để update không đè giá trị gieo sẵn.
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROWS = os.path.join(BASE, "meta", "dbrows")
PUSH = os.path.join(BASE, "meta", "push")
STATE = os.path.join(PUSH, "state.json")
NULLABLE = ("epochs", "eta_min", "eta_max", "best_epoch", "best_val_roc")
# 07/10 (người dùng: tab "Kết quả paper", "không cập nhật kết quả vào tab kết quả gốc"): run có "tab": "paper" trong runs.json đi vào
# collection RIÊNG `paper_folds`; mọi run khác vẫn vào `folds` (tab gốc chỉ đọc `folds`)
TAB = {r["id"]: r.get("tab") for r in json.load(open(os.path.join(HERE, "runs.json"), encoding="utf-8"))["runs"]}


def collection_of(doc):
    return "paper_folds" if TAB.get(doc.rsplit("__f", 1)[0]) == "paper" else "folds"


def load_state():
    return json.load(open(STATE, encoding="utf-8")) if os.path.exists(STATE) else {"default": 1, "docs": {}}


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def clean(row):
    return {k: v for k, v in row.items() if not (k in NULLABLE and v is None)}


def digest(row):
    # bỏ "updated" khi so: chỉ đổi giờ làm mới mà không đổi nội dung thì KHÔNG tính là kết quả mới
    return hashlib.md5(json.dumps({k: v for k, v in row.items() if k != "updated"}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cmd_init(versions, default="2"):
    st = {"default": int(default), "docs": {d: {"version": int(v)} for d, v in json.loads(versions).items()}}
    save(STATE, st)
    print("sổ phiên bản: %d doc ghi rõ, còn lại = %s" % (len(st["docs"]), default))


def cmd_plan():
    st = load_state()
    writes, pending = [], {}
    for p in sorted(glob.glob(os.path.join(ROWS, "*.json"))):
        doc = os.path.basename(p)[:-len(".json")]
        row = clean(json.load(open(p, encoding="utf-8")))
        h = digest(row)
        cur = st["docs"].get(doc, {})
        if cur.get("hash") == h:
            continue
        out = os.path.join(PUSH, "rows", doc + ".json")
        save(out, row)
        ver = cur.get("version", st["default"])
        writes.append({"op": "update", "collection": collection_of(doc), "doc_id": doc, "file_path": out, "if_version": ver})
        pending[doc] = {"version": ver + 1, "hash": h, "status": row.get("status")}
    save(os.path.join(PUSH, "writes.json"), writes)
    save(os.path.join(PUSH, "pending.json"), pending)
    print("%d dòng đổi so với lần đẩy trước%s" % (len(writes), (": " + ", ".join("%s(%s)" % (d, v["status"]) for d, v in pending.items())) if pending else ""))
    if writes:
        print("WRITES " + json.dumps(writes, ensure_ascii=False))


def cmd_commit():
    st = load_state()
    pend = json.load(open(os.path.join(PUSH, "pending.json"), encoding="utf-8"))
    for doc, v in pend.items():
        st["docs"][doc] = {"version": v["version"], "hash": v["hash"]}
    save(STATE, st)
    save(os.path.join(PUSH, "pending.json"), {})
    print("ghi nhận %d dòng đã đẩy" % len(pend))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"init": cmd_init, "plan": cmd_plan, "commit": cmd_commit}[cmd](*args)
