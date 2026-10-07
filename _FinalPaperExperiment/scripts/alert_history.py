#!/usr/bin/env python3
"""Lưu cảnh báo ĐÃ XEM của artifact vào lịch sử md (người dùng 04/10 22:0x: "Nếu đã xem thì nên lưu vào 1 md lịch sử nào đó").

    python3 scripts/alert_history.py [thư mục xuất] [file md]

Đầu vào: bản xuất collection `alerts` của artifact (ArtifactData list với out_dir -> <thư mục>/alerts/<id>.json).
Mỗi lần chạy DỰNG LẠI file từ mọi cảnh báo `acked: true` (cũ trước mới sau), kèm kết cục (`outcome`) và các dòng cập nhật
(`updates: [{at, text}]`) - người dùng 05/10: "nếu cảnh báo sai hãy bổ sung thông tin dưới cảnh báo" - nên cập nhật thêm SAU khi đã xem
vẫn vào được lịch sử. Không ghi trùng. In ra id lần đầu vào lịch sử (mỗi dòng một id) để phiên điều phối đánh dấu ở CURRENT_RUN.md.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "meta", "alerts_dump")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "ALERT_HISTORY.md")
HEAD = ("# Lịch sử cảnh báo đã xem - khối mwonly5\n\n"
        "Cảnh báo sập / nghi sập / kẹt / lỗi mà người dùng đã bấm \"Đã xem\" trên artifact "
        "https://claude.ai/artifact/Pouzqz7Q1Kd9vwwrnQDm51 . Cảnh báo chưa xem nằm ở dải đỏ đầu trang và mục \"⚠ CẢNH BÁO\" của CURRENT_RUN.md.\n"
        "Ghi bằng `_FinalPaperExperiment/scripts/alert_history.py`, cũ trước mới sau.\n")

old = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
alerts = []
for p in glob.glob(os.path.join(DUMP, "alerts", "*.json")):
    d = json.load(open(p, encoding="utf-8"))
    d["_id"] = os.path.basename(p)[:-len(".json")]
    alerts.append(d)
seen = sorted([a for a in alerts if a.get("acked")], key=lambda a: (a.get("at") or "", a["_id"]))
body = HEAD
for a in seen:
    where = "%s f%s" % (a.get("run") or "-", a.get("fold") if a.get("fold") is not None else "-")
    tag = " · BÁO NHẦM" if a.get("outcome") == "báo nhầm" else (" · SẬP THẬT" if a.get("outcome") == "sập thật" else "")
    body += ("\n## %s · %s · %s%s\n<!-- alert:%s -->\n\n- máy: %s\n- đã xem: %s\n- kết cục: %s\n\n%s\n"
             % (a.get("at") or "?", a.get("kind") or "khác", where, tag, a["_id"], a.get("host") or "-", a.get("acked_at") or "?",
                a.get("outcome") or "chưa ghi", (a.get("message") or "").strip()))
    for u in a.get("updates") or []:
        body += "\n- %s: %s" % (u.get("at") or "?", (u.get("text") or "").strip())
    if a.get("updates"):
        body += "\n"
if body != old:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(body)
for a in seen:
    if "<!-- alert:%s -->" % a["_id"] not in old:
        print(a["_id"])
