#!/usr/bin/env python3
"""Tập đích SVEN không comment: chép `data/sven_python_folds_norm` sang thư mục mới, chỉ thay `code` bằng bản đã xoá
comment `#` và docstring bằng `strip_comments.remove_comments` — bộ xoá comment DUY NHẤT, dùng chung cho mọi nguồn
và cho BABEL. Giữ nguyên cấu trúc thư mục (`data.jsonl`, `fold*/{train,val,test}.jsonl`), thứ tự dòng, nhãn, fold và
mọi trường khác.

Vì sao: `train_mwg.py` và `train_baseline.py` KHÔNG xoá comment trên văn bản đưa vào mô hình; BABEL chỉ bỏ qua comment
khi tìm định danh để dựng cạnh. Nên muốn so trên tập đích sạch comment thì phải sạch ở mức dữ liệu, cho mọi nhánh.

Chạy:  python tools/v4/make_sven_nocomment.py data/sven_python_folds_norm data/sven_python_folds_nocomment
"""
import glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strip_comments import remove_comments  # noqa: E402


def dumps_like_source(r):
    """Đúng định dạng file gốc: JSON gọn, không mã hoá ASCII, `/` viết thành `\\/` (tái tạo đúng từng byte
    4 560/4 560 dòng của sven_python_folds_norm) — dòng nào `code` không đổi thì trùng byte với file gốc."""
    return json.dumps(r, ensure_ascii=False, separators=(",", ":")).replace("/", "\\/")


def main():
    src, dst = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    if os.path.exists(dst):
        raise SystemExit("%s đã tồn tại — không ghi đè" % dst)
    files = sorted(glob.glob(src + "/*.jsonl") + glob.glob(src + "/fold*/*.jsonl"))
    stats = {}
    for f in files:
        out = dst + f[len(src):]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        n = changed = 0
        with open(out, "w", encoding="utf-8") as w:
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                code = remove_comments(r["code"], "python")
                if not code.strip():
                    raise ValueError("hàm rỗng sau khi xoá comment: %s dòng %d" % (f, n + 1))
                if remove_comments(code, "python") != code:
                    raise ValueError("xoá comment không lũy đẳng: %s dòng %d" % (f, n + 1))
                n += 1
                changed += code != r["code"]
                r["code"] = code
                w.write(dumps_like_source(r) + "\n")
        stats[f[len(src) + 1:]] = {"rows": n, "code_changed": changed}
    for k, v in stats.items():
        print("%-22s %s" % (k, v))


if __name__ == "__main__":
    main()
