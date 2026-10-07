"""Bí danh giữ tương thích cho mã cũ còn import `babel.remove_comments_re`.

Logic xoá comment CHUẨN duy nhất nằm ở `babel/strip_comments.py` (máy trạng thái đã chấm bằng gcc,
lexer javac 17 và Babel parser; hiểu chuỗi, regex JS, docstring Python; giữ chỉ thị `#` của C).
Bản regex cũ ở đây đã bỏ: nó cắt nhầm `//` trong regex JS và xoá cả dòng `#define`/`#if`.
"""
from .strip_comments import remove_comments  # noqa: F401
