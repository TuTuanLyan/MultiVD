"""MWG — CodeBERT nhiều cửa sổ (multi-window) + nhánh đồ thị dòng (BABEL) cho phát hiện lỗ hổng mức hàm.

    comments  bộ xoá comment duy nhất (dựng dữ liệu và dựng đồ thị dùng chung)
    symbols   chuẩn hoá định danh theo dòng -> VARk / FUNk
    graph     quan hệ giữa các dòng -> ma trận kề [R, L, L]
    data      GraphWindowDataset (cửa sổ token + đồ thị dòng), PairBatchSampler
    model     MWGraphModel
    optim     RecAdam, SAM / ASAM
"""
from .comments import blank_comments, remove_comments  # noqa: F401
from .graph import BRACKET_ONLY, build_relations       # noqa: F401
from .symbols import normalize_identifiers             # noqa: F401
