#!/usr/bin/env python3
"""Chạy một trainer của multibabel ở chế độ TẤT ĐỊNH NGHIÊM NGẶT, không sửa mã của trainer.

    python det_launch.py <src_dir> <trainer.py> [cờ của trainer...]

Trước khi trainer chạy (và trước mọi thao tác CUDA):
  - CUBLAS_WORKSPACE_CONFIG=:4096:8 (cuBLAS đọc MỘT lần khi tạo handle);
  - torch.use_deterministic_algorithms(True) — KHÔNG warn_only: gặp kernel không tất định là ném lỗi và dừng
    (người dùng chốt 24/09: "báo lỗi, dừng");
  - cudnn.deterministic = True, cudnn.benchmark = False; tắt TF32 ở matmul và cuDNN (hai máy A4000 giống nhau
    vẫn nên loại mọi đường tính gần đúng khác nhau giữa lần chạy).
PYTHONHASHSEED phải được đặt TRƯỚC khi trình thông dịch khởi động — run.sh export sẵn và file này kiểm tra lại.
Seed của random / numpy / torch do chính trainer đặt (`set_seed(args.seed)`); DataLoader dùng num_workers 0.

FPE_TF32=1: bật TF32 (matmul + cuDNN); mặc định tắt.

FPE_ORIG_CLEAN_GADGET=1: thay `babel.clean_gadget` bằng clean_gadget GỐC của gdufsnlp/BABEL
(`orig_babel_clean_gadget.py`, repo @ac53252, md5 a48eaa79…) cho nhánh so với BABEL gốc. Bản gốc BỎ HẲN dòng
có `/*…*/` nên danh sách ngắn đi; BABEL gốc để các cạnh lệch chỉ số trong im lặng, còn train_mwg sẽ ném lỗi vì hai
ma trận khác kích thước rồi gán đồ thị bằng 0 — nên đệm chuỗi rỗng ở CUỐI để giữ đúng hành vi gốc (lệch chỉ số).
"""
import os
import runpy
import sys

if os.environ.get("PYTHONHASHSEED") != "42":
    sys.exit("det_launch: PYTHONHASHSEED phải là 42 và phải đặt TRƯỚC khi chạy python (run.sh export sẵn)")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# FPE_TF32=1 (người dùng 04/10: "bật TF32 nếu độ lệch không quá lớn"): bật TF32 ở matmul và cuDNN - nhanh ~x1,84 trên A4000 (đo 28/09)
# nhưng đổi số học như đổi hạt giống; chỉ dùng cho CẢ một khối, không trộn với run FP32 khi ghép cặp. Mặc định: TẮT.
_tf32 = os.environ.get("FPE_TF32") == "1"
torch.backends.cuda.matmul.allow_tf32 = _tf32
torch.backends.cudnn.allow_tf32 = _tf32

src_dir, trainer = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.abspath(src_dir))

if os.environ.get("FPE_ORIG_CLEAN_GADGET") == "1":
    import importlib.util
    import babel
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("orig_babel_clean_gadget", os.path.join(here, "orig_babel_clean_gadget.py"))
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)

    def clean_gadget_orig(gadget, lang="c"):
        """clean_gadget gốc (không phân biệt ngôn ngữ); đệm cuối cho đủ số dòng như hành vi gốc."""
        out = orig.clean_gadget(list(gadget))
        return out + [""] * (len(gadget) - len(out))

    babel.clean_gadget = clean_gadget_orig          # train_mwg `from babel import clean_gadget` sẽ lấy bản này
    print("det_launch: dùng clean_gadget GỐC của BABEL", flush=True)

print("det_launch: deterministic=%s cudnn.det=%s benchmark=%s tf32(matmul/cudnn)=%s/%s CUBLAS=%s PYTHONHASHSEED=%s" % (
    torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, os.environ["CUBLAS_WORKSPACE_CONFIG"],
    os.environ["PYTHONHASHSEED"]), flush=True)
sys.argv = [trainer] + sys.argv[3:]
runpy.run_path(os.path.join(src_dir, trainer), run_name="__main__")
