"""Cong hai chieu cho phep tron theta_Pha1 (+) theta_Pha2 (RESEARCH §11.2).

Kiem _blend_state bang mot module do choi, khong dung GPU va khong can checkpoint that.
    python3 tests/test_interp.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_transfer import _blend_state   # noqa: E402

OK = True
def report(name, passed, detail=""):
    global OK
    OK = OK and passed
    print(f"  [{'OK ' if passed else 'SAI'}] {name}{('  | ' + detail) if detail else ''}")


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)
        # dem KHONG phai so thuc — phai bi bo qua, khong duoc tron
        self.register_buffer("counter", torch.tensor([7], dtype=torch.long))


m = Toy()
g = torch.Generator().manual_seed(0)
s2 = {k: (torch.randn(v.shape, generator=g) if v.is_floating_point() else v.clone())
      for k, v in m.state_dict().items()}
s1 = {k: (torch.randn(v.shape, generator=g) if v.is_floating_point() else v.clone())
      for k, v in m.state_dict().items()}
dev = torch.device("cpu")

print("1) alpha = 1 phai ra DUNG theta_Pha2")
b, sk = _blend_state(m, s2, s1, 1.0, dev)
same2 = all(torch.equal(m.state_dict()[k], s2[k]) for k in s2 if s2[k].is_floating_point())
report("alpha=1 == Pha2", same2, f"tron {b} tensor, bo qua {sk}")
report("dem khong bi tron", int(m.state_dict()["counter"]) == 7)

print("\n2) chieu LECH: alpha = 0 phai ra DUNG theta_Pha1, va KHAC theta_Pha2")
_blend_state(m, s2, s1, 0.0, dev)
same1 = all(torch.equal(m.state_dict()[k], s1[k]) for k in s1 if s1[k].is_floating_point())
diff = any(not torch.equal(m.state_dict()[k], s2[k]) for k in s2 if s2[k].is_floating_point())
report("alpha=0 == Pha1", same1)
report("va KHAC Pha2", diff)

print("\n3) alpha = 0.5 phai la trung diem chinh xac")
_blend_state(m, s2, s1, 0.5, dev)
mid = all(torch.allclose(m.state_dict()[k], 0.5 * s2[k] + 0.5 * s1[k], atol=1e-7)
          for k in s2 if s2[k].is_floating_point())
report("alpha=0.5 == trung diem", mid)

print("\n4) ten lech / shape lech phai bi BO QUA chu khong no")
bad = {k: (v[:1] if v.dim() and v.is_floating_point() else v) for k, v in s1.items()}
bad["khong_ton_tai"] = torch.zeros(2)
b2, sk2 = _blend_state(m, s2, bad, 0.5, dev)
report("bo qua tensor lech shape", sk2 >= 2, f"tron {b2}, bo qua {sk2}")

print("\n5) khoi phuc: tron ve alpha=1 phai lay lai DUNG Pha 2")
_blend_state(m, s2, s1, 1.0, dev)
report("khoi phuc duoc", all(torch.equal(m.state_dict()[k], s2[k])
                             for k in s2 if s2[k].is_floating_point()))

print("\n" + ("TAT CA CONG DEU DAT" if OK else "CO CONG THAT BAI"))
sys.exit(0 if OK else 1)
