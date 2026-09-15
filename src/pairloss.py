"""Mat mat BIEN TRONG CAP cho Pha 1 — "De xuat B" cua NEXT_CONTRIBUTION.md.

## Vi sao

FACTS §51: CleanVul TU CONG BO bo du lieu lo hong mang **40-75% nhieu nhan**, vi moi thay doi
trong mot commit va deu bi dan nhan "lien quan lo hong". So cua chinh du an khop: tac vu nguon
chi hoc duoc toi **0.57-0.59** macro-F1 tren ba seed o ba may.

FACTS §54.1 do duoc bien do cua van de: tien-huan-luyen bang nhan **sai** lam hai **-0.064** so
voi khong tien-huan-luyen; bang nhan **that** thi duoc **+0.024**. Tuc khoang **0.06** dang bi
tieu vao viec go lai thiet hai ma chinh viec huan luyen tren bo du lieu nay gay ra.

## Y tuong

Nhan TUYET DOI ("ham nay co lo hong") sai 40-75%. Nhung quan he TUONG DOI — "ban nay TRUOC ban
va, ban kia SAU" — **dung theo cau tao**: no den tu thu tu commit, khong tu phan doan bao mat.
Nhieu chi anh huong viec thay doi do **co phai bao mat khong**, khong dao chieu quan he.

Voi mot cap ma commit KHONG lien quan bao mat, hai ham gan nhu dong nghia:
  - Duoi BCE tung dong: mo hinh bi ep khang dinh CHAC CHAN ban truoc "co lo hong" => gradient SAI.
  - Duoi bien trong cap: rang buoc `s(truoc) >= s(sau) + m` duoc thoa bang mot khe NHO
    => gradient YEU.

**Nhieu bi ha cap tu "giam sat SAI" xuong "giam sat YEU".** Do la khac biet co the kiem chung.

## Ba dieu ky thuat quan trong

1. **Phai co sampler theo cap.** Xao ngau nhien thi xac suat hai nua cua mot cap roi vao cung
   mot batch 16 (tren ~3370 dong) chi khoang 0.4% — loss se gan nhu khong bao gio kich hoat.
   `PairBatchSampler` gom tung cap lai roi moi chia batch.
2. **Diem dung de so la BIEN QUYET DINH** `logit[1] - logit[0]`, khong phai xac suat: xac suat
   bao hoa nen gradient tat o dung cho mo hinh da tu tin.
3. **softplus thay vi hinge cung.** Hinge cho gradient bang 0 ngay khi vua du bien; softplus
   giam muot, khong co diem gay.

Mac dinh `beta = 0` => duong chay cu khong doi mot byte.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler


def build_pair_index(records, pair_field: str = "pair_id", label_field: str = "label"):
    """Tra ve (group, role, n_pairs).

    `group[i]`  = chi so cap cua hang i, hoac -1 neu hang do khong nam trong mot cap DAY DU.
    `role[i]`   = 1 neu hang i la ban CO lo hong (label 1), 0 neu la ban DA VA, -1 neu khong cap.

    "Cap day du" = dung hai hang cung `pair_id`, mot hang label 1 va mot hang label 0.
    Nhom ba hang tro len, hoac hai hang cung nhan, deu BI BO — im lang gop chung se tao ra
    nhung "cap" khong phai cap truoc/sau va lam ban chinh tin hieu dang do.
    """
    buckets: dict = {}
    for i, r in enumerate(records):
        pid = r.get(pair_field)
        if pid is None or pid == "":
            continue
        buckets.setdefault(pid, []).append(i)

    n = len(records)
    group = torch.full((n,), -1, dtype=torch.long)
    role = torch.full((n,), -1, dtype=torch.long)
    g = 0
    for _pid, idxs in buckets.items():
        if len(idxs) != 2:
            continue
        a, b = idxs
        la, lb = int(records[a][label_field]), int(records[b][label_field])
        if {la, lb} != {0, 1}:
            continue
        vul, fixed = (a, b) if la == 1 else (b, a)
        group[vul] = g
        group[fixed] = g
        role[vul] = 1
        role[fixed] = 0
        g += 1
    return group, role, g


class PairBatchSampler(Sampler):
    """Chia batch sao cho HAI NUA CUA MOT CAP luon cung batch.

    Xao thu tu CAC CAP (khong phai cac hang), moi batch lay `batch_size // 2` cap roi trai ra.
    Hang le (khong thuoc cap day du) duoc xao rieng va xep vao cuoi theo batch day.

    Xac dinh duoc theo seed: cung seed + cung epoch => cung thu tu.
    """

    def __init__(self, group: torch.Tensor, batch_size: int, seed: int = 0, drop_last: bool = False):
        if batch_size < 2:
            raise ValueError("batch_size phai >= 2 de chua tron mot cap")
        self.group = group
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        pairs: dict = {}
        singles = []
        for i, g in enumerate(group.tolist()):
            if g < 0:
                singles.append(i)
            else:
                pairs.setdefault(g, []).append(i)
        # chi giu nhom du hai; build_pair_index da bao dam nhung kiem lai cho chac
        self.pairs = [tuple(v) for v in pairs.values() if len(v) == 2]
        self.singles = singles

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed * 100003 + self.epoch)
        pair_order = torch.randperm(len(self.pairs), generator=gen).tolist()
        single_order = torch.randperm(len(self.singles), generator=gen).tolist()

        per_batch = self.batch_size // 2
        batch = []
        for k in pair_order:
            batch.extend(self.pairs[k])
            if len(batch) >= per_batch * 2:
                yield batch
                batch = []
        # hang le xep sau
        for k in single_order:
            batch.append(self.singles[k])
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        per_batch = self.batch_size // 2
        n_pair_batches = len(self.pairs) // per_batch if per_batch else 0
        rest_pairs = (len(self.pairs) % per_batch) * 2 if per_batch else 0
        n_rest = rest_pairs + len(self.singles)
        n_single_batches = n_rest // self.batch_size
        tail = 1 if (n_rest % self.batch_size) and not self.drop_last else 0
        return n_pair_batches + n_single_batches + tail


def pair_margin_loss(vul_logits: torch.Tensor, index: torch.Tensor,
                     group: torch.Tensor, role: torch.Tensor,
                     margin: float = 1.0):
    """`softplus(margin - (s_vul - s_fixed))` trung binh tren cac cap CO DU HAI NUA trong batch.

    `s = logit[1] - logit[0]` la bien quyet dinh, khong phai xac suat.
    Tra ve (loss, so_cap_dung_duoc). Khong cap nao thi tra ve 0 va giu nguyen do thi tinh toan
    (loss * 0) de `backward()` khong vo khi mot batch tinh co khong chua cap nao.
    """
    device = vul_logits.device
    idx = index.to(device)
    g = group.to(device)[idx]
    r = role.to(device)[idx]
    score = vul_logits[:, 1] - vul_logits[:, 0]

    keep = g >= 0
    if keep.sum() < 2:
        return vul_logits.sum() * 0.0, 0

    gk, rk, sk = g[keep], r[keep], score[keep]
    uniq, inv = torch.unique(gk, return_inverse=True)
    n_g = uniq.numel()

    # dem tung vai trong tung nhom; chi giu nhom co DUNG mot ban vul va mot ban fixed
    cnt_v = torch.zeros(n_g, device=device).index_add_(0, inv, (rk == 1).float())
    cnt_f = torch.zeros(n_g, device=device).index_add_(0, inv, (rk == 0).float())
    full = (cnt_v == 1) & (cnt_f == 1)
    if full.sum() == 0:
        return vul_logits.sum() * 0.0, 0

    s_v = torch.zeros(n_g, device=device, dtype=sk.dtype).index_add_(0, inv, sk * (rk == 1).to(sk.dtype))
    s_f = torch.zeros(n_g, device=device, dtype=sk.dtype).index_add_(0, inv, sk * (rk == 0).to(sk.dtype))
    diff = (s_v - s_f)[full]
    loss = F.softplus(margin - diff).mean()
    return loss, int(full.sum().item())
