# Đề xuất 1 — Ghép hai model (target-only + transfer) với tổ hợp học được, không dùng nhãn CWE

Ngày 15/09/2026. Đích SVEN (Python, 760 hàm, 5 fold rời). Tài liệu gồm: (A) đề xuất và cách kiểm; (B) mã nguồn đã triển khai mà đề xuất dựa vào — lọc dữ liệu JS và xoá comment, transfer hai pha, nhiều cửa sổ (MW), SAM/ASAM, RecAdam — kèm mô tả. Mã trích **nguyên văn** từ `MultiVD/src/*.py`, `reptile_codebert/code_clean.py`, `/drive1/cuongtm/tmp/build/build_jsc.py` (chỉ bỏ docstring dài và các nhánh không dùng, có ghi chú).

---

## A. Đề xuất

### A.1 Bằng chứng xuất phát (§127, §130)

Nguồn C+JS(+Java) đưa tri thức vào nhóm CWE yếu nhưng làm mất nhóm mạnh; ROC tổng bù trừ nhau:

| nhánh (3 fold, MW) | ROC | 022 | 078 | 079 | 089 | mF1@0,5 |
|---|---:|---:|---:|---:|---:|---:|
| SR = target-only, SAM, RecAdam-light | 0,9196 | 0,183 | 0,897 | 0,449 | 0,988 | 0,811 |
| TR-jv = init từ P1 C+JS+Java, SAM, RecAdam-light | 0,9205 | 0,781 | 0,802 | 0,749 | 0,987 | 0,864 |
| **ghép ½·p_SR + ½·p_TR-jv** | **0,9387** | 0,531 | 0,874 | 0,723 | 0,990 | **0,879** |
| đối chứng: ½·p_SR + ½·p_S (hai model không nguồn) | 0,9193 | 0,170 | 0,899 | 0,413 | 0,987 | 0,811 |

Theo commit: ghép − đối chứng ghép **+1,94 [−0,03 ; +4,85]**; ghép − mốc batch 4 `b4_sven` **+4,06 [+2,00 ; +6,61] \***. Ghép hai model không nguồn không tự tăng (+0,03 so SR) → phần tăng đến từ việc hai model **sai ở nhóm khác nhau**, không phải từ ensemble nói chung.

### A.2 Kiến trúc

```
hàm đích ──► SR  (MW, đông cứng) ──► p_A ─┐
                                          ├─► cổng g(·) ──► p = (1−g)·p_A + g·p_B
hàm đích ──► TR-jv (MW, đông cứng) ─► p_B ─┘
```

- Hai model giữ nguyên checkpoint đã có; **không huấn luyện lại encoder**.
- Cổng g ∈ [0,1] học trên **val** của cùng fold, nhãn 0/1 thường. Đầu vào cổng — ba mức, thử theo thứ tự:
  1. **g = hằng** (0,5 là bản đã đo; quét 0,3…0,7 trên val) — 1 tham số.
  2. **g = σ(w·[logit_A, logit_B, |p_A − p_B|, 1])** — 4 tham số, hồi quy logistic trên val.
  3. g từ đặc trưng mã (z_base 768 chiều của SR, đông cứng) — chỉ khi 2 không đủ; rủi ro quá khớp với val 152 mẫu.
- Suy luận: 2 encoder MW (≈ 2× SR). Không có head CWE, không định tuyến theo CWE.

### A.3 Cách kiểm (tiêu chí chặt)

- **Đối chứng cùng kiến trúc**: cùng cổng, cùng cách học, nhưng hai model đều không nguồn (SR + S). Δ nguồn = ghép(SR, TR-jv) − ghép(SR, S).
- Bootstrap theo commit (5 000 lần, `n5_commit_cwe.py`), 3 fold rồi 5 fold (SR và TR-jv fold 4–5 đang chờ trong q35 trên 158).
- Đọc thêm per-CWE và phân bố g theo CWE (chỉ để kiểm cơ chế, không dùng lúc huấn luyện).
- Ngưỡng mF1: dùng 0,5 (ngưỡng calib trên val nhảy 0,32→0,94 giữa fold, §130).

### A.4 Mã tham chiếu cho cổng (mới, CPU, chạy trên `*.probs.npz` đã lưu)

```python
# gate_fit.py — hoc cong tren val, ap len test; dau vao la probs npz cua hai model (co val_probabilities/val_labels)
import numpy as np, glob, os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
R = "/drive1/cuongtm/tmp/pr3_results"
def z(t, f): return np.load(glob.glob(f"{R}/{t}/*/seed_36/fold{f}.probs.npz")[0])
def feats(pa, pb):
    la, lb = np.log(pa/(1-pa+1e-9)+1e-9), np.log(pb/(1-pb+1e-9)+1e-9)
    return np.stack([la, lb, np.abs(pa-pb)], 1)
def run(A, B, folds=(1,2,3), mode="logreg"):
    out = []
    for f in folds:
        a, b = z(A, f), z(B, f)
        if mode == "const":                              # muc 1: quet g hang tren val
            gs = np.linspace(0.1, 0.9, 17)
            g = gs[np.argmax([roc_auc_score(a["val_labels"], (1-g)*a["val_probabilities"]+g*b["val_probabilities"]) for g in gs])]
            p = (1-g)*a["probabilities"] + g*b["probabilities"]
        else:                                            # muc 2: logistic tren [logit_A, logit_B, |pA-pB|]
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(feats(a["val_probabilities"], b["val_probabilities"]), a["val_labels"])
            p = clf.predict_proba(feats(a["probabilities"], b["probabilities"]))[:, 1]
        y = a["labels"]; out.append((roc_auc_score(y, p), f1_score(y, (p >= 0.5).astype(int), average="macro")))
        d = f"{R}/gate_{A}__{B}/multiwindow/seed_36"; os.makedirs(d, exist_ok=True)
        np.savez(f"{d}/fold{f}.probs.npz", probabilities=p, labels=y)   # de n5_commit_cwe.py doc
    return np.mean(out, 0)
for A, B in [("mw_bsamRl_sven_mean", "mw_xEAsRl_ccjscjv2sven"), ("mw_bsamRl_sven_mean", "mw_bsam_sven_mean")]:
    for mode in ("const", "logreg"): print(A, B, mode, run(A, B, mode=mode))
```

---

## B. Thành phần đã triển khai (mã + mô tả)

### B.1 Dữ liệu JS: lọc code nén/mã hoá → xoá comment → khử trùng → chia fold

**Quy tắc chốt 13/09 (§37):** lọc trên **mã thô** trước, rồi mới xoá comment (lọc sau khi xoá comment là sai thứ tự vì xoá comment làm đổi số dòng/độ dày). Áp cho cả đích lẫn nguồn JS. Kết quả: 1 836 hàm thô → loại 573 (31,2 %: one-liner 516 · long-line 52 · compress-function 3 · dense 2) → 1 263 → khử trùng md5 → **1 261** (629 dương / 632 âm) → `data/pr_jsc_folds/fold1..5` (6:2:2, seed 20260911 + f − 1) và nguồn `src_pr_jsc.jsonl`.

**Bộ lọc 8 tiêu chí** (`build_jsc.py::is_unreadable`, người dùng cung cấp; trả về lý do loại hoặc `None`):

```python
def is_unreadable(code):           # 8 tieu chi nguoi dung cung cap (build_js_cleanfull.py)
    lines = code.split('\n'); nl = len(lines)
    if nl == 1: return 'one-liner'
    if max((len(x) for x in lines), default=0) > 250: return 'long-line(minified)'
    if len(code) / max(1, nl) > 120: return 'dense(avg>120)'
    if re.search(r'__webpack_|webpackJsonp|__esModule', code): return 'webpack'
    if re.search(r'_interopRequireDefault|_classCallCheck|_createClass|_typeof2?=|_slicedToArray', code): return 'babel'
    if 'lime25' in code or 'limedev' in code: return 'lime-artifact'
    if code.count(';') / max(1, nl) > 3: return 'many-semicolons/line'
    if re.match(r'\s*function\s+[\w$]{1,2}\s*\(', code): return 'compress-function'
    return None
```

Ý nghĩa từng tiêu chí: cả hàm trên một dòng (bundle/minified); có dòng > 250 ký tự; trung bình > 120 ký tự/dòng; dấu vết webpack/babel (mã sinh máy); artifact riêng của bộ CleanVul (`lime25`); > 3 dấu `;` mỗi dòng; tên hàm bị rút còn 1–2 ký tự. Ghi chú từ script gốc: tín hiệu CWE-079 trước đây một phần LÀ artifact của bundle minified, nên lọc có thể làm điểm giảm — chấp nhận.

**Xoá comment** (`reptile_codebert/code_clean.py::strip_comments`) — máy trạng thái duyệt từng ký tự, không dùng regex vì `//` và `/*` có thể nằm trong chuỗi; xử lý escape và template literal JS; giữ ký tự xuống dòng để không dính dòng; bỏ dòng trắng thừa sau khi xoá:

```python
def strip_comments(code: str) -> str:
    out = []
    i, n = 0, len(code)
    state = "code"        # code | line | block | str
    quote = ""
    while i < n:
        c = code[i]
        nxt = code[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line"; i += 2; continue
            if c == "/" and nxt == "*":
                state = "block"; i += 2; continue
            if c in ('"', "'", "`"):
                state = "str"; quote = c; out.append(c); i += 1; continue
            out.append(c); i += 1; continue
        if state == "line":
            if c == "\n":
                state = "code"; out.append(c)      # giữ xuống dòng để không dính dòng
            i += 1; continue
        if state == "block":
            if c == "*" and nxt == "/":
                state = "code"; i += 2; continue
            if c == "\n":
                out.append(c)                       # giữ cấu trúc dòng
            i += 1; continue
        # state == "str"
        out.append(c)
        if c == "\\" and i + 1 < n:                 # escape: nuốt luôn ký tự sau
            out.append(code[i + 1]); i += 2; continue
        if c == quote:
            state = "code"
        i += 1
    text = "".join(out)
    # bỏ dòng trắng thừa sinh ra sau khi xoá comment
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln.strip())
```

Giới hạn đã biết: regex literal JS (`/.../`) không được nhận dạng, nên `//` bên trong regex literal bị coi là comment dòng; chưa thấy gây lỗi trên pool jsc (kiểm "vi phạm sau xoá comment" = 0). Áp cho C/C++, Java, JS; Python dùng `strip_comments_py` riêng.

**Vòng dựng pool** (`build_jsc.py`, phần chính):

```python
for score in (3, 4):
    with open(f"{CSV}/vulnerability_score_{score}.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r.get("extension") or "").strip().lower() != "js": continue
            cwe = canon_cwe(r.get("cwe_id"))
            for col, lab in (("func_before", 1), ("func_after", 0)):
                c0 = r.get(col) or ""
                if not c0.strip(): continue
                n_raw += 1
                why = is_unreadable(c0)              # (1) loc tren MA THO
                if why: drop[why] += 1; continue
                c = strip_comments(c0)               # (2) xoa comment
                if nows(c) != nows(c0): stripped += 1
                if not c.strip(): continue
                h = hashlib.md5(nows(c).encode()).hexdigest()
                if h in seen: continue               # (3) khu trung
                seen.add(h)
                recs.append({"code": c, "label": lab, "lang": "js",
                             "cwe": cwe, "original_CWE_ID": cwe, "score": score})
# (4) chia: 3 fold seed SEED+f-1, cat 6:2:2 (fold 4-5 dung them sau, cung cach)
for f in (1, 2, 3):
    r = list(recs); random.Random(SEED + f - 1).shuffle(r)
    n = len(r); ntr = int(n*0.6); nva = int(n*0.2)
    parts = {"train": r[:ntr], "val": r[ntr:ntr+nva], "test": r[ntr+nva:]}
```

Nguồn: CleanVul score 3+4, cột `func_before` (nhãn 1) / `func_after` (nhãn 0); `pair_id` = URL commit (bộ `src_pair_*` giữ đủ cặp). Chia fold là **paired_random** (ngẫu nhiên theo hàm, không theo cặp — chốt của người dùng); vì vậy fold jsc chồng nhau ≈ 20 % (repeated random holdout), khác 5 fold rời của SVEN.

### B.2 Transfer hai pha (`train_transfer.py`, `train.py`)

**Pha 1 — học trên nguồn (không chứa ngôn ngữ đích):** CE hai lớp, AdamW, chọn checkpoint theo val ROC trên 10 % nguồn giữ lại, lưu `cache/v2_<nguồn>_s36.pt` (state_dict + `training_args`). Trích `run_phase1` (bỏ nhánh head CWE phụ và RankNet — cả hai đã bị loại):

```python
def run_phase1(args, device):
    records = load_jsonl(args.data_path, trust_precomputed=args.cwe_vocab == "precomputed")
    train_records, val_records = split_source_records(records, args.seed)          # 90/10 ngau nhien theo ban ghi
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = make_model(args.model_name, device, args)                               # CodeBERT + vul_head Linear(768, 2)
    train_loader = build_dataloader(train_records, tokenizer, args.max_length, args.batch_size, True, args.seed,
                                    args.num_workers, args.truncation_strategy)   # head_middle_tail, 512 token
    val_loader   = build_dataloader(val_records, tokenizer, args.max_length, args.eval_batch_size, False, args.seed,
                                    args.num_workers, args.truncation_strategy)
    optimizer = torch.optim.AdamW([{"params": list(model.parameters())}], lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loop(args, model, train_loader, val_loader, optimizer, device, "phase1", save_checkpoint)
```

Nguồn "rỗng" (`n_*`): checkpoint `v2_NULL_s36.pt` = CodeBERT gốc + head khởi tạo, đi qua đúng đường pha 2 → đối chứng cùng optimizer/neo để tách hiệu ứng dữ liệu nguồn.

**Pha 2 — học trên đích:** nạp θ* nguồn, neo RecAdam = bản sao trọng số ngay sau khi nạp (`recadam_anchor=source`), warmup tuyến tính tuỳ chọn, replay nguồn tuỳ chọn. Trích `run_phase2` (bỏ nhánh `phase2_aux`, `lp_epochs`, `val_hard_ref`):

```python
def run_phase2(args, device):
    train_path, val_path, _ = python_paths(args)
    train_records = limit_records(load_jsonl(train_path, args.target_lang), args.max_train_samples, args.seed)
    val_records   = limit_records(load_jsonl(val_path,   args.target_lang), args.max_eval_samples,  args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.num_cwes = adopt_checkpoint_shape(args.source_checkpoint, device)
    model = make_model(args.model_name, device, args)
    source_checkpoint = load_checkpoint(args.source_checkpoint, model, device)   # theta* (hoac NULL)
    if args.source_interpolation < 1.0:
        interpolate_backbone(model, args, device)                                 # WiSE-FT, da dong (khong dung)
    assert_checkpoint_compatible(source_checkpoint, args, "source checkpoint")   # model_name phai khop (memory: cache chep may khac)
    freeze_aux_head(model)
    named_trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    current_params  = [p for _, p in named_trainable]
    pretrain_params = build_recadam_anchor(named_trainable, args, device)        # = clone ngay sau khi nap; 'pretrained' da dong
    assert_recadam_setup(current_params, pretrain_params, model)
    train_loader = build_dataloader(train_records, tokenizer, args.max_length, args.batch_size, True, args.seed, args.num_workers, args.truncation_strategy)
    val_loader   = build_dataloader(val_records,   tokenizer, args.max_length, args.eval_batch_size, False, args.seed, args.num_workers, args.truncation_strategy)
    total_steps = max(1, len(train_loader) * args.epochs)
    anneal_t0 = max(1, int(args.anneal_t0_ratio * total_steps))
    optimizer = RecAdam(current_params, lr=args.learning_rate, weight_decay=args.weight_decay, anneal_fun=args.anneal_fun,
                        anneal_k=args.anneal_k, anneal_t0=anneal_t0, anneal_w=args.anneal_w,
                        pretrain_cof=args.pretrain_cof, pretrain_params=pretrain_params)
    args._scheduler = None
    if getattr(args, "warmup_ratio", 0.0) > 0:
        from transformers import get_linear_schedule_with_warmup
        args._scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    replay = build_replay(args, model, tokenizer, device)                        # None neu --replay_mode none
    train_loop(args, model, train_loader, val_loader, optimizer, device, "phase2", save_checkpoint, pretrain_params, replay=replay)
```

```python
def build_recadam_anchor(named_trainable, args, device):
    anchor = [parameter.detach().clone() for _, parameter in named_trainable]
    if args.recadam_anchor == "source":
        return anchor
    # (nhanh 'pretrained': neo ve CodeBERT goc — da thu, -4,60*, dong; bo o day)
```

**Replay nguồn trong pha 2** (`build_replay`): loader nguồn lấy mẫu cân nhãn (WeightedRandomSampler), batch riêng (`--replay_batch_size`, mặc định = batch target → lưu ý 16 ở fold 1–3 cũ, 8 ở các nhánh sau), quay vòng vô hạn; `uniform` w = 1, `cosine` w = EMA(max(0, cos(g_S, g_T))) trên 2 tầng cuối + head (đã đóng):

```python
def build_replay(args, model, tokenizer, device):
    mode = getattr(args, "replay_mode", "none")
    if mode == "none":
        return None
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from dataset import CodeDataset
    src = load_jsonl(args.data_path)
    labels = [int(r["label"]) for r in src]
    n1 = sum(labels); n0 = len(labels) - n1
    weights = [1.0 / n1 if y == 1 else 1.0 / n0 for y in labels]          # batch can nhan (ky vong)
    g = torch.Generator(); g.manual_seed(args.seed + 7)
    sampler = WeightedRandomSampler(weights, num_samples=len(src), replacement=True, generator=g)
    bs = getattr(args, "replay_batch_size", 0) or args.batch_size
    loader = DataLoader(CodeDataset(src, tokenizer, args.max_length, args.truncation_strategy),
                        batch_size=bs, sampler=sampler, num_workers=0, drop_last=True)
    def cycle():
        while True:
            for b in loader:
                yield b
    L = model.backbone.config.num_hidden_layers
    nl = getattr(args, "replay_cos_layers", 2)
    want = tuple(f"backbone.encoder.layer.{i}." for i in range(L - nl, L)) + ("vul_head.",)
    params = [q for n, q in model.named_parameters() if q.requires_grad and n.startswith(want)]
    return {"iter": cycle(), "mode": mode, "lam": float(args.replay_lambda), "params": params,
            "ema_beta": float(getattr(args, "replay_ema", 0.9)), "ema": None, "stats": {"w": [], "cos": []}}
```

**Một epoch pha 2** (`train.py::train_one_epoch_phase2`, bỏ nhánh aux và cosine): loss = CE_T(+focal tuỳ chọn) + λ·w·CE_S; SAM hai lượt; clip 1,0; RecAdam bước; scheduler:

```python
def target_ce(logits, labels, focal_gamma=0.0):
    if focal_gamma <= 0:
        return F.cross_entropy(logits, labels)
    ce = F.cross_entropy(logits, labels, reduction="none")
    p_t = torch.exp(-ce)
    return ((1.0 - p_t) ** focal_gamma * ce).mean()

def train_one_epoch_phase2(model, dataloader, optimizer, device, max_grad_norm, pretrain_params,
                           check_first_step=False, sam=None, aux_lambda=0.0, scheduler=None, replay=None, focal_gamma=0.0):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device); attention_mask = batch["attention_mask"].to(device); labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask, return_cwe=False)
        loss = target_ce(outputs["vul_logits"], labels, focal_gamma)
        s_ids = s_am = s_lab = None; w_src = 0.0
        if replay is not None:
            sb = next(replay["iter"])
            s_ids, s_am, s_lab = sb["input_ids"].to(device), sb["attention_mask"].to(device), sb["labels"].to(device)
            loss_s = F.cross_entropy(model(s_ids, s_am, return_cwe=False)["vul_logits"], s_lab)
            w_src = 1.0                                                   # mode "uniform"
            total = loss + replay["lam"] * w_src * loss_s
        else:
            total = loss
        total.backward()
        if sam is not None:
            named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
            trainable = [q for _, q in named]
            if sam.ascend(trainable, names=[n for n, _ in named]):        # w -> w + eps
                optimizer.zero_grad(set_to_none=True)
                out2 = model(input_ids, attention_mask, return_cwe=False)
                loss2 = target_ce(out2["vul_logits"], labels, focal_gamma)
                if replay is not None and w_src > 0:
                    loss2 = loss2 + replay["lam"] * w_src * F.cross_entropy(model(s_ids, s_am, return_cwe=False)["vul_logits"], s_lab)
                loss2.backward()
                sam.restore(trainable)                                    # ve w truoc optimizer.step()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
```

Cờ chuẩn của loạt v2 (512): `--batch_size 16 --max_length 512 --truncation_strategy head_middle_tail --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.10 --patience 8 --min_epochs 3 --selection_metric roc_auc --pooling cls --sam_rho 0.02 --pretrain_cof 5000 --anneal_t0_ratio 0.05 --anneal_k 0.05 --recadam_anchor source`; replay: `--replay_mode uniform --replay_lambda 0.5 --replay_batch_size 8`. Mốc `b_*`: `train_baseline.py`, AdamW, không SAM (batch 16; `b4_*` batch 4).

### B.3 Nhiều cửa sổ (`train_mw.py`)

Mỗi hàm → tối đa K = 8 cửa sổ 510 token, bước 384; encoder CodeBERT dùng chung; vector cửa sổ = CLS + embedding chỉ số cửa sổ + chiếu vị trí tương đối; mean-pool có mặt nạ; head 2 lớp. Mã nguyên văn:

```python
class WindowDataset(Dataset):
    """Moi ham -> [K, max_length] input_ids/attention_mask + mat na cua so [K]."""
    def __init__(self, records, tokenizer, window, stride, max_windows, max_length=512):
        self.records = records; self.tok = tokenizer
        self.window, self.stride, self.K, self.max_length = window, stride, max_windows, max_length
        self.pad = tokenizer.pad_token_id
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        ids = self.tok(r["code"], add_special_tokens=False, truncation=False, return_attention_mask=False, verbose=False)["input_ids"]
        starts = [0] if len(ids) <= self.window else list(range(0, len(ids) - self.window + self.stride, self.stride))
        starts = starts[: self.K]
        wins = [ids[s: s + self.window] for s in starts]
        input_ids = torch.full((self.K, self.max_length), self.pad, dtype=torch.long)
        attn = torch.zeros((self.K, self.max_length), dtype=torch.long)
        wmask = torch.zeros(self.K, dtype=torch.bool)
        for k, w in enumerate(wins):
            seq = self.tok.build_inputs_with_special_tokens(w)
            input_ids[k, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[k, : len(seq)] = 1
            wmask[k] = True
        pos = torch.zeros(self.K, dtype=torch.float)          # vi tri tuong doi cua cua so trong ham (0..1)
        for k, s in enumerate(starts):
            pos[k] = s / max(1, len(ids) - 1)
        return {"input_ids": input_ids, "attention_mask": attn, "window_mask": wmask,
                "window_pos": pos, "labels": torch.tensor(int(r["label"]), dtype=torch.long),
                "index": torch.tensor(i, dtype=torch.long)}


class MultiWindowModel(nn.Module):
    def __init__(self, backbone, max_windows, agg="transformer", agg_layers=2, agg_heads=8, dropout=0.1, pooling="cls"):
        super().__init__()
        self.backbone = backbone; self.pooling = pooling; self.agg_kind = agg
        H = backbone.config.hidden_size
        self.win_pos_emb = nn.Embedding(max_windows, H)      # chi so cua so
        self.rel_pos = nn.Linear(1, H)                        # vi tri tuong doi 0..1
        if agg == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=H, nhead=agg_heads, dim_feedforward=4 * 256, dropout=dropout, batch_first=True, activation="gelu")
            self.agg = nn.TransformerEncoder(layer, num_layers=agg_layers)
        elif agg == "mean":
            self.agg = None                                   # cau hinh best: mean
        else:
            raise ValueError(agg)
        self.dropout = nn.Dropout(dropout)
        self.vul_head = nn.Linear(H, 2)

    def encode_windows(self, input_ids, attention_mask, window_mask, micro=16):
        """[B,K,L] -> [B,K,H]; chi dua cua so THAT qua backbone, theo microbatch."""
        B, K, L = input_ids.shape
        flat_idx = window_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        ids = input_ids.view(B * K, L)[flat_idx]
        am = attention_mask.view(B * K, L)[flat_idx]
        outs = []
        for s in range(0, ids.size(0), micro):
            o = self.backbone(input_ids=ids[s: s + micro], attention_mask=am[s: s + micro])
            outs.append(pool_hidden_states(o.last_hidden_state, am[s: s + micro], self.pooling))   # cls: vi tri 0
        h = torch.zeros(B * K, self.vul_head.in_features, device=input_ids.device, dtype=outs[0].dtype)
        h[flat_idx] = torch.cat(outs, 0)
        return h.view(B, K, -1)

    def forward(self, input_ids, attention_mask, window_mask, window_pos, micro=16):
        h = self.encode_windows(input_ids, attention_mask, window_mask, micro)      # [B,K,H]
        B, K, _ = h.shape
        h = h + self.win_pos_emb(torch.arange(K, device=h.device)).unsqueeze(0) + self.rel_pos(window_pos.unsqueeze(-1))
        if self.agg is not None:
            h = self.agg(h, src_key_padding_mask=~window_mask)
        m = window_mask.unsqueeze(-1).to(h.dtype)
        z = (h * m).sum(1) / m.sum(1).clamp(min=1.0)                                 # mean-pool co mat na
        return {"vul_logits": self.vul_head(self.dropout(z))}


def init_from(model, path, parts, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    assert ck.get("architecture") == "codebert_multiwindow", f"{path}: khong phai checkpoint multiwindow"
    sd = ck["model_state_dict"]
    keep = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    if parts == "pos":      # CHI chuyen embedding vi tri cua so / vi tri tuong doi; encoder = CodeBERT goc
        keep = {k: v for k, v in sd.items() if k.startswith(("win_pos_emb.", "rel_pos."))}
    if parts == "encoder_agg":
        keep.update({k: v for k, v in sd.items() if k.startswith(("agg.", "win_pos_emb.", "rel_pos."))})
    missing, unexpected = model.load_state_dict(keep, strict=False)
    assert not unexpected, unexpected
```

Vòng huấn luyện (trích `main`, phần train): RecAdam neo về trọng số ngay sau `init_from` (encoder + vị trí = θ* pha 1, head = khởi tạo), SAM hai lượt, warmup 10 %, chọn theo val ROC, patience 8:

```python
    if args.init != "none":
        init_from(model, args.init_ckpt, args.init, device)
    total = len(tr) * args.epochs
    if args.recadam:
        from RecAdam import RecAdam
        _params = [q for q in model.parameters() if q.requires_grad]
        _anchor = [q.detach().clone() for q in _params]          # neo = trong so ngay sau init_from
        _t0 = max(1, int(args.anneal_t0_ratio * total))
        opt = RecAdam(_params, lr=args.learning_rate, weight_decay=args.weight_decay, anneal_fun=args.anneal_fun,
                      anneal_k=args.anneal_k, anneal_t0=_t0, anneal_w=1.0, pretrain_cof=args.pretrain_cof, pretrain_params=_anchor)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sam = None
    if args.sam_rho > 0:
        from sam import SAMStep
        sam = SAMStep(args.sam_rho, variant="sam", eta=0.01)      # train_mw chi dung SAM goc (khong ASAM)
    sched = get_linear_schedule_with_warmup(opt, int(total * args.warmup_ratio), total)
    for ep in range(1, args.epochs + 1):
        model.train()
        for b in tr:
            out = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["window_mask"].to(device), b["window_pos"].to(device), args.micro)
            y = b["labels"].to(device); loss = F.cross_entropy(out["vul_logits"], y)
            opt.zero_grad(set_to_none=True); loss.backward()
            if sam is not None:
                named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
                trainable = [q for _, q in named]
                if sam.ascend(trainable, names=[n for n, _ in named]):
                    opt.zero_grad(set_to_none=True)
                    out2 = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["window_mask"].to(device), b["window_pos"].to(device), args.micro)
                    F.cross_entropy(out2["vul_logits"], y).backward()
                    sam.restore(trainable)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm); opt.step(); sched.step()
        v = run_eval(model, va, device, args.micro)
        score = v.get(args.selection_metric)
        if score > best:
            best, best_ep, wait = score, ep, 0; save_ckpt(args.checkpoint_path, model, ep, score, args)
        elif ep >= args.min_epochs:
            wait += 1
        if ep >= args.min_epochs and wait >= args.patience:
            break
```

Cờ chuẩn: `--window 510 --stride 384 --max_windows 8 --batch_size 4 --eval_batch_size 8 --micro 16 --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.10 --epochs 30 --min_epochs 3 --patience 8 --selection_metric roc_auc --agg mean --sam_rho 0.02`; RecAdam-light: `--recadam 1 --pretrain_cof 500 --anneal_t0_ratio 0.01`; nặng: `cof 5000, t0 0.05`. Pha 1 MW: `--data_root data/mwsrc_<nguồn>`, fold 1, `--sam_rho 0`, không init (pool dựng bằng `make_mwsrc.py`: chia 90/10 theo `pair_id`, seed 36). Trên SVEN: 30,5 % hàm > 510 token, trung bình 1,58 cửa sổ/hàm; hiệu ứng cửa sổ thuần ≈ 0 (§131), phần tăng của MW target-only đến từ batch 4 (+2,8 \*) và SAM (+1,9).

### B.4 SAM và ASAM (`sam.py`, nguyên văn phần thuật toán)

Port từ mã gốc Google (JAX). Bốn điểm giữ đúng: chuẩn L2 **toàn cục** qua mọi tensor; ρ là độ dài **tuyệt đối** của nhiễu; gradient lượt hai áp cho trọng số **gốc**; mọi thứ trước `optimizer.step()` nên ghép được với RecAdam hoặc AdamW. ASAM (Kwon et al. 2021): chuẩn hoá theo từng tham số qua T_w = |w| + η, chỉ tham số có `weight` trong tên; đã thử trên SVEN 512 (ρ 0,1) và không hơn SAM (§118).

```python
@torch.no_grad()
def _grad_global_norm(params):
    total = None
    for p in params:
        if p.grad is None:
            continue
        s = (p.grad.detach() ** 2).sum()
        total = s if total is None else total + s
    if total is None:
        return None
    return torch.sqrt(total)


class SAMStep:
    def __init__(self, rho, variant="sam", eta=0.01):
        if rho <= 0:
            raise ValueError("rho phai duong")
        if variant not in ("sam", "asam"):
            raise ValueError(f"variant phai la sam hoac asam, nhan {variant!r}")
        self.rho = rho; self.variant = variant; self.eta = eta; self._eps = []

    @torch.no_grad()
    def ascend(self, params, names=None):
        """Đi tới w + eps. Trả về False nếu gradient bằng 0 (bỏ qua bước)."""
        if self.variant == "asam":
            return self._ascend_asam(params, names)
        self._eps = []
        norm = _grad_global_norm(params)
        if norm is None or not torch.isfinite(norm) or norm.item() == 0.0:
            return False
        scale = self.rho / norm
        for p in params:
            if p.grad is None:
                self._eps.append(None); continue
            e = p.grad.detach() * scale
            p.add_(e)
            self._eps.append(e)
        return True

    @torch.no_grad()
    def _ascend_asam(self, params, names):
        if names is None:
            raise ValueError("ASAM can `names` de biet tham so nao la 'weight'")
        self._eps = []
        tws = []
        for p, n in zip(params, names):
            if p.grad is None:
                tws.append(None); continue
            tws.append(p.detach().abs().add(self.eta) if "weight" in n else None)
        total = None
        for p, tw in zip(params, tws):
            if p.grad is None:
                continue
            g = p.grad.detach() if tw is None else p.grad.detach() * tw
            sq = (g ** 2).sum()
            total = sq if total is None else total + sq
        if total is None:
            return False
        norm = torch.sqrt(total)
        if not torch.isfinite(norm) or norm.item() == 0.0:
            return False
        scale = self.rho / norm
        for p, tw in zip(params, tws):
            if p.grad is None:
                self._eps.append(None); continue
            g = p.grad.detach()
            e = (g if tw is None else g * tw * tw) * scale
            p.add_(e)
            self._eps.append(e)
        return True

    @torch.no_grad()
    def restore(self, params):
        """Trừ đúng eps đã cộng, đưa trọng số về w trước khi optimizer bước."""
        for p, e in zip(params, self._eps):
            if e is not None:
                p.sub_(e)
        self._eps = []
```

### B.5 RecAdam (`RecAdam.py`, Chen et al. 2020; phần lõi nguyên văn)

Mục tiêu: L = λ(t)·L_T + (1 − λ(t))·(cof/2)·Σ(θ − θ*)². λ(t) = sigmoid(k·(t − t0)) tăng từ 0 lên 1; lúc đầu neo mạnh, sau tự tắt. "Light" = cof 500, t0 = 1 % tổng bước; "nặng" = cof 5000, t0 = 5 %. Neo θ* = trọng số ngay sau khi nạp nguồn (head = khởi tạo ngẫu nhiên). Weight decay tách rời (không đi qua m/v của Adam).

```python
def anneal_function(function, step, k, t0, weight):
    if function == 'sigmoid':
        return float(1 / (1 + np.exp(-k * (step - t0)))) * weight
    elif function == 'linear':
        return min(1, step / t0) * weight
    elif function == 'constant':
        return weight
    else:
        raise ValueError("Unsupported anneal function: {}".format(function))


class RecAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True,
                 anneal_fun='sigmoid', anneal_k=0, anneal_t0=0, anneal_w=1.0, pretrain_cof=5000.0, pretrain_params=None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias,
                        anneal_fun=anneal_fun, anneal_k=anneal_k, anneal_t0=anneal_t0, anneal_w=anneal_w,
                        pretrain_cof=pretrain_cof, pretrain_params=pretrain_params)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p, pp in zip(group["params"], group["pretrain_params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                # Loss = lambda(t)*Loss_T + (1-lambda(t))*\gamma/2*\sum((\theta_i-\theta_i^*)^2)
                if group['anneal_w'] > 0.0:
                    anneal_lambda = anneal_function(group['anneal_fun'], state["step"], group['anneal_k'],
                                                    group['anneal_t0'], group['anneal_w'])
                    group["last_anneal_lambda"] = anneal_lambda
                    assert anneal_lambda <= group['anneal_w']
                    # The loss of the target task is multiplied by lambda(t)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size * anneal_lambda)
                    # Add the quadratic penalty to simulate the pretraining tasks
                    p.data.add_(p.data - pp.data, alpha=-group["lr"] * (group['anneal_w'] - anneal_lambda) * group["pretrain_cof"])
                else:
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
        return loss
```

Ghi chú đọc: với warmup tuyến tính, hệ số kéo về neo lớn nhất trong ba epoch đầu chỉ ≈ 3,8 %/bước ở cấu hình nặng; light gần như tắt neo sau ≈ 2·t0 bước (review 15/09 §3). Trên SVEN, tick RecAdam light/nặng/tắt đều trong ±1 ROC (§118).

---

## C. Lệnh chạy đã dùng (tham chiếu)

- 512: `v3f.sh` (`TGT=sven ARMS="x_ccjsc2sven:data/src_pr_ccpp_jsc.jsonl"`, thêm `r_*` qua `q3f.sh/q3h.sh` với `LAM`, `RBS`, `P2EXTRA`); mốc `DO_BASE=1`.
- MW: `q1r.sh` (`TGT=sven|js|src<nguồn>`, `ARMS="tag:mean"`, `INIT=encoder_agg INIT_CKPT=...`, `MWEXTRA="--recadam 1 --pretrain_cof 500 --anneal_t0_ratio 0.01"`, `SAM=0.02`, `FOLDS`).
- Ensemble/cổng: CPU trên `pr3_results/*/fold{f}.probs.npz`; bootstrap theo commit `review_audits/2026-09-14-afternoon/n5_commit_cwe.py` (py310).
- Tài liệu liên quan: `TAI_LAP_MW_RECADAM_LIGHT.md` (bản tự chứa để cài lại), `CHUYEN_GIAO_PAIRED_RANDOM.md` §127–§132, artifact sơ đồ đề xuất 2–3.

---

## A.5 Kết quả kiểm mức 1 và mức 2 (chạy 15/09 19:05, CPU, fold 1–3, `gate_fit.py` ở A.4; bootstrap theo commit `N_GATE.md`)

| tổ hợp | cách ghép | ROC | mF1@0,5 |
|---|---|---:|---:|
| SR + TR-jv | ½/½ cố định (§127) | 0,9387 | 0,879 |
| SR + TR-jv | mức 1: g hằng quét trên val | 0,9368 | 0,859 |
| **SR + TR-jv** | **mức 2: logistic [logit_A, logit_B, \|p_A − p_B\|] học trên val** | **0,9436** | 0,863 |
| SR + S (đối chứng, không nguồn) | mức 1 | 0,9203 | 0,815 |
| SR + S (đối chứng, không nguồn) | mức 2 | 0,9219 | 0,814 |

Theo commit (3 fold): **cổng(SR, TR-jv) − cổng(SR, S) = +2,17 [+0,20 ; +4,92] \*** ← tiêu chí chặt (cùng kiến trúc, cùng cổng, chỉ khác dữ liệu nguồn) **tách 0**;
cổng(SR, TR-jv) − SR **+2,40 [+0,37 ; +5,59] \***; cổng − ghép ½/½ +0,49 [+0,02 ; +1,01] \*.
Caveat: cổng học trên val (152 mẫu/fold) và đánh giá trên test — không rò rỉ test; 3 fold; mF1@0,5 của cổng thấp hơn ghép ½/½ (0,863 so 0,879) vì logistic
tối ưu log-loss chứ không tối ưu ngưỡng 0,5. Bước tiếp: n = 5 khi SR/TR-jv fold 4–5 xong (q35); thử cổng trên val gộp 3 fold để giảm phương sai.
