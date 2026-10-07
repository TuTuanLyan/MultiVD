"""Training epochs, RecAdam safety checks, and readable epoch reports."""

import math
import time
from collections import Counter

from pairloss import pair_margin_loss

import torch
import torch.nn.functional as F

from evaluate import classification_metrics, evaluate, print_classification_report
from logging_utils import get_logger
from model import auxiliary_loss


logger = get_logger()


def latent_diagnostics(assignments, cwe_targets, binary_labels, num_latent):
    """Is the auxiliary structure CWE-like, or has it collapsed onto the main task?

    High NMI against CWE with low NMI against the binary label is the outcome
    that justifies a latent auxiliary task at all: it means the head discovered
    vulnerability-type structure rather than re-learning what vul_head already
    predicts. Slot usage catches the other failure, collapse onto few slots.
    """
    from sklearn.metrics import normalized_mutual_info_score

    if not assignments:
        return {}
    valid = [i for i, c in enumerate(cwe_targets) if c != -100]
    used = sorted(set(assignments))
    counts = Counter(assignments)
    diagnostics = {
        "slots_used": len(used),
        "slots_total": num_latent,
        "largest_slot_share": max(counts.values()) / len(assignments),
        "nmi_binary": float(
            normalized_mutual_info_score(binary_labels, assignments)
        ),
    }
    if valid:
        diagnostics["nmi_cwe"] = float(
            normalized_mutual_info_score(
                [cwe_targets[i] for i in valid], [assignments[i] for i in valid]
            )
        )
    return diagnostics


def _combine_phase1_loss(vul_loss, aux_loss, lambda_cwe, aux_weighter):
    """Ham muc tieu cua Phase 1. Tach rieng vi SAM phai tinh LAI dung ham nay o w+eps."""
    if aux_weighter is None:
        return vul_loss if aux_loss is None else vul_loss + lambda_cwe * aux_loss
    # λ học được (Kendall/Gal/Cipolla). Hai vô hướng log-phương sai nằm trong
    # cùng optimizer, nên chúng được cập nhật cùng nhịp với trọng số.
    total, _ = aux_weighter(vul_loss, aux_loss)
    return total


def train_one_epoch_phase1(model, dataloader, optimizer, device, lambda_cwe, max_grad_norm,
                           aux_weighter=None, sam=None, aux_class_weight=None,
                           pair_ctx=None):
    """`sam=None` giu nguyen duong chay cu tung byte.

    Khi co SAM: nhieu loan phai lay tren DUNG ham muc tieu dang toi uu, ma o
    Phase 1 ham do la `vul + lambda*aux`, khong phai rieng `vul`. Nhieu loan chi
    theo loss nhi phan se leo doc theo mot mat khac voi mat that su dang huan
    luyen, va phep do thu duoc khong con la do nhon cua Phase 1 nua."""
    model.train()
    totals = Counter()
    examples = 0
    labels_seen, probabilities = [], []
    assignments, cwe_seen = [], []
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        cwes = batch["cwe_class"].to(device)
        outputs = model(input_ids, attention_mask, return_cwe=True)
        vul_loss = F.cross_entropy(outputs["vul_logits"], labels)
        aux_loss, assignment = auxiliary_loss(outputs, cwes, model.aux_mode,
            temperature=getattr(model, "latent_temperature", 0.1),
            class_weight=aux_class_weight)
        loss = _combine_phase1_loss(vul_loss, aux_loss, lambda_cwe, aux_weighter)
        # De xuat B: rang buoc TUONG DOI trong tung cap truoc-va/sau-va. Nhan tuyet doi sai
        # 40-75% (FACTS §51) nhung quan he "ban nao truoc ban va" dung theo cau tao, nen
        # nhieu bi ha cap tu "giam sat SAI" xuong "giam sat YEU". pair_ctx=None => duong cu.
        if pair_ctx is not None:
            pl, npair = pair_margin_loss(
                outputs["vul_logits"], batch["index"],
                pair_ctx["group"], pair_ctx["role"], pair_ctx["margin"])
            loss = loss + pair_ctx["beta"] * pl
            totals["pair"] += float(pl.item()) * labels.size(0)
            totals["pair_n"] += npair
        loss.backward()

        if sam is not None:
            # Luot 1 chi de lay HUONG leo; gia tri loss bao cao van la loss tai w.
            trainable = [p for p in model.parameters() if p.requires_grad]
            if sam.ascend(trainable):
                optimizer.zero_grad(set_to_none=True)
                out2 = model(input_ids, attention_mask, return_cwe=True)
                vul2 = F.cross_entropy(out2["vul_logits"], labels)
                aux2, _ = auxiliary_loss(out2, cwes, model.aux_mode,
                    temperature=getattr(model, "latent_temperature", 0.1),
                    class_weight=aux_class_weight)
                _combine_phase1_loss(vul2, aux2, lambda_cwe, aux_weighter).backward()
                # Ve w TRUOC optimizer.step(): gradient lay o w+eps nhung buoc
                # cap nhat phai ap cho w goc, dung nhu ma cua Google.
                sam.restore(trainable)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        size = labels.size(0)
        examples += size
        totals["loss"] += loss.item() * size
        totals["vul"] += vul_loss.item() * size
        totals["cwe"] += (0.0 if aux_loss is None else aux_loss.item()) * size
        labels_seen.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(outputs["vul_logits"].detach(), dim=-1)[:, 1].cpu().tolist())
        if assignment is not None:
            assignments.extend(assignment.detach().cpu().tolist())
            cwe_seen.extend(cwes.detach().cpu().tolist())
    # `pair_n` la SO DEM (bao nhieu cap dung duoc trong ca epoch), khong phai trung binh
    # theo hang — chia cho `examples` se bien no thanh mot phan so vo nghia.
    result = {key: value / examples for key, value in totals.items() if key != "pair_n"}
    if "pair_n" in totals:
        result["pair_n"] = totals["pair_n"]
    result.update(classification_metrics(labels_seen, probabilities, 0.5))
    result.update(labels=labels_seen, probabilities=probabilities)
    result["latent"] = latent_diagnostics(
        assignments, cwe_seen, labels_seen, getattr(model, "num_latent", 0)
    )
    return result


def assert_recadam_setup(current_params, pretrain_params, model, aux_trainable=False):
    assert len(current_params) == len(pretrain_params), "RecAdam parameter counts differ"
    for index, (current, source) in enumerate(zip(current_params, pretrain_params)):
        assert current.shape == source.shape, f"RecAdam parameter shape differs at index {index}"
        assert not source.requires_grad, f"source parameter {index} unexpectedly requires gradients"
    if aux_trainable:
        # Cầu CWE (--phase2_lambda_cwe / --replay_lambda_cwe): head phụ HỌC TIẾP ở Pha 2.
        assert all(
            parameter.requires_grad for parameter in model.aux_parameters()
        ), "auxiliary head should be trainable but some parameter is frozen"
    else:
        assert all(
            not parameter.requires_grad for parameter in model.aux_parameters()
        ), "auxiliary head not frozen"


def _phase2_target_loss(model, batch, device, lambda_cwe, teacher=None, beta=0.0, mode="cos",
                        gate_alpha=0.0):
    """Mục tiêu trên một batch ĐÍCH. `lambda_cwe=0`, `teacher=None` ⇒ đúng đường cũ từng byte.

    `lambda_cwe>0` ⇒ cộng λ·L_cwe trên `cwe_class` của đích (4 lớp, cùng bảng CWE_MAPPING
    với nguồn 4cwe) — head phụ Pha 1 học tiếp trên đích, là nửa "đích" của cầu CWE.

    `teacher` (FACTS §36) ⇒ cộng β·d(f_θ(x), f_θ*(x)): NEO TRONG KHÔNG GIAN ĐẶC TRƯNG về
    chính mô hình Pha 1, trên input ĐÍCH. Khác neo trọng số (RecAdam/L2-SP/SPD — bốn khối
    đã bác) ở chỗ ràng buộc này biết dữ liệu đích, và nó neo đúng đại lượng đã ĐO ĐƯỢC là
    chuyển giao: đặc trưng Pha 1 cho linear probe +0.1125 ROC (5/5 fold, codebert) trong khi
    `vul_head` của nó chỉ ~0.54 F1 zero-shot. `teacher` là bộ đệm đặc trưng ĐÃ TÍNH SẴN
    (mô hình tại thời điểm khởi tạo Pha 2), tra theo `batch["index"]` — 0 VRAM thêm, 0 giây
    thêm mỗi bước. `mode`: cos = 1 − cosine (chỉ giữ HƯỚNG, bất biến thang đo — đúng thứ
    linear probe dùng); mse = L2 bình phương trung bình (giữ cả độ dài)."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    want_feat = teacher is not None and beta > 0
    if lambda_cwe > 0:
        outputs = model(input_ids, attention_mask, return_cwe=True, return_features=want_feat)
        vul_loss = F.cross_entropy(outputs["vul_logits"], labels)
        cwes = batch["cwe_class"].to(device)
        aux_loss, _ = auxiliary_loss(outputs, cwes, model.aux_mode,
                                     temperature=getattr(model, "latent_temperature", 0.1))
        loss = vul_loss if aux_loss is None else vul_loss + lambda_cwe * aux_loss
    else:
        outputs = model(input_ids, attention_mask, return_cwe=False, return_features=want_feat)
        vul_loss = F.cross_entropy(outputs["vul_logits"], labels)
        aux_loss, loss = None, vul_loss
    feat_loss = None
    if want_feat:
        ref = teacher[batch["index"].to(teacher.device)].to(outputs["pooled"].dtype)
        cur = outputs["pooled"]
        if mode == "mse":
            feat_loss = F.mse_loss(cur, ref)
        else:
            feat_loss = (1.0 - F.cosine_similarity(cur, ref, dim=-1)).mean()
        loss = loss + beta * feat_loss
    # MANH 2: giam sat TRUC TIEP nhanh 8 chieu. Khong co no thi cong phai cham diem mot
    # nhanh chua duoc huan luyen, va se dim no ve 0 ngay epoch dau — phep do thanh vo nghia.
    if gate_alpha > 0 and "lat_vul_logits" in outputs:
        loss = loss + gate_alpha * F.cross_entropy(outputs["lat_vul_logits"], labels)
    return loss, vul_loss, aux_loss, outputs, labels, feat_loss


def _phase2_replay_loss(model, batch, device, mu, lambda_cwe):
    """μ·(L_vul_nguồn + λ·L_cwe_nguồn) trên MỘT batch nguồn (src/replay.py). Trả (đã nhân μ, chưa nhân, aux)."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids, attention_mask, return_cwe=(lambda_cwe > 0))
    loss = F.cross_entropy(outputs["vul_logits"], labels)
    aux_loss = None
    if lambda_cwe > 0:
        cwes = batch["cwe_class"].to(device)
        aux_loss, _ = auxiliary_loss(outputs, cwes, model.aux_mode,
                                     temperature=getattr(model, "latent_temperature", 0.1))
        if aux_loss is not None:
            loss = loss + lambda_cwe * aux_loss
    return mu * loss, loss, aux_loss


def train_one_epoch_phase2(
    model, dataloader, optimizer, device, max_grad_norm, pretrain_params,
    check_first_step=False, sam=None, replay=None, lambda_cwe_target=0.0, epoch=1,
    teacher_feats=None, feat_beta=0.0, feat_mode="cos", gate_alpha=0.0,
):
    """`sam=None`, `replay=None`, `lambda_cwe_target=0` giữ nguyên đường chạy cũ từng byte.

    replay (src/replay.py): mỗi bước lấy thêm MỘT batch NGUỒN và cộng μ(e)·L_nguồn vào mục
    tiêu. Hai backward NỐI TIẾP (đích rồi nguồn) cho đúng gradient tổng như backward của
    tổng loss (tuyến tính), nhưng chỉ giữ MỘT đồ thị trong bộ nhớ — t5p đã đo 13,4 GB cho
    một batch, gộp hai batch vào một đồ thị là tràn A4000 16 GB.
    Với SAM: lượt 2 tại w+ε phải tính lại ĐÚNG mục tiêu (đích + nguồn) như Phase 1 làm với
    `_combine_phase1_loss`; chỉ tính lại L_đích là xoá phần nguồn im lặng."""
    model.train()
    aux_trainable = any(p.requires_grad for p in model.aux_parameters())
    mu = replay.mu(epoch) if replay is not None else 0.0
    lam_s = replay.lambda_cwe if replay is not None else 0.0
    total_loss = 0.0
    total_vul = 0.0
    total_cwe = 0.0
    total_replay = 0.0
    total_feat = 0.0
    examples = 0
    labels_seen, probabilities = [], []
    source_versions = [parameter._version for parameter in pretrain_params]
    checked = False
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        loss, vul_loss, aux_loss, outputs, labels, feat_loss = _phase2_target_loss(
            model, batch, device, lambda_cwe_target, teacher_feats, feat_beta, feat_mode,
            gate_alpha
        )
        loss.backward()
        replay_batch, replay_loss = None, None
        if mu > 0:
            replay_batch = replay.next_batch()
            scaled, replay_loss, _ = _phase2_replay_loss(model, replay_batch, device, mu, lam_s)
            scaled.backward()

        if sam is not None:
            # Lượt 1 chỉ để lấy HƯỚNG leo; giá trị loss báo cáo vẫn là loss tại w.
            trainable = [p for p in model.parameters() if p.requires_grad]
            if sam.ascend(trainable):
                optimizer.zero_grad(set_to_none=True)
                loss2, _, _, _, _, _ = _phase2_target_loss(
                    model, batch, device, lambda_cwe_target, teacher_feats, feat_beta, feat_mode,
                    gate_alpha
                )
                loss2.backward()
                if replay_batch is not None:
                    scaled2, _, _ = _phase2_replay_loss(model, replay_batch, device, mu, lam_s)
                    scaled2.backward()
                # Về w TRƯỚC optimizer.step(): gradient lấy ở w+eps nhưng bước
                # cập nhật phải áp cho w gốc, đúng như mã của Google.
                sam.restore(trainable)

        if check_first_step and not checked:
            if aux_trainable:
                # Cầu CWE: head phụ đang học, và phải THẬT SỰ nhận gradient khi có λ — nếu
                # không thì cờ bật mà không có tác dụng, kết quả sẽ trông y hệt "plain".
                if lambda_cwe_target > 0 or (mu > 0 and lam_s > 0):
                    assert any(p.grad is not None for p in model.aux_parameters()), (
                        "auxiliary head is trainable but received no gradient"
                    )
            else:
                assert all(parameter.grad is None for parameter in model.aux_parameters()), (
                    "auxiliary head received a gradient in Phase 2"
                )
            useful = [
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and parameter.grad is not None
                and (name.startswith("backbone.") or name.startswith("vul_head."))
            ]
            assert useful, "neither backbone nor vul_head received gradients"
            probe = min(useful, key=lambda parameter: parameter.numel())
            probe_before = probe.detach().clone()

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step()

        if check_first_step and not checked:
            assert not torch.equal(probe_before, probe.detach()), "optimizer step changed no probed parameter"
            assert source_versions == [parameter._version for parameter in pretrain_params], (
                "cloned source parameters changed during RecAdam step"
            )
            logger.info("RecAdam first-step safety checks passed")
            checked = True

        size = labels.size(0)
        examples += size
        total_loss += loss.item() * size
        total_vul += vul_loss.item() * size
        total_cwe += (0.0 if aux_loss is None else aux_loss.item()) * size
        total_replay += (0.0 if replay_loss is None else replay_loss.item()) * size
        total_feat += (0.0 if feat_loss is None else feat_loss.item()) * size
        labels_seen.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(outputs["vul_logits"].detach(), dim=-1)[:, 1].cpu().tolist())
    result = {"loss": total_loss / examples, "vul": total_vul / examples}
    if lambda_cwe_target > 0:
        result["cwe"] = total_cwe / examples
    if replay is not None:
        # L_nguồn CHƯA nhân μ (để đọc được độ khó của nguồn qua các epoch) và μ(e) đã dùng.
        result["replay"] = total_replay / examples
        result["replay_mu"] = mu
    if teacher_feats is not None and feat_beta > 0:
        # Khoảng cách đặc trưng CHƯA nhân β: đọc được mô hình đã rời khỏi đặc trưng Pha 1 bao xa.
        result["feat"] = total_feat / examples
    result.update(classification_metrics(labels_seen, probabilities, 0.5))
    result.update(labels=labels_seen, probabilities=probabilities)
    return result


def train_one_epoch_binary(model, dataloader, optimizer, device, max_grad_norm):
    """Plain binary fine-tuning used by the no-transfer CodeBERT baseline."""
    model.train()
    total_loss = 0.0
    examples = 0
    labels_seen, probabilities = [], []
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask, return_cwe=False)
        loss = F.cross_entropy(outputs["vul_logits"], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        size = labels.size(0)
        examples += size
        total_loss += loss.item() * size
        labels_seen.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(outputs["vul_logits"].detach(), dim=-1)[:, 1].cpu().tolist())
    result = {"loss": total_loss / examples, "vul": total_loss / examples}
    result.update(classification_metrics(labels_seen, probabilities, 0.5))
    result.update(labels=labels_seen, probabilities=probabilities)
    return result


def print_epoch(
    epoch, total_epochs, train, val, optimizer, best_epoch, patience_counter, include_cwe,
    train_seconds, validation_seconds, epoch_seconds, total_seconds
):
    anneal_lambda = optimizer.param_groups[0].get("last_anneal_lambda")
    logger.info(
        "Epoch %d/%d | Train loss: %.6f | Val loss: %.6f | LR: %.2e | "
        "Best epoch: %d | Patience: %d | Train: %.2fs | Validation: %.2fs | "
        "Epoch: %.2fs | Total: %.2fs",
        epoch,
        total_epochs,
        train["loss"],
        val["loss"],
        optimizer.param_groups[0]["lr"],
        best_epoch,
        patience_counter,
        train_seconds,
        validation_seconds,
        epoch_seconds,
        total_seconds,
    )
    if include_cwe:
        logger.info(
            "Phase-1 losses | Vulnerability: %.6f | CWE: %.6f | Total: %.6f",
            train["vul"],
            train["cwe"],
            train["loss"],
        )
    latent = train.get("latent") or {}
    if latent:
        logger.info(
            "Latent structure | Slots used: %d/%d | Largest slot: %.3f | "
            "NMI vs CWE: %s | NMI vs binary: %.4f",
            latent["slots_used"],
            latent["slots_total"],
            latent["largest_slot_share"],
            f"{latent['nmi_cwe']:.4f}" if "nmi_cwe" in latent else "n/a",
            latent["nmi_binary"],
        )
    if anneal_lambda is not None:
        logger.info("RecAdam target-task weight at epoch end: %.6f", anneal_lambda)
    print_classification_report("train", train["labels"], train["probabilities"], epoch=epoch)
    print_classification_report("validation", val["labels"], val["probabilities"], epoch=epoch)


def train_loop(
    args,
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    phase,
    save_checkpoint,
    pretrain_params=None,
    aux_weighter=None,
    replay=None,
    teacher_feats=None,
    aux_class_weight=None,
    # `pair_ctx` phai o CUOI: chen vao giua chu ky se lam cac loi goi truyen
    # THEO VI TRI bi lech mot bac. Da xay ra that 15/09: `pretrain_params` o dong
    # 1322/1344 duoc truyen theo vi tri, bi gan thanh `pair_ctx`, va Pha 2 chet bang
    # `TypeError: 'NoneType' object is not iterable` — mot loi KHONG lien quan gi
    # toi tinh nang vua them, nen rat kho lan ra.
    pair_ctx=None,
):
    # SAM chi bat khi --sam_rho > 0. Mac dinh 0 -> sam=None -> duong chay cu.
    sam = None
    if phase in ("phase1", "phase2") and getattr(args, "sam_rho", 0.0) > 0:
        # `--sam_variant asam` doi hinh dang vung nhieu loan tu qua cau ban kinh
        # tuyet doi sang ellipsoid ti le theo |w| (Kwon et al., ICML 2021).
        # LUU Y rho KHONG cung thang do giua hai bien the — xem src/sam.py.
        variant = getattr(args, "sam_variant", "sam")
        if variant == "asam":
            from sam import ASAMStep
            sam = ASAMStep(args.sam_rho, eta=getattr(args, "asam_eta", 0.01))
        else:
            from sam import SAMStep
            sam = SAMStep(args.sam_rho)
        logger.info(
            "%s bat o %s | rho=%.4f (%s) | moi buoc 2 luot forward-backward, "
            "thoi gian huan luyen ~2x",
            "ASAM" if variant == "asam" else "SAM", phase, args.sam_rho,
            "chuan hoa theo |w|, bat bien thang do" if variant == "asam"
            else "tuyet doi, chuan L2 toan cuc",
        )
    best_score, best_epoch, patience_counter = -math.inf, 0, 0
    training_started = time.perf_counter()
    # Gom gio tung epoch de cuoi lan chay ghi ra <checkpoint>.runtime.json. Truoc day
    # nhung con so nay chi ra console roi troi mat cung may thue, trong khi muc setup
    # cua bai lai can chinh chung.
    _rt_epoch, _rt_train, _rt_val, _rt_epochs_run = [], [], [], 0
    # Lich su val theo epoch (F1@0.5, ROC-AUC, PR-AUC, loss, lambda(t)). Ghi vao sidecar
    # runtime -> JSON ket qua. Muc dich: tra loi KHONG TON GPU cau "chon checkpoint theo
    # AUC thi co ra epoch khac khong, khac bao xa" truoc khi mo mot khoi chay lai
    # (RESEARCH_2026-09-06 §5.2). Tieu chi chon van la selection_metric, khong doi.
    _val_history = []
    logger.info("Training started | Phase: %s | Epochs: %d", phase, args.epochs)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        if phase == "phase1":
            train = train_one_epoch_phase1(
                model, train_loader, optimizer, device, args.lambda_cwe, args.max_grad_norm,
                aux_weighter=aux_weighter, sam=sam, aux_class_weight=aux_class_weight,
                pair_ctx=pair_ctx,
            )
            if pair_ctx is not None:
                logger.info("bien-trong-cap | beta %.3f margin %.2f | loss %.4f | %d cap dung duoc/epoch",
                            pair_ctx["beta"], pair_ctx["margin"],
                            train.get("pair", 0.0), int(train.get("pair_n", 0)))
            if aux_weighter is not None:
                d = aux_weighter.diagnostics()
                # λ_eff là đại lượng đáng đọc nhất của thí nghiệm này: nó nói mô
                # hình MUỐN λ bằng bao nhiêu, so trực tiếp được với 0.2 và 0.05
                # đã quét tay.
                logger.info(
                    "lambda hoc duoc | epoch %d | lambda_eff %.4f | w_bin %.4f | w_aux %.4f "
                    "| s_bin %+.4f | s_aux %+.4f",
                    epoch, d["lambda_eff"], d["w_binary"], d["w_aux"], d["s_binary"], d["s_aux"],
                )
                train["lambda_eff"] = d["lambda_eff"]
                # Gan vao args NGAY moi epoch: save_checkpoint chay ngay sau day khi
                # epoch nay la epoch tot nhat, nen dat sau vong lap thi checkpoint
                # se khong mang gia tri nao.
                args.learned_lambda_eff = d["lambda_eff"]
                args.learned_s_binary = d["s_binary"]
                args.learned_s_aux = d["s_aux"]
        elif phase == "phase2":
            train = train_one_epoch_phase2(
                model,
                train_loader,
                optimizer,
                device,
                args.max_grad_norm,
                pretrain_params,
                check_first_step=(epoch == 1),
                sam=sam,
                replay=replay,
                lambda_cwe_target=float(getattr(args, "phase2_lambda_cwe", 0.0) or 0.0),
                epoch=epoch,
                teacher_feats=teacher_feats,
                feat_beta=float(getattr(args, "feat_distill_beta", 0.0) or 0.0),
                feat_mode=getattr(args, "feat_distill_mode", "cos"),
                gate_alpha=float(getattr(args, "phase2_gate_alpha", 0.0) or 0.0),
            )
            gv = model.gate_value() if hasattr(model, "gate_value") else None
            if gv is not None:
                logger.info("CONG 8 chieu | epoch %d | g = %.4f (0 = toan bo qua 768 chieu, "
                            "1 = toan bo qua nut that neo vao nguon)", epoch, gv)
            if "feat" in train:
                logger.info("Neo dac trung | epoch %d | beta %.3f | d(f,f*) %.4f | L_dich %.4f",
                            epoch, float(getattr(args, "feat_distill_beta", 0.0) or 0.0),
                            train["feat"], train["vul"])
            if replay is not None:
                logger.info(
                    "Replay nguon | epoch %d | mu %.3f | L_nguon %.4f | L_dich %.4f%s",
                    epoch, train.get("replay_mu", 0.0), train.get("replay", 0.0), train["vul"],
                    "" if "cwe" not in train else " | L_cwe_dich %.4f" % train["cwe"],
                )
        elif phase == "baseline":
            train = train_one_epoch_binary(
                model, train_loader, optimizer, device, args.max_grad_norm
            )
        else:
            raise ValueError(f"unsupported training phase: {phase}")
        train_seconds = time.perf_counter() - train_started

        validation_started = time.perf_counter()
        if phase == "phase1":
            val = evaluate(model, val_loader, device, return_cwe=True, lambda_cwe=args.lambda_cwe)
        else:
            val = evaluate(model, val_loader, device, return_cwe=False)
        validation_seconds = time.perf_counter() - validation_started

        metric_name = getattr(args, "selection_metric", "macro_f1")
        score = val.get(metric_name)
        if score is None:
            # ROC/PR AUC are undefined when a validation split is single-class.
            score = val["macro_f1"]
        new_best = False
        tied_best = False
        # `--save_last_epoch`: LUON ghi checkpoint cua epoch vua xong va KHONG dung som.
        #
        # Vi sao can: doi chung "xao nhan" lam tac vu khong hoc duoc, nen chon-theo-val
        # se chot o mot epoch ngau nhien rat som, trong khi nhanh nhan THAT chot o epoch
        # 11-12. Khi do hai nhanh khac nhau CA nhan LAN so buoc gradient, va neu nhanh
        # xao chuyen giao kem thi khong biet tai cai nao. Co nay ep ca hai nhanh chay
        # dung cung so epoch va lay checkpoint CUOI.
        # Duong mac dinh (co tat) khong doi mot byte.
        if getattr(args, "save_last_epoch", False):
            best_score, best_epoch, patience_counter = score, epoch, 0
            save_checkpoint(args.checkpoint_path, model, epoch, score, args)
        elif score > best_score:
            best_score, best_epoch, patience_counter = score, epoch, 0
            save_checkpoint(args.checkpoint_path, model, epoch, score, args)
            new_best = True
        else:
            # Macro-F1 is discrete and often ties while probabilities are still moving.
            # Keep the latest checkpoint among exact ties without treating it as a new best.
            if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12):
                best_epoch = epoch
                save_checkpoint(args.checkpoint_path, model, epoch, score, args)
                tied_best = True
            if epoch >= args.min_epochs:
                patience_counter += 1
        epoch_seconds = time.perf_counter() - epoch_started
        total_seconds = time.perf_counter() - training_started
        _val_history.append({
            "epoch": epoch,
            "train_loss": round(float(train["loss"]), 6),
            "val_loss": round(float(val["loss"]), 6),
            "val_macro_f1": round(float(val["macro_f1"]), 6),
            "val_roc_auc": None if val.get("roc_auc") is None else round(float(val["roc_auc"]), 6),
            "val_pr_auc": None if val.get("pr_auc") is None else round(float(val["pr_auc"]), 6),
            "anneal_lambda": optimizer.param_groups[0].get("last_anneal_lambda"),
            "lr": optimizer.param_groups[0].get("lr"),
            "new_best": bool(new_best or tied_best),
        })
        if phase == "phase2" and (replay is not None or "cwe" in train or "feat" in train):
            # Chỉ thêm khoá khi cầu CWE bật, để JSON của đường cũ không đổi một byte.
            _val_history[-1].update({
                "replay_mu": train.get("replay_mu"),
                "train_replay_loss": None if "replay" not in train else round(float(train["replay"]), 6),
                "train_cwe_loss": None if "cwe" not in train else round(float(train["cwe"]), 6),
                "train_feat_dist": None if "feat" not in train else round(float(train["feat"]), 6),
            })
        _rt_epoch.append(epoch_seconds)
        _rt_train.append(train_seconds)
        _rt_val.append(validation_seconds)
        _rt_epochs_run = epoch
        print_epoch(
            epoch, args.epochs, train, val, optimizer, best_epoch, patience_counter,
            phase == "phase1",
            train_seconds, validation_seconds, epoch_seconds, total_seconds
        )
        if new_best:
            logger.info("New best model | Macro-F1: %.6f | Epoch: %d", best_score, epoch)
        elif tied_best:
            logger.info(
                "Latest tied-best checkpoint saved | Macro-F1: %.6f | Epoch: %d",
                best_score,
                epoch,
            )
        if (not getattr(args, "save_last_epoch", False)
                and epoch >= args.min_epochs and patience_counter >= args.patience):
            logger.info("Early stopping | Epoch: %d | Best epoch: %d", epoch, best_epoch)
            break
    logger.info(
        "Best checkpoint saved | Path: %s | Epoch: %d | Val %s: %.6f",
        args.checkpoint_path,
        best_epoch,
        getattr(args, "selection_metric", "macro_f1"),
        best_score,
    )
    _rt_total = time.perf_counter() - training_started
    logger.info("Training finished | Elapsed: %.2fs", _rt_total)
    try:
        from runtime_env import build_runtime_record, write_runtime_sidecar
        _rt = build_runtime_record(
            phase=phase, args=args, device=device,
            epochs_planned=args.epochs, epochs_run=_rt_epochs_run, best_epoch=best_epoch,
            epoch_seconds=_rt_epoch, train_seconds=_rt_train, val_seconds=_rt_val,
            total_seconds=_rt_total,
            n_train=len(getattr(train_loader, "dataset", []) or []),
            n_val=len(getattr(val_loader, "dataset", []) or []),
            extra={
                "selection_metric": getattr(args, "selection_metric", "macro_f1"),
                "val_history": _val_history,
            },
        )
        _p = write_runtime_sidecar(args.checkpoint_path, _rt)
        if _p:
            logger.info(
                "Runtime ghi lai | %s | %d epoch | %.1fs/epoch | %s",
                _p, _rt_epochs_run, _rt["seconds"]["per_epoch"].get("mean", 0.0),
                _rt["hardware"].get("gpu_name"),
            )
    except Exception as exc:
        # Ghi nhat ky hong khong duoc lam hong lan chay: mot o co so ma thieu gio van
        # dung hon mot o trong.
        logger.warning("Khong ghi duoc runtime sidecar: %s", exc)
