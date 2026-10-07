"""Training epochs, RecAdam safety checks, and readable epoch reports."""

import math
import time
from collections import Counter

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


def train_one_epoch_phase1(model, dataloader, optimizer, device, lambda_cwe, max_grad_norm,
                           log_vars=None):
    """lambda_cwe co dinh, HOAC log_vars -> trong so hoc duoc theo do bat dinh.

    Kendall, Gal & Cipolla, CVPR 2018 (arXiv:1705.07115): thay vi do tay he so
    giua cac ham mat mat, hoc log(sigma^2) cho tung tac vu:

        L = sum_i [ exp(-s_i) * L_i + s_i ],   s_i = log(sigma_i^2)

    So hang +s_i chan nghiem tam thuong sigma -> vo cung. Voi hai tac vu thi thu
    thuc su hoc duoc la TY LE lambda = exp(s_vul - s_cwe) — dung dai luong ma
    du an nay dang do tay, va do an do khac nhau theo backbone
    (CodeT5+ va CodeBERT deu tot hon o 0.05 so voi 0.2, nhung khong cung muc).

    log_vars nam NGOAI model nen checkpoint khong doi va Buoc 2 khong biet den no.
    """
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
            temperature=getattr(model, "latent_temperature", 0.1))
        if log_vars is None:
            loss = vul_loss if aux_loss is None else vul_loss + lambda_cwe * aux_loss
        else:
            loss = torch.exp(-log_vars[0]) * vul_loss + log_vars[0]
            if aux_loss is not None:
                loss = loss + torch.exp(-log_vars[1]) * aux_loss + log_vars[1]
        loss.backward()
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
    result = {key: value / examples for key, value in totals.items()}
    result.update(classification_metrics(labels_seen, probabilities, 0.5))
    result.update(labels=labels_seen, probabilities=probabilities)
    result["latent"] = latent_diagnostics(
        assignments, cwe_seen, labels_seen, getattr(model, "num_latent", 0)
    )
    return result


def assert_recadam_setup(current_params, pretrain_params, model, allow_trainable_aux=False):
    """`allow_trainable_aux` chi dung cho --phase2_aux target4.

    Mac dinh buoc 2 dong bang head phu (no thuoc ve taxonomy cua NGUON, khong
    phai cua dich), nen khang dinh nay bat moi truong hop quen dong bang.
    Voi target4 thi head phu la head MOI 4 lop cua chinh DICH va PHAI hoc duoc,
    nen khang dinh do khong con ap dung.
    """
    assert len(current_params) == len(pretrain_params), "RecAdam parameter counts differ"
    for index, (current, source) in enumerate(zip(current_params, pretrain_params)):
        assert current.shape == source.shape, f"RecAdam parameter shape differs at index {index}"
        assert not source.requires_grad, f"source parameter {index} unexpectedly requires gradients"
    if not allow_trainable_aux:
        assert all(
            not parameter.requires_grad for parameter in model.aux_parameters()
        ), "auxiliary head not frozen"


def train_one_epoch_phase1_rank(model, dataloader, optimizer, device, max_grad_norm, rank_lambda=1.0):
    """RankNet O NGUON (nguoi dung de xuat 13/09): nguon van giu du cap (loi, va) nen dung duoc
    loss so cap:  L = CE(ca hai) + rank_lambda * BCE( sigma(s_loi - s_va), 1 ),  s = logit1 - logit0.
    Chi o pha 1; pha 2 khong doi. Khong dung head CWE (aux_mode phai la none)."""
    model.train()
    totals = Counter(); examples = 0; labels_seen, probabilities = [], []; n_pairs = 0; rank_acc = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        ids = torch.cat([batch["input_ids_v"], batch["input_ids_f"]]).to(device)
        am = torch.cat([batch["attention_mask_v"], batch["attention_mask_f"]]).to(device)
        B = batch["input_ids_v"].size(0)
        labels = torch.cat([torch.ones(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)]).to(device)
        out = model(ids, am, return_cwe=False)
        logits = out["vul_logits"]
        ce = F.cross_entropy(logits, labels)
        s = logits[:, 1] - logits[:, 0]
        diff = s[:B] - s[B:]
        rank = F.binary_cross_entropy_with_logits(diff, torch.ones_like(diff))
        loss = ce + rank_lambda * rank
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        examples += 2 * B; n_pairs += B; rank_acc += int((diff > 0).sum())
        totals["loss"] += loss.item() * 2 * B; totals["vul"] += ce.item() * 2 * B; totals["rank"] += rank.item() * 2 * B
        labels_seen.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(logits.detach(), dim=-1)[:, 1].cpu().tolist())
    result = {k: v / examples for k, v in totals.items()}
    result["cwe"] = 0.0; result["pair_acc"] = rank_acc / max(1, n_pairs)
    result.update(classification_metrics(labels_seen, probabilities, 0.5))
    result.update(labels=labels_seen, probabilities=probabilities)
    return result


def target_ce(logits, labels, focal_gamma=0.0):
    """CE cua TARGET o pha 2. focal_gamma=0 -> F.cross_entropy y het duong cu. gamma>0: focal loss
    (Lin et al. 2017) — mau da phan loai dung (p_t cao) bi giam trong so, gradient don ve mau kho
    (de xuat 14/09 muc 90: nhom CWE yeu 022/079 chiem 20 % pool, bi 078/089 gan tran chi phoi)."""
    if focal_gamma <= 0:
        return F.cross_entropy(logits, labels)
    ce = F.cross_entropy(logits, labels, reduction="none")
    p_t = torch.exp(-ce)
    return ((1.0 - p_t) ** focal_gamma * ce).mean()


def train_one_epoch_phase2(
    model, dataloader, optimizer, device, max_grad_norm, pretrain_params,
    check_first_step=False, sam=None, aux_lambda=0.0, scheduler=None, replay=None, focal_gamma=0.0
):
    """`sam=None` giữ nguyên đường chạy cũ từng byte; chỉ khi truyền vào một
    SAMStep thì mỗi bước mới thành hai lượt forward-backward.

    `aux_lambda > 0` (chi khi --phase2_aux target4) BAT tac vu phu o buoc 2:
    head CWE 4 lop cua chinh DICH, hoc cung luc voi tac vu chinh. Mac dinh 0.0
    giu nguyen hanh vi cu — buoc 2 KHONG he tinh loss phu (muc 40).
    """
    model.train()
    total_loss = 0.0
    examples = 0
    labels_seen, probabilities = [], []
    source_versions = [parameter._version for parameter in pretrain_params]
    checked = False
    # ---- Q3 (DE_XUAT_CAI_TIEN_TRANSFER.md muc 1): replay nguon trong pha 2 ----
    # replay=None -> duong chay cu tung byte. replay = dict(iter, mode, lam, params, ema_beta,
    # ema[None|float], stats{w, cos, fb}). mode "uniform": w=1; "cosine": w = EMA(max(0, cos(g_S, g_T)))
    # do tren `params` (cac tang encoder cuoi + head), w duoc DETACH (hang so) khi cap nhat that.
    fb_passes = 0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        use_aux = aux_lambda > 0.0
        outputs = model(input_ids, attention_mask, return_cwe=use_aux)
        loss = target_ce(outputs["vul_logits"], labels, focal_gamma)
        if use_aux:
            aux, _ = auxiliary_loss(outputs, batch["cwe_class"].to(device), model.aux_mode,
                                    temperature=getattr(model, "latent_temperature", 0.1))
            if aux is not None:
                loss = loss + aux_lambda * aux
        fb_passes += 1
        s_ids = s_am = s_lab = None
        w_src = 0.0
        if replay is not None:
            sb = next(replay["iter"])
            s_ids, s_am, s_lab = sb["input_ids"].to(device), sb["attention_mask"].to(device), sb["labels"].to(device)
            loss_s = F.cross_entropy(model(s_ids, s_am, return_cwe=False)["vul_logits"], s_lab)
            fb_passes += 1
            if replay["mode"] == "uniform":
                w_src = 1.0
            else:
                gT = torch.autograd.grad(loss, replay["params"], retain_graph=True, allow_unused=True)
                gS = torch.autograd.grad(loss_s, replay["params"], retain_graph=True, allow_unused=True)
                vt = torch.cat([g.reshape(-1) for g in gT if g is not None])
                vs = torch.cat([g.reshape(-1) for g in gS if g is not None])
                cos = float(F.cosine_similarity(vt, vs, dim=0).item())
                w_raw = max(0.0, cos)
                replay["ema"] = w_raw if replay["ema"] is None else replay["ema_beta"] * replay["ema"] + (1 - replay["ema_beta"]) * w_raw
                w_src = float(replay["ema"])
                replay["stats"]["cos"].append(cos)
            replay["stats"]["w"].append(w_src)
            total = loss + replay["lam"] * w_src * loss_s if w_src > 0 else loss
        else:
            total = loss
        total.backward()

        if sam is not None:
            # Lượt 1 chỉ để lấy HƯỚNG leo; giá trị loss báo cáo vẫn là loss tại w.
            named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
            trainable = [q for _, q in named]
            if sam.ascend(trainable, names=[n for n, _ in named]):
                optimizer.zero_grad(set_to_none=True)
                out2 = model(input_ids, attention_mask, return_cwe=use_aux)
                loss2 = target_ce(out2["vul_logits"], labels, focal_gamma)
                if use_aux:
                    aux2, _ = auxiliary_loss(out2, batch["cwe_class"].to(device), model.aux_mode,
                                             temperature=getattr(model, "latent_temperature", 0.1))
                    if aux2 is not None:
                        loss2 = loss2 + aux_lambda * aux2
                fb_passes += 1
                if replay is not None and w_src > 0:
                    loss2_s = F.cross_entropy(model(s_ids, s_am, return_cwe=False)["vul_logits"], s_lab)
                    loss2 = loss2 + replay["lam"] * w_src * loss2_s
                    fb_passes += 1
                loss2.backward()
                # Về w TRƯỚC optimizer.step(): gradient lấy ở w+eps nhưng bước
                # cập nhật phải áp cho w gốc, đúng như mã của Google.
                sam.restore(trainable)

        if check_first_step and not checked:
            if aux_lambda <= 0.0:
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
            # Phai doc lr TRUOC optimizer.step(): scheduler.step() chay ngay sau do
            # nen doc sau se ra lr cua buoc KE TIEP, khong phai buoc vua chay.
            lr_used = optimizer.param_groups[0].get("lr", 0.0)

        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if check_first_step and not checked:
            # Voi warmup tuyen tinh, lr o BUOC DAU bang 0 theo dung dinh nghia cua
            # get_linear_schedule_with_warmup -> optimizer.step() khong doi tham so nao,
            # va assert duoi day se no oan. Chi kiem tra khi lr thuc su > 0.
            if lr_used > 0:
                assert not torch.equal(probe_before, probe.detach()), (
                    "optimizer step changed no probed parameter"
                )
            assert source_versions == [parameter._version for parameter in pretrain_params], (
                "cloned source parameters changed during RecAdam step"
            )
            logger.info("RecAdam first-step safety checks passed")
            checked = True

        size = labels.size(0)
        examples += size
        total_loss += loss.item() * size
        labels_seen.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(outputs["vul_logits"].detach(), dim=-1)[:, 1].cpu().tolist())
    result = {"loss": total_loss / examples, "vul": total_loss / examples, "fb_passes": fb_passes}
    if replay is not None:
        ws = replay["stats"]["w"]; cs = replay["stats"]["cos"]
        result.update(replay_w_mean=(sum(ws) / len(ws) if ws else 0.0),
                      replay_zero_frac=(sum(1 for w in ws if w <= 0) / len(ws) if ws else 0.0),
                      replay_cos_mean=(sum(cs) / len(cs) if cs else float("nan")))
        replay["stats"]["w"].clear(); replay["stats"]["cos"].clear()
    result.update(classification_metrics(labels_seen, probabilities, 0.5))
    result.update(labels=labels_seen, probabilities=probabilities)
    return result


def train_one_epoch_binary(model, dataloader, optimizer, device, max_grad_norm, scheduler=None):
    """Plain binary fine-tuning used by the no-transfer CodeBERT baseline.

    `scheduler` tuy chon, mac dinh None -> duong chay cu khong doi mot buoc nao.
    MultiVD von KHONG co lr scheduler: lr hang 2e-5 tu dau den cuoi, khong warmup.
    Voi patience nho dieu do lam nhanh moc bi cat som ngau nhien — do duoc: ba fold
    cua cung mot cau hinh dung o epoch 4 / 15 / 5."""
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
        if scheduler is not None:
            scheduler.step()
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
    replay=None,
):
    # SAM chi bat khi --sam_rho > 0. Mac dinh 0 -> sam=None -> duong chay cu.
    sam = None
    if phase == "phase2" and getattr(args, "sam_rho", 0.0) > 0:
        from sam import SAMStep

        _variant = getattr(args, "sam_variant", "sam")
        sam = SAMStep(args.sam_rho, variant=_variant, eta=getattr(args, "sam_eta", 0.01))
        if _variant == "asam":
            logger.info(
                "ASAM bat | rho=%.4f eta=%.4f | chuan hoa THEO TUNG THAM SO qua "
                "T_w=|w|+eta nen rho bat bien theo thang | 2 luot forward-backward",
                args.sam_rho, getattr(args, "sam_eta", 0.01),
            )
        else:
            logger.info(
                "SAM bat | rho=%.4f (tuyet doi, chuan L2 toan cuc) | moi buoc 2 luot "
                "forward-backward, thoi gian huan luyen ~2x", args.sam_rho,
            )
    best_score, best_epoch, patience_counter = -math.inf, 0, 0
    training_started = time.perf_counter()
    logger.info("Training started | Phase: %s | Epochs: %d", phase, args.epochs)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        if phase == "phase1" and getattr(args, "phase1_loss", "ce") == "ranknet":
            train = train_one_epoch_phase1_rank(
                model, train_loader, optimizer, device, args.max_grad_norm,
                rank_lambda=getattr(args, "rank_lambda", 1.0),
            )
        elif phase == "phase1":
            train = train_one_epoch_phase1(
                model, train_loader, optimizer, device, args.lambda_cwe, args.max_grad_norm,
                log_vars=getattr(args, "_log_vars", None),
            )
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
                # Chi khac 0 khi --phase2_aux target4: bat tac vu phu o buoc 2
                # voi head CWE 4 lop cua chinh DICH.
                aux_lambda=(args.lambda_cwe
                            if getattr(args, "phase2_aux", "frozen_source") == "target4" else 0.0),
                scheduler=getattr(args, "_scheduler", None),
                replay=replay,
                focal_gamma=getattr(args, "focal_gamma", 0.0),
            )
        elif phase == "baseline":
            train = train_one_epoch_binary(
                model, train_loader, optimizer, device, args.max_grad_norm,
                scheduler=getattr(args, "_scheduler", None),
            )
        else:
            raise ValueError(f"unsupported training phase: {phase}")
        train_seconds = time.perf_counter() - train_started

        validation_started = time.perf_counter()
        if phase == "phase1":
            # lambda_cwe < 0 la co "hoc trong so": val loss phai dung lambda HIEU DUNG
            # hien tai, khong phai gia tri co (-1 se TRU aux loss -> loss am).
            _lv = getattr(args, "_log_vars", None)
            _lam = args.lambda_cwe
            if _lv is not None:
                _s = _lv.detach().tolist()
                _lam = math.exp(_s[0] - _s[1])
            val = evaluate(model, val_loader, device, return_cwe=True, lambda_cwe=_lam)
        else:
            val = evaluate(model, val_loader, device, return_cwe=False)
        validation_seconds = time.perf_counter() - validation_started

        metric_name = getattr(args, "selection_metric", "macro_f1")
        score = val.get(metric_name)
        _keep = getattr(args, "_val_keep", None)
        if _keep is not None and len(_keep) != len(val["labels"]):
            logger.warning("Val mask bo qua: mask %d mau, val %d mau (max_eval_samples?)", len(_keep), len(val["labels"])); _keep = None
        if _keep is not None:
            # De xuat (1) cua agent 13/09: chon epoch bang val DA MASK cac mau co partner cung commit
            # trong train (khong dung split, khong dung test). Chi anh huong viec chon checkpoint.
            import numpy as _np
            _y = _np.asarray(val["labels"])[_keep]; _p = _np.asarray(val["probabilities"])[_keep]
            _sub = classification_metrics(_y.tolist(), _p.tolist(), 0.5)
            score = _sub.get(metric_name)
            logger.info("Val mask | giu %d/%d mau | %s tren phan giu: %s | toan val: %s",
                        int(_keep.sum()), len(_keep), metric_name,
                        "%.4f" % score if score is not None else "-", "%.4f" % val.get(metric_name) if val.get(metric_name) is not None else "-")
        if score is None:
            # ROC/PR AUC are undefined when a validation split is single-class.
            score = val["macro_f1"]
        new_best = False
        tied_best = False
        if score > best_score:
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
        print_epoch(
            epoch, args.epochs, train, val, optimizer, best_epoch, patience_counter,
            phase == "phase1",
            train_seconds, validation_seconds, epoch_seconds, total_seconds
        )
        if "pair_acc" in train:
            logger.info("RankNet nguon | rank loss %.4f | pair_acc %.3f", train.get("rank", 0.0), train["pair_acc"])
        if "replay_w_mean" in train:
            logger.info("Replay | w trung binh %.3f | ty le w=0 %.2f | cos trung binh %s | forward-backward %d",
                        train["replay_w_mean"], train["replay_zero_frac"],
                        ("%.3f" % train["replay_cos_mean"]) if train["replay_cos_mean"] == train["replay_cos_mean"] else "-",
                        train["fb_passes"])
        if new_best:
            logger.info("New best model | Macro-F1: %.6f | Epoch: %d", best_score, epoch)
        elif tied_best:
            logger.info(
                "Latest tied-best checkpoint saved | Macro-F1: %.6f | Epoch: %d",
                best_score,
                epoch,
            )
        if epoch >= args.min_epochs and patience_counter >= args.patience:
            logger.info("Early stopping | Epoch: %d | Best epoch: %d", epoch, best_epoch)
            break
    logger.info(
        "Best checkpoint saved | Path: %s | Epoch: %d | Val %s: %.6f",
        args.checkpoint_path,
        best_epoch,
        getattr(args, "selection_metric", "macro_f1"),
        best_score,
    )
    logger.info(
        "Training finished | Elapsed: %.2fs",
        time.perf_counter() - training_started,
    )
