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


def train_one_epoch_phase1(model, dataloader, optimizer, device, lambda_cwe, max_grad_norm):
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
        loss = vul_loss if aux_loss is None else vul_loss + lambda_cwe * aux_loss
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


def assert_recadam_setup(current_params, pretrain_params, model):
    assert len(current_params) == len(pretrain_params), "RecAdam parameter counts differ"
    for index, (current, source) in enumerate(zip(current_params, pretrain_params)):
        assert current.shape == source.shape, f"RecAdam parameter shape differs at index {index}"
        assert not source.requires_grad, f"source parameter {index} unexpectedly requires gradients"
    assert all(
        not parameter.requires_grad for parameter in model.aux_parameters()
    ), "auxiliary head not frozen"


def train_one_epoch_phase2(
    model, dataloader, optimizer, device, max_grad_norm, pretrain_params, check_first_step=False
):
    model.train()
    total_loss = 0.0
    examples = 0
    labels_seen, probabilities = [], []
    source_versions = [parameter._version for parameter in pretrain_params]
    checked = False
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask, return_cwe=False)
        loss = F.cross_entropy(outputs["vul_logits"], labels)
        loss.backward()

        if check_first_step and not checked:
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
        labels_seen.extend(labels.detach().cpu().tolist())
        probabilities.extend(torch.softmax(outputs["vul_logits"].detach(), dim=-1)[:, 1].cpu().tolist())
    result = {"loss": total_loss / examples, "vul": total_loss / examples}
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
):
    best_score, best_epoch, patience_counter = -math.inf, 0, 0
    training_started = time.perf_counter()
    logger.info("Training started | Phase: %s | Epochs: %d", phase, args.epochs)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_started = time.perf_counter()
        if phase == "phase1":
            train = train_one_epoch_phase1(
                model, train_loader, optimizer, device, args.lambda_cwe, args.max_grad_norm
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
        "Best checkpoint saved | Path: %s | Epoch: %d | Val Macro-F1@0.5: %.6f",
        args.checkpoint_path,
        best_epoch,
        best_score,
    )
    logger.info(
        "Training finished | Elapsed: %.2fs",
        time.perf_counter() - training_started,
    )
