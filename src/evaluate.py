"""Evaluation, reporting, threshold calibration, and per-CWE metrics."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from logging_utils import get_logger


CLASS_NAMES = ("safe (0)", "vulnerable (1)")
logger = get_logger()


def safe_auc(labels, probabilities, metric):
    if len(set(labels)) < 2:
        return None
    return float(metric(labels, probabilities))


def classification_metrics(labels, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    labels = np.asarray(labels)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": safe_auc(labels, probabilities, roc_auc_score),
        "pr_auc": safe_auc(labels, probabilities, average_precision_score),
    }


def print_classification_report(split, labels, probabilities, threshold=0.5, epoch=None):
    """Log an sklearn report plus probability diagnostics."""
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    predictions = (probabilities_array >= threshold).astype(int)
    location = f" epoch={epoch}" if epoch is not None else ""
    report = classification_report(
            labels_array,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ).rstrip()
    logger.info(
        "[Classification Report - split=%s%s - threshold=%.2f]",
        split,
        location,
        threshold,
    )
    for line in report.splitlines():
        logger.info("%s", line)
    matrix = confusion_matrix(labels_array, predictions, labels=[0, 1])
    logger.info(
        "Confusion matrix - [[tn=%d, fp=%d], [fn=%d, tp=%d]]",
        matrix[0, 0],
        matrix[0, 1],
        matrix[1, 0],
        matrix[1, 1],
    )
    logger.info(
        "probability_positive "
        f"min={probabilities_array.min():.4f} mean={probabilities_array.mean():.4f} "
        f"std={probabilities_array.std():.4f} max={probabilities_array.max():.4f} "
        f"predicted_0={(predictions == 0).sum()} predicted_1={(predictions == 1).sum()}"
    )


def log_per_cwe_reports(labels, probabilities, cwe_classes, threshold, class_to_cwe, split):
    """Log one binary classification report for every available CWE subset."""
    for cwe_class, cwe_name in class_to_cwe.items():
        indices = [index for index, value in enumerate(cwe_classes) if value == cwe_class]
        if not indices:
            logger.info("Per-CWE test - split=%s cwe=%s samples=0 (skipped)", split, cwe_name)
            continue
        subset_labels = [labels[index] for index in indices]
        subset_probabilities = [probabilities[index] for index in indices]
        print_classification_report(
            f"{split}_{cwe_name}", subset_labels, subset_probabilities, threshold
        )


@torch.no_grad()
def evaluate(model, dataloader, device, return_cwe=False, lambda_cwe=0.2):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels, probabilities, cwe_classes = [], [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)
        batch_cwes = batch["cwe_class"].to(device)
        outputs = model(input_ids, attention_mask, return_cwe=return_cwe)
        loss = F.cross_entropy(outputs["vul_logits"], batch_labels)
        valid_cwe = batch_cwes != -100
        if return_cwe and valid_cwe.any():
            loss = loss + lambda_cwe * F.cross_entropy(
                outputs["cwe_logits"][valid_cwe], batch_cwes[valid_cwe]
            )
        batch_size = batch_labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        labels.extend(batch_labels.cpu().tolist())
        probabilities.extend(torch.softmax(outputs["vul_logits"], dim=-1)[:, 1].cpu().tolist())
        cwe_classes.extend(batch_cwes.cpu().tolist())
    result = classification_metrics(labels, probabilities, 0.5)
    result.update(
        loss=total_loss / total_examples,
        labels=labels,
        probabilities=probabilities,
        cwe_classes=cwe_classes,
    )
    return result


def find_best_threshold(labels, probabilities):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.05, 0.951, 0.01):
        score = classification_metrics(labels, probabilities, float(threshold))["macro_f1"]
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return round(best_threshold, 2), best_f1


def per_cwe_metrics(labels, probabilities, cwe_classes, threshold, class_to_cwe):
    output = {}
    for cwe_class, cwe_name in class_to_cwe.items():
        indices = [index for index, value in enumerate(cwe_classes) if value == cwe_class]
        subset_labels = [labels[index] for index in indices]
        subset_probabilities = [probabilities[index] for index in indices]
        if not indices:
            output[cwe_name] = {
                "number_of_samples": 0,
                "positive_ratio": None,
                "macro_f1": None,
                "positive_f1": None,
                "precision": None,
                "recall": None,
                "accuracy": None,
            }
            continue
        metrics = classification_metrics(subset_labels, subset_probabilities, threshold)
        output[cwe_name] = {
            "number_of_samples": len(indices),
            "positive_ratio": float(np.mean(subset_labels)),
            "macro_f1": metrics["macro_f1"],
            "positive_f1": metrics["positive_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
        }
    return output
