"""Chỉ số phân loại và chọn ngưỡng theo val."""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def _safe_auc(labels, probabilities, metric):
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
        "roc_auc": _safe_auc(labels, probabilities, roc_auc_score),
        "pr_auc": _safe_auc(labels, probabilities, average_precision_score),
    }


def find_best_threshold(labels, probabilities):
    """Ngưỡng 0,05…0,95 (bước 0,01) cho macro-F1 cao nhất trên val."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.05, 0.951, 0.01):
        score = classification_metrics(labels, probabilities, float(threshold))["macro_f1"]
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return round(best_threshold, 2), best_f1
