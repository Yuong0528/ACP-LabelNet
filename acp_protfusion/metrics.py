"""Metrics and validation-derived decision-threshold utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CANCER_TYPES


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "fpr": false_positive / max(1, false_positive + true_negative),
        "auroc": _safe_score(roc_auc_score, labels, probabilities),
        "auprc": _safe_score(average_precision_score, labels, probabilities),
    }


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray, grid_size: int = 200) -> float:
    thresholds = np.linspace(0.05, 0.95, grid_size)
    return float(max(thresholds, key=lambda value: f1_score(
        labels, probabilities >= value, zero_division=0
    )))


def best_youden_thresholds(labels: np.ndarray, probabilities: np.ndarray, grid_size: int = 50) -> np.ndarray:
    thresholds = np.linspace(0.05, 0.95, grid_size)
    selected = np.full(labels.shape[1], 0.5, dtype=float)
    for label_index in range(labels.shape[1]):
        best_score = -np.inf
        for threshold in thresholds:
            prediction = probabilities[:, label_index] >= threshold
            truth = labels[:, label_index].astype(bool)
            sensitivity = (prediction & truth).sum() / max(1, truth.sum())
            specificity = ((~prediction) & (~truth)).sum() / max(1, (~truth).sum())
            score = sensitivity + specificity - 1.0
            if score > best_score:
                best_score, selected[label_index] = score, threshold
    return selected


def multilabel_metrics(labels: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict:
    result = {
        name: binary_metrics(labels[:, index].astype(int), probabilities[:, index], thresholds[index])
        for index, name in enumerate(CANCER_TYPES)
    }
    keys = next(iter(result.values())).keys()
    result["macro"] = {
        key: float(np.nanmean([result[name][key] for name in CANCER_TYPES])) for key in keys
    }
    return result


def selection_score(metrics: dict) -> float:
    """Harmonic mean of macro-F1 and the lowest label-specific F1."""
    macro_f1 = metrics["macro"]["f1"]
    minimum_f1 = min(metrics[name]["f1"] for name in CANCER_TYPES)
    if macro_f1 <= 0 or minimum_f1 <= 0:
        return 0.0
    return float(2.0 * macro_f1 * minimum_f1 / (macro_f1 + minimum_f1))


def _safe_score(function, labels, probabilities) -> float:
    try:
        return float(function(labels, probabilities))
    except ValueError:
        return float("nan")

