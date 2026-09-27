from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)


CLASS_NAMES = ("Negative", "Neutral", "Positive")


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(
    class_target: np.ndarray,
    class_probability: np.ndarray,
    regression_target: np.ndarray,
    regression_prediction: np.ndarray,
) -> Dict[str, Any]:
    class_prediction = np.asarray(class_probability).argmax(axis=1)
    class_target = np.asarray(class_target, dtype=np.int64)
    regression_target = np.asarray(regression_target, dtype=np.float64)
    regression_prediction = np.asarray(regression_prediction, dtype=np.float64)
    macro_f1 = float(
        f1_score(class_target, class_prediction, average="macro", zero_division=0)
    )
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        class_target,
        class_prediction,
        labels=[0, 1, 2],
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_NAMES)
    }
    residual = regression_prediction - regression_target
    return {
        # The competition asks for Accuracy and F1.  Macro-F1 is the primary
        # three-class F1 because it gives every sentiment class equal weight.
        "accuracy": float(accuracy_score(class_target, class_prediction)),
        "f1": macro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": float(
            f1_score(class_target, class_prediction, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(class_target, class_prediction)),
        "mae": float(mean_absolute_error(regression_target, regression_prediction)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "regression_bias": float(np.mean(residual)),
        "pearson": safe_pearson(regression_target, regression_prediction),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            class_target, class_prediction, labels=[0, 1, 2]
        ).tolist(),
    }


def selection_score(metrics: Mapping[str, float], weights: Mapping[str, float]) -> float:
    # All four terms are mapped to approximately [0,1]. MAE is clipped because labels lie in [-3,3].
    normalized = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "mae": float(np.clip(1.0 - metrics["mae"] / 3.0, 0.0, 1.0)),
        "pearson": float(np.clip((metrics["pearson"] + 1.0) / 2.0, 0.0, 1.0)),
    }
    denominator = sum(float(weights.get(k, 0.0)) for k in normalized)
    if denominator <= 0:
        raise ValueError("At least one model-selection weight must be positive")
    return sum(float(weights.get(k, 0.0)) * v for k, v in normalized.items()) / denominator
