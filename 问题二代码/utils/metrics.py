"""Dependency-free classification and regression metrics."""

from __future__ import annotations

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    return float((predictions == targets).float().mean().item())


def macro_f1(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 3) -> float:
    scores: list[torch.Tensor] = []
    for class_id in range(num_classes):
        predicted = predictions == class_id
        actual = targets == class_id
        tp = (predicted & actual).sum().double()
        fp = (predicted & ~actual).sum().double()
        fn = (~predicted & actual).sum().double()
        denominator = 2 * tp + fp + fn
        scores.append(torch.where(denominator > 0, 2 * tp / denominator, 0.0))
    return float(torch.stack(scores).mean().item())


def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    return float(torch.mean(torch.abs(predictions.double() - targets.double())).item())


def pearson(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Pearson correlation over complete collected vectors."""
    x = predictions.double().flatten()
    y = targets.double().flatten()
    if x.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    if denominator <= 0:
        return 0.0
    return float((torch.sum(x * y) / denominator).item())


def compute_metrics(
    logits: torch.Tensor,
    classification_targets: torch.Tensor,
    regression_predictions: torch.Tensor,
    regression_targets: torch.Tensor,
) -> dict[str, float]:
    classes = logits.argmax(dim=-1)
    return {
        "accuracy": accuracy(classes, classification_targets),
        "macro_f1": macro_f1(classes, classification_targets),
        "mae": mae(regression_predictions, regression_targets),
        "pearson": pearson(regression_predictions, regression_targets),
    }
