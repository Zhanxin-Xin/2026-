from __future__ import annotations

import torch

from src.train_pretrained_fusion import hierarchical_best_view_consistency


def _outputs(probability: torch.Tensor, regression: torch.Tensor):
    return {"class_probabilities": probability, "regression": regression}


def _batch(labels: list[int], regression: list[float]):
    return {
        "class_label": torch.tensor(labels),
        "regression_label": torch.tensor(regression),
    }


def test_identical_views_have_exact_zero_loss() -> None:
    probability = torch.tensor([[0.2, 0.6, 0.2], [0.6, 0.1, 0.3]])
    regression = torch.tensor([0.0, -1.0])
    total, components = hierarchical_best_view_consistency(
        _outputs(probability, regression),
        _outputs(probability.clone(), regression.clone()),
        _batch([1, 0], [0.0, -1.0]),
        {"hierarchical_best_view_regression_weight": 0.1},
    )
    assert total.item() == 0.0
    assert components["hierarchical_best_view_neutral"].item() == 0.0
    assert components["hierarchical_best_view_polarity"].item() == 0.0
    assert components["hierarchical_best_view_regression"].item() == 0.0


def test_neutral_samples_are_excluded_from_polarity_axis() -> None:
    first = torch.tensor([[0.10, 0.70, 0.20]])
    second = torch.tensor([[0.30, 0.40, 0.30]])
    _, components = hierarchical_best_view_consistency(
        _outputs(first, torch.tensor([0.1])),
        _outputs(second, torch.tensor([-0.1])),
        _batch([1], [0.0]),
    )
    assert components["hierarchical_best_view_neutral"].item() > 0.0
    assert components["hierarchical_best_view_polarity"].item() == 0.0


def test_more_label_aligned_view_is_detached_teacher() -> None:
    first_logits = torch.tensor([[0.0, 2.0, -0.5]], requires_grad=True)
    second_logits = torch.tensor([[0.2, 0.1, -0.1]], requires_grad=True)
    first = torch.softmax(first_logits, dim=-1)
    second = torch.softmax(second_logits, dim=-1)
    total, components = hierarchical_best_view_consistency(
        _outputs(first, torch.tensor([0.0])),
        _outputs(second, torch.tensor([0.0])),
        _batch([1], [0.0]),
    )
    assert components[
        "hierarchical_best_view_boundary_teacher_first_fraction"
    ].item() == 1.0
    total.backward()
    assert first_logits.grad is None or first_logits.grad.abs().sum().item() == 0.0
    assert second_logits.grad is not None
    assert torch.isfinite(second_logits.grad).all()
    assert second_logits.grad.abs().sum().item() > 0.0


def test_polarity_and_regression_choose_their_own_best_views() -> None:
    first_logits = torch.tensor([[1.8, -0.4, 0.1]], requires_grad=True)
    second_logits = torch.tensor([[0.2, -0.2, 1.2]], requires_grad=True)
    first_regression = torch.tensor([-0.8], requires_grad=True)
    second_regression = torch.tensor([0.4], requires_grad=True)
    total, components = hierarchical_best_view_consistency(
        _outputs(torch.softmax(first_logits, dim=-1), first_regression),
        _outputs(torch.softmax(second_logits, dim=-1), second_regression),
        _batch([0], [-1.0]),
        {"hierarchical_best_view_regression_weight": 0.1},
    )
    assert total.item() > 0.0
    assert components["hierarchical_best_view_polarity"].item() > 0.0
    assert components[
        "hierarchical_best_view_polarity_teacher_first_fraction"
    ].item() == 1.0
    assert components[
        "hierarchical_best_view_regression_teacher_first_fraction"
    ].item() == 1.0
    total.backward()
    assert second_logits.grad is not None
    assert second_regression.grad is not None
    assert torch.isfinite(second_logits.grad).all()
    assert torch.isfinite(second_regression.grad).all()
