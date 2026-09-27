from __future__ import annotations

import torch

from src.train_pretrained_fusion import hierarchical_rdrop_consistency


def _outputs(probability: torch.Tensor, regression: torch.Tensor | None = None):
    if regression is None:
        regression = probability[:, 2] - probability[:, 0]
    return {"class_probabilities": probability, "regression": regression}


def test_identical_views_have_exactly_zero_consistency() -> None:
    probability = torch.tensor(
        [[0.20, 0.60, 0.20], [0.55, 0.10, 0.35]], dtype=torch.float32
    )
    total, components = hierarchical_rdrop_consistency(
        _outputs(probability), _outputs(probability.clone())
    )
    assert total.item() == 0.0
    assert all(value.item() == 0.0 for value in components.values())


def test_neutral_and_conditional_polarity_axes_are_separated() -> None:
    neutral_shift_a = torch.tensor([[0.20, 0.60, 0.20]])
    neutral_shift_b = torch.tensor([[0.10, 0.80, 0.10]])
    _, neutral_components = hierarchical_rdrop_consistency(
        _outputs(neutral_shift_a), _outputs(neutral_shift_b)
    )
    assert neutral_components["hierarchical_rdrop_neutral"].item() > 0.0
    assert neutral_components["hierarchical_rdrop_polarity"].item() == 0.0

    polarity_shift_a = torch.tensor([[0.40, 0.20, 0.40]])
    polarity_shift_b = torch.tensor([[0.60, 0.20, 0.20]])
    _, polarity_components = hierarchical_rdrop_consistency(
        _outputs(polarity_shift_a), _outputs(polarity_shift_b)
    )
    assert polarity_components["hierarchical_rdrop_neutral"].abs().item() < 1e-7
    assert polarity_components["hierarchical_rdrop_polarity"].item() > 0.0


def test_changed_views_produce_positive_finite_gradients() -> None:
    first_logits = torch.tensor(
        [[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]], requires_grad=True
    )
    second_logits = torch.tensor(
        [[-0.1, 0.3, 0.2], [0.1, -0.2, 0.6]], requires_grad=True
    )
    first_regression = torch.tensor([0.2, -0.4], requires_grad=True)
    second_regression = torch.tensor([-0.1, 0.1], requires_grad=True)
    total, _ = hierarchical_rdrop_consistency(
        _outputs(torch.softmax(first_logits, dim=-1), first_regression),
        _outputs(torch.softmax(second_logits, dim=-1), second_regression),
        {"hierarchical_rdrop_regression_weight": 0.1},
    )
    assert torch.isfinite(total)
    assert total.item() > 0.0
    total.backward()
    for value in (
        first_logits,
        second_logits,
        first_regression,
        second_regression,
    ):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().sum().item() > 0.0
