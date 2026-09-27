from __future__ import annotations

import numpy as np
import torch

from src.train_explainable_hierarchical_nam import (
    HierarchicalExplainableNAM,
    TrainConfig,
    predict,
    train_model,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    concepts = torch.randn(9, 5)
    parent = torch.softmax(torch.randn(9, 3), dim=1)
    return concepts, parent


def test_initialization_is_exact_parent_and_probabilities_are_normalized() -> None:
    concepts, parent = _fixture()
    model = HierarchicalExplainableNAM(feature_count=concepts.shape[1])
    output = model(concepts, parent)

    torch.testing.assert_close(output["probability"], parent, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(
        output["probability"].sum(dim=1), torch.ones(len(parent)), atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(
        output["neutral_contributions"].sum(dim=1), output["neutral_residual"]
    )
    torch.testing.assert_close(
        output["polarity_contributions"].sum(dim=1), output["polarity_residual"]
    )


def test_deleting_one_concept_removes_exactly_its_axis_contribution() -> None:
    concepts, parent = _fixture()
    model = HierarchicalExplainableNAM(feature_count=concepts.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(3):
        output = model(concepts, parent)
        loss = -torch.log(output["probability"][:, 1].clamp(min=1e-7)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    original = model(concepts, parent)
    feature = 2
    deleted = concepts.clone()
    deleted[:, feature] = 0.0
    counterfactual = model(deleted, parent)
    torch.testing.assert_close(
        original["neutral_logit"] - counterfactual["neutral_logit"],
        original["neutral_contributions"][:, feature],
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        original["polarity_logit"] - counterfactual["polarity_logit"],
        original["polarity_contributions"][:, feature],
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.count_nonzero(counterfactual["neutral_contributions"][:, feature]) == 0
    assert torch.count_nonzero(counterfactual["polarity_contributions"][:, feature]) == 0


def test_gradients_reach_every_shape_output() -> None:
    concepts, parent = _fixture()
    model = HierarchicalExplainableNAM(feature_count=concepts.shape[1])
    output = model(concepts, parent)
    loss = -torch.log(output["probability"][:, 0].clamp(min=1e-7)).mean()
    loss.backward()
    for axis in (model.neutral_axis, model.polarity_axis):
        for shape in axis.shapes:
            assert shape.output.weight.grad is not None
            assert torch.isfinite(shape.output.weight.grad).all()
            assert float(shape.output.weight.grad.abs().sum()) > 0.0


def test_fixed_seed_training_is_deterministic() -> None:
    rng = np.random.default_rng(11)
    concepts = rng.normal(size=(18, 4)).astype(np.float32)
    parent = rng.dirichlet([2.0, 2.0, 2.0], size=18).astype(np.float32)
    targets = np.asarray([0, 1, 2] * 6, dtype=np.int64)
    config = TrainConfig(epochs=4, hidden_dim=4, feature_dropout=0.10)
    device = torch.device("cpu")
    first, _ = train_model(concepts, parent, targets, config, device, seed=123)
    second, _ = train_model(concepts, parent, targets, config, device, seed=123)
    first_result = predict(first, concepts, parent, device)["probability"]
    second_result = predict(second, concepts, parent, device)["probability"]
    np.testing.assert_array_equal(first_result, second_result)
