from __future__ import annotations

import numpy as np
import torch

from src.train_task_conditional_convex_router import (
    TaskConditionalConvexRouter,
    soft_macro_f1_loss,
)


def main() -> None:
    rng = np.random.default_rng(20260924)
    raw = rng.uniform(0.01, 1.0, size=(17, 4, 3))
    experts = raw / raw.sum(axis=2, keepdims=True)
    # Exercise the array math without constructing project CSVs.
    parent = experts.mean(axis=1)
    ordered = np.sort(experts, axis=2)
    confidence = experts.max(axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    certainty = 1.0 + (
        experts * np.log(experts)
    ).sum(axis=2) / np.log(3.0)
    agreement = 1.0 - 0.5 * np.abs(experts - parent[:, None, :]).sum(axis=2)
    diagnostics = np.stack([confidence, margin, certainty, agreement], axis=2)
    trust_features = np.zeros((len(experts), 4), dtype=np.float64)

    model = TaskConditionalConvexRouter(expert_count=4, maximum_acceptance=0.45)
    result = model(
        torch.tensor(experts, dtype=torch.float32),
        torch.tensor(diagnostics, dtype=torch.float32),
        torch.tensor(trust_features, dtype=torch.float32),
    )
    probability = result["probabilities"].detach().numpy()
    routes = result["route_weights"].detach().numpy()
    trust = result["trust"].detach().numpy()
    np.testing.assert_allclose(probability, parent, atol=1e-6)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(routes.sum(axis=2), 1.0, atol=1e-6)
    assert (routes >= 0.0).all()
    assert (trust >= 0.0).all() and (trust <= 0.45).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 29

    polar_router = TaskConditionalConvexRouter(
        expert_count=4,
        maximum_acceptance=0.45,
        preserve_neutral_probability=True,
    )
    with torch.no_grad():
        polar_router.route_bias[0] = torch.tensor([3.0, -1.0, 0.5, -2.0])
        polar_router.route_bias[2] = torch.tensor([-2.0, 0.5, -1.0, 3.0])
    polar_result = polar_router(
        torch.tensor(experts, dtype=torch.float32),
        torch.tensor(diagnostics, dtype=torch.float32),
        torch.tensor(trust_features, dtype=torch.float32),
    )
    polar_probability = polar_result["probabilities"].detach().numpy()
    polar_routes = polar_result["route_weights"].detach().numpy()
    np.testing.assert_allclose(polar_probability[:, 1], parent[:, 1], atol=1e-6)
    np.testing.assert_allclose(polar_routes[:, 1, :], 0.25, atol=1e-7)
    np.testing.assert_allclose(polar_probability.sum(axis=1), 1.0, atol=1e-6)

    eligibility = np.asarray(
        [[True, True, False, True], [True, True, True, False], [False, True, True, True]]
    )
    sparse_router = TaskConditionalConvexRouter(
        expert_count=4, route_eligibility=eligibility
    )
    sparse_result = sparse_router(
        torch.tensor(experts, dtype=torch.float32),
        torch.tensor(diagnostics, dtype=torch.float32),
        torch.tensor(trust_features, dtype=torch.float32),
    )
    sparse_routes = sparse_result["route_weights"].detach().numpy()
    sparse_probability = sparse_result["probabilities"].detach().numpy()
    np.testing.assert_allclose(sparse_routes[:, ~eligibility], 0.0, atol=1e-7)
    np.testing.assert_allclose(sparse_routes.sum(axis=2), 1.0, atol=1e-6)
    # Sparse eligibility changes the zero-parameter reference distribution.
    assert not np.allclose(sparse_probability, parent, atol=1e-7)

    targets = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    uniform_probability = torch.full((4, 3), 1.0 / 3.0, requires_grad=True)
    perfect_probability = torch.nn.functional.one_hot(targets, 3).float()
    uniform_loss, uniform_class_f1 = soft_macro_f1_loss(
        uniform_probability, targets, neutral_multiplier=1.5
    )
    perfect_loss, perfect_class_f1 = soft_macro_f1_loss(
        perfect_probability, targets, neutral_multiplier=1.5
    )
    assert perfect_loss < uniform_loss
    np.testing.assert_allclose(perfect_class_f1.numpy(), 1.0, atol=1e-6)
    assert torch.isfinite(uniform_class_f1).all()
    uniform_loss.backward()
    assert uniform_probability.grad is not None
    assert torch.isfinite(uniform_probability.grad).all()
    print("task_conditional_convex_router_smoke_test: PASS")


if __name__ == "__main__":
    main()
