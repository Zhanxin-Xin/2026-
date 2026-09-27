from __future__ import annotations

import numpy as np
import torch

from src.train_hierarchical_expert_hurdle import HierarchicalExpertHurdle


def main() -> None:
    rng = np.random.default_rng(20260925)
    raw = rng.uniform(0.01, 1.0, size=(19, 7, 3))
    experts = raw / raw.sum(axis=2, keepdims=True)
    tokens = rng.normal(size=(19, 7, 11))
    global_features = rng.normal(size=(19, 10))
    model = HierarchicalExpertHurdle(
        expert_count=7,
        token_dimension=11,
        global_dimension=10,
        hidden_dimension=24,
        maximum_log_odds_residual=0.75,
    )
    result = model(
        torch.tensor(experts, dtype=torch.float32),
        torch.tensor(tokens, dtype=torch.float32),
        torch.tensor(global_features, dtype=torch.float32),
    )
    probability = result["probabilities"].detach().numpy()
    parent = experts.mean(axis=1)
    np.testing.assert_allclose(probability, parent, atol=1e-6)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        result["attention"].detach().numpy().sum(axis=2), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(result["residual"].detach().numpy(), 0.0, atol=1e-7)

    with torch.no_grad():
        model.neutral_head[-1].bias.fill_(2.0)
        model.polarity_head[-1].bias.fill_(-2.0)
    shifted = model(
        torch.tensor(experts, dtype=torch.float32),
        torch.tensor(tokens, dtype=torch.float32),
        torch.tensor(global_features, dtype=torch.float32),
    )
    assert torch.all(shifted["probabilities"][:, 1] > result["probabilities"][:, 1])
    assert torch.all(shifted["residual"].abs() <= 0.75 + 1e-6)
    loss = -shifted["probabilities"].clamp_min(1e-8).log().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    enveloped = HierarchicalExpertHurdle(
        expert_count=7,
        token_dimension=11,
        global_dimension=10,
        hidden_dimension=24,
        maximum_log_odds_residual=0.75,
        uncertainty_envelope=True,
    )
    enveloped.load_state_dict(model.state_dict())
    enveloped_result = enveloped(
        torch.tensor(experts, dtype=torch.float32),
        torch.tensor(tokens, dtype=torch.float32),
        torch.tensor(global_features, dtype=torch.float32),
    )
    assert torch.all(
        enveloped_result["residual"].abs()
        <= enveloped_result["raw_residual"].abs() + 1e-7
    )
    assert torch.all(enveloped_result["uncertainty_envelope"] >= 0.0)
    assert torch.all(enveloped_result["uncertainty_envelope"] <= 1.0 + 1e-7)
    print("hierarchical_expert_hurdle_smoke_test: PASS")


if __name__ == "__main__":
    main()
