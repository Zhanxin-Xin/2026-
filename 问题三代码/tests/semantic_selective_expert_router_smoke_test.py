from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.train_semantic_selective_expert_router import SemanticSelectiveExpertRouter


def test_semantic_router_contract() -> None:
    torch.manual_seed(3)
    batch, experts_count = 8, 5
    experts = torch.softmax(torch.randn(batch, experts_count, 3), dim=-1)
    model = SemanticSelectiveExpertRouter(
        text_dimension=24,
        diagnostic_dimension=12,
        global_dimension=9,
        expert_count=experts_count,
        hidden_dimension=16,
        dropout=0.0,
        maximum_acceptance=0.35,
    )
    output = model(
        torch.randn(batch, 24),
        experts,
        torch.randn(batch, experts_count, 12),
        torch.randn(batch, 9),
    )
    parent = experts.mean(dim=1)
    assert torch.allclose(output["probabilities"], parent, atol=1e-6)
    assert torch.allclose(output["route_weights"].sum(dim=2), torch.ones(batch, 3))
    assert torch.all(output["trust"] >= 0.0)
    assert torch.all(output["trust"] <= 0.35)
    loss = -output["probabilities"].clamp_min(1e-8).log().mean()
    loss.backward()
    assert model.route_head.weight.grad is not None
    assert torch.isfinite(model.route_head.weight.grad).all()


if __name__ == "__main__":
    test_semantic_router_contract()
    print("Semantic selective expert router smoke test passed.")
