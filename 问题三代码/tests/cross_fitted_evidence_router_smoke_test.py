from __future__ import annotations

import torch

from src.cross_fitted_evidence_router import CrossFittedEvidenceRouter


def main() -> None:
    torch.manual_seed(7)
    features = torch.randn(6, 11)
    experts = torch.softmax(torch.randn(6, 3, 3), dim=-1)
    experts[:, 0] = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.70, 0.15],
            [0.10, 0.20, 0.70],
            [0.40, 0.35, 0.25],
            [0.25, 0.50, 0.25],
            [0.20, 0.30, 0.50],
        ]
    )
    identity = CrossFittedEvidenceRouter(
        11, hidden_dimension=8, parent_bias=30.0, maximum_logit_residual=0.75
    )
    identity.eval()
    output = identity(features, experts)
    assert torch.allclose(output["probabilities"], experts[:, 0], atol=1e-6)
    assert torch.allclose(
        output["probabilities"].sum(dim=-1), torch.ones(6), atol=1e-6
    )
    assert (output["route_weights"] >= 0).all()
    assert torch.allclose(
        output["route_weights"].sum(dim=-1), torch.ones(6), atol=1e-6
    )

    routed = CrossFittedEvidenceRouter(
        11, hidden_dimension=8, parent_bias=0.0, maximum_logit_residual=0.75
    )
    with torch.no_grad():
        routed.residual.bias.copy_(torch.tensor([10.0, -10.0, 10.0]))
    result = routed(features, experts)
    assert torch.isfinite(result["probabilities"]).all()
    assert torch.allclose(
        result["probabilities"].sum(dim=-1), torch.ones(6), atol=1e-6
    )
    # Centering can at most double the raw tanh bound before accept gating.
    assert float(result["bounded_correction"].abs().max()) <= 1.5 + 1e-6
    loss = -result["probabilities"][:, 1].clamp_min(1e-8).log().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in routed.parameters()
    )
    print("cross-fitted evidence router smoke test passed")


if __name__ == "__main__":
    main()
