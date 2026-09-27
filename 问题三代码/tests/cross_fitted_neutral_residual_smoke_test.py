from __future__ import annotations

import torch

from src.cross_fitted_neutral_residual import NeutralOnlyResidual


def main() -> None:
    torch.manual_seed(3)
    features = torch.randn(8, 9)
    parent = torch.softmax(torch.randn(8, 3), dim=-1)
    model = NeutralOnlyResidual(9, hidden_dimension=12)
    identity = model(features, parent)
    assert torch.allclose(identity["probabilities"], parent, atol=1e-6)
    assert torch.allclose(
        identity["probabilities"].sum(dim=-1), torch.ones(8), atol=1e-6
    )
    old_polar_odds = parent[:, 0] / parent[:, 2]
    with torch.no_grad():
        model.shift.bias.fill_(4.0)
    shifted = model(features, parent)
    new_polar_odds = shifted["probabilities"][:, 0] / shifted["probabilities"][:, 2]
    assert torch.allclose(old_polar_odds, new_polar_odds, atol=1e-5)
    assert float(shifted["neutral_logit_shift"].abs().max()) <= 1.0 + 1e-6
    loss = -shifted["probabilities"][:, 1].clamp_min(1e-8).log().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    print("cross-fitted neutral residual smoke test passed")


if __name__ == "__main__":
    main()
