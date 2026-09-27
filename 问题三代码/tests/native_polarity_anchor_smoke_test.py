from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PolarDistanceNeutralExpert


def main() -> None:
    torch.manual_seed(20260924)
    maximum_shift = 1.0
    model = PolarDistanceNeutralExpert(
        dimension=32,
        dropout=0.0,
        polarity_prior_maximum_shift=maximum_shift,
    )
    anchor = torch.randn(5, 32)
    boundary = torch.randn(5, 32)
    prior = torch.linspace(-2.0, 2.0, 5)
    probability = torch.softmax(torch.randn(5, 3), dim=-1)
    reliability = torch.rand(5)
    output = model(
        anchor,
        boundary,
        probability,
        reliability,
        reliability,
        reliability,
        polarity_prior_logit=prior,
    )
    assert torch.allclose(output["polarity_prior_logit"], prior)
    assert torch.allclose(
        output["polarity_anchor_logit"],
        output["polarity_prior_logit"] + output["polarity_prior_residual"],
        atol=1e-6,
    )
    assert output["polarity_prior_residual"].abs().max().item() <= maximum_shift + 1e-6
    positive_given_polar = output["probabilities"][:, 2] / (
        output["probabilities"][:, 0] + output["probabilities"][:, 2]
    ).clamp_min(1e-8)
    assert torch.allclose(
        positive_given_polar,
        torch.sigmoid(output["polarity_anchor_logit"]),
        atol=1e-6,
    )
    (-output["probabilities"].clamp_min(1e-8).log().mean()).backward()
    assert model.polarity[-1].weight.grad is not None
    assert torch.isfinite(model.polarity[-1].weight.grad).all()
    print(
        {
            "maximum_shift": maximum_shift,
            "observed_shift": float(output["polarity_prior_residual"].abs().max()),
        }
    )
    print("Native polarity-anchor smoke test passed.")


if __name__ == "__main__":
    main()
