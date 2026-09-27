from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PolarDistanceNeutralExpert


def main() -> None:
    torch.manual_seed(20260924)
    maximum_shift = 0.75
    model = PolarDistanceNeutralExpert(
        dimension=32,
        dropout=0.0,
        maximum_boundary_mix=0.5,
        maximum_evidence_shift=1.5,
        maximum_polarity_shift=maximum_shift,
    )
    anchor = torch.randn(6, 32)
    boundary = torch.randn(6, 32)
    base_probability = torch.softmax(torch.randn(6, 3), dim=-1)
    reliability = torch.rand(6)
    output = model(
        anchor,
        boundary,
        base_probability,
        reliability,
        reliability,
        reliability,
    )
    assert torch.allclose(
        output["polarity_logit"],
        output["polarity_anchor_logit"] + output["polarity_residual"],
        atol=1e-6,
    )
    assert output["polarity_residual"].abs().max().item() <= maximum_shift + 1e-6
    assert ((0.0 <= output["polarity_trust"]) & (output["polarity_trust"] <= 1.0)).all()
    assert torch.allclose(
        output["probabilities"].sum(dim=-1), torch.ones(6), atol=1e-6
    )
    loss = -output["probabilities"].clamp_min(1e-8).log().mean()
    loss.backward()
    assert model.polarity_correction[-1].weight.grad is not None
    assert model.polarity_trust[-1].weight.grad is not None
    assert torch.isfinite(model.polarity_correction[-1].weight.grad).all()
    assert torch.isfinite(model.polarity_trust[-1].weight.grad).all()
    print(
        {
            "maximum_shift": maximum_shift,
            "observed_shift": float(output["polarity_residual"].abs().max()),
            "mean_trust": float(output["polarity_trust"].mean()),
        }
    )
    print("Soft-anchored valence-boundary smoke test passed.")


if __name__ == "__main__":
    main()
