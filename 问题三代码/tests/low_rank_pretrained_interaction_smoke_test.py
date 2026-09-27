from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import LowRankTriModalInteraction


def test_low_rank_interaction_is_bounded_and_trainable() -> None:
    torch.manual_seed(29)
    batch, dimension = 7, 16
    model = LowRankTriModalInteraction(
        dimension=dimension,
        rank=3,
        dropout=0.0,
        maximum_mix=0.35,
        initial_mix_logit=-2.0,
    )
    shared_private = torch.randn(batch, dimension)
    text = torch.randn(batch, dimension)
    audio = torch.randn(batch, dimension)
    vision = torch.randn(batch, dimension)
    reliability = torch.rand(batch)
    fused, mix, candidate = model(
        shared_private,
        text,
        audio,
        vision,
        reliability,
        1.0 - reliability,
    )
    assert fused.shape == shared_private.shape
    assert candidate.shape == shared_private.shape
    assert torch.all(mix >= 0.0)
    assert torch.all(mix <= 0.35)
    expected = shared_private + mix.unsqueeze(-1) * (candidate - shared_private)
    assert torch.allclose(fused, expected, atol=1e-6)
    loss = fused.square().mean()
    loss.backward()
    assert model.text_factor.grad is not None
    assert model.gate[-1].weight.grad is not None
    assert torch.isfinite(model.text_factor.grad).all()


if __name__ == "__main__":
    test_low_rank_interaction_is_bounded_and_trainable()
    print("Low-rank pretrained interaction smoke test passed.")
