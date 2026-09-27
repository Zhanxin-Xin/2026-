from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.model import GatedTemporalEncoderV2


def test_encoder_depends_on_modality_features() -> None:
    torch.manual_seed(11)
    encoder = GatedTemporalEncoderV2(
        input_dim=7,
        d_model=16,
        n_layers=1,
        ff_multiplier=2,
        dropout=0.0,
        max_sequence_length=32,
    ).eval()
    mask = torch.ones(3, 6, dtype=torch.bool)
    first = torch.randn(3, 6, 7)
    second = first + 2.0 * torch.randn(3, 6, 7)
    first_tokens, first_pooled, _ = encoder(first, mask)
    second_tokens, second_pooled, _ = encoder(second, mask)
    assert not torch.allclose(first_tokens, second_tokens)
    assert not torch.allclose(first_pooled, second_pooled)


if __name__ == "__main__":
    test_encoder_depends_on_modality_features()
    print("Gated V2 input-dependence smoke test passed.")
