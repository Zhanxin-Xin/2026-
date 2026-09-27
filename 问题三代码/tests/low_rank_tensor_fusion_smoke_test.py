from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import MODALITIES
from src.losses import MultitaskEvidenceLoss
from src.model import LowRankTensorFusionNet


def test_low_rank_tensor_fusion_forward_backward() -> None:
    torch.manual_seed(7)
    batch_size, length = 5, 9
    dims = {"text": 12, "audio": 6, "vision": 5}
    mask = torch.arange(length).unsqueeze(0) < torch.tensor([9, 8, 7, 6, 5]).unsqueeze(1)
    batch = {
        modality: torch.randn(batch_size, length, dims[modality])
        * mask.unsqueeze(-1)
        for modality in MODALITIES
    }
    batch.update({f"{modality}_mask": mask.clone() for modality in MODALITIES})
    cfg = {
        "input_dims": dims,
        "d_model": 24,
        "fusion_dim": 32,
        "rank": 4,
        "n_layers": 1,
        "ff_multiplier": 2,
        "dropout": 0.1,
        "conv_kernel_size": 3,
        "modality_dropout": 0.0,
    }
    model = LowRankTensorFusionNet(cfg)
    outputs = model(batch)
    assert outputs["class_logits"].shape == (batch_size, 3)
    assert outputs["neutral_logit"].shape == (batch_size,)
    assert outputs["regression"].shape == (batch_size,)
    assert outputs["temporal_weights"].shape == (batch_size, 3, length)
    assert torch.allclose(
        outputs["class_probabilities"].sum(dim=1), torch.ones(batch_size), atol=1e-6
    )
    assert torch.allclose(
        outputs["temporal_weights"].sum(dim=2),
        torch.ones(batch_size, 3),
        atol=1e-6,
    )
    assert torch.allclose(
        outputs["class_logits"] - outputs["class_bias"],
        outputs["classification_contributions"].sum(dim=1),
        atol=1e-5,
    )
    assert torch.allclose(
        outputs["regression_raw"] - outputs["regression_bias"],
        outputs["regression_contributions"].sum(dim=1),
        atol=1e-5,
    )
    criterion = MultitaskEvidenceLoss(
        {
            "classification": 1.0,
            "regression": 0.5,
            "pearson": 0.1,
            "consistency": 0.1,
            "attention_entropy": 0.003,
            "attention_total_variation": 0.003,
            "gate_balance": 0.002,
            "unimodal_auxiliary": 0.1,
            "neutral_auxiliary": 0.2,
        }
    )
    labels = torch.tensor([0, 1, 2, 1, 2])
    intensity = torch.tensor([-1.0, 0.0, 1.5, 0.0, 0.7])
    loss, components = criterion(outputs, labels, intensity)
    assert torch.isfinite(loss)
    assert components["neutral_auxiliary"] > 0
    loss.backward()
    assert model.factors["text"].grad is not None
    assert torch.isfinite(model.factors["text"].grad).all()


if __name__ == "__main__":
    test_low_rank_tensor_fusion_forward_backward()
    print("Low-rank tensor fusion smoke test passed.")
