from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.losses import MultitaskEvidenceLoss
from src.model import HAFusionNet


def tiny_learnable_batch(device: torch.device):
    batch_size, length = 24, 10
    latent = torch.linspace(-2.5, 2.5, batch_size, device=device)
    labels = torch.where(
        latent < -0.25,
        torch.zeros_like(latent, dtype=torch.long),
        torch.where(
            latent > 0.25,
            torch.full_like(latent, 2, dtype=torch.long),
            torch.ones_like(latent, dtype=torch.long),
        ),
    )

    def feature(dim: int, scale: float) -> torch.Tensor:
        values = torch.randn(batch_size, length, dim, device=device) * 0.15
        values[:, :, 0] += latent[:, None] * scale
        return values

    mask = torch.ones(batch_size, length, dtype=torch.bool, device=device)
    return {
        "text": feature(12, 1.0),
        "audio": feature(6, 0.8),
        "vision": feature(5, 0.6),
        "text_mask": mask,
        "audio_mask": mask,
        "vision_mask": mask,
        "class_label": labels,
        "regression_label": latent,
    }


def test_tiny_batch_is_finite_and_learnable() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = tiny_learnable_batch(device)
    model = HAFusionNet(
        {
            "input_dims": {"text": 12, "audio": 6, "vision": 5},
            "d_model": 24,
            "n_heads": 4,
            "n_layers": 1,
            "ff_multiplier": 2,
            "dropout": 0.05,
            "conv_kernel_sizes": [3, 5],
            "sparse_attention": True,
            "evidence_temperature": 0.8,
            "use_cross_context": True,
            "use_conflict_context": True,
            "use_pairwise_evidence": True,
            "fixed_modality_gate": False,
        }
    ).to(device)
    criterion = MultitaskEvidenceLoss(
        {
            "classification": 1.0,
            "regression": 1.0,
            "pearson": 0.2,
            "consistency": 0.12,
            "attention_entropy": 0.002,
            "attention_total_variation": 0.002,
            "gate_balance": 0.001,
            "faithfulness": 0.0,
            "distillation": 0.0,
            "interaction_l1": 0.001,
            "unimodal_auxiliary": 0.15,
        }
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(batch)
            loss, components = criterion(
                outputs, batch["class_label"], batch["regression_label"]
            )
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        assert all(torch.isfinite(value) for value in components.values())
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ), f"non-finite gradient at step {step}"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < 0.8 * losses[0], (losses[0], losses[-1])
    print(
        {
            "device": str(device),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
        }
    )


if __name__ == "__main__":
    test_tiny_batch_is_finite_and_learnable()
    print("Training stability test passed.")
