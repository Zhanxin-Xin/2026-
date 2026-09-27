from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import AlignedTemporalIncongruityHead


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this architecture smoke test")
    device = torch.device("cuda")
    torch.manual_seed(20260924)
    batch, length = 5, 7
    head = AlignedTemporalIncongruityHead(
        text_dimension=12,
        audio_dimension=6,
        vision_dimension=4,
        hidden_dimension=16,
        dropout=0.0,
    ).to(device)
    parent = torch.softmax(torch.randn(batch, 3, device=device), dim=-1)
    text = torch.randn(batch, length, 12, device=device)
    audio = torch.randn(batch, length, 6, device=device)
    vision = torch.randn(batch, length, 4, device=device)
    mask = torch.ones(batch, length, dtype=torch.bool, device=device)
    mask[0, -2:] = False
    output = head(
        parent,
        text,
        audio,
        vision,
        mask,
        mask,
        mask,
        torch.rand(batch, device=device),
        torch.rand(batch, device=device),
    )
    if not torch.allclose(output["probabilities"], parent, atol=2e-6, rtol=2e-6):
        raise AssertionError("zero initialization must preserve the parent")
    if not torch.allclose(
        output["probabilities"].sum(dim=-1),
        torch.ones(batch, device=device),
        atol=1e-6,
    ):
        raise AssertionError("probabilities must be normalized")
    target = torch.tensor([0, 1, 2, 1, 2], device=device)
    torch.nn.functional.nll_loss(
        output["probabilities"].clamp_min(1e-8).log(), target
    ).backward()
    final = head.axes[-1]
    if final.weight.grad is None or final.weight.grad.abs().sum() == 0:
        raise AssertionError("both aligned decision axes must receive gradients")
    if not torch.isfinite(output["attention_entropy"]).all():
        raise AssertionError("temporal diagnostics must remain finite")
    print("aligned temporal incongruity CUDA smoke test passed")


if __name__ == "__main__":
    main()
