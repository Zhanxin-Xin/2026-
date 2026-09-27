from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import ConflictIgnoranceNeutralHead


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this architecture smoke test")
    device = torch.device("cuda")
    torch.manual_seed(20260924)
    head = ConflictIgnoranceNeutralHead(
        dimension=32,
        dropout=0.0,
        maximum_neutral_logit_shift=0.75,
    ).to(device)
    batch = 6
    parent_logits = torch.randn(batch, 3, device=device)
    parent = torch.softmax(parent_logits, dim=-1)
    states = [torch.randn(batch, 32, device=device) for _ in range(3)]
    reliability = [torch.rand(batch, device=device) for _ in range(2)]

    initial = head(parent, *states, *reliability)
    if not torch.allclose(initial["probabilities"], parent, atol=2e-6, rtol=2e-6):
        raise AssertionError("zero initialization must preserve parent probabilities")
    initial_odds = parent[:, 2] / parent[:, 0]
    corrected_odds = initial["probabilities"][:, 2] / initial["probabilities"][:, 0]
    if not torch.allclose(initial_odds, corrected_odds, atol=2e-6, rtol=2e-6):
        raise AssertionError("the head must preserve Positive/Negative odds")
    if not torch.allclose(
        initial["probabilities"].sum(dim=-1),
        torch.ones(batch, device=device),
        atol=1e-6,
    ):
        raise AssertionError("probabilities must be normalized")

    target = torch.tensor([0, 1, 2, 1, 0, 2], device=device)
    loss = torch.nn.functional.nll_loss(
        initial["probabilities"].clamp_min(1e-8).log(), target
    )
    loss.backward()
    final_layer = head.boundary[-1]
    if final_layer.weight.grad is None or final_layer.weight.grad.abs().sum() == 0:
        raise AssertionError("Neutral boundary must receive task gradients")
    if not torch.isfinite(initial["conflict"]).all():
        raise AssertionError("conflict diagnostics must remain finite")
    if not torch.isfinite(initial["ignorance"]).all():
        raise AssertionError("ignorance diagnostics must remain finite")
    print("conflict/ignorance Neutral head CUDA smoke test passed")


if __name__ == "__main__":
    main()
