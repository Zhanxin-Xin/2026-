from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import LatentAffectSubcenterHead


def main() -> None:
    generator = torch.Generator().manual_seed(20260925)
    embedding = torch.randn(6, 32, generator=generator, requires_grad=True)
    head = LatentAffectSubcenterHead(
        dimension=32, subcenters=3, maximum_mix=0.35,
        initial_mix_logit=-2.0, initial_scale=10.0,
    )
    output = head(embedding)
    assert output["logits"].shape == (6, 3)
    assert 0.0 < float(output["mix"]) < 0.35
    assert torch.isfinite(output["diversity"])
    loss = F.cross_entropy(output["logits"], torch.tensor([0, 1, 2, 0, 1, 2]))
    loss = loss + 0.02 * output["diversity"]
    loss.backward()
    assert head.centers.grad is not None
    assert torch.isfinite(head.centers.grad).all()
    print(
        {
            "loss": float(loss.detach()),
            "mix": float(output["mix"].detach()),
            "scale": float(output["scale"].detach()),
        }
    )
    print("Latent affect subcenter smoke test passed.")


if __name__ == "__main__":
    main()
