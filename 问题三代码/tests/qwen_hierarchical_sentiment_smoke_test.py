from __future__ import annotations

import torch

from src.train_qwen_hierarchical_sentiment import hierarchical_loss


def main() -> None:
    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [-1.0, 2.0, -0.5], [-1.0, 0.0, 2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([0, 1, 2], dtype=torch.long)
    loss, components = hierarchical_loss(
        logits,
        target,
        torch.ones(3),
        torch.tensor(1.5),
        {
            "classification": 1.0,
            "neutral_hurdle": 0.35,
            "polarity": 0.20,
            "label_smoothing": 0.03,
        },
    )
    assert set(components) == {"classification", "neutral_hurdle", "polarity"}
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    print("qwen hierarchical sentiment smoke test passed")


if __name__ == "__main__":
    main()
