from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.train_dialogue_graph_emotion_oof import build_directional_neighbors
from src.train_parent_anchored_dialogue_graph_oof import ParentAnchoredDialogueGraphNet


def main() -> None:
    torch.manual_seed(20260924)
    ids = np.asarray(["a$_$0", "a$_$1", "a$_$2", "b$_$0"])
    neighbors = build_directional_neighbors(ids, 2)
    parent = torch.softmax(torch.randn(4, 3), dim=-1)
    model = ParentAnchoredDialogueGraphNet(
        semantic_dimension=16,
        audio_dimension=5,
        vision_dimension=6,
        hidden_dimension=32,
        window=2,
        depth=2,
        dropout=0.0,
    )
    arguments = {
        "emotion_logits": torch.randn(4, 7),
        "semantic": torch.randn(4, 16),
        "audio": torch.randn(4, 5),
        "vision": torch.randn(4, 6),
        "parent_probability": parent,
        "parent_regression": torch.randn(4),
        "neighbor_index": torch.from_numpy(neighbors[0]),
        "direction": torch.from_numpy(neighbors[1]),
        "distance": torch.from_numpy(neighbors[2]),
    }
    initial = model(**arguments)
    assert torch.allclose(initial["probabilities"], parent, atol=1e-6)
    assert torch.allclose(initial["neutral_residual"], torch.zeros(4), atol=1e-7)
    assert initial["graph_gate"][3].item() == 0.0
    polar_parent = parent[:, 2] / (parent[:, 0] + parent[:, 2])
    polar_output = initial["probabilities"][:, 2] / (
        initial["probabilities"][:, 0] + initial["probabilities"][:, 2]
    )
    assert torch.allclose(polar_parent, polar_output, atol=1e-6)
    loss = F.nll_loss(initial["probabilities"].log(), torch.tensor([0, 1, 2, 1]))
    loss.backward()
    assert model.neutral_residual[-1].weight.grad is not None
    assert torch.isfinite(model.neutral_residual[-1].weight.grad).all()
    print({"loss": float(loss.detach()), "exact_parent_start": True})
    print("Parent-anchored dialogue graph smoke test passed.")


if __name__ == "__main__":
    main()
