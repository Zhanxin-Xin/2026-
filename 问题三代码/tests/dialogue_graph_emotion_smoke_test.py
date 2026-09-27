from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.train_dialogue_graph_emotion_oof import (
    DialogueGraphEmotionNet,
    build_directional_neighbors,
)


def main() -> None:
    ids = np.asarray(
        ["videoA$_$0", "videoA$_$1", "videoA$_$2", "videoB$_$0"]
    )
    neighbor, direction, distance = build_directional_neighbors(ids, window=2)
    assert neighbor.shape == (4, 4)
    assert set(neighbor[3].tolist()) == {-1}
    assert 3 not in neighbor[:3]
    model = DialogueGraphEmotionNet(
        semantic_dimension=16,
        audio_dimension=5,
        vision_dimension=6,
        hidden_dimension=32,
        window=2,
        depth=2,
        dropout=0.1,
    )
    generator = torch.Generator().manual_seed(20260924)
    output = model(
        emotion_logits=torch.randn(4, 7, generator=generator),
        semantic=torch.randn(4, 16, generator=generator),
        audio=torch.randn(4, 5, generator=generator),
        vision=torch.randn(4, 6, generator=generator),
        neighbor_index=torch.from_numpy(neighbor),
        direction=torch.from_numpy(direction),
        distance=torch.from_numpy(distance),
    )
    assert output["probabilities"].shape == (4, 3)
    assert torch.isfinite(output["probabilities"]).all()
    assert torch.allclose(
        output["probabilities"].sum(dim=1), torch.ones(4), atol=1e-6
    )
    assert output["graph_gate"][3].item() == 0.0
    loss = F.nll_loss(
        output["probabilities"].log(), torch.tensor([0, 1, 2, 1])
    ) + F.smooth_l1_loss(output["regression"], torch.zeros(4))
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.graph.parameters()
    )
    print(
        {
            "loss": float(loss.detach()),
            "graph_gate": output["graph_gate"].detach().tolist(),
        }
    )
    print("Dialogue graph emotion smoke test passed.")


if __name__ == "__main__":
    main()
