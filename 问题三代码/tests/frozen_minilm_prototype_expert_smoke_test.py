from __future__ import annotations

import numpy as np
import torch

from src.train_frozen_minilm_prototype_expert import (
    FrozenSentencePrototypeHead,
    class_centroids,
)


def main() -> None:
    rng = np.random.default_rng(20260925)
    embeddings = rng.normal(size=(30, 32)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    targets = np.repeat(np.arange(3), 10)
    centroids = class_centroids(embeddings, targets)
    np.testing.assert_allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-6)
    model = FrozenSentencePrototypeHead(32, centroids, hidden_dimension=16, dropout=0.0)
    output = model(torch.tensor(embeddings))
    np.testing.assert_allclose(
        output["probabilities"].detach().numpy().sum(axis=1), 1.0, atol=1e-6
    )
    assert torch.all(output["regression"].abs() <= 3.0 + 1e-6)
    loss = -output["probabilities"].clamp_min(1e-8).log().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    print("frozen_minilm_prototype_expert_smoke_test: PASS")


if __name__ == "__main__":
    main()
