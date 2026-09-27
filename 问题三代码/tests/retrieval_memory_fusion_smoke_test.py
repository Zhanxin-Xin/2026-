from __future__ import annotations

import numpy as np

from src.retrieval_memory_fusion import (
    class_balanced_posterior,
    standardized_cosine_similarity,
    uncertainty_gated_fusion,
)


def main() -> None:
    memory = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, 0.1]],
        dtype=np.float32,
    )
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    identifiers = np.asarray(["b", "a", "d", "c", "f", "e"], dtype=object)
    similarity = standardized_cosine_similarity(memory, query)
    probability, neighbors, confidence = class_balanced_posterior(
        similarity, labels, identifiers, k=1, temperature=0.1
    )
    assert probability.shape == (2, 3)
    assert neighbors.shape == (2, 3, 1)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.all(np.isfinite(probability))
    assert np.all((0.0 <= confidence) & (confidence <= 1.0))

    # Deliberately tied similarity: the lexicographically smaller ID must win.
    tied = np.ones((1, len(labels)), dtype=np.float64)
    _, tied_neighbors, _ = class_balanced_posterior(
        tied, labels, identifiers, k=1, temperature=1.0
    )
    assert identifiers[tied_neighbors[0, 0, 0]] == "a"
    assert identifiers[tied_neighbors[0, 1, 0]] == "c"
    assert identifiers[tied_neighbors[0, 2, 0]] == "e"

    anchor = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], dtype=np.float64)
    identity, zero_weight = uncertainty_gated_fusion(
        anchor, probability, confidence, max_weight=0.0
    )
    assert np.allclose(identity, anchor)
    assert np.allclose(zero_weight, 0.0)
    fused, weight = uncertainty_gated_fusion(anchor, probability, confidence, max_weight=0.25)
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert np.all((0.0 <= weight) & (weight <= 0.25))
    print("retrieval memory fusion smoke test passed")


if __name__ == "__main__":
    main()
