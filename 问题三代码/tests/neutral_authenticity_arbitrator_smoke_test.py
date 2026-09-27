from __future__ import annotations

import numpy as np

from src.train_neutral_authenticity_arbitrator import apply_arbitration


def test_neutral_arbitration_preserves_declared_invariants() -> None:
    parent = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.60, 0.30],
            [0.10, 0.20, 0.70],
        ],
        dtype=np.float64,
    )
    authenticity = np.asarray([1.0, 0.10, 0.90, 1.0])
    output, rejected = apply_arbitration(parent, authenticity, threshold=0.50)

    assert rejected.tolist() == [False, True, False, False]
    np.testing.assert_allclose(output[[0, 2, 3]], parent[[0, 2, 3]])
    assert output[1, 1] == 0.0
    np.testing.assert_allclose(output.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        output[1, 0] / output[1, 2], parent[1, 0] / parent[1, 2]
    )
