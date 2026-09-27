from __future__ import annotations

import numpy as np

from src.train_neutral_context_cascade import (
    CascadeParameters,
    cascade_probability,
    select_cascade,
)


def main() -> None:
    parent = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.1, 0.2, 0.7],
            [0.2, 0.45, 0.35],
        ],
        dtype=np.float64,
    )
    expert = np.asarray([0.05, 0.85, 0.10, 0.75])
    target = np.asarray([0, 1, 2, 1], dtype=np.int64)
    identity = CascadeParameters(0.0, 0.0, 0.0)
    corrected = CascadeParameters(0.3, 0.0, 1.0)
    probability = cascade_probability(parent, expert, corrected)
    assert probability.shape == parent.shape
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.allclose(
        probability[:, 0] / probability[:, 2],
        parent[:, 0] / parent[:, 2],
    )
    assert np.allclose(cascade_probability(parent, expert, identity), parent)
    selected, metrics = select_cascade(
        parent, expert, target, [identity, corrected]
    )
    assert selected in (identity, corrected)
    assert metrics["accuracy"] >= metrics["parent_accuracy"]
    print("neutral_context_cascade_smoke_test: PASS")


if __name__ == "__main__":
    main()
