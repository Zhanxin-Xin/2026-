from __future__ import annotations

import numpy as np

from src.train_heteroscedastic_ordinal_distribution import (
    DistributionParameters,
    distribution_probability,
    fast_classification_metrics,
    select_parameters,
)


def main() -> None:
    parent = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.55, 0.25],
            [0.10, 0.25, 0.65],
            [0.30, 0.38, 0.32],
            [0.32, 0.36, 0.32],
            [0.20, 0.30, 0.50],
        ],
        dtype=np.float64,
    )
    regression_mean = np.asarray([-1.0, 0.0, 1.0, -0.1, 0.1, 0.8])
    disagreement = np.asarray([0.4, 0.1, 0.3, 0.2, 0.2, 0.4])
    target = np.asarray([0, 1, 2, 1, 1, 2], dtype=np.int64)
    candidates = [
        DistributionParameters(-0.3, 0.35, 0.15, 1.5, 0.1),
        DistributionParameters(-0.3, 0.35, 0.15, 1.5, 0.4),
    ]
    probability = distribution_probability(
        parent, regression_mean, disagreement, candidates[1]
    )
    assert probability.shape == (6, 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert np.isfinite(probability).all()
    accuracy, macro_f1 = fast_classification_metrics(target, probability.argmax(1))
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= macro_f1 <= 1.0
    selected, metrics = select_parameters(
        parent,
        regression_mean,
        disagreement,
        target,
        candidates,
        accuracy_tolerance=1.0,
    )
    assert selected in candidates
    assert set(metrics) == {
        "accuracy",
        "macro_f1",
        "parent_accuracy",
        "parent_macro_f1",
    }
    print("heteroscedastic_ordinal_distribution_smoke_test: PASS")


if __name__ == "__main__":
    main()
