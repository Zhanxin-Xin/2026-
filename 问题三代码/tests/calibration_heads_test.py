from __future__ import annotations

import numpy as np

from src.freeze_zero_consistency_neutral_rescue import apply_rescue
from src.train_explainable_hierarchical_nam import ConceptBundle
from src.train_validation_sparse_neutral_refit import apply_sparse_neutral_head


class _FixedNeutralModel:
    def __init__(self, neutral: np.ndarray) -> None:
        self.neutral = neutral

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        assert len(values) == len(self.neutral)
        return np.column_stack([1.0 - self.neutral, self.neutral])


def _bundle() -> ConceptBundle:
    return ConceptBundle(
        values=np.asarray([[3.0], [1.0], [4.0]], dtype=np.float64) / 7.0,
        names=["consensus_vote_neutral"],
        groups=["consensus_vote"],
        parent_probability=np.asarray(
            [[0.45, 0.35, 0.20], [0.20, 0.60, 0.20], [0.15, 0.34, 0.51]],
            dtype=np.float64,
        ),
        mean_probability=np.full((3, 3), 1.0 / 3.0),
        regression=np.asarray([0.05, 0.00, 0.50], dtype=np.float64),
    )


def test_sparse_head_preserves_parent_polar_odds() -> None:
    bundle = _bundle()
    probability, _ = apply_sparse_neutral_head(
        bundle,
        _FixedNeutralModel(np.asarray([0.40, 0.70, 0.20])),
        neutral_logit_bias=-0.10,
    )
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        probability[:, 0] / probability[:, 2],
        bundle.parent_probability[:, 0] / bundle.parent_probability[:, 2],
    )


def test_zero_consistency_rescue_is_selective_and_preserves_polar_odds() -> None:
    bundle = _bundle()
    probability, diagnostics = apply_rescue(
        bundle,
        {
            "neutral_bias": 0.0,
            "maximum_absolute_intensity": 0.15,
            "minimum_neutral_votes": 3,
            "maximum_neutral_logit_deficit": 0.30,
        },
    )
    assert diagnostics["zero_consistency_rescued"].tolist() == [True, False, False]
    assert probability[0].argmax() == 1
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    np.testing.assert_allclose(
        probability[:, 0] / probability[:, 2],
        bundle.parent_probability[:, 0] / bundle.parent_probability[:, 2],
    )
