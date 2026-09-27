from __future__ import annotations

import numpy as np

from src.train_frozen_emoberta_lda_expert import combine, fit_predict


def main() -> None:
    rng = np.random.default_rng(7)
    target = np.repeat(np.arange(3), 16)
    features = rng.normal(scale=0.15, size=(48, 12)).astype(np.float32)
    features[:, :3] += np.eye(3, dtype=np.float32)[target] * 2.0
    _, probability = fit_predict(features, target, features, seed=11)
    assert probability.shape == (48, 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert (probability.argmax(axis=1) == target).mean() > 0.95
    parent = np.full((48, 3), 1.0 / 3.0)
    fused = combine(parent, probability, parent_member_count=7)
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert np.allclose(fused, (7.0 * parent + probability) / 8.0)
    print("frozen EmoBERTa LDA smoke test passed")


if __name__ == "__main__":
    main()
