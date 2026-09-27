from __future__ import annotations

import numpy as np
import pandas as pd

from src.train_ontology_neutral_calibrator import (
    build_features,
    neutral_residual_fusion,
)


def main() -> None:
    frame = pd.DataFrame(
        {
            "emotion_logit_neutral": [0.0, 1.0],
            "emotion_logit_joy": [1.0, -1.0],
            "emotion_logit_sadness": [-1.0, 1.0],
            "emotion_probability_neutral": [0.5, 0.73],
            "emotion_probability_joy": [0.73, 0.27],
            "emotion_probability_sadness": [0.27, 0.73],
        }
    )
    names = ["joy", "neutral", "sadness"]
    assert build_features(frame, "ontology", names).shape == (2, 3)
    assert build_features(frame, "semantic", names).shape == (2, 10)
    assert build_features(frame, "hybrid", names).shape == (2, 13)

    anchor = np.asarray([[0.60, 0.20, 0.20], [0.10, 0.80, 0.10]], dtype=np.float64)
    fused, correction, uncertainty = neutral_residual_fusion(
        anchor,
        np.asarray([0.80, 0.20]),
        prior=0.25,
        max_logit_residual=1.5,
    )
    assert np.all(np.isfinite(fused))
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert np.max(np.abs(correction)) <= 1.5 + 1e-12
    assert np.all((0.0 <= uncertainty) & (uncertainty <= 1.0))
    before = np.log(anchor[:, 2] / anchor[:, 0])
    after = np.log(fused[:, 2] / fused[:, 0])
    assert np.allclose(before, after, atol=1e-12)

    identity, identity_correction, _ = neutral_residual_fusion(
        anchor,
        np.full(2, 0.25),
        prior=0.25,
        max_logit_residual=1.5,
    )
    assert np.allclose(identity, anchor, atol=1e-12)
    assert np.allclose(identity_correction, 0.0, atol=1e-12)
    print("ontology neutral calibrator smoke test passed")


if __name__ == "__main__":
    main()
