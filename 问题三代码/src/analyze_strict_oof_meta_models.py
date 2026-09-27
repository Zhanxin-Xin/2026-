from __future__ import annotations

"""Train-only grouped-OOF feasibility audit for compact meta-model families."""

import argparse
import json
import math
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_sample_weight

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics


def _targets(frame: Any) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = frame["true_label"].map(mapping)
    if labels.isna().any():
        raise ValueError("Unknown target label")
    return labels.to_numpy(np.int64)


def _features(frames: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.stack(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    probability /= probability.sum(axis=2, keepdims=True).clip(min=1e-12)
    intensity = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    centered_log = np.log(probability.clip(min=1e-8))
    centered_log -= centered_log.mean(axis=2, keepdims=True)
    ordered = np.sort(probability, axis=2)
    confidence = probability.max(axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    entropy = -(
        probability.clip(min=1e-8) * np.log(probability.clip(min=1e-8))
    ).sum(axis=2) / math.log(probability.shape[2])
    parent = probability.mean(axis=1)
    vote_fraction = np.stack(
        [np.mean(probability.argmax(axis=2) == index, axis=1) for index in range(3)],
        axis=1,
    )
    features = np.concatenate(
        [
            probability.reshape(len(probability), -1),
            centered_log.reshape(len(probability), -1),
            confidence,
            margin,
            entropy,
            intensity,
            np.abs(intensity),
            parent,
            probability.std(axis=1),
            vote_fraction,
            intensity.mean(axis=1, keepdims=True),
            intensity.std(axis=1, keepdims=True),
        ],
        axis=1,
    )
    if not np.isfinite(features).all():
        raise ValueError("Meta-model features contain NaN/Inf")
    return features, probability, intensity.mean(axis=1)


def _models(seed: int) -> dict[str, Any]:
    return {
        "balanced_logistic": LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        ),
        "shallow_hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "shallow_extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    frames = load_aligned(args.oof, require_fold=True)
    reference = frames[0]
    features, probability, regression = _features(frames)
    targets = _targets(reference)
    folds = reference["fold"].to_numpy(np.int64)
    parent_probability = probability.mean(axis=1)
    report: dict[str, Any] = {
        "scope": "train_only_existing_grouped_oof_feasibility_no_valid_or_test_access",
        "source_count": len(frames),
        "feature_count": int(features.shape[1]),
        "parent": compute_metrics(
            targets,
            parent_probability,
            reference["true_intensity"].to_numpy(np.float64),
            regression,
        ),
        "models": {},
    }
    for name, prototype in _models(args.seed).items():
        prediction = np.zeros_like(parent_probability)
        fold_log_loss: list[float] = []
        for fold in sorted(np.unique(folds).tolist()):
            fit = folds != fold
            heldout = folds == fold
            model = prototype.__class__(**prototype.get_params())
            fit_kwargs: dict[str, Any] = {}
            if name == "shallow_hist_gradient_boosting":
                fit_kwargs["sample_weight"] = compute_sample_weight(
                    "balanced", targets[fit]
                )
            model.fit(features[fit], targets[fit], **fit_kwargs)
            probability_fold = model.predict_proba(features[heldout])
            prediction[heldout] = probability_fold
            fold_log_loss.append(float(log_loss(targets[heldout], probability_fold)))
        report["models"][name] = {
            "metrics": compute_metrics(
                targets,
                prediction,
                reference["true_intensity"].to_numpy(np.float64),
                regression,
            ),
            "fold_log_loss": fold_log_loss,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
