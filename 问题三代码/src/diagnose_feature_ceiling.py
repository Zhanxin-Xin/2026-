from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FeatureNormalizer, attach_labels_from_excel, load_pickle, parse_split


def masked_statistics(features: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    mask_f = mask[..., None].astype(np.float32)
    count = mask_f.sum(axis=1).clip(min=1.0)
    mean = (features * mask_f).sum(axis=1) / count
    variance = (((features - mean[:, None]) * mask_f) ** 2).sum(axis=1) / count
    masked = np.where(mask[..., None], features, -np.inf)
    maximum = masked.max(axis=1)
    maximum[~np.isfinite(maximum)] = 0.0
    first = features[:, 0]
    return {
        "mean": mean.astype(np.float32),
        "mean_std": np.concatenate([mean, np.sqrt(variance + 1e-6)], axis=1).astype(
            np.float32
        ),
        "mean_std_max_first": np.concatenate(
            [mean, np.sqrt(variance + 1e-6), maximum, first], axis=1
        ).astype(np.float32),
    }


def score(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "weighted_f1": float(f1_score(target, prediction, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
    }


def fit_direct(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    c: float,
    class_weight: str | None,
) -> dict[str, Any]:
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight=class_weight,
            max_iter=1500,
            solver="lbfgs",
        ),
    )
    classifier.fit(train_x, train_y)
    return {
        "train": score(train_y, classifier.predict(train_x)),
        "valid": score(valid_y, classifier.predict(valid_x)),
    }


def fit_hurdle(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    c: float,
    class_weight: str | None,
) -> dict[str, Any]:
    neutral_target = (train_y == 1).astype(np.int64)
    polar_mask = train_y != 1
    polarity_target = (train_y[polar_mask] == 2).astype(np.int64)
    neutral = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight=class_weight,
            max_iter=1500,
            solver="lbfgs",
        ),
    )
    polarity = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight=class_weight,
            max_iter=1500,
            solver="lbfgs",
        ),
    )
    neutral.fit(train_x, neutral_target)
    polarity.fit(train_x[polar_mask], polarity_target)

    def predict(features: np.ndarray) -> np.ndarray:
        p_neutral = neutral.predict_proba(features)[:, 1]
        p_positive_given_polar = polarity.predict_proba(features)[:, 1]
        probabilities = np.stack(
            [
                (1.0 - p_neutral) * (1.0 - p_positive_given_polar),
                p_neutral,
                (1.0 - p_neutral) * p_positive_given_polar,
            ],
            axis=1,
        )
        return probabilities.argmax(axis=1)

    return {
        "train": score(train_y, predict(train_x)),
        "valid": score(valid_y, predict(valid_x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the linear ceiling of masked pooled multimodal features"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw = attach_labels_from_excel(load_pickle(args.data), args.labels)
    train = parse_split(raw, "train", "text_shared", require_labels=True)
    valid = parse_split(raw, "valid", "text_shared", require_labels=True)
    normalizer = FeatureNormalizer(normalize_text=False, clip_value=10.0).fit(train)
    train = normalizer.transform(train)
    valid = normalizer.transform(valid)

    train_stats = {
        name: masked_statistics(train.features[name], train.masks[name])
        for name in ("text", "audio", "vision")
    }
    valid_stats = {
        name: masked_statistics(valid.features[name], valid.masks[name])
        for name in ("text", "audio", "vision")
    }
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "text_mean": (train_stats["text"]["mean"], valid_stats["text"]["mean"]),
        "text_statistics": (
            train_stats["text"]["mean_std_max_first"],
            valid_stats["text"]["mean_std_max_first"],
        ),
        "text_vision_mean": (
            np.concatenate(
                [train_stats["text"]["mean"], train_stats["vision"]["mean"]], axis=1
            ),
            np.concatenate(
                [valid_stats["text"]["mean"], valid_stats["vision"]["mean"]], axis=1
            ),
        ),
        "text_vision_statistics": (
            np.concatenate(
                [
                    train_stats["text"]["mean_std"],
                    train_stats["vision"]["mean_std_max_first"],
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    valid_stats["text"]["mean_std"],
                    valid_stats["vision"]["mean_std_max_first"],
                ],
                axis=1,
            ),
        ),
        "all_mean": (
            np.concatenate([train_stats[name]["mean"] for name in ("text", "audio", "vision")], axis=1),
            np.concatenate([valid_stats[name]["mean"] for name in ("text", "audio", "vision")], axis=1),
        ),
    }

    report: dict[str, Any] = {"experiments": []}
    for feature_name, (train_x, valid_x) in feature_sets.items():
        for c in (0.01, 0.1, 1.0):
            for class_weight in (None, "balanced"):
                settings = {
                    "features": feature_name,
                    "dimensions": int(train_x.shape[1]),
                    "c": c,
                    "class_weight": class_weight,
                }
                report["experiments"].append(
                    {
                        **settings,
                        "head": "direct_three_class",
                        **fit_direct(
                            train_x,
                            train.class_labels,
                            valid_x,
                            valid.class_labels,
                            c,
                            class_weight,
                        ),
                    }
                )
                report["experiments"].append(
                    {
                        **settings,
                        "head": "zero_inflated_hurdle",
                        **fit_hurdle(
                            train_x,
                            train.class_labels,
                            valid_x,
                            valid.class_labels,
                            c,
                            class_weight,
                        ),
                    }
                )

    report["experiments"].sort(
        key=lambda row: (row["valid"]["macro_f1"], row["valid"]["accuracy"]),
        reverse=True,
    )
    report["best"] = report["experiments"][0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["best"], indent=2))


if __name__ == "__main__":
    main()
