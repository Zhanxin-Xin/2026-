"""Cross-fitted Neutral-authenticity arbitration over a fixed expert bank.

The parent is the equal consensus of the mean-posterior and majority-vote
channels.  The arbitrator is deliberately asymmetric: it may only reject a
parent Neutral decision, and a rejection preserves the parent's Negative vs
Positive odds.  This creates a small error-correcting decision graph instead
of another unconstrained three-class stacker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics


LABELS = np.asarray(CLASS_NAMES)
LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}


def parent_and_features(
    frames: list[pd.DataFrame], neutral_bias: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.stack(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    probability = np.clip(probability, 1e-8, 1.0)
    mean_probability = probability.mean(axis=1)
    vote_probability = np.eye(3, dtype=np.float64)[
        probability.argmax(axis=-1)
    ].mean(axis=1)
    parent = 0.5 * (mean_probability + vote_probability)
    if neutral_bias:
        parent[:, 1] *= np.exp(float(neutral_bias))
    parent /= parent.sum(axis=1, keepdims=True)

    entropy = -(probability * np.log(probability)).sum(axis=-1)
    intensity = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    # Keep expert identities because their Neutral errors are complementary,
    # then add permutation-stable consensus/disagreement coordinates.
    features = np.concatenate(
        [
            np.log(probability).reshape(len(parent), -1),
            mean_probability,
            probability.std(axis=1),
            probability.min(axis=1),
            probability.max(axis=1),
            vote_probability,
            parent,
            entropy,
            intensity,
            intensity.mean(axis=1, keepdims=True),
            intensity.std(axis=1, keepdims=True),
        ],
        axis=1,
    )
    return parent, features, mean_probability


def make_arbitrator(c_value: float) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs"),
    )


def apply_arbitration(
    parent: np.ndarray,
    authenticity: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    output = parent.copy()
    parent_prediction = parent.argmax(axis=1)
    rejected = (parent_prediction == 1) & (authenticity < threshold)
    # Removing Neutral mass exactly preserves Negative/Positive odds.
    output[rejected, 1] = 0.0
    output /= output.sum(axis=1, keepdims=True).clip(min=1e-12)
    return output, rejected


def select_threshold(
    targets: np.ndarray,
    parent: np.ndarray,
    authenticity: np.ndarray,
    regression_target: np.ndarray,
    regression_prediction: np.ndarray,
) -> tuple[float, dict[str, object], np.ndarray, np.ndarray]:
    parent_metrics = compute_metrics(
        targets, parent, regression_target, regression_prediction
    )
    candidates = np.unique(
        np.concatenate(([0.0, 1.0], authenticity[parent.argmax(axis=1) == 1]))
    )
    rows: list[tuple[tuple[float, float, float, float], float, dict[str, object], np.ndarray, np.ndarray]] = []
    for threshold in candidates:
        probability, rejected = apply_arbitration(parent, authenticity, float(threshold))
        metrics = compute_metrics(
            targets, probability, regression_target, regression_prediction
        )
        non_degrading = float(
            metrics["accuracy"] >= parent_metrics["accuracy"]
            and metrics["macro_f1"] >= parent_metrics["macro_f1"]
        )
        # Prefer candidates that improve both parent metrics, then maximize the
        # weaker primary metric and finally minimize decision churn.
        key = (
            non_degrading,
            min(float(metrics["accuracy"]), float(metrics["macro_f1"])),
            float(metrics["accuracy"]) + float(metrics["macro_f1"]),
            -float(rejected.sum()),
        )
        rows.append((key, float(threshold), metrics, probability, rejected))
    _, threshold, metrics, probability, rejected = max(rows, key=lambda row: row[0])
    return threshold, metrics, probability, rejected


def prediction_frame(
    reference: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
    authenticity: np.ndarray,
    rejected: np.ndarray,
    include_fold: bool,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": LABELS[probability.argmax(axis=1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "neutral_authenticity": authenticity,
            "neutral_rejected": rejected.astype(np.int64),
            "true_label": reference["true_label"].astype(str),
            "true_intensity": reference["true_intensity"].to_numpy(np.float64),
        }
    )
    if include_fold:
        output.insert(1, "fold", reference["fold"].to_numpy(np.int64))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fitted asymmetric Neutral-authenticity arbitrator"
    )
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--valid", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--oof-only", action="store_true")
    parser.add_argument("--neutral-bias", type=float, default=0.0)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument(
        "--fit-valid-threshold",
        action="store_true",
        help=(
            "Fit only the final authenticity threshold on validation labels; "
            "the arbitrator weights remain trained exclusively on train OOF data"
        ),
    )
    args = parser.parse_args()
    if len(args.oof) < 2:
        raise ValueError("At least two OOF expert sources are required")
    if not args.oof_only and len(args.oof) != len(args.valid):
        raise ValueError("Every OOF source needs a matching valid source")

    oof_frames = load_aligned(args.oof, require_fold=True)
    reference = oof_frames[0]
    parent, features, _ = parent_and_features(oof_frames, args.neutral_bias)
    folds = reference["fold"].to_numpy(np.int64)
    targets = reference["true_label"].map(LABEL_MAP).to_numpy(np.int64)
    regression_target = reference["true_intensity"].to_numpy(np.float64)
    regression = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in oof_frames],
        axis=1,
    ).mean(axis=1)
    parent_neutral = parent.argmax(axis=1) == 1
    authenticity = np.ones(len(parent), dtype=np.float64)
    for fold in np.unique(folds):
        train = (folds != fold) & parent_neutral
        heldout = (folds == fold) & parent_neutral
        model = make_arbitrator(args.c)
        model.fit(features[train], targets[train] == 1)
        authenticity[heldout] = model.predict_proba(features[heldout])[:, 1]

    threshold, oof_metrics, oof_probability, oof_rejected = select_threshold(
        targets,
        parent,
        authenticity,
        regression_target,
        regression,
    )
    parent_metrics = compute_metrics(targets, parent, regression_target, regression)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame(
        reference,
        oof_probability,
        regression,
        authenticity,
        oof_rejected,
        include_fold=True,
    ).to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")

    report: dict[str, object] = {
        "scope": (
            "grouped_meta_oof_only_no_valid_or_test_access"
            if args.oof_only
            else "grouped_meta_oof_then_locked_valid_no_test_access"
        ),
        "method": "asymmetric_neutral_authenticity_arbitrator",
        "invariants": [
            "parent_non_neutral_predictions_are_unchanged",
            "neutral_rejection_preserves_negative_positive_odds",
        ],
        "oof_sources": args.oof,
        "valid_sources": None if args.oof_only else args.valid,
        "source_count": len(args.oof),
        "neutral_bias": args.neutral_bias,
        "logistic_c": args.c,
        "selected_oof_authenticity_threshold": threshold,
        "oof_rejected_neutral_count": int(oof_rejected.sum()),
        "oof_parent": parent_metrics,
        "oof": oof_metrics,
    }

    if not args.oof_only:
        valid_frames = load_aligned(args.valid, require_fold=False)
        valid_reference = valid_frames[0]
        valid_parent, valid_features, _ = parent_and_features(
            valid_frames, args.neutral_bias
        )
        full_train = parent_neutral
        deployment_model = make_arbitrator(args.c)
        deployment_model.fit(features[full_train], targets[full_train] == 1)
        joblib.dump(deployment_model, output / "neutral_authenticity_arbitrator.joblib")
        valid_authenticity = np.ones(len(valid_parent), dtype=np.float64)
        valid_neutral = valid_parent.argmax(axis=1) == 1
        valid_authenticity[valid_neutral] = deployment_model.predict_proba(
            valid_features[valid_neutral]
        )[:, 1]
        valid_targets = (
            valid_reference["true_label"].map(LABEL_MAP).to_numpy(np.int64)
        )
        valid_regression_target = valid_reference["true_intensity"].to_numpy(
            np.float64
        )
        valid_regression = np.stack(
            [
                frame["predicted_intensity"].to_numpy(np.float64)
                for frame in valid_frames
            ],
            axis=1,
        ).mean(axis=1)
        deployment_threshold = threshold
        if args.fit_valid_threshold:
            (
                deployment_threshold,
                valid_metrics,
                valid_probability,
                valid_rejected,
            ) = select_threshold(
                valid_targets,
                valid_parent,
                valid_authenticity,
                valid_regression_target,
                valid_regression,
            )
        else:
            valid_probability, valid_rejected = apply_arbitration(
                valid_parent, valid_authenticity, deployment_threshold
            )
            valid_metrics = compute_metrics(
                valid_targets,
                valid_probability,
                valid_regression_target,
                valid_regression,
            )
        valid_parent_metrics = compute_metrics(
            valid_targets,
            valid_parent,
            valid_regression_target,
            valid_regression,
        )
        prediction_frame(
            valid_reference,
            valid_probability,
            valid_regression,
            valid_authenticity,
            valid_rejected,
            include_fold=False,
        ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
        report.update(
            {
                "valid_parent": valid_parent_metrics,
                "deployment_authenticity_threshold": deployment_threshold,
                "deployment_threshold_fitted_on_valid": bool(
                    args.fit_valid_threshold
                ),
                "deployment_model": "neutral_authenticity_arbitrator.joblib",
                "feature_dimension": int(features.shape[1]),
                "valid_rejected_neutral_count": int(valid_rejected.sum()),
                "valid": valid_metrics,
            }
        )
    else:
        report.update({"valid_parent": None, "valid": None})

    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
