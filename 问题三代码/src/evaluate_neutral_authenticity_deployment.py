"""Evaluate one frozen Neutral-authenticity deployment on the final test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .build_cross_fitted_ensemble import load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics
from .train_neutral_authenticity_arbitrator import (
    apply_arbitration,
    parent_and_features,
    prediction_frame,
)


LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot evaluation of a frozen Neutral arbitrator"
    )
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    deployment = Path(args.deployment)
    report = json.loads((deployment / "final_metrics.json").read_text("utf-8"))
    if not report.get("deployment_threshold_fitted_on_valid", False):
        raise ValueError("Deployment threshold was not frozen on validation")
    if len(args.test) != int(report["source_count"]):
        raise ValueError("Test source count does not match the frozen deployment")
    threshold = float(report["deployment_authenticity_threshold"])
    neutral_bias = float(report["neutral_bias"])
    model = joblib.load(deployment / str(report["deployment_model"]))
    frames = load_aligned(args.test, require_fold=False)
    reference = frames[0]
    parent, features, _ = parent_and_features(frames, neutral_bias)
    if features.shape[1] != int(report["feature_dimension"]):
        raise ValueError("Frozen arbitrator feature dimension mismatch")

    parent_neutral = parent.argmax(axis=1) == 1
    authenticity = np.ones(len(parent), dtype=np.float64)
    authenticity[parent_neutral] = model.predict_proba(features[parent_neutral])[:, 1]
    probability, rejected = apply_arbitration(parent, authenticity, threshold)
    regression = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=1,
    ).mean(axis=1)
    targets = reference["true_label"].map(LABEL_MAP).to_numpy(np.int64)
    regression_targets = reference["true_intensity"].to_numpy(np.float64)
    parent_metrics = compute_metrics(
        targets, parent, regression_targets, regression
    )
    metrics = compute_metrics(targets, probability, regression_targets, regression)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame(
        reference,
        probability,
        regression,
        authenticity,
        rejected,
        include_fold=False,
    ).to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    result = {
        "scope": "frozen_after_valid_then_one_shot_test",
        "deployment": str(deployment),
        "test_sources": args.test,
        "neutral_bias": neutral_bias,
        "authenticity_threshold": threshold,
        "test_rejected_neutral_count": int(rejected.sum()),
        "test_parent": parent_metrics,
        "test": metrics,
    }
    (output / "final_test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
