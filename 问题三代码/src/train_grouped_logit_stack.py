from __future__ import annotations

"""Regularized grouped-OOF logit stacking for task-trained experts."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics


def _groups(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.rsplit("$_$", 1)[0] if "$_$" in value else value for value in ids],
        dtype=object,
    )


def features(frames: list[pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []
    predictions: list[np.ndarray] = []
    for expert, frame in enumerate(frames):
        probability = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
        probability /= probability.sum(axis=1, keepdims=True).clip(min=1e-12)
        logit = np.log(probability.clip(min=1e-8))
        centered = logit - logit.mean(axis=1, keepdims=True)
        ordered = np.sort(probability, axis=1)
        entropy = -(
            probability.clip(min=1e-8) * np.log(probability.clip(min=1e-8))
        ).sum(axis=1) / np.log(3.0)
        diagnostic = np.column_stack(
            [probability.max(axis=1), ordered[:, -1] - ordered[:, -2], entropy]
        )
        parts.extend([centered, diagnostic])
        names.extend([f"expert_{expert}_{label}_logit" for label in CLASS_NAMES])
        names.extend(
            [
                f"expert_{expert}_confidence",
                f"expert_{expert}_margin",
                f"expert_{expert}_entropy",
            ]
        )
        predictions.append(probability.argmax(1))
    for left in range(len(frames)):
        for right in range(left + 1, len(frames)):
            parts.append((predictions[left] != predictions[right]).astype(np.float64)[:, None])
            names.append(f"expert_{left}_{right}_disagreement")
    return np.concatenate(parts, axis=1), names


def make_model(targets: np.ndarray, seed: int) -> Pipeline:
    counts = np.bincount(targets, minlength=3).astype(np.float64)
    weights = np.sqrt(len(targets) / (3.0 * counts.clip(min=1.0)))
    weights /= weights.mean()
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.10,
                    class_weight={index: float(value) for index, value in enumerate(weights)},
                    max_iter=2000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
) -> dict[str, Any]:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    targets = frame["true_label"].map(mapping).to_numpy(np.int64)
    return compute_metrics(
        targets,
        probability,
        frame["true_intensity"].to_numpy(np.float64),
        regression,
    )


def prediction_frame(
    reference: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
    fold: np.ndarray | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": reference["true_label"],
            "true_intensity": reference["true_intensity"],
        }
    )
    if fold is not None:
        output.insert(1, "stack_fold", fold)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped cross-fitted logit stack")
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--valid", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    if len(args.oof) != len(args.valid) or len(args.oof) < 2:
        raise ValueError("Need at least two matched OOF/valid expert sources")
    oof_frames = load_aligned(args.oof, require_fold=True)
    valid_frames = load_aligned(args.valid, require_fold=False)
    x_oof, feature_names = features(oof_frames)
    x_valid, valid_feature_names = features(valid_frames)
    if valid_feature_names != feature_names:
        raise RuntimeError("OOF/valid stack feature schemas differ")
    reference = oof_frames[0]
    target_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    targets = reference["true_label"].map(target_map).to_numpy(np.int64)
    groups = _groups(reference["id"].astype(str).to_numpy())
    split = list(
        StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed
        ).split(x_oof, targets, groups)
    )
    oof_probability = np.zeros((len(targets), 3), dtype=np.float64)
    stack_fold = np.full(len(targets), -1, dtype=np.int64)
    fold_reports: list[dict[str, Any]] = []
    regression_oof = np.mean(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in oof_frames],
        axis=0,
    )
    for fold, (fit_index, heldout_index) in enumerate(split):
        if set(groups[fit_index]).intersection(groups[heldout_index]):
            raise RuntimeError(f"Stack fold {fold} has group leakage")
        model = make_model(targets[fit_index], args.seed + fold)
        model.fit(x_oof[fit_index], targets[fit_index])
        oof_probability[heldout_index] = model.predict_proba(x_oof[heldout_index])
        stack_fold[heldout_index] = fold
        fold_reports.append(
            {
                "fold": fold,
                "fit_samples": int(len(fit_index)),
                "heldout_samples": int(len(heldout_index)),
                "group_overlap": 0,
                "metrics": compute_metrics(
                    targets[heldout_index],
                    oof_probability[heldout_index],
                    reference["true_intensity"].to_numpy(np.float64)[heldout_index],
                    regression_oof[heldout_index],
                ),
            }
        )
    if (stack_fold < 0).any():
        raise RuntimeError("Stack OOF did not cover all samples")
    final_model = make_model(targets, args.seed)
    final_model.fit(x_oof, targets)
    valid_probability = final_model.predict_proba(x_valid)
    regression_valid = np.mean(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in valid_frames],
        axis=0,
    )
    uniform_oof = np.mean(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in oof_frames],
        axis=0,
    )
    uniform_valid = np.mean(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in valid_frames],
        axis=0,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output / "stack.joblib")
    prediction_frame(reference, oof_probability, regression_oof, stack_fold).to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    prediction_frame(valid_frames[0], valid_probability, regression_valid).to_csv(
        output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "scope": "task_expert_grouped_oof_stack_then_locked_validation_no_test_access",
        "architecture": "regularized_multinomial_centered_logit_stack",
        "seed": args.seed,
        "folds": args.folds,
        "regularization_c": 0.10,
        "class_weighting": "square_root_inverse_frequency",
        "feature_names": feature_names,
        "oof_sources": args.oof,
        "valid_sources": args.valid,
        "uniform_oof": metrics(reference, uniform_oof, regression_oof),
        "stack_oof": metrics(reference, oof_probability, regression_oof),
        "fold_reports": fold_reports,
        "uniform_valid": metrics(valid_frames[0], uniform_valid, regression_valid),
        "stack_valid": metrics(valid_frames[0], valid_probability, regression_valid),
    }
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
