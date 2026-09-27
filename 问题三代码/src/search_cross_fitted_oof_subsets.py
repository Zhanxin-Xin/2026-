from __future__ import annotations

"""Leakage-safe subset search over an existing grouped-OOF expert library.

Each held-out fold is predicted by a uniform subset selected only on the other
four folds.  This estimates whether discrete architecture selection transfers
better than the previously rejected learned routers.  Official validation and
test artifacts are never opened by this program.
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics


def fast_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    encoded = 3 * target.astype(np.int64) + prediction.astype(np.int64)
    matrix = np.bincount(encoded, minlength=9).reshape(3, 3)
    accuracy = float(np.trace(matrix) / max(1, matrix.sum()))
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(
        2.0 * np.diag(matrix),
        denominator,
        out=np.zeros(3, dtype=np.float64),
        where=denominator > 0,
    )
    return accuracy, float(f1.mean())


def select_subset(
    probabilities: np.ndarray,
    target: np.ndarray,
    parent_probability: np.ndarray,
    accuracy_tolerance: float,
    minimum_size: int,
    maximum_size: int,
) -> tuple[tuple[int, ...], dict[str, float]]:
    parent_accuracy, parent_macro_f1 = fast_metrics(
        target, parent_probability.argmax(axis=1)
    )
    selected: tuple[tuple[float, float, int], tuple[int, ...]] | None = None
    selected_metrics: dict[str, float] | None = None
    for size in range(minimum_size, maximum_size + 1):
        for indices in itertools.combinations(range(probabilities.shape[1]), size):
            probability = probabilities[:, indices, :].mean(axis=1)
            accuracy, macro_f1 = fast_metrics(target, probability.argmax(axis=1))
            if accuracy + 1e-12 < parent_accuracy - accuracy_tolerance:
                continue
            key = (macro_f1, accuracy, -size)
            if selected is None or key > selected[0]:
                selected = (key, indices)
                selected_metrics = {
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "parent_accuracy": parent_accuracy,
                    "parent_macro_f1": parent_macro_f1,
                }
    if selected is None or selected_metrics is None:
        raise RuntimeError("No subset satisfied the accuracy guard")
    return selected[1], selected_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-oof", action="append", required=True)
    parser.add_argument("--parent-oof", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.0)
    parser.add_argument("--minimum-size", type=int, default=2)
    parser.add_argument("--maximum-size", type=int, default=10)
    args = parser.parse_args()
    if args.accuracy_tolerance < 0.0:
        raise ValueError("accuracy-tolerance must be non-negative")
    if len(set(args.candidate_oof)) != len(args.candidate_oof):
        raise ValueError("candidate-oof paths must be unique")
    frames = load_aligned(args.candidate_oof + args.parent_oof, require_fold=True)
    candidate_frames = frames[: len(args.candidate_oof)]
    parent_frames = frames[len(args.candidate_oof) :]
    count = len(candidate_frames)
    maximum_size = min(int(args.maximum_size), count)
    minimum_size = int(args.minimum_size)
    if not 1 <= minimum_size <= maximum_size:
        raise ValueError("invalid subset-size interval")

    probability = np.stack(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in candidate_frames],
        axis=1,
    )
    regression = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in candidate_frames],
        axis=1,
    )
    parent_probability = np.mean(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in parent_frames],
        axis=0,
    )
    parent_regression = np.mean(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in parent_frames],
        axis=0,
    )
    reference = candidate_frames[0]
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    target = reference["true_label"].map(mapping).to_numpy(np.int64)
    intensity_target = reference["true_intensity"].to_numpy(np.float64)
    folds = reference["fold"].to_numpy(np.int64)

    cross_fitted_probability = np.empty_like(parent_probability)
    cross_fitted_regression = np.empty_like(parent_regression)
    fold_reports: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds).tolist()):
        fit = folds != fold
        heldout = folds == fold
        indices, fit_metrics = select_subset(
            probability[fit],
            target[fit],
            parent_probability[fit],
            args.accuracy_tolerance,
            minimum_size,
            maximum_size,
        )
        cross_fitted_probability[heldout] = probability[heldout][:, indices, :].mean(axis=1)
        cross_fitted_regression[heldout] = regression[heldout][:, indices].mean(axis=1)
        heldout_accuracy, heldout_macro_f1 = fast_metrics(
            target[heldout], cross_fitted_probability[heldout].argmax(axis=1)
        )
        fold_reports.append(
            {
                "fold": int(fold),
                "selected_indices": list(indices),
                "selected_sources": [args.candidate_oof[index] for index in indices],
                "fit_metrics": fit_metrics,
                "heldout_accuracy": heldout_accuracy,
                "heldout_macro_f1": heldout_macro_f1,
            }
        )

    full_indices, full_selection_metrics = select_subset(
        probability,
        target,
        parent_probability,
        args.accuracy_tolerance,
        minimum_size,
        maximum_size,
    )
    parent_metrics = compute_metrics(
        target, parent_probability, intensity_target, parent_regression
    )
    oof_metrics = compute_metrics(
        target,
        cross_fitted_probability,
        intensity_target,
        cross_fitted_regression,
    )
    gate_passed = bool(
        oof_metrics["accuracy"] + 1e-12 >= parent_metrics["accuracy"]
        and oof_metrics["macro_f1"] > parent_metrics["macro_f1"] + 1e-12
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "fold": folds,
            "predicted_label": [
                CLASS_NAMES[index] for index in cross_fitted_probability.argmax(axis=1)
            ],
            "negative_probability": cross_fitted_probability[:, 0],
            "neutral_probability": cross_fitted_probability[:, 1],
            "positive_probability": cross_fitted_probability[:, 2],
            "predicted_intensity": cross_fitted_regression,
            "true_label": reference["true_label"].astype(str),
            "true_intensity": intensity_target,
        }
    )
    prediction_frame.to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "scope": "train_only_nested_grouped_oof_subset_search_no_valid_or_test_access",
        "candidate_sources": args.candidate_oof,
        "parent_sources": args.parent_oof,
        "candidate_count": count,
        "subset_count": int(
            sum(
                len(list(itertools.combinations(range(count), size)))
                for size in range(minimum_size, maximum_size + 1)
            )
        ),
        "accuracy_tolerance": args.accuracy_tolerance,
        "fold_reports": fold_reports,
        "full_oof_selected_indices": list(full_indices),
        "full_oof_selected_sources_for_valid": [
            args.candidate_oof[index] for index in full_indices
        ],
        "full_oof_selection_metrics": full_selection_metrics,
        "parent_oof": parent_metrics,
        "oof": oof_metrics,
        "oof_delta": {
            "accuracy": oof_metrics["accuracy"] - parent_metrics["accuracy"],
            "macro_f1": oof_metrics["macro_f1"] - parent_metrics["macro_f1"],
        },
        "oof_gate_passed": gate_passed,
        "decision": (
            "eligible_for_locked_valid_subset_materialization"
            if gate_passed
            else "closed_before_loading_official_valid"
        ),
    }
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
