from __future__ import annotations

"""Cross-fitted heteroscedastic ordinal distribution expert.

The expert turns the ensemble's continuous sentiment predictions into an
ordered three-class distribution.  Its location is the mean regression
prediction and its sample-wise scale is an affine function of regression
disagreement.  The resulting interval probabilities are mixed with the
uniform classification parent through a bounded convex residual.

Parameters are selected inside each grouped OOF training partition before
predicting its held-out fold.  Official validation files are not opened until
the assembled cross-fitted predictions improve Macro-F1 without reducing
Accuracy relative to the parent.
"""

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics


@dataclass(frozen=True)
class DistributionParameters:
    lower_threshold: float
    upper_threshold: float
    base_scale: float
    disagreement_scale: float
    residual_mix: float


DEFAULT_LOWER_THRESHOLDS = (-0.60, -0.45, -0.30, -0.15, 0.00)
DEFAULT_UPPER_THRESHOLDS = (0.05, 0.20, 0.35, 0.50, 0.65)
DEFAULT_BASE_SCALES = (0.15, 0.30, 0.45, 0.60, 0.80)
DEFAULT_DISAGREEMENT_SCALES = (0.0, 0.5, 1.0, 1.5)
DEFAULT_RESIDUAL_MIXES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def parameter_grid() -> list[DistributionParameters]:
    return [
        DistributionParameters(*values)
        for values in itertools.product(
            DEFAULT_LOWER_THRESHOLDS,
            DEFAULT_UPPER_THRESHOLDS,
            DEFAULT_BASE_SCALES,
            DEFAULT_DISAGREEMENT_SCALES,
            DEFAULT_RESIDUAL_MIXES,
        )
    ]


def targets(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    result = frame["true_label"].map(mapping)
    if result.isna().any():
        raise ValueError("Unknown class label in prediction source")
    return result.to_numpy(np.int64)


def evidence(
    frames: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probability = np.stack(
        [
            frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
            for frame in frames
        ],
        axis=1,
    )
    probability /= probability.sum(axis=2, keepdims=True).clip(min=1e-12)
    regression = np.stack(
        [
            frame["predicted_intensity"].to_numpy(np.float64)
            for frame in frames
        ],
        axis=1,
    )
    if not np.isfinite(probability).all() or not np.isfinite(regression).all():
        raise ValueError("Prediction source contains NaN/Inf")
    return (
        probability.mean(axis=1),
        regression.mean(axis=1),
        regression.std(axis=1),
        regression,
    )


def distribution_probability(
    parent_probability: np.ndarray,
    regression_mean: np.ndarray,
    regression_disagreement: np.ndarray,
    parameters: DistributionParameters,
) -> np.ndarray:
    if parameters.lower_threshold >= parameters.upper_threshold:
        raise ValueError("lower_threshold must be less than upper_threshold")
    scale = np.maximum(
        0.08,
        parameters.base_scale
        + parameters.disagreement_scale * regression_disagreement,
    )
    below_lower = ndtr(
        (parameters.lower_threshold - regression_mean) / scale
    )
    below_upper = ndtr(
        (parameters.upper_threshold - regression_mean) / scale
    )
    ordinal = np.column_stack(
        [below_lower, below_upper - below_lower, 1.0 - below_upper]
    )
    probability = (
        (1.0 - parameters.residual_mix) * parent_probability
        + parameters.residual_mix * ordinal
    )
    probability = np.clip(probability, 1e-12, None)
    return probability / probability.sum(axis=1, keepdims=True)


def fast_classification_metrics(
    class_target: np.ndarray, class_prediction: np.ndarray
) -> tuple[float, float]:
    encoded = 3 * class_target.astype(np.int64) + class_prediction.astype(np.int64)
    matrix = np.bincount(encoded, minlength=9).reshape(3, 3)
    accuracy = float(np.trace(matrix) / max(1, matrix.sum()))
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    denominator = true_count + predicted_count
    class_f1 = np.divide(
        2.0 * np.diag(matrix),
        denominator,
        out=np.zeros(3, dtype=np.float64),
        where=denominator > 0,
    )
    return accuracy, float(class_f1.mean())


def select_parameters(
    parent_probability: np.ndarray,
    regression_mean: np.ndarray,
    regression_disagreement: np.ndarray,
    class_target: np.ndarray,
    candidate_parameters: Iterable[DistributionParameters],
    accuracy_tolerance: float = 0.0,
) -> tuple[DistributionParameters, dict[str, float]]:
    parent_prediction = parent_probability.argmax(axis=1)
    parent_accuracy, parent_macro_f1 = fast_classification_metrics(
        class_target, parent_prediction
    )
    selected: tuple[tuple[float, float, float], DistributionParameters] | None = None
    selected_metrics: dict[str, float] | None = None
    for parameters in candidate_parameters:
        probability = distribution_probability(
            parent_probability,
            regression_mean,
            regression_disagreement,
            parameters,
        )
        accuracy, macro_f1 = fast_classification_metrics(
            class_target, probability.argmax(axis=1)
        )
        if accuracy + 1e-12 < parent_accuracy - accuracy_tolerance:
            continue
        key = (macro_f1, accuracy, -parameters.residual_mix)
        if selected is None or key > selected[0]:
            selected = (key, parameters)
            selected_metrics = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "parent_accuracy": parent_accuracy,
                "parent_macro_f1": parent_macro_f1,
            }
    if selected is None or selected_metrics is None:
        raise RuntimeError("No parameter candidate satisfied the accuracy guard")
    return selected[1], selected_metrics


def prediction_frame(
    reference: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
    include_fold: bool,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": reference["true_label"].astype(str),
            "true_intensity": reference["true_intensity"].to_numpy(np.float64),
        }
    )
    if include_fold:
        output.insert(1, "fold", reference["fold"].to_numpy(np.int64))
    return output


def evaluate(
    reference: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
) -> dict[str, Any]:
    return compute_metrics(
        targets(reference),
        probability,
        reference["true_intensity"].to_numpy(np.float64),
        regression,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fitted heteroscedastic ordinal distribution expert"
    )
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--valid", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    if len(args.oof) != len(args.valid):
        raise ValueError("Every OOF source needs one matching locked-valid source")
    if len(args.oof) < 2:
        raise ValueError("At least two ensemble members are required")
    if args.accuracy_tolerance < 0.0:
        raise ValueError("accuracy-tolerance must be non-negative")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = parameter_grid()
    oof_frames = load_aligned(args.oof, require_fold=True)
    reference = oof_frames[0]
    parent, regression_mean, disagreement, _ = evidence(oof_frames)
    class_target = targets(reference)
    folds = reference["fold"].to_numpy(np.int64)

    cross_fitted_probability = np.empty_like(parent)
    fold_reports: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds).tolist()):
        fit = folds != fold
        heldout = folds == fold
        parameters, fit_metrics = select_parameters(
            parent[fit],
            regression_mean[fit],
            disagreement[fit],
            class_target[fit],
            candidates,
            accuracy_tolerance=args.accuracy_tolerance,
        )
        cross_fitted_probability[heldout] = distribution_probability(
            parent[heldout],
            regression_mean[heldout],
            disagreement[heldout],
            parameters,
        )
        heldout_accuracy, heldout_macro_f1 = fast_classification_metrics(
            class_target[heldout], cross_fitted_probability[heldout].argmax(axis=1)
        )
        fold_reports.append(
            {
                "fold": int(fold),
                "fit_samples": int(fit.sum()),
                "heldout_samples": int(heldout.sum()),
                "parameters": asdict(parameters),
                "fit_selection_metrics": fit_metrics,
                "heldout_accuracy": heldout_accuracy,
                "heldout_macro_f1": heldout_macro_f1,
            }
        )

    parent_metrics = evaluate(reference, parent, regression_mean)
    oof_metrics = evaluate(reference, cross_fitted_probability, regression_mean)
    oof_delta = {
        "accuracy": oof_metrics["accuracy"] - parent_metrics["accuracy"],
        "macro_f1": oof_metrics["macro_f1"] - parent_metrics["macro_f1"],
    }
    gate_passed = bool(
        oof_metrics["accuracy"] + 1e-12 >= parent_metrics["accuracy"]
        and oof_metrics["macro_f1"] > parent_metrics["macro_f1"] + 1e-12
    )
    oof_output = prediction_frame(
        reference, cross_fitted_probability, regression_mean, include_fold=True
    )
    oof_output.to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )

    full_parameters, full_selection_metrics = select_parameters(
        parent,
        regression_mean,
        disagreement,
        class_target,
        candidates,
        accuracy_tolerance=args.accuracy_tolerance,
    )
    report: dict[str, Any] = {
        "scope": "grouped_cross_fitted_train_oof_gate_before_locked_valid_no_test_access",
        "architecture": "heteroscedastic_gaussian_interval_ordinal_residual",
        "research_basis": {
            "DORN_CVPR_2018": {
                "idea": "ordered interval probabilities",
                "repository": "https://github.com/hufu6371/DORN",
                "commit": "ce9ee7b9f6560d964436db97e02990fca7fd68b6",
            },
            "Deep_Evidential_Regression_NeurIPS_2020": {
                "idea": "continuous predictive location and uncertainty",
                "repository": "https://github.com/aamini/evidential-deep-learning",
                "commit": "d1d8e395fb083308d14fa92c5ce766e97b2a066a",
                "license": "Apache-2.0",
            },
            "implementation_note": (
                "Original NumPy implementation; no external source code copied. "
                "Regression ensemble disagreement supplies sample-wise scale."
            ),
        },
        "oof_sources": args.oof,
        "source_count": len(args.oof),
        "candidate_count": len(candidates),
        "accuracy_tolerance": args.accuracy_tolerance,
        "fold_alignment_verified": True,
        "fold_reports": fold_reports,
        "full_oof_parameters_for_valid": asdict(full_parameters),
        "full_oof_selection_metrics": full_selection_metrics,
        "parent_oof": parent_metrics,
        "oof": oof_metrics,
        "oof_delta": oof_delta,
        "oof_gate_passed": gate_passed,
    }

    if not gate_passed:
        report["valid_sources"] = None
        report["valid"] = None
        report["decision"] = "closed_before_loading_official_valid"
    else:
        # The official validation prediction files are deliberately opened only
        # after the assembled cross-fitted OOF result clears the promotion gate.
        valid_frames = load_aligned(args.valid, require_fold=False)
        valid_reference = valid_frames[0]
        valid_parent, valid_regression, valid_disagreement, _ = evidence(valid_frames)
        valid_probability = distribution_probability(
            valid_parent,
            valid_regression,
            valid_disagreement,
            full_parameters,
        )
        valid_metrics = evaluate(
            valid_reference, valid_probability, valid_regression
        )
        prediction_frame(
            valid_reference,
            valid_probability,
            valid_regression,
            include_fold=False,
        ).to_csv(
            output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report["valid_sources"] = args.valid
        report["valid"] = valid_metrics
        report["decision"] = "promoted_after_oof_gate"

    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
