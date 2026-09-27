from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .data import CLASS_NAMES
from .metrics import compute_metrics


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)


def _targets(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    labels = frame["true_label"].map(mapping)
    if labels.isna().any():
        raise ValueError("Unknown true_label in prediction table")
    return labels.to_numpy(np.int64)


def _probabilities(
    frame: pd.DataFrame, bias: np.ndarray, polarity_scale: float
) -> np.ndarray:
    probability = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    logits = np.log(np.clip(probability, 1e-12, 1.0)) + bias[None, :]
    intensity = frame["predicted_intensity"].to_numpy(np.float64)
    logits += (
        float(polarity_scale)
        * intensity[:, None]
        * np.asarray([-1.0, 0.0, 1.0])[None, :]
    )
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    return calibrated / calibrated.sum(axis=1, keepdims=True)


def _select_bias(
    valid: pd.DataFrame,
    minimum: float,
    maximum: float,
    step: float,
    polarity_maximum: float,
    polarity_step: float,
    maximin: bool = False,
) -> tuple[np.ndarray, float, dict[str, float]]:
    y_true = _targets(valid)
    base_logits = np.log(
        np.clip(valid.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64), 1e-12, 1.0)
    )
    values = np.arange(minimum, maximum + 0.5 * step, step)
    best_key: tuple[float, float, float, float] | None = None
    best_bias: np.ndarray | None = None
    best_polarity_scale: float | None = None
    best_metrics: dict[str, float] | None = None
    # The positive-class bias is fixed to zero because adding a constant to all
    # logits is unidentifiable. Only two calibration degrees of freedom remain.
    polarity_values = np.arange(
        0.0, polarity_maximum + 0.5 * polarity_step, polarity_step
    )
    intensity = valid["predicted_intensity"].to_numpy(np.float64)
    polarity_basis = np.asarray([-1.0, 0.0, 1.0])
    for polarity_scale in polarity_values:
        structured_logits = (
            base_logits
            + polarity_scale * intensity[:, None] * polarity_basis[None, :]
        )
        for negative_bias in values:
            for neutral_bias in values:
                bias = np.asarray([negative_bias, neutral_bias, 0.0])
                prediction = np.argmax(structured_logits + bias[None, :], axis=1)
                accuracy = float(accuracy_score(y_true, prediction))
                macro_f1 = float(f1_score(y_true, prediction, average="macro"))
                objective = (
                    min(accuracy, macro_f1)
                    if maximin
                    else 0.4 * accuracy + 0.6 * macro_f1
                )
                # Prefer fewer/smaller calibration corrections when metrics tie.
                correction_norm = float(np.square(bias).sum() + polarity_scale**2)
                key = (
                    objective,
                    accuracy + macro_f1,
                    macro_f1,
                    -correction_norm,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_bias = bias
                    best_polarity_scale = float(polarity_scale)
                    best_metrics = {
                        "objective": objective,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                    }
    assert (
        best_bias is not None
        and best_polarity_scale is not None
        and best_metrics is not None
    )
    return best_bias, best_polarity_scale, best_metrics


def _evaluate(
    frame: pd.DataFrame, bias: np.ndarray, polarity_scale: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    probabilities = _probabilities(frame, bias, polarity_scale)
    regression = frame["predicted_intensity"].to_numpy(np.float64)
    regression_target = frame["true_intensity"].to_numpy(np.float64)
    metrics = compute_metrics(
        _targets(frame), probabilities, regression_target, regression
    )
    output = frame.copy()
    output.loc[:, PROBABILITY_COLUMNS] = probabilities
    output["predicted_label"] = [
        CLASS_NAMES[index] for index in probabilities.argmax(axis=1)
    ]
    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit additive class-logit biases on validation predictions only"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum", type=float, default=-0.5)
    parser.add_argument("--maximum", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--polarity-maximum", type=float, default=0.0)
    parser.add_argument("--polarity-step", type=float, default=0.05)
    parser.add_argument(
        "--maximin",
        action="store_true",
        help="Maximize the smaller of Accuracy and Macro-F1",
    )
    args = parser.parse_args()
    if (
        args.step <= 0
        or args.minimum >= args.maximum
        or args.polarity_maximum < 0
        or args.polarity_step <= 0
    ):
        raise ValueError("Require step > 0 and minimum < maximum")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = pd.read_csv(input_dir / "valid_predictions.csv")
    bias, polarity_scale, selection = _select_bias(
        valid,
        args.minimum,
        args.maximum,
        args.step,
        args.polarity_maximum,
        args.polarity_step,
        maximin=args.maximin,
    )
    report: dict[str, Any] = {
        "method": "validation_fitted_additive_class_logit_bias",
        "source": str(input_dir),
        "bias_negative_neutral_positive": bias.tolist(),
        "polarity_scale": polarity_scale,
        "search": {
            "minimum": args.minimum,
            "maximum": args.maximum,
            "step": args.step,
            "polarity_maximum": args.polarity_maximum,
            "polarity_step": args.polarity_step,
            "objective": "maximin_accuracy_macro_f1" if args.maximin else "0.4_accuracy_plus_0.6_macro_f1",
            "selection": selection,
        },
    }
    for split in ("train", "valid", "test"):
        path = input_dir / f"{split}_predictions.csv"
        if not path.is_file():
            continue
        metrics, predictions = _evaluate(
            pd.read_csv(path), bias, polarity_scale
        )
        report[split] = metrics
        predictions.to_csv(
            output_dir / path.name, index=False, encoding="utf-8-sig"
        )
    (output_dir / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
