from __future__ import annotations

"""Build a uniform ensemble from aligned OOF and locked valid predictions."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import CLASS_NAMES
from .metrics import compute_metrics


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)


def load_aligned(
    paths: list[str],
    require_fold: bool,
    require_fold_alignment: bool = True,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for value in paths:
        frame = pd.read_csv(value)
        required = {
            "id",
            "true_label",
            "true_intensity",
            "predicted_intensity",
            *PROBABILITY_COLUMNS,
        }
        if require_fold:
            required.add("fold")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{value} is missing columns: {sorted(missing)}")
        frame["id"] = frame["id"].astype(str)
        if frame["id"].duplicated().any():
            raise ValueError(f"{value} has duplicate ids")
        frames.append(frame.set_index("id", drop=False))
    reference_ids = frames[0].index
    for path, frame in zip(paths[1:], frames[1:]):
        if set(frame.index) != set(reference_ids):
            raise ValueError(f"ID mismatch in {path}")
        frame = frame.loc[reference_ids]
        if not np.array_equal(
            frame["true_label"].astype(str).to_numpy(),
            frames[0]["true_label"].astype(str).to_numpy(),
        ):
            raise ValueError(f"Label mismatch in {path}")
        if require_fold and require_fold_alignment and not np.array_equal(
            frame["fold"].to_numpy(np.int64), frames[0]["fold"].to_numpy(np.int64)
        ):
            raise ValueError(f"Grouped fold assignment mismatch in {path}")
    return [frame.loc[reference_ids].reset_index(drop=True) for frame in frames]


def average(frames: list[pd.DataFrame], include_fold: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    probability = np.mean(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in frames],
        axis=0,
    )
    probability /= probability.sum(axis=1, keepdims=True).clip(min=1e-12)
    regression = np.mean(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=0,
    )
    reference = frames[0]
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    targets = reference["true_label"].map(label_map).to_numpy(np.int64)
    true_intensity = reference["true_intensity"].to_numpy(np.float64)
    metrics = compute_metrics(targets, probability, true_intensity, regression)
    output = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": reference["true_label"],
            "true_intensity": true_intensity,
        }
    )
    if include_fold:
        output.insert(1, "fold", reference["fold"].to_numpy(np.int64))
    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe OOF/valid ensemble builder")
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--valid", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--oof-only", action="store_true")
    parser.add_argument(
        "--allow-independent-folds",
        action="store_true",
        help=(
            "Allow sources produced by different grouped fold partitions; "
            "every source must still contain strict per-sample OOF predictions"
        ),
    )
    args = parser.parse_args()
    if not args.oof_only and len(args.oof) != len(args.valid):
        raise ValueError("Every OOF source needs one matching locked valid source")
    if len(args.oof) < 2:
        raise ValueError("At least two sources are required")
    oof_frames = load_aligned(
        args.oof,
        require_fold=True,
        require_fold_alignment=not args.allow_independent_folds,
    )
    oof_metrics, oof = average(oof_frames, include_fold=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "scope": (
            "uniform_grouped_oof_only_no_valid_or_test_access"
            if args.oof_only
            else "uniform_grouped_oof_ensemble_then_locked_validation_no_test_access"
        ),
        "method": "uniform_probability_and_regression_average",
        "oof_sources": args.oof,
        "source_count": len(args.oof),
        "fold_alignment_verified": not args.allow_independent_folds,
        "independently_cross_fitted_sources": bool(args.allow_independent_folds),
        "oof": oof_metrics,
    }
    if args.oof_only:
        report["valid_sources"] = None
        report["valid"] = None
    else:
        # Official valid predictions are intentionally loaded only after the
        # caller's OOF-only selection gate has passed.
        valid_frames = load_aligned(args.valid, require_fold=False)
        valid_metrics, valid = average(valid_frames, include_fold=False)
        valid.to_csv(
            output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report["valid_sources"] = args.valid
        report["valid"] = valid_metrics
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
