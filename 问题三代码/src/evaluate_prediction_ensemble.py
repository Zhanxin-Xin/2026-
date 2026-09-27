from __future__ import annotations

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


def _load_aligned(run_dirs: list[Path], split: str) -> list[pd.DataFrame]:
    frames = []
    for run_dir in run_dirs:
        path = run_dir / f"{split}_predictions.csv"
        frame = pd.read_csv(path)
        required = {
            "id",
            "true_label",
            "true_intensity",
            "predicted_intensity",
            *PROBABILITY_COLUMNS,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate IDs")
        frames.append(frame.sort_values("id").reset_index(drop=True))

    reference = frames[0]
    for run_dir, frame in zip(run_dirs[1:], frames[1:]):
        if not reference["id"].equals(frame["id"]):
            raise ValueError(f"ID mismatch in {run_dir} for split '{split}'")
        if not reference["true_label"].equals(frame["true_label"]):
            raise ValueError(f"Class-target mismatch in {run_dir} for split '{split}'")
        if not np.allclose(reference["true_intensity"], frame["true_intensity"]):
            raise ValueError(f"Regression-target mismatch in {run_dir} for split '{split}'")
    return frames


def _ensemble_split(run_dirs: list[Path], split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    frames = _load_aligned(run_dirs, split)
    probabilities = np.mean(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in frames],
        axis=0,
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True).clip(min=1e-12)
    regression = np.mean(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=0,
    )
    label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    reference = frames[0]
    class_target = reference["true_label"].map(label_to_index).to_numpy(np.int64)
    regression_target = reference["true_intensity"].to_numpy(np.float64)
    metrics = compute_metrics(
        class_target, probabilities, regression_target, regression
    )
    output = pd.DataFrame(
        {
            "id": reference["id"],
            "predicted_label": [CLASS_NAMES[index] for index in probabilities.argmax(1)],
            "negative_probability": probabilities[:, 0],
            "neutral_probability": probabilities[:, 1],
            "positive_probability": probabilities[:, 2],
            "predicted_intensity": regression,
            "true_label": reference["true_label"],
            "true_intensity": regression_target,
        }
    )
    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uniformly average prediction probabilities and regression outputs"
    )
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"]
    )
    args = parser.parse_args()

    run_dirs = [Path(value) for value in args.run]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "method": "uniform_probability_and_regression_average",
        "runs": [str(path) for path in run_dirs],
    }
    for split in args.splits:
        metrics, predictions = _ensemble_split(run_dirs, split)
        report[split] = metrics
        predictions.to_csv(
            output_dir / f"{split}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    (output_dir / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
