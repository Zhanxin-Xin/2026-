"""Post-freeze modality occlusion audit for the frozen EXP183 test deployment.

This script is descriptive only: it never changes a checkpoint, threshold, model
member, or feature.  The already materialized full-condition test predictions
are treated as the reference and the three zero/blank interventions are passed
through the same seven members and the same frozen Neutral arbitrator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_attachment4_exp183 import (  # noqa: E402
    MEMBER_FILES,
    MODALITIES,
    PROBABILITY_COLUMNS,
    deploy,
    intervention,
    predict_hafusion,
    predict_nli,
    predict_pretrained,
)
from src.data import FeatureNormalizer, load_pickle, parse_split  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.utils import resolve_device, save_json, seed_everything  # noqa: E402


LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--full-member", action="append", type=Path, required=True)
    parser.add_argument("--frozen-full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hafusion-batch-size", type=int, default=64)
    return parser.parse_args()


def metric_row(
    condition: str,
    metrics: dict,
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict:
    reference_labels = reference["predicted_label"].astype(str).to_numpy()
    changed_labels = frame["predicted_label"].astype(str).to_numpy()
    reference_probability = reference.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    changed_probability = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    predicted = reference_probability.argmax(1)
    rows = np.arange(len(reference))
    confidence_delta = (
        reference_probability[rows, predicted] - changed_probability[rows, predicted]
    )
    intensity_delta = (
        reference["predicted_intensity"].to_numpy(float)
        - frame["predicted_intensity"].to_numpy(float)
    )
    return {
        "evaluation_split": "test",
        "condition": condition,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "regression_bias": metrics["regression_bias"],
        "pearson": metrics["pearson"],
        "accuracy_delta_from_full": metrics["accuracy"] - FULL_METRICS["accuracy"],
        "macro_f1_delta_from_full": metrics["macro_f1"] - FULL_METRICS["macro_f1"],
        "mae_delta_from_full": metrics["mae"] - FULL_METRICS["mae"],
        "pearson_delta_from_full": metrics["pearson"] - FULL_METRICS["pearson"],
        "prediction_flip_rate": float(np.mean(reference_labels != changed_labels)),
        "mean_frozen_class_confidence_drop": float(confidence_delta.mean()),
        "mean_absolute_intensity_change": float(np.abs(intensity_delta).mean()),
    }


FULL_METRICS: dict = {}


def main() -> None:
    global FULL_METRICS
    args = parse_args()
    if len(args.full_member) != len(MEMBER_FILES):
        raise ValueError(f"Expected {len(MEMBER_FILES)} --full-member inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    condition_dir = args.output / "condition_predictions"
    member_dir = args.output / "member_predictions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    member_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(20260924)

    data = load_pickle(args.data)
    test_arrays = parse_split(data, "test", "text_shared", require_labels=True)
    train_arrays = parse_split(data, "train", "text_shared", require_labels=True)
    normalizer = FeatureNormalizer(normalize_text=False, clip_value=10.0).fit(train_arrays)
    normalized_test = normalizer.transform(test_arrays)
    arrays_by_condition = {
        modality: intervention(normalized_test, modality) for modality in MODALITIES
    }

    all_predictions: dict[str, list[pd.DataFrame]] = {
        "full": [pd.read_csv(path) for path in args.full_member],
        **{modality: [] for modality in MODALITIES},
    }
    checkpoints = [args.checkpoint_dir / name for name in MEMBER_FILES]
    for member_index, checkpoint in enumerate(checkpoints, start=1):
        print(f"[{member_index}/7] {checkpoint.name}", flush=True)
        if member_index <= 5:
            predictions = predict_pretrained(checkpoint, arrays_by_condition, device)
        elif member_index == 6:
            predictions = predict_hafusion(
                checkpoint,
                test_arrays,
                device,
                args.hafusion_batch_size,
                conditions=MODALITIES,
            )
        else:
            predictions = predict_nli(checkpoint, arrays_by_condition, device)
        for modality, frame in predictions.items():
            all_predictions[modality].append(frame)
            target = member_dir / modality
            target.mkdir(parents=True, exist_ok=True)
            frame.to_csv(target / f"member_{member_index:02d}.csv", index=False)

    deployed: dict[str, pd.DataFrame] = {}
    for condition, frames in all_predictions.items():
        deployed[condition], _, _ = deploy(frames, args.deployment)
        deployed[condition].to_csv(
            condition_dir / f"exp183_{condition}_test_predictions.csv", index=False
        )

    frozen = pd.read_csv(args.frozen_full)
    reference = deployed["full"]
    frozen = frozen.set_index("id").loc[reference["id"].astype(str)].reset_index()
    probability_error = np.abs(
        frozen.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
        - reference.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    ).max()
    regression_error = np.abs(
        frozen["predicted_intensity"].to_numpy(float)
        - reference["predicted_intensity"].to_numpy(float)
    ).max()
    if probability_error > 2e-5 or regression_error > 2e-5:
        raise ValueError(
            "Saved member predictions do not reproduce frozen EXP183: "
            f"probability_error={probability_error}, regression_error={regression_error}"
        )

    targets = frozen["true_label"].map(LABEL_MAP).to_numpy(np.int64)
    regression_targets = frozen["true_intensity"].to_numpy(float)
    metrics: dict[str, dict] = {}
    for condition, frame in deployed.items():
        metrics[condition] = compute_metrics(
            targets,
            frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float),
            regression_targets,
            frame["predicted_intensity"].to_numpy(float),
        )
    FULL_METRICS = metrics["full"]

    rows = [metric_row("full", metrics["full"], reference, reference)]
    rows.extend(
        metric_row(f"without_{modality}", metrics[modality], deployed[modality], reference)
        for modality in MODALITIES
    )
    result_table = pd.DataFrame(rows)
    result_table.to_csv(args.output / "occlusion_results.csv", index=False)

    full_probability = reference.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    predicted = full_probability.argmax(1)
    indices = np.arange(len(reference))
    effects = []
    sample = frozen[["id", "true_label", "true_intensity"]].copy()
    sample["predicted_label"] = reference["predicted_label"].to_numpy()
    sample["predicted_intensity"] = reference["predicted_intensity"].to_numpy(float)
    for modality in MODALITIES:
        changed = deployed[modality]
        changed_probability = changed.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
        confidence_delta = (
            full_probability[indices, predicted] - changed_probability[indices, predicted]
        )
        intensity_delta = (
            reference["predicted_intensity"].to_numpy(float)
            - changed["predicted_intensity"].to_numpy(float)
        )
        effect = 0.5 * (np.abs(confidence_delta) + np.abs(intensity_delta) / 3.0)
        effects.append(effect)
        sample[f"{modality}_confidence_delta"] = confidence_delta
        sample[f"{modality}_intensity_delta"] = intensity_delta
    effect_matrix = np.stack(effects, axis=1)
    totals = effect_matrix.sum(1, keepdims=True)
    importance = np.divide(
        effect_matrix,
        totals,
        out=np.full_like(effect_matrix, 1.0 / len(MODALITIES)),
        where=totals > 1e-12,
    )
    for index, modality in enumerate(MODALITIES):
        sample[f"{modality}_importance"] = importance[:, index]
    sample["dominant_modality"] = [MODALITIES[index] for index in importance.argmax(1)]
    sample.to_csv(args.output / "test_modality_importance.csv", index=False)

    summary = {
        "scope": "post_freeze_descriptive_test_occlusion",
        "test_used_for_selection": False,
        "sample_count": int(len(sample)),
        "frozen_reproduction_max_probability_error": float(probability_error),
        "frozen_reproduction_max_regression_error": float(regression_error),
        "metrics": metrics,
        "mean_modality_importance": {
            modality: float(sample[f"{modality}_importance"].mean())
            for modality in MODALITIES
        },
        "dominant_modality_counts": {
            modality: int((sample["dominant_modality"] == modality).sum())
            for modality in MODALITIES
        },
    }
    save_json(summary, args.output / "occlusion_manifest.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
