from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .data import MODALITIES, load_feature_source, parse_split
from .utils import save_json


REQUIRED = {
    "id",
    "predicted_label",
    "negative_probability",
    "neutral_probability",
    "positive_probability",
    "predicted_intensity",
    "main_modality",
    "text_importance",
    "audio_importance",
    "vision_importance",
    "text_evidence",
    "audio_evidence",
    "vision_evidence",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate attachment-4 prediction/explanation CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data", default=None, help="Optional attachment-4 PKL for ID coverage")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mask-strategy", default="text_shared")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    # Competition sample IDs such as 01 must retain leading zeros.
    frame = pd.read_csv(args.csv, dtype={"id": str})
    errors, warnings = [], []
    if frame.empty:
        errors.append("Prediction file contains no samples")
    missing = REQUIRED - set(frame.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    if "id" in frame:
        if frame["id"].astype(str).duplicated().any():
            errors.append("Duplicate sample IDs exist")
    if frame[list(REQUIRED & set(frame.columns))].isna().any().any():
        warnings.append("Some required fields contain empty values; inspect evidence/time mapping")

    probability_columns = [
        "negative_probability",
        "neutral_probability",
        "positive_probability",
    ]
    if all(column in frame for column in probability_columns):
        try:
            probability = frame[probability_columns].to_numpy(dtype=float)
            if not np.isfinite(probability).all():
                errors.append("Class probabilities contain non-finite values")
            if np.any((probability < -1e-6) | (probability > 1 + 1e-6)):
                errors.append("Class probabilities fall outside [0,1]")
            if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-4):
                errors.append("Class probabilities do not sum to one")
            if "predicted_label" in frame:
                class_names = np.asarray(["Negative", "Neutral", "Positive"])
                probability_label = class_names[probability.argmax(axis=1)]
                if not np.array_equal(probability_label, frame["predicted_label"].astype(str)):
                    errors.append("predicted_label is inconsistent with the largest class probability")
        except (TypeError, ValueError):
            errors.append("Class probabilities must be numeric")
    if "predicted_intensity" in frame:
        intensity = pd.to_numeric(frame["predicted_intensity"], errors="coerce")
        if intensity.isna().any() or not np.isfinite(intensity.to_numpy()).all():
            errors.append("Predicted intensity contains non-numeric or non-finite values")
        elif not intensity.between(-3, 3).all():
            errors.append("Predicted intensity falls outside [-3,3]")
    if "predicted_label" in frame and not frame["predicted_label"].isin(
        ["Negative", "Neutral", "Positive"]
    ).all():
        errors.append("predicted_label contains an unknown class")
    if "main_modality" in frame and not frame["main_modality"].isin(MODALITIES).all():
        errors.append("main_modality contains an unknown modality")

    importance_columns = [f"{m}_importance" for m in MODALITIES]
    if all(column in frame for column in importance_columns):
        try:
            importance = frame[importance_columns].to_numpy(dtype=float)
            if not np.isfinite(importance).all():
                errors.append("Modality importance contains non-finite values")
            if np.any((importance < -1e-6) | (importance > 1 + 1e-6)):
                errors.append("Modality importance falls outside [0,1]")
            if not np.allclose(importance.sum(axis=1), 1.0, atol=1e-4):
                errors.append("Three modality importance values do not sum to one")
            if "main_modality" in frame:
                expected_main = np.asarray(MODALITIES)[importance.argmax(axis=1)]
                if not np.array_equal(expected_main, frame["main_modality"].astype(str)):
                    errors.append("main_modality is inconsistent with the largest modality importance")
        except (TypeError, ValueError):
            errors.append("Modality importance must be numeric")

    for column in ("text_evidence", "audio_evidence", "vision_evidence"):
        if column not in frame:
            continue
        empty_count = 0
        for row_index, value in enumerate(frame[column]):
            try:
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("evidence is not a list")
                if not parsed:
                    empty_count += 1
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"Invalid JSON in {column}, row {row_index + 2}: {exc}")
                break
        if len(frame) and empty_count:
            warnings.append(f"{column} is empty for {empty_count}/{len(frame)} samples")

    if args.data:
        raw = load_feature_source(args.data, split_name=args.split)
        if args.split not in raw and all(m in raw for m in MODALITIES):
            raw = {args.split: raw}
        arrays = parse_split(raw, args.split, args.mask_strategy, require_labels=False)
        if "id" in frame:
            expected, actual = set(map(str, arrays.ids)), set(frame["id"].astype(str))
            if expected != actual:
                errors.append(
                    f"ID coverage mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}"
                )

    for column in ("regression_conservation_error", "classification_conservation_error"):
        if column in frame and float(frame[column].max()) > 1e-4:
            errors.append(f"{column} exceeds 1e-4")

    report: Dict[str, Any] = {
        "csv": str(Path(args.csv).resolve()),
        "rows": len(frame),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        save_json(report, args.report)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
