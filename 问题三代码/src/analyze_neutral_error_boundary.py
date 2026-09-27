from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_pickle, parse_split


PROBABILITY_COLUMNS = [
    "negative_probability",
    "neutral_probability",
    "positive_probability",
]


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("id").reset_index(drop=True)
    required = {
        "id",
        "true_label",
        "predicted_label",
        *PROBABILITY_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the validation-only Neutral decision boundary"
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--expert", action="append", default=[])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_path = Path(args.reference) / "valid_predictions.csv"
    reference = load_predictions(reference_path)
    arrays = parse_split(load_pickle(args.data), "valid", "text_shared", False)
    text_by_id = {
        str(identifier): str(text)
        for identifier, text in zip(arrays.ids, arrays.raw_text)
    }

    true_neutral = reference["true_label"].eq("Neutral").to_numpy()
    predicted_neutral = reference["predicted_label"].eq("Neutral").to_numpy()
    correct = reference["true_label"].eq(reference["predicted_label"]).to_numpy()
    probability = reference[PROBABILITY_COLUMNS].to_numpy(np.float64)
    neutral_probability = probability[:, 1]
    strongest_polar_probability = probability[:, [0, 2]].max(axis=1)
    neutral_margin = neutral_probability - strongest_polar_probability

    missed_neutral = true_neutral & ~predicted_neutral
    false_neutral = ~true_neutral & predicted_neutral
    report: dict[str, Any] = {
        "scope": "validation_only_no_test_access",
        "reference": str(reference_path),
        "counts": {
            "samples": int(len(reference)),
            "errors": int((~correct).sum()),
            "true_neutral": int(true_neutral.sum()),
            "predicted_neutral": int(predicted_neutral.sum()),
            "missed_neutral": int(missed_neutral.sum()),
            "false_neutral": int(false_neutral.sum()),
        },
        "neutral_probability": {
            "true_neutral_mean": float(neutral_probability[true_neutral].mean()),
            "missed_neutral_mean": float(neutral_probability[missed_neutral].mean()),
            "false_neutral_mean": float(neutral_probability[false_neutral].mean()),
            "true_polar_mean": float(neutral_probability[~true_neutral].mean()),
        },
        "near_boundary": {
            "missed_neutral_within_0.02": int(
                (missed_neutral & (neutral_margin >= -0.02)).sum()
            ),
            "missed_neutral_within_0.05": int(
                (missed_neutral & (neutral_margin >= -0.05)).sum()
            ),
            "missed_neutral_within_0.10": int(
                (missed_neutral & (neutral_margin >= -0.10)).sum()
            ),
            "false_neutral_with_margin_below_0.02": int(
                (false_neutral & (neutral_margin <= 0.02)).sum()
            ),
            "false_neutral_with_margin_below_0.05": int(
                (false_neutral & (neutral_margin <= 0.05)).sum()
            ),
        },
        "experts": {},
    }

    oracle_correct = correct.copy()
    for value in args.expert:
        expert_path = Path(value) / "valid_predictions.csv"
        expert = load_predictions(expert_path)
        if not reference["id"].equals(expert["id"]):
            raise ValueError(f"ID mismatch for {expert_path}")
        expert_correct = expert["true_label"].eq(expert["predicted_label"]).to_numpy()
        expert_neutral = expert["predicted_label"].eq("Neutral").to_numpy()
        oracle_correct |= expert_correct
        report["experts"][str(Path(value))] = {
            "rescues_all_reference_errors": int((~correct & expert_correct).sum()),
            "rescues_missed_neutral": int((missed_neutral & expert_neutral).sum()),
            "introduces_false_neutral_vs_reference": int(
                (correct & ~true_neutral & expert_neutral).sum()
            ),
            "disagreement_count": int(
                reference["predicted_label"].ne(expert["predicted_label"]).sum()
            ),
        }
    report["oracle_any_correct_accuracy"] = float(oracle_correct.mean())

    order = np.argsort(np.abs(neutral_margin))
    examples = []
    for index in order:
        if not (missed_neutral[index] or false_neutral[index]):
            continue
        row = reference.iloc[index]
        examples.append(
            {
                "id": str(row["id"]),
                "text": text_by_id.get(str(row["id"]), ""),
                "true_label": str(row["true_label"]),
                "predicted_label": str(row["predicted_label"]),
                "neutral_probability": float(neutral_probability[index]),
                "neutral_margin_vs_best_polar": float(neutral_margin[index]),
            }
        )
        if len(examples) >= 20:
            break
    report["closest_neutral_boundary_errors"] = examples

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
