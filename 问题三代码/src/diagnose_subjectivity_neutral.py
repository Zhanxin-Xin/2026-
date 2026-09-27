from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import CLASS_NAMES, load_pickle, parse_split
from .metrics import compute_metrics
from .utils import resolve_device


PROBABILITY_COLUMNS = [
    "negative_probability",
    "neutral_probability",
    "positive_probability",
]


class EncodedTextDataset(Dataset):
    def __init__(self, encoded: dict[str, torch.Tensor]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return int(self.encoded["input_ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.encoded.items()}


def binary_neutral_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = probability >= 0.5
    true_positive = int((prediction & target).sum())
    false_positive = int((prediction & ~target).sum())
    false_negative = int((~prediction & target).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float((prediction == target).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [
            [int((~target & ~prediction).sum()), false_positive],
            [false_negative, true_positive],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only subjectivity-as-Neutral diagnostic"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = resolve_device(args.device)
    arrays = parse_split(load_pickle(args.data), "valid", "text_shared", False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=True,
    )
    encoded = tokenizer(
        [str(value) for value in arrays.raw_text],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    loader = DataLoader(
        EncodedTextDataset(dict(encoded)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
    ).to(device)
    if str(model.config.id2label.get(0, "")).upper() != "OBJ":
        raise ValueError(f"Unexpected subjectivity labels: {model.config.id2label}")
    model.eval()
    objective_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            objective_chunks.append(
                torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()
            )
    objective = np.concatenate(objective_chunks)
    subjectivity = pd.DataFrame(
        {"id": [str(value) for value in arrays.ids], "objective_probability": objective}
    ).sort_values("id").reset_index(drop=True)

    reference_path = Path(args.reference) / "valid_predictions.csv"
    reference = pd.read_csv(reference_path).sort_values("id").reset_index(drop=True)
    if not reference["id"].astype(str).equals(subjectivity["id"]):
        raise ValueError("Subjectivity/reference validation IDs do not align")
    reference_probability = reference[PROBABILITY_COLUMNS].to_numpy(np.float64)
    polar = reference_probability[:, [0, 2]]
    polar /= polar.sum(axis=1, keepdims=True).clip(min=1e-12)
    remaining = 1.0 - subjectivity["objective_probability"].to_numpy(np.float64)
    probability = np.column_stack(
        [remaining * polar[:, 0], 1.0 - remaining, remaining * polar[:, 1]]
    )
    label_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    target = reference["true_label"].map(label_to_index).to_numpy(np.int64)
    regression_target = reference["true_intensity"].to_numpy(np.float64)
    regression = reference["predicted_intensity"].to_numpy(np.float64)
    metrics = compute_metrics(
        target, probability, regression_target, regression
    )
    neutral_binary = binary_neutral_metrics(
        target == 1,
        subjectivity["objective_probability"].to_numpy(np.float64),
    )
    reference_correct = reference["predicted_label"].to_numpy() == np.asarray(
        CLASS_NAMES, dtype=object
    )[target]
    subjectivity_prediction = probability.argmax(axis=1)
    diagnostic_correct = subjectivity_prediction == target
    report = {
        "scope": "validation_only_no_test_access",
        "model": args.model,
        "revision": args.revision,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "license": "cc-by-4.0",
        "reference": str(reference_path),
        "neutral_binary": neutral_binary,
        "valid": metrics,
        "rescues_reference_errors": int(
            ((~reference_correct) & diagnostic_correct).sum()
        ),
        "oracle_any_correct_accuracy": float(
            (reference_correct | diagnostic_correct).mean()
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "id": reference["id"],
            "predicted_label": [CLASS_NAMES[index] for index in subjectivity_prediction],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "objective_probability": subjectivity["objective_probability"],
            "predicted_intensity": regression,
            "true_label": reference["true_label"],
            "true_intensity": regression_target,
        }
    )
    predictions.to_csv(
        output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
    )
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
