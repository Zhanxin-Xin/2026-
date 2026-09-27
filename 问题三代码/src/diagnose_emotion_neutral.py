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
from .diagnose_nli_sentiment import _complementarity
from .metrics import compute_metrics
from .train_prompt_neutral_router import binary_metrics, hurdle_probability
from .utils import resolve_device


class EncodedTextDataset(Dataset):
    def __init__(self, encoded: dict[str, torch.Tensor]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.encoded.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only emotion-Neutral transfer diagnostic"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = resolve_device(args.device)
    arrays = parse_split(load_pickle(args.data), "valid", "text_shared", False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    encoded = tokenizer(
        [str(value) for value in arrays.raw_text],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    loader = DataLoader(
        EncodedTextDataset(dict(encoded)), batch_size=args.batch_size, shuffle=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    ).to(device)
    labels = {
        int(index): str(label).lower() for index, label in model.config.id2label.items()
    }
    neutral_indices = [index for index, label in labels.items() if label == "neutral"]
    if len(neutral_indices) != 1:
        raise ValueError(f"Expected one Neutral emotion label, received {labels}")
    neutral_index = neutral_indices[0]
    model.eval()
    emotion_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            emotion_chunks.append(torch.sigmoid(logits).cpu().numpy())
    emotion = np.concatenate(emotion_chunks)
    neutral_probability = emotion[:, neutral_index]

    reference_path = Path(args.reference) / "valid_predictions.csv"
    reference = pd.read_csv(reference_path)
    reference = reference.set_index(reference["id"].astype(str), drop=False)
    expected_ids = [str(value) for value in arrays.ids]
    if set(reference.index) != set(expected_ids):
        raise ValueError("Emotion/reference validation IDs do not align")
    reference = reference.loc[expected_ids].reset_index(drop=True)
    target_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    target = reference["true_label"].map(target_map).to_numpy(np.int64)
    if arrays.class_labels is not None and not np.array_equal(target, arrays.class_labels):
        raise ValueError("Reference labels do not match embedded validation labels")
    reference_probability = reference[
        ["negative_probability", "neutral_probability", "positive_probability"]
    ].to_numpy(np.float64)
    probability = hurdle_probability(neutral_probability, reference_probability)
    regression_target = reference["true_intensity"].to_numpy(np.float64)
    regression = reference["predicted_intensity"].to_numpy(np.float64)
    metrics = compute_metrics(target, probability, regression_target, regression)
    report: dict[str, Any] = {
        "scope": "validation_only_no_test_access",
        "model": args.model,
        "revision": args.revision,
        "license": "mit",
        "reference": str(reference_path),
        "neutral_label_index": neutral_index,
        "neutral_binary": binary_metrics(target == 1, neutral_probability),
        "valid": metrics,
        "complementarity": _complementarity(
            target, reference_probability, probability
        ),
        "emotion_probability_mean": {
            label: float(emotion[:, index].mean()) for index, label in labels.items()
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": expected_ids,
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "emotion_neutral_probability": neutral_probability,
            "predicted_intensity": regression,
            "true_label": reference["true_label"],
            "true_intensity": regression_target,
        }
    ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

