from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import attach_labels_from_excel, load_pickle, parse_split
from .metrics import CLASS_NAMES, compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only audit of an external three-class sentiment checkpoint"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)
    loaded = load_pickle(args.data)
    # Remove test before labels are attached so this diagnostic cannot materialize
    # held-out targets.
    raw = attach_labels_from_excel({"valid": loaded["valid"]}, args.labels)
    valid = parse_split(raw, "valid", "text_shared", require_labels=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    ).to(device)
    model.eval()

    label_names = [str(model.config.id2label[index]).lower() for index in range(3)]
    target_order = ["negative", "neutral", "positive"]
    reorder = [label_names.index(name) for name in target_order]
    rows: list[np.ndarray] = []
    texts = [str(value) for value in valid.raw_text]
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            probability = torch.softmax(model(**encoded).logits.float(), dim=-1)
            rows.append(probability[:, reorder].cpu().numpy())
    probabilities = np.concatenate(rows)
    regression = 3.0 * (probabilities[:, 2] - probabilities[:, 0])
    metrics = compute_metrics(
        valid.class_labels,
        probabilities,
        valid.regression_labels,
        regression,
    )
    report = {
        "scope": "validation_only_no_test_access",
        "model": args.model,
        "revision": args.revision,
        "samples": valid.size,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "id": [str(value) for value in valid.ids],
            "predicted_label": [
                CLASS_NAMES[index] for index in probabilities.argmax(axis=1)
            ],
            "negative_probability": probabilities[:, 0],
            "neutral_probability": probabilities[:, 1],
            "positive_probability": probabilities[:, 2],
            "predicted_intensity": regression,
            "true_label": [CLASS_NAMES[index] for index in valid.class_labels],
            "true_intensity": valid.regression_labels,
        }
    ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
