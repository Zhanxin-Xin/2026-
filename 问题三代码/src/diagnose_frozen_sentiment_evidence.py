from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import CLASS_NAMES, attach_labels_from_excel, load_pickle, parse_split
from .metrics import compute_metrics
from .utils import resolve_device


class TokenDataset(Dataset):
    def __init__(self, encoded: dict[str, torch.Tensor]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return int(self.encoded["input_ids"].size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.encoded.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-only diagnostic for a fixed three-class sentiment model"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = resolve_device(args.device)
    loaded = load_pickle(args.data)
    raw = attach_labels_from_excel({"train": loaded["train"]}, args.labels)
    arrays = parse_split(raw, "train", "text_shared", require_labels=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encoded = tokenizer(
        [str(value) for value in arrays.raw_text],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    loader = DataLoader(
        TokenDataset(dict(encoded)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True
    ).to(device).eval()
    labels = {
        int(index): str(label).strip().lower()
        for index, label in model.config.id2label.items()
    }
    required = ["negative", "neutral", "positive"]
    if set(labels.values()) != set(required):
        raise ValueError(f"expected three sentiment labels, received {labels}")
    order = [next(index for index, label in labels.items() if label == name) for name in required]
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = {name: value.to(device) for name, value in cpu_batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits.float()
            chunks.append(torch.softmax(logits, dim=-1)[:, order].cpu().numpy())
    teacher = np.concatenate(chunks)
    reference = pd.read_csv(args.reference)
    if reference["id"].astype(str).duplicated().any():
        raise ValueError("reference OOF contains duplicate IDs")
    reference = reference.set_index(reference["id"].astype(str), drop=False)
    expected_ids = arrays.ids.astype(str)
    if set(reference.index) != set(expected_ids):
        raise ValueError("reference OOF IDs do not match the train ID set")
    reference = reference.loc[expected_ids].reset_index(drop=True)
    parent = reference[
        ["negative_probability", "neutral_probability", "positive_probability"]
    ].to_numpy(np.float64)
    target = arrays.class_labels
    intensity = arrays.regression_labels
    regression = reference["predicted_intensity"].to_numpy(np.float64)
    probability_mean = 0.5 * (parent + teacher)
    parent_vote = parent.argmax(axis=1)
    teacher_vote = teacher.argmax(axis=1)
    mean_vote = probability_mean.argmax(axis=1)
    # Two-member plurality ties resolve by the corresponding probability mean.
    consensus = np.where(parent_vote == teacher_vote, parent_vote, mean_vote)
    consensus_probability = np.zeros_like(probability_mean)
    consensus_probability[np.arange(len(consensus)), consensus] = 1.0

    metrics = {
        "teacher": compute_metrics(target, teacher, intensity, regression),
        "fixed_probability_mean": compute_metrics(
            target, probability_mean, intensity, regression
        ),
        "fixed_vote_probability_consensus": compute_metrics(
            target, consensus_probability, intensity, regression
        ),
        "agreement_rate": float((parent_vote == teacher_vote).mean()),
        "teacher_rescues_parent": int(
            ((teacher_vote == target) & (parent_vote != target)).sum()
        ),
        "teacher_breaks_parent": int(
            ((teacher_vote != target) & (parent_vote == target)).sum()
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "source_index": np.arange(arrays.size),
            "id": arrays.ids.astype(str),
            "teacher_negative_probability": teacher[:, 0],
            "teacher_neutral_probability": teacher[:, 1],
            "teacher_positive_probability": teacher[:, 2],
            "parent_predicted_label": [CLASS_NAMES[index] for index in parent_vote],
            "teacher_predicted_label": [CLASS_NAMES[index] for index in teacher_vote],
            "true_label": [CLASS_NAMES[index] for index in target],
        }
    )
    frame.to_csv(output / "train_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "scope": "train_only_fixed_pretrained_teacher_no_valid_or_test_access",
        "model": args.model,
        "reference": args.reference,
        "label_order": required,
        "metrics": metrics,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
