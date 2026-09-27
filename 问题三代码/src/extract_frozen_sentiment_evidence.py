from __future__ import annotations

"""Extract train/valid evidence from a revision-pinned frozen classifier."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import attach_labels_from_excel, load_pickle, parse_split
from .metrics import CLASS_NAMES, compute_metrics


def class_order(config: Any) -> list[int]:
    labels = {
        int(index): str(value).strip().lower()
        for index, value in dict(config.id2label).items()
    }
    expected = ("negative", "neutral", "positive")
    order: list[int] = []
    for name in expected:
        matches = [index for index, value in labels.items() if value == name]
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve class '{name}' from id2label={labels}")
        order.append(matches[0])
    return order


@torch.inference_mode()
def predict(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    order: list[int],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.amp.autocast(
            "cuda", enabled=device.type == "cuda", dtype=torch.float16
        ):
            logits = model(**encoded).logits.float()
        chunks.append(torch.softmax(logits[:, order], dim=-1).cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen sentiment evidence extractor")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loaded = load_pickle(args.data)
    # Explicitly omit test.  This extractor has no code path that reads it.
    raw = attach_labels_from_excel(
        {name: loaded[name] for name in ("train", "valid")}, args.labels
    )
    arrays = {
        name: parse_split(raw, name, "text_shared", require_labels=True)
        for name in ("train", "valid")
    }
    kwargs = {
        "revision": args.revision,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, **kwargs
    ).to(device)
    order = class_order(model.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "scope": "frozen_train_and_validation_evidence_no_test_access",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_commit_hash": getattr(model.config, "_commit_hash", None),
        "class_order": order,
        "trainable_parameters": 0,
        "splits": {},
    }
    for name, split in arrays.items():
        probability = predict(
            model,
            tokenizer,
            [str(value) for value in split.raw_text],
            order,
            device,
            args.batch_size,
            args.max_length,
        )
        regression = 3.0 * (probability[:, 2] - probability[:, 0])
        metrics = compute_metrics(
            split.class_labels,
            probability,
            split.regression_labels,
            regression,
        )
        frame = pd.DataFrame(
            {
                "id": split.ids.astype(str),
                "predicted_label": [
                    CLASS_NAMES[index] for index in probability.argmax(axis=1)
                ],
                "negative_probability": probability[:, 0],
                "neutral_probability": probability[:, 1],
                "positive_probability": probability[:, 2],
                "predicted_intensity": regression,
                "true_label": [CLASS_NAMES[index] for index in split.class_labels],
                "true_intensity": split.regression_labels,
            }
        )
        frame.to_csv(
            output / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report["splits"][name] = metrics
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
