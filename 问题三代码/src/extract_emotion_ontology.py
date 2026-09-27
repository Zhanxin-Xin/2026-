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
from .utils import resolve_device


class EncodedTextDataset(Dataset):
    def __init__(self, encoded: dict[str, torch.Tensor]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.encoded.items()}


def _safe_label(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value.lower())


@torch.inference_mode()
def extract_split(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    loader = DataLoader(
        EncodedTextDataset(dict(encoded)),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks: list[np.ndarray] = []
    embedding_chunks: list[np.ndarray] = []
    model.eval()
    for cpu_batch in loader:
        batch = {name: value.to(device, non_blocking=True) for name, value in cpu_batch.items()}
        with torch.amp.autocast(
            "cuda", enabled=amp and device.type == "cuda", dtype=torch.float16
        ):
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            logits = outputs.logits.float()
        chunks.append(logits.cpu().numpy())
        embedding_chunks.append(outputs.hidden_states[-1][:, 0].float().cpu().numpy())
    logits = np.concatenate(chunks, axis=0).astype(np.float32)
    embeddings = np.concatenate(embedding_chunks, axis=0).astype(np.float32)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return logits, probabilities.astype(np.float32), embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen multi-label emotion-ontology evidence for train/valid. "
            "The command never reads test unless --split test is explicitly supplied."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", choices=("train", "valid", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    splits = args.split or ["train", "valid"]
    if args.batch_size < 1 or args.max_length < 8:
        raise ValueError("batch-size must be positive and max-length must be at least 8")
    device = resolve_device(args.device)
    loaded = load_pickle(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    ).to(device)
    labels = {
        int(index): str(label).lower()
        for index, label in dict(model.config.id2label).items()
    }
    if sorted(labels) != list(range(len(labels))):
        raise ValueError(f"Emotion label indices must be contiguous: {labels}")
    neutral = [index for index, label in labels.items() if label == "neutral"]
    if len(neutral) != 1:
        raise ValueError(f"Expected exactly one Neutral label: {labels}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "scope": "explicit_splits_only",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_commit_hash": getattr(model.config, "_commit_hash", None),
        "labels": labels,
        "neutral_label_index": neutral[0],
        "splits": {},
    }
    for split_name in splits:
        arrays = parse_split(
            {split_name: loaded[split_name]},
            split_name,
            "text_shared",
            require_labels=False,
        )
        logits, probabilities, embeddings = extract_split(
            model=model,
            tokenizer=tokenizer,
            texts=[str(value) for value in arrays.raw_text],
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            amp=not args.no_amp,
        )
        columns: dict[str, Any] = {"id": [str(value) for value in arrays.ids]}
        for index, label in labels.items():
            safe = _safe_label(label)
            columns[f"emotion_logit_{safe}"] = logits[:, index]
            columns[f"emotion_probability_{safe}"] = probabilities[:, index]
        for index in range(embeddings.shape[1]):
            columns[f"emotion_embedding_{index:04d}"] = embeddings[:, index]
        if arrays.class_labels is not None:
            columns["true_label"] = [CLASS_NAMES[index] for index in arrays.class_labels]
        if arrays.regression_labels is not None:
            columns["true_intensity"] = arrays.regression_labels
        frame = pd.DataFrame(columns)
        frame.to_csv(
            output / f"{split_name}_ontology.csv",
            index=False,
            encoding="utf-8-sig",
        )
        report["splits"][split_name] = {
            "samples": arrays.size,
            "neutral_probability_mean": float(probabilities[:, neutral[0]].mean()),
            "embedding_dimension": int(embeddings.shape[1]),
            "output": str(output / f"{split_name}_ontology.csv"),
        }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
