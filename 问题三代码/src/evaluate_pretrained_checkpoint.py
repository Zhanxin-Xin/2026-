from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import FeatureNormalizer, attach_labels_from_excel, load_pickle, parse_split
from .pretrained_fusion import PretrainedTextFusionNet
from .train_pretrained_fusion import EncodedTextDataset, encode_split, evaluate
from .utils import resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a validation-selected pretrained-fusion checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate test only after a scheme has been locked on validation",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Evaluate only the already-authorized frozen test split",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg: dict[str, Any] = copy.deepcopy(checkpoint["config"])
    raw = attach_labels_from_excel(load_pickle(args.data), args.labels)
    split_names = (
        ["test"]
        if args.test_only
        else ["train", "valid"] + (["test"] if args.evaluate_test else [])
    )
    arrays = {
        split: parse_split(raw, split, "text_shared", require_labels=True)
        for split in split_names
    }
    train_arrays = parse_split(raw, "train", "text_shared", require_labels=True)
    normalizer = FeatureNormalizer(normalize_text=False, clip_value=10.0).fit(
        train_arrays
    )
    arrays = {name: normalizer.transform(value) for name, value in arrays.items()}
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    max_length = int(cfg["data"].get("max_length", 128))
    context_window = int(cfg["data"].get("context_window", 0))
    eval_batch_size = int(cfg["training"].get("eval_batch_size", 8))
    loaders = {
        name: DataLoader(
            EncodedTextDataset(
                value,
                encode_split(tokenizer, value, max_length, context_window),
            ),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for name, value in arrays.items()
    }
    model = PretrainedTextFusionNet(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "best_epoch": checkpoint["epoch"],
        "best_selection_score": checkpoint["selection_score"],
    }
    for split, loader in loaders.items():
        split_metrics, predictions = evaluate(model, loader, device, amp)
        result[split] = split_metrics
        predictions.to_csv(
            output / f"{split}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    save_json(result, output / "final_metrics.json")
    print(result)


if __name__ == "__main__":
    main()
