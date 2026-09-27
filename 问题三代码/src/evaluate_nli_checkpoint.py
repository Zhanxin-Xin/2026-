"""Evaluate a validation-selected NLI semantic checkpoint, including locked test."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import attach_labels_from_excel, load_pickle, parse_split
from .train_nli_semantic_expert import (
    LabelPairDataset,
    NLILabelSemanticExpert,
    encode_label_pairs,
    evaluate,
)
from .utils import resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one locked NLI semantic checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg: dict[str, Any] = copy.deepcopy(checkpoint["config"])
    loaded = load_pickle(args.data)
    split_names = (
        ["test"]
        if args.test_only
        else ["train", "valid"] + (["test"] if args.evaluate_test else [])
    )
    raw = attach_labels_from_excel(
        {name: loaded[name] for name in split_names}, args.labels
    )
    arrays = {
        name: parse_split(raw, name, "text_shared", require_labels=True)
        for name in split_names
    }
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    hypotheses = [str(value) for value in model_cfg["hypotheses"]]
    max_length = int(cfg["data"].get("max_length", 128))
    context_window = int(cfg["data"].get("context_window", 0))
    batch_size = int(cfg["training"].get("eval_batch_size", 8))
    loaders = {
        name: DataLoader(
            LabelPairDataset(
                value,
                encode_label_pairs(
                    tokenizer, value, hypotheses, max_length, context_window
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for name, value in arrays.items()
    }
    model = NLILabelSemanticExpert(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "best_epoch": checkpoint["epoch"],
        "best_selection_score": checkpoint["selection_score"],
    }
    for split, loader in loaders.items():
        metrics, predictions = evaluate(model, loader, device, amp)
        result[split] = metrics
        predictions.to_csv(
            output / f"{split}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    save_json(result, output / "final_metrics.json")
    print(result)


if __name__ == "__main__":
    main()
