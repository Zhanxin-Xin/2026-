from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import CLASS_NAMES, attach_labels_from_excel, load_pickle, parse_split
from .train_qwen_hierarchical_sentiment import PromptDataset, encode, evaluate
from .utils import resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a locked Qwen hierarchical adapter")
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.evaluate_test:
        raise ValueError("Test access is disabled until validation and multi-seed gates pass")

    run = Path(args.run)
    cfg: dict[str, Any] = json.loads(
        (run / "resolved_config.json").read_text(encoding="utf-8")
    )
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    best = max(history, key=lambda row: float(row["selection_score"]))
    model_cfg = cfg["model"]
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(run / "best_adapter", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=True,
        num_labels=3,
        id2label={index: name for index, name in enumerate(CLASS_NAMES)},
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, run / "best_adapter").to(device)

    loaded = load_pickle(args.data)
    raw = attach_labels_from_excel(
        {name: loaded[name] for name in ("train", "valid")}, args.labels
    )
    arrays = {
        name: parse_split(raw, name, "text_shared", require_labels=True)
        for name in ("train", "valid")
    }
    max_length = int(cfg["data"].get("max_length", 128))
    eval_batch_size = int(cfg["training"].get("eval_batch_size", 4))
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    report: dict[str, Any] = {
        "scope": "train_and_validation_only_no_test_access",
        "architecture": "qwen25_lora_verbalizer_hierarchical_sentiment",
        "model": str(model_cfg["pretrained_model"]),
        "revision": model_cfg.get("revision"),
        "license": "apache-2.0",
        "best_epoch": int(best["epoch"]),
        "best_selection_score": float(best["selection_score"]),
        "training_stopped_after_epoch": int(max(row["epoch"] for row in history)),
    }
    for name, split in arrays.items():
        dataset = PromptDataset(split, encode(tokenizer, split, max_length))
        loader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        metrics, predictions = evaluate(model, loader, device, amp)
        report[name] = metrics
        predictions.to_csv(
            run / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    save_json(report, run / "final_metrics.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
