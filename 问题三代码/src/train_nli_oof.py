from __future__ import annotations

"""Grouped fixed-epoch OOF training for the LoRA NLI semantic expert."""

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import attach_labels_from_excel, compute_class_weights, load_pickle, parse_split
from .metrics import compute_metrics
from .train_nli_semantic_expert import (
    LabelPairDataset,
    NLILabelSemanticExpert,
    compute_loss,
    encode_label_pairs,
    evaluate,
    move_batch,
)
from .train_pretrained_oof import subset_arrays, video_groups
from .utils import apply_overrides, load_config, resolve_device, save_json, seed_everything


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)


def _validate_hypotheses(model_cfg: Mapping[str, Any]) -> list[str]:
    hypotheses = [str(value) for value in model_cfg["hypotheses"]]
    semantic_mode = str(model_cfg.get("semantic_mode", "label_hypotheses"))
    expected = {
        "dual_polarity": 2,
        "label_hypotheses": 3,
        "neutral_residual_verbalizer": 4,
    }.get(semantic_mode)
    if expected is not None and len(hypotheses) != expected:
        raise ValueError(f"{semantic_mode} requires {expected} ordered hypotheses")
    if semantic_mode == "multi_verbalizer" and len(
        model_cfg.get("verbalizer_class_indices", [])
    ) != len(hypotheses):
        raise ValueError(
            "multi_verbalizer requires one verbalizer_class_indices entry per hypothesis"
        )
    return hypotheses


def _loader(
    dataset: LabelPairDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def train_fold(
    cfg: Mapping[str, Any],
    train_arrays: Any,
    heldout_arrays: Any,
    tokenizer: Any,
    device: torch.device,
    seed: int,
    fixed_epochs: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, float]]]:
    seed_everything(seed)
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    hypotheses = _validate_hypotheses(model_cfg)
    max_length = int(cfg["data"].get("max_length", 128))
    context_window = int(cfg["data"].get("context_window", 0))
    if bool(model_cfg.get("context_dual_view", False)) != (context_window > 0):
        raise ValueError(
            "model.context_dual_view must be true exactly when data.context_window > 0"
        )
    train_dataset = LabelPairDataset(
        train_arrays,
        encode_label_pairs(
            tokenizer,
            train_arrays,
            hypotheses,
            max_length,
            context_window=context_window,
        ),
    )
    heldout_dataset = LabelPairDataset(
        heldout_arrays,
        encode_label_pairs(
            tokenizer,
            heldout_arrays,
            hypotheses,
            max_length,
            context_window=context_window,
        ),
    )
    train_loader = _loader(
        train_dataset,
        int(training_cfg["batch_size"]),
        True,
        seed,
        device,
    )
    heldout_loader = _loader(
        heldout_dataset,
        int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        False,
        seed,
        device,
    )

    model = NLILabelSemanticExpert(model_cfg).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder_parameters = [
        parameter for parameter in trainable if id(parameter) in encoder_ids
    ]
    head_parameters = [
        parameter for parameter in trainable if id(parameter) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(training_cfg["adapter_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(training_cfg["head_learning_rate"]),
            },
        ],
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    accumulation = int(training_cfg.get("gradient_accumulation", 1))
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    total_updates = int(training_cfg["epochs"]) * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_updates * float(training_cfg.get("warmup_ratio", 0.1))
        ),
        num_training_steps=total_updates,
    )
    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    class_weights = compute_class_weights(
        train_arrays.class_labels,
        power=float(training_cfg.get("class_weight_power", 0.5)),
        max_weight=float(training_cfg.get("class_weight_max", 3.0)),
    ).to(device)
    history: list[dict[str, float]] = []
    for epoch in range(1, fixed_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, cpu_batch in enumerate(train_loader, start=1):
            batch = move_batch(cpu_batch, device)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(batch)
                loss, _ = compute_loss(outputs, batch, class_weights, cfg["loss"])
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite NLI OOF loss at epoch={epoch}, step={step}"
                )
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(training_cfg.get("grad_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_sum += float(loss.detach())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(1, len(train_loader)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    metrics, predictions = evaluate(model, heldout_loader, device, amp)
    del model, optimizer, scheduler, scaler, train_loader, heldout_loader
    del train_dataset, heldout_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, predictions, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped fixed-epoch NLI expert OOF")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fixed-epochs", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    if args.folds < 3:
        raise ValueError("At least three grouped folds are required")

    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg["seed"] = args.seed
    scheduled_epochs = int(cfg["training"]["epochs"])
    if not 1 <= args.fixed_epochs <= scheduled_epochs:
        raise ValueError("fixed-epochs must be within the configured schedule")
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "resolved_config.json")

    loaded = load_pickle(args.data)
    raw = attach_labels_from_excel({"train": loaded["train"]}, args.labels)
    arrays = parse_split(raw, "train", "text_shared", require_labels=True)
    groups = video_groups(arrays.ids)
    folds = list(
        StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed
        ).split(np.zeros(arrays.size), arrays.class_labels, groups)
    )
    model_cfg = cfg["model"]
    _validate_hypotheses(model_cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )

    started = time.time()
    chunks: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for fold, (fit_index, heldout_index) in enumerate(folds):
        overlap = set(groups[fit_index]).intersection(groups[heldout_index])
        if overlap:
            raise RuntimeError(f"Fold {fold} has video-group leakage")
        metrics, frame, history = train_fold(
            cfg,
            subset_arrays(arrays, fit_index),
            subset_arrays(arrays, heldout_index),
            tokenizer,
            device,
            args.seed + fold,
            args.fixed_epochs,
        )
        frame.insert(0, "source_index", heldout_index)
        frame.insert(1, "fold", fold)
        chunks.append(frame)
        report = {
            "fold": fold,
            "seed": args.seed + fold,
            "fit_samples": int(len(fit_index)),
            "heldout_samples": int(len(heldout_index)),
            "group_overlap": 0,
            "metrics": metrics,
            "history": history,
        }
        fold_reports.append(report)
        save_json(report, output / f"fold_{fold}_report.json")
        print(
            f"fold={fold} heldout={len(heldout_index)} "
            f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    oof = pd.concat(chunks, ignore_index=True).sort_values("source_index")
    if not np.array_equal(oof["source_index"].to_numpy(), np.arange(arrays.size)):
        raise RuntimeError("OOF rows do not cover every train sample exactly once")
    if not np.array_equal(oof["id"].astype(str).to_numpy(), arrays.ids.astype(str)):
        raise RuntimeError("OOF IDs do not preserve original train order")
    oof_metrics = compute_metrics(
        arrays.class_labels,
        oof.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64),
        arrays.regression_labels,
        oof["predicted_intensity"].to_numpy(np.float64),
    )
    oof.to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "scope": "train_only_grouped_oof_no_valid_or_test_access",
        "architecture": "lora_nli_neutral_hurdle_fixed_epoch_oof",
        "base_config": str(Path(args.config)),
        "pretrained_model": str(model_cfg["pretrained_model"]),
        "revision": model_cfg.get("revision"),
        "seed": args.seed,
        "folds": args.folds,
        "fixed_epochs": args.fixed_epochs,
        "checkpoint_selection_on_heldout_fold": False,
        "samples": arrays.size,
        "groups": int(len(set(groups))),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "oof_metrics": oof_metrics,
        "fold_reports": fold_reports,
        "artifact": "train_oof_predictions.csv",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
