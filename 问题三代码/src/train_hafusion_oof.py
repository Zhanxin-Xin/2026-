from __future__ import annotations

"""Fixed-epoch, video-grouped OOF generation for the native HAFusion model."""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .data import (
    FeatureNormalizer,
    MultimodalDataset,
    SplitArrays,
    attach_labels_from_excel,
    compute_class_weights,
    load_pickle,
    parse_split,
)
from .losses import MultitaskEvidenceLoss
from .metrics import compute_metrics
from .model import build_model
from .train import evaluate, make_loader, train_one_epoch
from .train_pretrained_oof import subset_arrays, video_groups
from .utils import (
    ModelEMA,
    apply_overrides,
    cosine_schedule_with_warmup,
    load_config,
    resolve_device,
    save_json,
    seed_everything,
)


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)


def train_fold(
    cfg: Mapping[str, Any],
    train_arrays: SplitArrays,
    heldout_arrays: SplitArrays,
    device: torch.device,
    seed: int,
    fixed_epochs: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, float]]]:
    seed_everything(seed)
    normalizer = FeatureNormalizer(
        bool(cfg["data"].get("normalize_text", False)),
        clip_value=cfg["data"].get("normalization_clip", 10.0),
    ).fit(train_arrays)
    train_arrays = normalizer.transform(train_arrays)
    heldout_arrays = normalizer.transform(heldout_arrays)
    train_cfg = cfg["training"]
    loader_args = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "seed": seed,
    }
    train_loader = make_loader(
        MultimodalDataset(train_arrays), shuffle=True, **loader_args
    )
    heldout_loader = make_loader(
        MultimodalDataset(heldout_arrays), shuffle=False, **loader_args
    )
    model = build_model(cfg["model"]).to(device)
    class_weights = compute_class_weights(
        train_arrays.class_labels,
        power=float(train_cfg.get("class_weight_power", 0.5)),
        max_weight=(
            None
            if train_cfg.get("class_weight_max") is None
            else float(train_cfg["class_weight_max"])
        ),
    ).to(device)
    criterion = MultitaskEvidenceLoss(
        cfg["loss"],
        class_weights=class_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        distillation_temperature=float(train_cfg.get("distillation_temperature", 2.0)),
        focal_gamma=float(train_cfg.get("focal_gamma", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    total_steps = int(train_cfg["epochs"]) * max(1, len(train_loader))
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=int(total_steps * float(train_cfg.get("warmup_ratio", 0.0))),
        total_steps=total_steps,
    )
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp,
        init_scale=float(train_cfg.get("amp_init_scale", 16384.0)),
        growth_interval=int(train_cfg.get("amp_growth_interval", 2000)),
    )
    ema = ModelEMA(model, decay=float(train_cfg.get("ema_decay", 0.995)))
    history: list[dict[str, float]] = []
    for epoch in range(1, fixed_epochs + 1):
        stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            train_cfg,
            epoch,
            ema,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(stats["loss"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "amp_skipped_steps": float(stats.get("amp_skipped_steps", 0.0)),
            }
        )
    metrics, predictions = evaluate(ema.module, heldout_loader, criterion, device, amp)
    del model, ema, criterion, optimizer, scheduler, scaler
    del train_loader, heldout_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, predictions, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped fixed-epoch HAFusion OOF")
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
    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg["seed"] = args.seed
    if not 1 <= args.fixed_epochs <= int(cfg["training"]["epochs"]):
        raise ValueError("fixed-epochs must lie inside the configured schedule")
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "resolved_config.json")
    loaded = load_pickle(args.data)
    raw = attach_labels_from_excel({"train": loaded["train"]}, args.labels)
    arrays = parse_split(raw, "train", cfg["data"].get("mask_strategy", "text_shared"), True)
    groups = video_groups(arrays.ids)
    split = list(
        StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed
        ).split(np.zeros(arrays.size), arrays.class_labels, groups)
    )
    chunks: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    started = time.time()
    for fold, (fit_index, heldout_index) in enumerate(split):
        overlap = set(groups[fit_index]).intersection(groups[heldout_index])
        if overlap:
            raise RuntimeError(f"Fold {fold} has video-group leakage")
        fold_metrics, frame, history = train_fold(
            cfg,
            subset_arrays(arrays, fit_index),
            subset_arrays(arrays, heldout_index),
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
            "metrics": fold_metrics,
            "history": history,
        }
        reports.append(report)
        save_json(report, output / f"fold_{fold}_report.json")
        print(
            f"fold={fold} heldout={len(heldout_index)} "
            f"accuracy={fold_metrics['accuracy']:.4f} "
            f"macro_f1={fold_metrics['macro_f1']:.4f}",
            flush=True,
        )
    oof = pd.concat(chunks, ignore_index=True).sort_values("source_index")
    if not np.array_equal(oof["source_index"].to_numpy(), np.arange(arrays.size)):
        raise RuntimeError("OOF coverage is incomplete or duplicated")
    probability = oof.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    metrics = compute_metrics(
        arrays.class_labels,
        probability,
        arrays.regression_labels,
        oof["predicted_intensity"].to_numpy(np.float64),
    )
    oof.to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "scope": "train_only_grouped_oof_no_valid_or_test_access",
        "architecture": "native_hafusion_fixed_epoch_oof",
        "seed": args.seed,
        "folds": args.folds,
        "fixed_epochs": args.fixed_epochs,
        "checkpoint_selection_on_heldout_fold": False,
        "groups": int(len(set(groups))),
        "samples": arrays.size,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "oof_metrics": metrics,
        "fold_reports": reports,
        "artifact": "train_oof_predictions.csv",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
