from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    CLASS_NAMES,
    MODALITIES,
    FeatureNormalizer,
    MultimodalDataset,
    augment_batch,
    compute_class_weights,
    load_pickle,
    attach_labels_from_excel,
    find_sibling_label_excel,
    parse_split,
)
from .losses import MultitaskEvidenceLoss, ablation_modality_importance
from .metrics import compute_metrics, selection_score
from .model import HAFusionNet, build_model, count_parameters
from .q3_reporting import export_labeled_prediction_views
from .utils import (
    AverageMeter,
    ModelEMA,
    apply_overrides,
    atomic_torch_save,
    cosine_schedule_with_warmup,
    load_config,
    make_logger,
    move_to_device,
    resolve_device,
    save_json,
    seed_everything,
    worker_init_fn,
)


def make_loader(
    dataset: MultimodalDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle and len(dataset) >= batch_size,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def compute_faithfulness_target(
    model: HAFusionNet, batch: Mapping[str, Any]
) -> Tensor:
    was_training = model.training
    model.eval()
    full = model(batch)
    ablated = []
    for modality in MODALITIES:
        counterfactual = dict(batch)
        counterfactual[modality] = torch.zeros_like(batch[modality])
        ablated.append(model(counterfactual))
    target = ablation_modality_importance(full, ablated).detach()
    model.train(was_training)
    return target


def train_one_epoch(
    model: HAFusionNet,
    loader: DataLoader,
    criterion: MultitaskEvidenceLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: Mapping[str, Any],
    epoch: int,
    ema: Optional[ModelEMA] = None,
) -> Dict[str, float]:
    model.train()
    meters: Dict[str, AverageMeter] = {"loss": AverageMeter()}
    skipped_steps = 0
    global_frequency = int(cfg.get("faithfulness_every", 0))
    amp_enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
    evidence_start = int(cfg.get("evidence_start_epoch", 1))
    faithfulness_start = int(cfg.get("faithfulness_start_epoch", evidence_start))
    distillation_start = int(cfg.get("distillation_start_epoch", evidence_start))
    ramp_epochs = max(1, int(cfg.get("curriculum_ramp_epochs", 1)))
    if epoch < evidence_start:
        regularizer_scale = 0.0
    else:
        regularizer_scale = min(1.0, (epoch - evidence_start + 1) / ramp_epochs)
    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    for step, cpu_batch in enumerate(progress):
        clean_batch = move_to_device(cpu_batch, device)
        batch = augment_batch(
            clean_batch,
            span_probability=float(cfg.get("span_mask_probability", 0.0)),
            max_ratio=float(cfg.get("span_mask_max_ratio", 0.0)),
            noise_std=float(cfg.get("feature_noise_std", 0.0)),
        )
        faith_target = None
        if (
            epoch >= faithfulness_start
            and global_frequency > 0
            and step % global_frequency == 0
        ):
            faith_target = compute_faithfulness_target(model, batch)

        teacher_outputs = None
        if ema is not None and epoch >= distillation_start:
            ema.module.eval()
            with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", enabled=amp_enabled
            ):
                teacher_outputs = ema.module(clean_batch)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(batch)
            loss, components = criterion(
                outputs,
                batch["class_label"],
                batch["regression_label"],
                faithfulness_target=faith_target,
                teacher_outputs=teacher_outputs,
                regularizer_scale=regularizer_scale,
            )
        if not torch.isfinite(loss):
            component_values = {
                name: float(value.detach().float().cpu())
                for name, value in components.items()
            }
            bad_outputs = [
                name
                for name, value in outputs.items()
                if torch.is_tensor(value) and not torch.isfinite(value).all()
            ]
            sample_ids = [str(value) for value in cpu_batch.get("id", [])[:5]]
            raise FloatingPointError(
                "Non-finite training loss detected; "
                f"epoch={epoch}, step={step}, sample_ids={sample_ids}, "
                f"components={component_values}, nonfinite_outputs={bad_outputs}. "
                "Do not continue this run."
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.get("grad_clip", 1.0))
        )
        if not torch.isfinite(grad_norm):
            if not amp_enabled:
                raise FloatingPointError(
                    f"Non-finite gradient detected with AMP disabled at epoch={epoch}, step={step}"
                )
            # GradScaler records the overflow during unscale_ and scaler.step
            # safely skips this optimizer update.  Do not update EMA or the LR
            # schedule for a skipped step.
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            skipped_steps += 1
            if skipped_steps > int(cfg.get("max_amp_skipped_steps_per_epoch", 8)):
                raise FloatingPointError(
                    f"Too many AMP gradient overflows in epoch {epoch}: {skipped_steps}"
                )
            progress.set_postfix(
                avg_loss=f"{meters['loss'].average:.4f}", amp_skips=skipped_steps
            )
            continue
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        scheduler.step()

        n = batch["text"].size(0)
        meters["loss"].update(float(loss.detach()), n)
        for name, value in components.items():
            meters.setdefault(name, AverageMeter()).update(float(value.detach()), n)
        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            avg=f"{meters['loss'].average:.4f}",
        )
    meters["regularizer_scale"] = AverageMeter()
    meters["regularizer_scale"].update(regularizer_scale)
    result = {name: meter.average for name, meter in meters.items()}
    result["amp_skipped_steps"] = float(skipped_steps)
    return result


@torch.no_grad()
def evaluate(
    model: HAFusionNet,
    loader: DataLoader,
    criterion: Optional[MultitaskEvidenceLoss],
    device: torch.device,
    amp: bool,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()
    losses = AverageMeter()
    all_probs, all_reg, all_cls_target, all_reg_target, all_ids = [], [], [], [], []
    amp_enabled = amp and device.type == "cuda"
    for cpu_batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_to_device(cpu_batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(batch)
            if criterion is not None and "class_label" in batch:
                loss, _ = criterion(
                    outputs, batch["class_label"], batch["regression_label"]
                )
                losses.update(float(loss), batch["text"].size(0))
        all_probs.append(outputs["class_probabilities"].cpu().numpy())
        all_reg.append(outputs["regression"].cpu().numpy())
        all_ids.extend(cpu_batch["id"])
        if "class_label" in batch:
            all_cls_target.append(batch["class_label"].cpu().numpy())
            all_reg_target.append(batch["regression_label"].cpu().numpy())

    probabilities = np.concatenate(all_probs)
    regression = np.concatenate(all_reg)
    frame = pd.DataFrame(
        {
            "id": [str(x) for x in all_ids],
            "predicted_label": [CLASS_NAMES[x] for x in probabilities.argmax(axis=1)],
            "negative_probability": probabilities[:, 0],
            "neutral_probability": probabilities[:, 1],
            "positive_probability": probabilities[:, 2],
            "predicted_intensity": regression,
        }
    )
    if all_cls_target:
        cls_target = np.concatenate(all_cls_target)
        reg_target = np.concatenate(all_reg_target)
        frame["true_label"] = [CLASS_NAMES[x] for x in cls_target]
        frame["true_intensity"] = reg_target
        metrics = compute_metrics(cls_target, probabilities, reg_target, regression)
        metrics["loss"] = losses.average
    else:
        metrics = {}
    return metrics, frame


def validate_dimensions(cfg: Dict[str, Any], train_arrays) -> None:
    actual = {m: int(train_arrays.features[m].shape[-1]) for m in MODALITIES}
    configured = {m: int(cfg["model"]["input_dims"][m]) for m in MODALITIES}
    if actual != configured:
        raise ValueError(f"Feature dimensions differ: config={configured}, data={actual}")
    lengths = {train_arrays.features[m].shape[1] for m in MODALITIES}
    if len(lengths) != 1:
        raise ValueError("This model requires aligned features with equal sequence lengths")


def audit_labeled_split(name: str, arrays, logger) -> None:
    """Fail early when labels cannot represent the competition definition."""
    class_labels = np.asarray(arrays.class_labels, dtype=np.int64)
    regression_labels = np.asarray(arrays.regression_labels, dtype=np.float32)
    if not np.isfinite(regression_labels).all():
        raise ValueError(f"Split '{name}' contains NaN/Inf regression labels")
    expected = np.where(
        regression_labels < 0.0,
        0,
        np.where(regression_labels > 0.0, 2, 1),
    )
    mismatches = np.flatnonzero(class_labels != expected)
    if len(mismatches):
        examples = [
            {
                "id": str(arrays.ids[index]),
                "class": int(class_labels[index]),
                "intensity": float(regression_labels[index]),
            }
            for index in mismatches[:5]
        ]
        raise ValueError(
            f"Split '{name}' has {len(mismatches)} classification/regression label "
            f"mismatches; examples={examples}"
        )
    unique_ids = np.unique(arrays.ids.astype(str))
    if len(unique_ids) != arrays.size:
        raise ValueError(
            f"Split '{name}' contains {arrays.size - len(unique_ids)} duplicate sample IDs"
        )
    counts = np.bincount(class_labels, minlength=3)
    logger.info(
        "Label audit %s | counts[N,Neu,P]=%s | intensity min=%.3f mean=%.3f "
        "std=%.3f max=%.3f | label mismatches=0",
        name,
        counts.tolist(),
        float(regression_labels.min()),
        float(regression_labels.mean()),
        float(regression_labels.std()),
        float(regression_labels.max()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C3-HAFusion for problem 3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None, help="Override data.pkl_path")
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional label.xlsx; automatically discovered beside the feature PKL",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config with dotted key=value; may be repeated",
    )
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.data:
        cfg["data"]["pkl_path"] = args.data
    if args.seed is not None:
        cfg["seed"] = args.seed
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(output_dir)
    save_json(cfg, output_dir / "resolved_config.json")
    seed_everything(int(cfg["seed"]))
    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    raw = load_pickle(cfg["data"]["pkl_path"])
    label_path = Path(args.labels) if args.labels else find_sibling_label_excel(
        cfg["data"]["pkl_path"]
    )
    if label_path is not None:
        raw = attach_labels_from_excel(raw, label_path)
        logger.info("Label spreadsheet: %s", label_path)
    mask_strategy = cfg["data"].get("mask_strategy", "text_shared")
    train_arrays = parse_split(raw, "train", mask_strategy, require_labels=True)
    valid_arrays = parse_split(raw, "valid", mask_strategy, require_labels=True)
    test_arrays = (
        parse_split(raw, "test", mask_strategy, require_labels=True) if "test" in raw else None
    )
    audit_labeled_split("train", train_arrays, logger)
    audit_labeled_split("valid", valid_arrays, logger)
    if test_arrays is not None:
        audit_labeled_split("test", test_arrays, logger)
    validate_dimensions(cfg, train_arrays)
    normalizer = FeatureNormalizer(
        bool(cfg["data"].get("normalize_text", False)),
        clip_value=cfg["data"].get("normalization_clip", 10.0),
    ).fit(train_arrays)
    train_arrays = normalizer.transform(train_arrays)
    valid_arrays = normalizer.transform(valid_arrays)
    if test_arrays is not None:
        test_arrays = normalizer.transform(test_arrays)
    logger.info(
        "Samples: train=%d valid=%d test=%s",
        train_arrays.size,
        valid_arrays.size,
        "none" if test_arrays is None else str(test_arrays.size),
    )

    train_cfg = cfg["training"]
    loader_args = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(cfg["data"].get("num_workers", 0)),
        "pin_memory": bool(cfg["data"].get("pin_memory", True)) and device.type == "cuda",
        "seed": int(cfg["seed"]),
    }
    train_loader = make_loader(MultimodalDataset(train_arrays), shuffle=True, **loader_args)
    train_eval_loader = make_loader(
        MultimodalDataset(train_arrays), shuffle=False, **loader_args
    )
    valid_loader = make_loader(MultimodalDataset(valid_arrays), shuffle=False, **loader_args)
    test_loader = (
        make_loader(MultimodalDataset(test_arrays), shuffle=False, **loader_args)
        if test_arrays is not None
        else None
    )

    model = build_model(cfg["model"]).to(device)
    logger.info("Trainable parameters: %s", f"{count_parameters(model):,}")
    class_weights = compute_class_weights(
        train_arrays.class_labels,
        power=float(train_cfg.get("class_weight_power", 0.5)),
        max_weight=(
            None
            if train_cfg.get("class_weight_max") is None
            else float(train_cfg.get("class_weight_max"))
        ),
    ).to(device)
    logger.info("Class weights [N,Neu,P]: %s", class_weights.tolist())
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
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=float(train_cfg.get("amp_init_scale", 16384.0)),
        growth_interval=int(train_cfg.get("amp_growth_interval", 2000)),
    )
    ema = ModelEMA(model, decay=float(train_cfg.get("ema_decay", 0.995)))

    best_score = -float("inf")
    patience = 0
    history = []
    checkpoint_path = output_dir / "best_model.pt"
    started = time.time()
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_stats = train_one_epoch(
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
        valid_metrics, valid_predictions = evaluate(
            ema.module, valid_loader, criterion, device, bool(train_cfg.get("amp", True))
        )
        score = selection_score(valid_metrics, cfg["selection"])
        record = {
            "epoch": epoch,
            "train": train_stats,
            "valid": valid_metrics,
            "selection_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        save_json(history, output_dir / "history.json")
        logger.info(
            "Epoch %03d | train_loss %.4f | score %.5f | Acc %.4f F1 %.4f "
            "MAE %.4f r %.4f | lr %.3e | amp_skips %d",
            epoch,
            train_stats["loss"],
            score,
            valid_metrics["accuracy"],
            valid_metrics["macro_f1"],
            valid_metrics["mae"],
            valid_metrics["pearson"],
            optimizer.param_groups[0]["lr"],
            int(train_stats.get("amp_skipped_steps", 0)),
        )

        if score > best_score + float(train_cfg.get("min_delta", 0.0)):
            best_score = score
            patience = 0
            atomic_torch_save(
                {
                    "model_state": ema.module.state_dict(),
                    "config": copy.deepcopy(cfg),
                    "normalizer": normalizer.state_dict(),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "selection_score": score,
                    "class_names": CLASS_NAMES,
                },
                checkpoint_path,
            )
            valid_predictions.to_csv(
                output_dir / "best_valid_predictions.csv", index=False, encoding="utf-8-sig"
            )
        else:
            patience += 1
            if patience >= int(train_cfg["early_stopping_patience"]):
                logger.info("Early stopping at epoch %d", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_report: Dict[str, Any] = {
        "best_epoch": checkpoint["epoch"],
        "best_selection_score": checkpoint["selection_score"],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    split_loaders = {"train": train_eval_loader, "valid": valid_loader}
    if test_loader is not None:
        split_loaders["test"] = test_loader
    metric_rows = []
    for split_name, split_loader in split_loaders.items():
        split_metrics, split_predictions = evaluate(
            model,
            split_loader,
            criterion,
            device,
            bool(train_cfg.get("amp", True)),
        )
        final_report[split_name] = split_metrics
        enriched = export_labeled_prediction_views(
            split_predictions, output_dir, split_name
        )
        enriched.to_csv(
            output_dir / f"{split_name}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if split_name == "valid":
            enriched.to_csv(
                output_dir / "best_valid_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
        metric_row: Dict[str, Any] = {"split": split_name}
        for metric_name in (
            "accuracy",
            "f1",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "mae",
            "rmse",
            "regression_bias",
            "pearson",
        ):
            if metric_name in split_metrics:
                metric_row[metric_name] = split_metrics[metric_name]
        metric_rows.append(metric_row)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "all_split_metrics.csv", index=False, encoding="utf-8-sig"
    )
    save_json(final_report, output_dir / "final_metrics.json")
    logger.info("Finished. Best checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
