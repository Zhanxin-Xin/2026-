#!/usr/bin/env python3
"""Train TASP-MSA with full, single-missing and double-missing views."""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from data.dataset import MOSEIAlignedDataset
from data.robustness_augmentation import RobustnessMissingTransform
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_metrics
from utils.seed import seed_everything
from tasp_msa.multiview import MultiViewMissingTransform
from tasp_msa.model import MODALITIES, TASPMsa


CONDITIONS = {
    "full": (),
    "text": ("text",),
    "audio": ("audio",),
    "vision": ("vision",),
    "text_audio": ("text", "audio"),
    "text_vision": ("text", "vision"),
    "audio_vision": ("audio", "vision"),
}


def make_inputs(
    batch: dict[str, Any], device: torch.device, view: str = "full"
) -> dict[str, torch.Tensor]:
    source = batch if view == "full" else batch[view]
    result: dict[str, torch.Tensor] = {}
    for modality in MODALITIES:
        valid = batch[f"valid_mask_{modality}"].to(device).bool()
        result[modality] = (
            batch[modality].to(device)
            if view == "full"
            else source[f"masked_{modality}"].to(device)
        )
        result[f"valid_mask_{modality}"] = valid
        result[f"missing_mask_{modality}"] = (
            torch.ones_like(valid)
            if view == "full"
            else source[f"missing_mask_{modality}"].to(device).bool()
        )
    return result


def make_evaluation_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for modality in MODALITIES:
        result[modality] = batch[f"masked_{modality}"].to(device)
        result[f"valid_mask_{modality}"] = batch[f"valid_mask_{modality}"].to(device).bool()
        result[f"missing_mask_{modality}"] = batch[f"missing_mask_{modality}"].to(device).bool()
    return result


def task_loss(output: dict[str, Any], labels: torch.Tensor, targets: torch.Tensor, cfg: dict[str, Any]):
    classification = F.cross_entropy(
        output["classification_logits"], labels,
        label_smoothing=float(cfg["label_smoothing"]),
    )
    regression = F.smooth_l1_loss(output["regression"], targets)
    return classification, regression


def proxy_loss(view: dict[str, Any], full: dict[str, Any]) -> torch.Tensor:
    anchor = view["regression"].sum() * 0
    total = anchor
    active = 0
    for modality in MODALITIES:
        selected = view["missing_ratios"][modality] > 0
        if not selected.any():
            continue
        active += 1
        mean = view["proxy_mean"][modality][selected]
        logvar = view["proxy_logvar"][modality][selected]
        target = full["shared_observed"][modality].detach()[selected]
        nll = 0.5 * ((mean - target).pow(2) * torch.exp(-logvar) + logvar).mean()
        semantic = F.smooth_l1_loss(mean, target)
        total = total + nll + semantic
    return total / max(active, 1)


def consistency_loss(view: dict[str, Any], full: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = F.kl_div(
        F.log_softmax(view["classification_logits"], -1),
        F.softmax(full["classification_logits"].detach(), -1),
        reduction="batchmean",
    ) + F.l1_loss(view["regression"], full["regression"].detach())
    representation = (
        1 - F.cosine_similarity(view["fused_features"], full["fused_features"].detach(), dim=-1)
    ).mean()
    return prediction, representation


def structural_losses(
    model: TASPMsa, full: dict[str, Any], batch_size: int, device: torch.device
):
    modality_targets = torch.cat([
        torch.full((batch_size,), index, device=device, dtype=torch.long)
        for index in range(3)
    ])
    modality = F.cross_entropy(full["private_modality_logits"], modality_targets)
    orthogonality = full["regression"].sum() * 0
    for name in MODALITIES:
        private = full["private_observed"][name]
        projected = model.private_orthogonal_projection(private)
        orthogonality = orthogonality + F.cosine_similarity(
            full["shared_observed"][name], projected, dim=-1
        ).pow(2).mean()
    orthogonality = orthogonality / 3
    alignment = sum(
        (1 - F.cosine_similarity(full["shared_observed"][left], full["shared_observed"][right], -1)).mean()
        for left, right in (("text", "audio"), ("text", "vision"), ("audio", "vision"))
    ) / 3
    return modality, orthogonality, alignment


def all_losses(
    model: TASPMsa,
    full: dict[str, Any],
    single: dict[str, Any],
    double: dict[str, Any],
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    device = full["classification_logits"].device
    labels = batch["classification_label"].to(device)
    targets = batch["regression_label"].to(device).unsqueeze(-1)
    task_components = [task_loss(output, labels, targets, cfg) for output in (full, single, double)]
    classification = sum(item[0] for item in task_components) / 3
    regression = sum(item[1] for item in task_components) / 3
    proxy = (proxy_loss(single, full) + proxy_loss(double, full)) / 2
    single_consistency, single_representation = consistency_loss(single, full)
    double_consistency, double_representation = consistency_loss(double, full)
    consistency = (single_consistency + double_consistency) / 2
    representation = (single_representation + double_representation) / 2
    modality, orthogonality, alignment = structural_losses(model, full, labels.shape[0], device)
    total = (
        classification
        + float(cfg["lambda_regression"]) * regression
        + float(cfg["lambda_proxy"]) * proxy
        + float(cfg["lambda_consistency"]) * consistency
        + float(cfg["lambda_representation"]) * representation
        + float(cfg["lambda_private_modality"]) * modality
        + float(cfg["lambda_orthogonality"]) * orthogonality
        + float(cfg["lambda_shared_alignment"]) * alignment
    )
    return {
        "total": total,
        "classification": classification,
        "regression": regression,
        "proxy": proxy,
        "consistency": consistency,
        "representation": representation,
        "private_modality": modality,
        "orthogonality": orthogonality,
        "shared_alignment": alignment,
    }


@torch.no_grad()
def evaluate_loader(model: TASPMsa, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    data = {key: [] for key in ("logits", "labels", "regression", "targets")}
    for batch in loader:
        output = model(**make_evaluation_inputs(batch, device))
        data["logits"].append(output["classification_logits"].cpu())
        data["labels"].append(batch["classification_label"])
        data["regression"].append(output["regression"].cpu())
        data["targets"].append(batch["regression_label"].unsqueeze(-1))
    return compute_metrics(
        torch.cat(data["logits"]), torch.cat(data["labels"]),
        torch.cat(data["regression"]), torch.cat(data["targets"]),
    )


def evaluate_all(
    model: TASPMsa, loaders: dict[str, DataLoader], device: torch.device
) -> tuple[dict[str, dict[str, float]], float]:
    metrics = {name: evaluate_loader(model, loader, device) for name, loader in loaders.items()}
    missing_accuracy = sum(metrics[name]["accuracy"] for name in CONDITIONS if name != "full") / 6
    selection_score = 0.5 * metrics["full"]["accuracy"] + 0.5 * missing_accuracy
    metrics["aggregate"] = {
        key: sum(metrics[name][key] for name in CONDITIONS if name != "full") / 6
        for key in ("accuracy", "macro_f1", "mae", "pearson")
    }
    return metrics, selection_score


def make_loader(dataset, batch_size: int, config: dict[str, Any], device: torch.device, shuffle=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=int(config["data"]["num_workers"]), pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(int(config["training"]["seed"])) if shuffle else None,
    )


def build_validation_loaders(
    config: dict[str, Any], split: str, device: torch.device, limit: int | None = None
) -> dict[str, DataLoader]:
    augmentation = config["augmentation"]
    result = {}
    for name, modalities in CONDITIONS.items():
        transform = RobustnessMissingTransform(
            modalities=modalities,
            missing_ratio=0.0 if name == "full" else float(augmentation["validation_ratio"]),
            position="random",
            seed=int(augmentation["validation_seed"]),
        )
        dataset = MOSEIAlignedDataset(config["data"]["path"], split, transform)
        if limit:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
        result[name] = make_loader(dataset, int(config["training"]["batch_size"]), config, device)
    return result


def write_history(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--anchor-mode", choices=("text", "audio", "vision", "symmetric"),
        help="train a parameter-matched anchor-selection ablation",
    )
    parser.add_argument(
        "--variant",
        choices=("full", "no_proxy", "no_shared_specific", "no_reliability", "no_hierarchical", "no_consistency"),
        default="full",
    )
    args = parser.parse_args()
    # Multiple ablation jobs may run on separate GPUs. Limiting host threads
    # prevents four small GPU models from oversubscribing all CPU cores during
    # tensor collation and Transformer dispatch.
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.anchor_mode:
        config["model"]["anchor_mode"] = args.anchor_mode
        config["anchor_ablation"] = True
        if not args.run_name:
            args.run_name = f"anchor_{args.anchor_mode}"
    variant = args.variant
    if variant == "no_proxy":
        config["model"]["use_proxy"] = False
        config["loss"]["lambda_proxy"] = 0.0
    elif variant == "no_shared_specific":
        config["model"]["use_shared_specific"] = False
        config["loss"]["lambda_private_modality"] = 0.0
        config["loss"]["lambda_orthogonality"] = 0.0
    elif variant == "no_reliability":
        config["model"]["use_reliability"] = False
    elif variant == "no_hierarchical":
        config["model"]["use_hierarchical"] = False
    elif variant == "no_consistency":
        config["loss"]["lambda_consistency"] = 0.0
        config["loss"]["lambda_representation"] = 0.0
    config["ablation_variant"] = variant
    if variant != "full" and not args.run_name:
        args.run_name = variant
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.run_name:
        config["output"]["best_checkpoint"] = f"checkpoints/{args.run_name}_best.pth"
        config["output"]["last_checkpoint"] = f"checkpoints/{args.run_name}_last.pth"
        config["output"]["history_path"] = f"results/{args.run_name}_history.csv"
        config["output"]["log_path"] = f"logs/{args.run_name}.log"
    if args.smoke_test:
        config["training"]["epochs"] = 2
        args.train_limit = args.train_limit or 48
        args.valid_limit = args.valid_limit or 24
        for key in ("best_checkpoint", "last_checkpoint"):
            config["output"][key] = config["output"][key].replace(".pth", "_smoke.pth")
        config["output"]["history_path"] = "results/smoke_history.csv"
        config["output"]["log_path"] = "logs/smoke.log"
    seed_everything(int(config["training"]["seed"]))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    amp = bool(config["training"]["amp"] and device.type == "cuda")
    log_path = Path(config["output"]["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tasp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, mode="w")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    transform = MultiViewMissingTransform(
        tuple(config["augmentation"]["train_ratio_range"]),
        config["augmentation"]["double_interval_mode"],
    )
    train_dataset = MOSEIAlignedDataset(config["data"]["path"], "train", transform)
    if args.train_limit:
        train_dataset = Subset(train_dataset, range(min(args.train_limit, len(train_dataset))))
    train_loader = make_loader(
        train_dataset, int(config["training"]["batch_size"]), config, device, shuffle=True
    )
    valid_loaders = build_validation_loaders(config, "valid", device, args.valid_limit)
    model = TASPMsa(**config["model"]).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = AdamW(
        model.parameters(), lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    decay = float(config["training"]["ema_decay"])
    logger.info(
        "device=%s params=%d train=%d valid=%d patience=%d",
        device, sum(p.numel() for p in model.parameters()), len(train_dataset),
        len(valid_loaders["full"].dataset), int(config["training"]["early_stopping"]),
    )
    best = -1.0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        sums: dict[str, float] = {}
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            context = torch.amp.autocast("cuda") if amp else nullcontext()
            with context:
                full = model(**make_inputs(batch, device, "full"))
                single = model(**make_inputs(batch, device, "single_view"), sample_proxy=True)
                double = model(**make_inputs(batch, device, "double_view"), sample_proxy=True)
                losses = all_losses(model, full, single, double, batch, config["loss"])
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1 - decay)
                for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
                    ema_buffer.copy_(buffer)
            count = batch["classification_label"].shape[0]
            seen += count
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * count
        metrics, score = evaluate_all(ema, valid_loaders, device)
        scheduler.step(score)
        row: dict[str, Any] = {
            "epoch": epoch,
            **{f"train_{key}": value / seen for key, value in sums.items()},
            "selection_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for condition, values in metrics.items():
            for key, value in values.items():
                row[f"{condition}_valid_{key}"] = value
        history.append(row)
        write_history(config["output"]["history_path"], history)
        improved = score > best + 1e-8
        stale = 0 if improved else stale + 1
        best = max(best, score)
        state = {
            "model": ema.state_dict(), "train_model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_metric": best, "metrics": metrics,
            "config": config, "model_family": "TASP-MSA",
            "ablation_variant": variant,
            "selection_protocol": "0.5*full_accuracy + 0.5*mean_six_missing_accuracy",
        }
        save_checkpoint(state, config["output"]["last_checkpoint"])
        if improved:
            save_checkpoint(state, config["output"]["best_checkpoint"])
        logger.info(
            "Epoch %03d loss %.5f score %.4f Full Acc %.4f F1 %.4f MissingMean Acc %.4f F1 %.4f stale %d/%d",
            epoch, row["train_total"], score, metrics["full"]["accuracy"],
            metrics["full"]["macro_f1"], metrics["aggregate"]["accuracy"],
            metrics["aggregate"]["macro_f1"], stale, int(config["training"]["early_stopping"]),
        )
        if stale >= int(config["training"]["early_stopping"]):
            break
    logger.info("Finished best_selection_score=%.5f", best)


if __name__ == "__main__":
    main()
