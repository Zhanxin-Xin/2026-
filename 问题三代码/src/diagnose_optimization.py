from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import (
    FeatureNormalizer,
    MultimodalDataset,
    attach_labels_from_excel,
    compute_class_weights,
    load_pickle,
    parse_split,
)
from .losses import MultitaskEvidenceLoss
from .metrics import compute_metrics
from .model import build_model
from .utils import apply_overrides, load_config, move_to_device, seed_everything


def _balanced_indices(labels: np.ndarray, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_index in range(3):
        candidates = np.flatnonzero(labels == class_index)
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {class_index} has {len(candidates)} rows, fewer than {per_class}"
            )
        selected.extend(rng.choice(candidates, per_class, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def _group_name(parameter_name: str) -> str:
    prefixes = (
        "encoders",
        "pair_projections",
        "context_adapters",
        "gates",
        "evidence_heads",
    )
    for prefix in prefixes:
        if parameter_name.startswith(prefix):
            return prefix
    return "other"


def _tensor_norms(values: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for name, value in values.items():
        group = _group_name(name)
        totals[group] = totals.get(group, 0.0) + float(value.float().square().sum())
    return {name: value**0.5 for name, value in totals.items()}


def _snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }


@torch.no_grad()
def _summarize(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    criterion: MultitaskEvidenceLoss,
) -> Dict[str, Any]:
    model.eval()
    outputs = model(batch)
    loss, components = criterion(
        outputs, batch["class_label"], batch["regression_label"]
    )
    probabilities = outputs["class_probabilities"].float().cpu().numpy()
    regression = outputs["regression"].float().cpu().numpy()
    metrics = compute_metrics(
        batch["class_label"].cpu().numpy(),
        probabilities,
        batch["regression_label"].cpu().numpy(),
        regression,
    )
    return {
        "loss": float(loss),
        "components": {name: float(value) for name, value in components.items()},
        "metrics": metrics,
        "logit_std": float(outputs["class_logits"].float().std()),
        "regression_std": float(outputs["regression"].float().std()),
        "mean_modality_gates": outputs["modality_gates"].float().mean(dim=0).cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify gradients and fixed-batch overfitting for a configured model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    cfg = apply_overrides(load_config(args.config), args.overrides)
    raw = attach_labels_from_excel(load_pickle(args.data), args.labels)
    train_arrays = parse_split(
        raw,
        "train",
        cfg["data"].get("mask_strategy", "text_shared"),
        require_labels=True,
    )
    normalizer = FeatureNormalizer(
        bool(cfg["data"].get("normalize_text", False)),
        clip_value=cfg["data"].get("normalization_clip", 10.0),
    ).fit(train_arrays)
    train_arrays = normalizer.transform(train_arrays)
    indices = _balanced_indices(train_arrays.class_labels, args.per_class, args.seed)
    loader = DataLoader(
        Subset(MultimodalDataset(train_arrays), indices),
        batch_size=len(indices),
        shuffle=False,
        num_workers=0,
    )
    batch = move_to_device(next(iter(loader)), device)

    model = build_model(cfg["model"]).to(device)
    class_weights = compute_class_weights(
        train_arrays.class_labels,
        power=float(cfg["training"].get("class_weight_power", 0.5)),
        max_weight=cfg["training"].get("class_weight_max", 3.0),
    ).to(device)
    criterion = MultitaskEvidenceLoss(
        cfg["loss"],
        class_weights=class_weights,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )

    initial = _summarize(model, batch, criterion)
    before = _snapshot(model)
    first_gradients: Dict[str, torch.Tensor] = {}
    trace = []
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, _ = criterion(
            outputs,
            batch["class_label"],
            batch["regression_label"],
            regularizer_scale=0.0,
        )
        loss.backward()
        if step == 1:
            first_gradients = {
                name: parameter.grad.detach().float().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            trace.append({"step": step, "train_loss": float(loss.detach())})

    final = _summarize(model, batch, criterion)
    after = _snapshot(model)
    parameter_deltas = {
        name: after[name] - before[name]
        for name in before
    }
    report = {
        "architecture": cfg["model"].get("architecture", "hafusion"),
        "samples": len(indices),
        "class_counts": np.bincount(
            train_arrays.class_labels[indices], minlength=3
        ).tolist(),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial": initial,
        "first_step_gradient_norms": _tensor_norms(first_gradients),
        "parameter_delta_norms": _tensor_norms(parameter_deltas),
        "trace": trace,
        "final": final,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
