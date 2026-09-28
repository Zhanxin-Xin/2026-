#!/usr/bin/env python3
"""Evaluate separately trained anchor-selection variants on Attachment 2 test."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import MOSEIAlignedDataset
from data.missing_augmentation import DeterministicMissingAugmentation
from data.robustness_augmentation import RobustnessMissingTransform
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_metrics
from tasp_msa.model import MODALITIES, TASPMsa

VARIANTS = ("text", "audio", "vision", "symmetric")
DISPLAY = {
    "text": "Text Anchor", "audio": "Audio Anchor",
    "vision": "Vision Anchor", "symmetric": "Symmetric",
}
METRICS = ("accuracy", "macro_f1", "mae", "pearson")


@torch.inference_mode()
def evaluate(model: TASPMsa, loader: DataLoader, device: torch.device) -> dict[str, float]:
    logits, labels, regression, targets = [], [], [], []
    for batch in loader:
        kwargs: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            kwargs[modality] = batch[f"masked_{modality}"].to(device)
            kwargs[f"valid_mask_{modality}"] = batch[f"valid_mask_{modality}"].to(device).bool()
            kwargs[f"missing_mask_{modality}"] = batch[f"missing_mask_{modality}"].to(device).bool()
        output = model(**kwargs)
        logits.append(output["classification_logits"].cpu())
        labels.append(batch["classification_label"])
        regression.append(output["regression"].cpu())
        targets.append(batch["regression_label"].unsqueeze(-1))
    return compute_metrics(
        torch.cat(logits), torch.cat(labels), torch.cat(regression), torch.cat(targets)
    )


def loader(data_path: str, transform, batch_size: int, device: torch.device) -> DataLoader:
    dataset = MOSEIAlignedDataset(data_path, "test", transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )


def plot(rows: list[dict[str, Any]], metric: str) -> None:
    labels = [row["variant"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.8, 4.8))
    axis.bar(x - width / 2, [row[f"full_{metric}"] for row in rows], width, label="Full")
    axis.bar(x + width / 2, [row[f"mixed_{metric}"] for row in rows], width,
             label="30% Mixed Missing")
    axis.set_xticks(x, labels)
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(f"Anchor Selection Ablation: {metric.replace('_', ' ').title()}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    destination = ROOT / "figures" / f"anchor_ablation_{metric}"
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1042)
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    checkpoints = {
        mode: ROOT / "checkpoints" / f"anchor_{mode}_best.pth" for mode in VARIANTS
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing retrained checkpoints: " + ", ".join(missing))
    first = load_checkpoint(checkpoints["text"], map_location="cpu")
    data_path = first["config"]["data"]["path"]
    loaders = {
        "full": loader(data_path, RobustnessMissingTransform((), 0.0, "random", 0),
                       args.batch_size, device),
        "mixed": loader(
            data_path,
            DeterministicMissingAugmentation(
                probabilities=(0.0, 0.625, 0.375), ratio_range=(0.30, 0.30),
                double_interval_mode="random", seed=args.seed,
            ), args.batch_size, device,
        ),
    }
    for modality in MODALITIES:
        loaders[f"{modality}_missing"] = loader(
            data_path, RobustnessMissingTransform((modality,), 0.30, "random", args.seed),
            args.batch_size, device,
        )

    rows: list[dict[str, Any]] = []
    for mode in VARIANTS:
        checkpoint = load_checkpoint(checkpoints[mode], map_location=device)
        model = TASPMsa(**checkpoint["config"]["model"]).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        scores = {name: evaluate(model, item, device) for name, item in loaders.items()}
        row: dict[str, Any] = {
            "variant": DISPLAY[mode], "anchor_mode": mode,
            "checkpoint": str(checkpoints[mode].relative_to(ROOT)),
            "best_epoch": checkpoint["epoch"],
            "valid_selection_score": checkpoint["best_metric"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "evaluation_split": "test", "mixed_missing_ratio": 0.30,
            "mixed_seed": args.seed,
        }
        for condition, values in scores.items():
            for metric, value in values.items():
                row[f"{condition}_{metric}"] = value
        anchor_condition = f"{mode}_missing" if mode != "symmetric" else None
        if anchor_condition:
            for metric in METRICS:
                row[f"anchor_missing_{metric}"] = scores[anchor_condition][metric]
        else:
            for metric in METRICS:
                row[f"anchor_missing_{metric}"] = float(np.mean([
                    scores[f"{m}_missing"][metric] for m in MODALITIES
                ]))
        rows.append(row)

    destination = ROOT / "results" / "anchor_ablation.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for metric in METRICS:
        plot(rows, metric)
    summary = {
        "evaluation_split": "test", "test_used_for_selection": False,
        "best_full_accuracy": max(rows, key=lambda row: row["full_accuracy"])["variant"],
        "best_mixed_accuracy": max(rows, key=lambda row: row["mixed_accuracy"])["variant"],
        "rows": rows,
    }
    output = ROOT / "results" / "anchor_ablation.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
