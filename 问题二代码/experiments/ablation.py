#!/usr/bin/env python3
"""Evaluate retrained TASP-MSA ablations on the locked test protocols."""
from __future__ import annotations

import csv
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

VARIANTS = (
    ("Full TASP-MSA", ROOT / "checkpoints/best_model.pth"),
    ("w/o Semantic Proxy", ROOT / "checkpoints/no_proxy_best.pth"),
    ("w/o Shared-Specific", ROOT / "checkpoints/no_shared_specific_best.pth"),
    ("w/o Reliability Gate", ROOT / "checkpoints/no_reliability_best.pth"),
    ("w/o Hierarchical Head", ROOT / "checkpoints/no_hierarchical_best.pth"),
    ("w/o View Consistency", ROOT / "checkpoints/no_consistency_best.pth"),
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(model: TASPMsa, loader: DataLoader, device: torch.device):
    logits, labels, regression, targets = [], [], [], []
    for batch in loader:
        kwargs = {}
        for modality in MODALITIES:
            kwargs[modality] = batch[f"masked_{modality}"].to(device)
            kwargs[f"valid_mask_{modality}"] = batch[f"valid_mask_{modality}"].to(device).bool()
            kwargs[f"missing_mask_{modality}"] = batch[f"missing_mask_{modality}"].to(device).bool()
        output = model(**kwargs)
        logits.append(output["classification_logits"].cpu())
        labels.append(batch["classification_label"])
        regression.append(output["regression"].cpu())
        targets.append(batch["regression_label"].unsqueeze(-1))
    return compute_metrics(torch.cat(logits), torch.cat(labels), torch.cat(regression), torch.cat(targets))


def make_loader(dataset, device):
    return DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0,
                      pin_memory=device.type == "cuda")


def plot(rows: list[dict[str, Any]], metric: str) -> None:
    labels = [row["variant"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(x - width / 2, [row[f"full_{metric}"] for row in rows], width, label="Full")
    axis.bar(x + width / 2, [row[f"missing_{metric}"] for row in rows], width,
             label="30% Mixed Missing")
    axis.set_xticks(x, labels, rotation=22, ha="right")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(f"TASP-MSA Ablation: {metric.replace('_', ' ').title()}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    destination = ROOT / "figures" / f"ablation_{metric}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_checkpoint(ROOT / "checkpoints/best_model.pth", map_location="cpu")
    data_path = base["config"]["data"]["path"]
    full_dataset = MOSEIAlignedDataset(
        data_path, "test", RobustnessMissingTransform((), 0.0, "random", 0)
    )
    mixed_dataset = MOSEIAlignedDataset(
        data_path, "test",
        DeterministicMissingAugmentation(
            probabilities=(0.0, 0.625, 0.375), ratio_range=(0.30, 0.30),
            double_interval_mode="random", seed=1042,
        ),
    )
    full_loader = make_loader(full_dataset, device)
    mixed_loader = make_loader(mixed_dataset, device)
    rows = []
    for label, path in VARIANTS:
        checkpoint = load_checkpoint(path, map_location=device)
        model = TASPMsa(**checkpoint["config"]["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        full = evaluate(model, full_loader, device)
        missing = evaluate(model, mixed_loader, device)
        row: dict[str, Any] = {
            "variant": label,
            "checkpoint": str(path.relative_to(ROOT)),
            "best_epoch": checkpoint["epoch"],
            "valid_selection_score": checkpoint["best_metric"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "evaluation_split": "test",
            "missing_protocol": "30% mixed: single=0.625,double=0.375,seed=1042",
        }
        for key, value in full.items():
            row[f"full_{key}"] = value
        for key, value in missing.items():
            row[f"missing_{key}"] = value
        row["delta_f1"] = full["macro_f1"] - missing["macro_f1"]
        rows.append(row)
        print(label, full, missing)
    write_csv(ROOT / "results/ablation.csv", rows)
    for metric in ("accuracy", "macro_f1", "mae", "pearson"):
        plot(rows, metric)
    print("saved ablation.csv and figures")


if __name__ == "__main__":
    main()
