#!/usr/bin/env python3
"""Run TASP-MSA missing-type, missing-rate, and missing-position experiments."""
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
from data.robustness_augmentation import RobustnessMissingTransform
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_metrics
from utils.seed import seed_everything
from tasp_msa.model import MODALITIES, TASPMsa

METRICS = ("accuracy", "macro_f1", "mae", "pearson")
SEEDS = (42, 43, 44, 45, 46)
DISPLAY = {"text": "Text", "audio": "Audio", "vision": "Vision"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries = []
    for group, members in groups.items():
        item = {key: value for key, value in zip(keys, group)}
        item.update(seed="mean", n_seeds=len(members), evaluation_split="test")
        for metric in METRICS:
            values = np.asarray([member[metric] for member in members], dtype=float)
            item[metric] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        summaries.append(item)
    return summaries


@torch.inference_mode()
def evaluate(
    model: TASPMsa, data_path: str, transform: RobustnessMissingTransform,
    batch_size: int, device: torch.device,
) -> dict[str, float]:
    dataset = MOSEIAlignedDataset(data_path, "test", transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    logits, labels, regression, targets = [], [], [], []
    for batch in loader:
        inputs = {}
        for modality in MODALITIES:
            inputs[modality] = batch[f"masked_{modality}"].to(device)
            inputs[f"valid_mask_{modality}"] = batch[f"valid_mask_{modality}"].to(device).bool()
            inputs[f"missing_mask_{modality}"] = batch[f"missing_mask_{modality}"].to(device).bool()
        output = model(**inputs)
        logits.append(output["classification_logits"].cpu())
        labels.append(batch["classification_label"])
        regression.append(output["regression"].cpu())
        targets.append(batch["regression_label"].unsqueeze(-1))
    return compute_metrics(torch.cat(logits), torch.cat(labels), torch.cat(regression), torch.cat(targets))


def save(fig: plt.Figure, name: str) -> None:
    destination = ROOT / "figures" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_type(summary: list[dict[str, Any]], metric: str) -> None:
    labels = [row["setting"] for row in summary]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(labels, [row[metric] for row in summary],
             yerr=[row[f"{metric}_std"] for row in summary], capsize=3, color="#4C78A8")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title("Robustness by Missing Modality Type (ratio=0.30)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    save(fig, f"missing_type_{metric}")


def plot_rate(summary: list[dict[str, Any]], metric: str) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for modality in MODALITIES:
        rows = sorted((row for row in summary if row["modality"] == modality),
                      key=lambda row: row["ratio"])
        x = np.asarray([row["ratio"] for row in rows])
        y = np.asarray([row[metric] for row in rows])
        std = np.asarray([row[f"{metric}_std"] for row in rows])
        axis.plot(x, y, marker="o", label=DISPLAY[modality])
        axis.fill_between(x, y - std, y + std, alpha=0.15)
    axis.set_xlabel("Missing Ratio")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title(f"Missing Rate vs {metric.replace('_', ' ').title()}")
    axis.legend()
    axis.grid(alpha=0.25)
    save(fig, f"missing_rate_{metric}")


def plot_position(summary: list[dict[str, Any]], metric: str) -> None:
    positions = ("beginning", "middle", "end", "random")
    x = np.arange(4)
    width = 0.25
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for index, modality in enumerate(MODALITIES):
        lookup = {row["position"]: row for row in summary if row["modality"] == modality}
        axis.bar(x + (index - 1) * width, [lookup[p][metric] for p in positions], width,
                 yerr=[lookup[p][f"{metric}_std"] for p in positions], capsize=2,
                 label=DISPLAY[modality])
    axis.set_xticks(x, [position.title() for position in positions])
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.set_title("Missing Position Robustness (ratio=0.30)")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save(fig, f"missing_position_{metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_model.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    seed_everything(int(config["training"]["seed"]))
    model = TASPMsa(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data_path = config["data"]["path"]
    cache: dict[tuple[Any, ...], dict[str, float]] = {}

    def run(modalities=(), ratio=0.0, position="random", seed=42):
        effective_seed = seed if position == "random" and ratio > 0 else 0
        key = (tuple(modalities), float(ratio), position, effective_seed)
        if key not in cache:
            cache[key] = evaluate(
                model, data_path,
                RobustnessMissingTransform(modalities, ratio, position, effective_seed),
                args.batch_size, device,
            )
        return cache[key]

    type_settings = [
        ("None", ()), ("Text", ("text",)), ("Audio", ("audio",)),
        ("Vision", ("vision",)), ("Text+Audio", ("text", "audio")),
        ("Text+Vision", ("text", "vision")), ("Audio+Vision", ("audio", "vision")),
    ]
    type_raw = [
        {"setting": label, "seed": seed, "evaluation_split": "test",
         **run(modalities, 0.3, "random", seed)}
        for label, modalities in type_settings for seed in SEEDS
    ]
    type_summary = summarize(type_raw, ("setting",))
    write_csv(ROOT / "results/missing_type.csv", type_raw + type_summary)
    for metric in METRICS:
        plot_type(type_summary, metric)
    print("completed missing type")

    rates = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    rate_raw = [
        {"modality": modality, "ratio": ratio, "seed": seed,
         "evaluation_split": "test", **run((modality,), ratio, "random", seed)}
        for modality in MODALITIES for ratio in rates for seed in SEEDS
    ]
    rate_summary = summarize(rate_raw, ("modality", "ratio"))
    baseline = next(row["macro_f1"] for row in rate_summary
                    if row["modality"] == "text" and row["ratio"] == 0.0)
    for row in rate_raw + rate_summary:
        row["degradation_rate"] = (baseline - row["macro_f1"]) / baseline
        row["f1_retention"] = row["macro_f1"] / baseline
    write_csv(ROOT / "results/missing_rate.csv", rate_raw + rate_summary)
    for metric in METRICS:
        plot_rate(rate_summary, metric)

    sensitivity_rows, retention_rows = [], []
    for modality in MODALITIES:
        rows = sorted((row for row in rate_summary if row["modality"] == modality),
                      key=lambda row: row["ratio"])
        x = np.asarray([row["ratio"] for row in rows], dtype=float)
        y = np.asarray([row["macro_f1"] for row in rows], dtype=float)
        beta1, beta0 = np.polyfit(x, y, 1)
        fitted = beta0 + beta1 * x
        denominator = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - fitted) ** 2) / denominator if denominator > 0 else 1.0
        sensitivity_rows.append({
            "evaluation_split": "test", "modality": modality, "beta0": beta0,
            "beta1": beta1, "r_squared": r2, "sensitivity": abs(beta1),
        })
        retention_rows.append({
            "evaluation_split": "test", "modality": modality,
            "average_f1_retention": float(np.mean(y[1:] / baseline)),
        })
    sensitivity_rows.sort(key=lambda row: row["sensitivity"], reverse=True)
    write_csv(ROOT / "results/missing_sensitivity.csv", sensitivity_rows)
    write_csv(ROOT / "results/missing_retention.csv", retention_rows)
    print("completed missing rate")

    position_raw = [
        {"modality": modality, "position": position, "seed": seed,
         "evaluation_split": "test", **run((modality,), 0.3, position, seed)}
        for modality in MODALITIES for position in ("beginning", "middle", "end", "random")
        for seed in SEEDS
    ]
    position_summary = summarize(position_raw, ("modality", "position"))
    write_csv(ROOT / "results/missing_position.csv", position_raw + position_summary)
    for metric in METRICS:
        plot_position(position_summary, metric)
    print("completed missing position")

    manifest = {
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
        "evaluation_split": "test", "seeds": list(SEEDS), "baseline_f1": baseline,
        "sensitivity": sensitivity_rows, "retention": retention_rows,
        "test_used_for_model_selection": False,
    }
    (ROOT / "results/robustness_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
