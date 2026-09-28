#!/usr/bin/env python3
"""Reliability, confusion, regression-error, and sensitive-sample analysis."""
from __future__ import annotations

import argparse
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
from tasp_msa.model import MODALITIES, TASPMsa

CLASS_NAMES = ("Negative", "Neutral", "Positive")
SEEDS = (42, 43, 44, 45, 46)


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


def save(fig: plt.Figure, name: str) -> None:
    destination = ROOT / "figures" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_loader(dataset, batch_size: int, device: torch.device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                      pin_memory=device.type == "cuda")


def inputs(batch: dict[str, Any], device: torch.device, masked=True):
    result = {}
    for modality in MODALITIES:
        valid = batch[f"valid_mask_{modality}"].to(device).bool()
        result[modality] = batch[f"masked_{modality}" if masked else modality].to(device)
        result[f"valid_mask_{modality}"] = valid
        result[f"missing_mask_{modality}"] = (
            batch[f"missing_mask_{modality}"].to(device).bool()
            if masked else torch.ones_like(valid)
        )
    return result


@torch.inference_mode()
def reliability_one(
    model: TASPMsa, data_path: str, modalities: tuple[str, ...], ratio: float,
    seed: int, batch_size: int, device: torch.device,
) -> dict[str, float]:
    dataset = MOSEIAlignedDataset(
        data_path, "test", RobustnessMissingTransform(modalities, ratio, "random", seed)
    )
    sums = {"reliability_audio": 0.0, "reliability_vision": 0.0,
            "uncertainty_audio": 0.0, "uncertainty_vision": 0.0}
    count = 0
    for batch in make_loader(dataset, batch_size, device):
        output = model(**inputs(batch, device))
        size = batch["classification_label"].shape[0]
        count += size
        for modality in ("audio", "vision"):
            sums[f"reliability_{modality}"] += float(output["reliability"][modality].sum())
            uncertainty = torch.exp(output["proxy_logvar"][modality]).mean(-1)
            sums[f"uncertainty_{modality}"] += float(uncertainty.sum())
    return {key: value / count for key, value in sums.items()}


def summarize_reliability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["missing_type"], row["missing_rate"]), []).append(row)
    output = []
    value_keys = ("reliability_audio", "reliability_vision", "uncertainty_audio", "uncertainty_vision")
    for (missing_type, rate), members in groups.items():
        item: dict[str, Any] = {
            "evaluation_split": "test", "missing_type": missing_type,
            "missing_rate": rate, "seed": "mean", "n_seeds": len(members),
        }
        for key in value_keys:
            values = np.asarray([member[key] for member in members])
            item[key] = float(values.mean())
            item[f"{key}_std"] = float(values.std(ddof=0))
        output.append(item)
    return output


def plot_reliability(summary: list[dict[str, Any]], uncertainty=False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for axis, missing_type in zip(axes, ("Audio", "Vision", "Audio+Vision")):
        rows = sorted((row for row in summary if row["missing_type"] == missing_type),
                      key=lambda row: row["missing_rate"])
        for modality, color in (("audio", "#F58518"), ("vision", "#54A24B")):
            prefix = "uncertainty" if uncertainty else "reliability"
            axis.plot([row["missing_rate"] for row in rows],
                      [row[f"{prefix}_{modality}"] for row in rows], marker="o",
                      label=modality.title(), color=color)
        axis.set_title(f"{missing_type} Missing")
        axis.set_xlabel("Missing Ratio")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Proxy Uncertainty" if uncertainty else "Mean Reliability")
    axes[-1].legend()
    save(fig, "uncertainty_vs_missing_rate" if uncertainty else "reliability_vs_missing_rate")


def confusion(rows: list[dict[str, Any]], key: str, name: str) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    for row in rows:
        matrix[int(row["ground_truth_class"]), int(row[key])] += 1
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color)
    axis.set_xticks(range(3), CLASS_NAMES)
    axis.set_yticks(range(3), CLASS_NAMES)
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    axis.set_title(name.replace("_", " ").title())
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save(fig, name)
    return matrix


@torch.inference_mode()
def prediction_rows(
    model: TASPMsa, data_path: str, batch_size: int, device: torch.device
) -> list[dict[str, Any]]:
    transform = DeterministicMissingAugmentation(
        probabilities=(0.0, 0.625, 0.375), ratio_range=(0.30, 0.30),
        double_interval_mode="random", seed=1042,
    )
    dataset = MOSEIAlignedDataset(data_path, "test", transform)
    raw_lookup = {
        str(sample_id): str(text)
        for sample_id, text in zip(dataset.data["id"], dataset.data.get("raw_text", [""] * len(dataset)))
    }
    rows: list[dict[str, Any]] = []
    for batch in make_loader(dataset, batch_size, device):
        full = model(**inputs(batch, device, masked=False))
        missing = model(**inputs(batch, device, masked=True))
        full_class = full["classification_logits"].argmax(-1).cpu()
        missing_class = missing["classification_logits"].argmax(-1).cpu()
        full_reg = full["regression"].squeeze(-1).cpu()
        missing_reg = missing["regression"].squeeze(-1).cpu()
        for index, sample_id in enumerate(batch["id"]):
            identity = str(sample_id)
            truth = float(batch["regression_label"][index])
            rates = {m: float(batch[f"missing_ratio_{m}"][index]) for m in MODALITIES}
            intervals = {
                m: [int(value) for value in batch[f"missing_interval_{m}"][index].tolist()]
                for m in MODALITIES
            }
            rows.append({
                "id": identity, "raw_text": raw_lookup.get(identity, ""),
                "ground_truth_class": int(batch["classification_label"][index]),
                "ground_truth_class_name": CLASS_NAMES[int(batch["classification_label"][index])],
                "ground_truth": truth,
                "full_predicted_class": int(full_class[index]),
                "full_predicted_class_name": CLASS_NAMES[int(full_class[index])],
                "full_prediction": float(full_reg[index]),
                "full_absolute_error": abs(float(full_reg[index]) - truth),
                "missing_predicted_class": int(missing_class[index]),
                "missing_predicted_class_name": CLASS_NAMES[int(missing_class[index])],
                "missing_prediction": float(missing_reg[index]),
                "missing_absolute_error": abs(float(missing_reg[index]) - truth),
                "difference": abs(float(full_reg[index]) - float(missing_reg[index])),
                "classification_changed": bool(full_class[index] != missing_class[index]),
                "missing_type": str(batch["missing_modalities"][index]) or "None",
                "missing_rate": 0.30,
                "actual_missing_rate": max(rates.values()),
                "text_interval": intervals["text"], "audio_interval": intervals["audio"],
                "vision_interval": intervals["vision"],
            })
    return rows


def regression_plots(rows: list[dict[str, Any]]) -> None:
    truth = np.asarray([row["ground_truth"] for row in rows])
    prediction = np.asarray([row["full_prediction"] for row in rows])
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.scatter(truth, prediction, s=13, alpha=0.45, color="#4C78A8")
    axis.plot([-3, 3], [-3, 3], "--", color="black", linewidth=1)
    axis.set(xlabel="Ground Truth", ylabel="Prediction", xlim=(-3, 3), ylim=(-3, 3),
             title="Regression: Ground Truth vs Prediction")
    axis.grid(alpha=0.2)
    save(fig, "regression_scatter")
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.hist(np.abs(prediction - truth), bins=30, color="#4C78A8", alpha=0.85)
    axis.set(xlabel="Absolute Error", ylabel="Sample Count", title="Regression Error Distribution")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "regression_error_distribution")


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
    model = TASPMsa(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data_path = config["data"]["path"]

    raw = []
    for label, modalities in (("Audio", ("audio",)), ("Vision", ("vision",)),
                              ("Audio+Vision", ("audio", "vision"))):
        for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            for seed in SEEDS:
                raw.append({
                    "evaluation_split": "test", "missing_type": label,
                    "missing_rate": ratio, "seed": seed,
                    **reliability_one(model, data_path, modalities, ratio, seed,
                                      args.batch_size, device),
                })
    summary = summarize_reliability(raw)
    write_csv(ROOT / "results/reliability_analysis.csv", raw + summary)
    plot_reliability(summary, uncertainty=False)
    plot_reliability(summary, uncertainty=True)

    rows = prediction_rows(model, data_path, args.batch_size, device)
    confusion(rows, "full_predicted_class", "confusion_matrix_full")
    confusion(rows, "missing_predicted_class", "confusion_matrix_missing")
    regression_plots(rows)
    top_regression = sorted(rows, key=lambda row: row["full_absolute_error"], reverse=True)[:20]
    top_sensitive = sorted(rows, key=lambda row: row["difference"], reverse=True)[:20]
    write_csv(ROOT / "results/top_regression_errors.csv", top_regression)
    write_csv(ROOT / "results/top_missing_sensitive_samples.csv", top_sensitive)
    write_csv(ROOT / "results/test_sample_predictions_full_and_missing.csv", rows)
    print("saved reliability, uncertainty, confusion, regression, and error analyses")


if __name__ == "__main__":
    main()
