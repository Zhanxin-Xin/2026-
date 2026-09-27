from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import (
    MODALITIES,
    FeatureNormalizer,
    MultimodalDataset,
    attach_labels_from_excel,
    load_pickle,
    parse_split,
)
from .metrics import compute_metrics
from .model import HAFusionNet, build_model
from .utils import move_to_device, resolve_device, save_json


@torch.no_grad()
def evaluate_subset(
    model: HAFusionNet,
    loader: DataLoader,
    enabled: Iterable[str],
    device: torch.device,
    amp: bool,
) -> Dict[str, Any]:
    enabled_set = set(enabled)
    probabilities = []
    regression = []
    class_targets = []
    regression_targets = []
    model.eval()
    for cpu_batch in loader:
        batch = move_to_device(cpu_batch, device)
        for modality in MODALITIES:
            if modality not in enabled_set:
                batch[modality] = torch.zeros_like(batch[modality])
        with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
            outputs = model(batch)
        probabilities.append(outputs["class_probabilities"].cpu())
        regression.append(outputs["regression"].cpu())
        class_targets.append(batch["class_label"].cpu())
        regression_targets.append(batch["regression_label"].cpu())
    return compute_metrics(
        torch.cat(class_targets).numpy(),
        torch.cat(probabilities).numpy(),
        torch.cat(regression_targets).numpy(),
        torch.cat(regression).numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only modality ablation")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    raw = attach_labels_from_excel(load_pickle(args.data), args.labels)
    arrays = parse_split(
        raw,
        args.split,
        cfg["data"].get("mask_strategy", "text_shared"),
        require_labels=True,
    )
    arrays = FeatureNormalizer.from_state_dict(checkpoint["normalizer"]).transform(arrays)
    loader = DataLoader(
        MultimodalDataset(arrays),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    subsets = {
        "all": MODALITIES,
        "text": ("text",),
        "audio": ("audio",),
        "vision": ("vision",),
        "text_audio": ("text", "audio"),
        "text_vision": ("text", "vision"),
        "audio_vision": ("audio", "vision"),
    }
    report = {
        name: evaluate_subset(model, loader, enabled, device, bool(cfg["training"].get("amp", True)))
        for name, enabled in subsets.items()
    }
    save_json(report, args.output)
    rows = []
    for name, metrics in report.items():
        rows.append(
            {
                "subset": name,
                **{
                    key: metrics[key]
                    for key in (
                        "accuracy",
                        "macro_f1",
                        "weighted_f1",
                        "balanced_accuracy",
                        "mae",
                        "rmse",
                        "pearson",
                    )
                },
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
