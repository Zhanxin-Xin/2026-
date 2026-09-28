#!/usr/bin/env python3
"""Evaluate one frozen TASP-MSA checkpoint on valid or Attachment 2 test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from utils.checkpoint import load_checkpoint
from utils.seed import seed_everything
from tasp_msa.model import TASPMsa
from train import build_validation_loaders, evaluate_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_model.pth")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "results/test_metrics.json")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    seed_everything(int(config["training"]["seed"]))
    loaders = build_validation_loaders(config, args.split, device)
    model = TASPMsa(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, score = evaluate_all(model, loaders, device)
    result = {
        "model_family": "TASP-MSA",
        "checkpoint": str(args.checkpoint),
        "selected_epoch": checkpoint["epoch"],
        "selection_split": "valid",
        "selection_protocol": checkpoint["selection_protocol"],
        "evaluation_split": args.split,
        "samples": len(loaders["full"].dataset),
        "test_used_for_tuning": False,
        "missing_ratio": float(config["augmentation"]["validation_ratio"]),
        "missing_seed": int(config["augmentation"]["validation_seed"]),
        "selection_score": score,
        "metrics": metrics,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
