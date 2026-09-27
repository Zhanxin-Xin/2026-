from __future__ import annotations

"""Train a native architecture once, then evaluate only the locked valid split."""

import argparse
import json
import time
from pathlib import Path

from .data import attach_labels_from_excel, load_pickle, parse_split
from .train_hafusion_oof import train_fold
from .utils import apply_overrides, load_config, resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-epoch full-train fit followed by one locked-valid evaluation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--fixed-epochs", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg["seed"] = args.seed
    if not 1 <= args.fixed_epochs <= int(cfg["training"]["epochs"]):
        raise ValueError("fixed-epochs must lie inside the configured schedule")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "resolved_config.json")
    loaded = load_pickle(args.data)
    raw = attach_labels_from_excel(
        {"train": loaded["train"], "valid": loaded["valid"]}, args.labels
    )
    mask_strategy = cfg["data"].get("mask_strategy", "text_shared")
    train_arrays = parse_split(raw, "train", mask_strategy, require_labels=True)
    valid_arrays = parse_split(raw, "valid", mask_strategy, require_labels=True)
    started = time.time()
    metrics, predictions, history = train_fold(
        cfg,
        train_arrays,
        valid_arrays,
        resolve_device(args.device),
        args.seed,
        args.fixed_epochs,
    )
    predictions.to_csv(
        output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "scope": "full_train_fixed_epoch_then_single_locked_valid_no_test_access",
        "architecture": str(cfg["model"].get("architecture", "hafusion")),
        "seed": args.seed,
        "fixed_epochs": args.fixed_epochs,
        "checkpoint_selection_on_valid": False,
        "train_samples": train_arrays.size,
        "valid_samples": valid_arrays.size,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "valid": metrics,
        "history": history,
        "artifact": "valid_predictions.csv",
    }
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
