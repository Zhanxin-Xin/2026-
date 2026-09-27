from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MODALITIES, FeatureNormalizer, MultimodalDataset, load_feature_source, parse_split
from .explain import (
    build_video_mapping,
    build_explanation_rows,
    extract_vision_keyframes,
    load_timestamps,
    plot_explanation_cards,
)
from .losses import ablation_modality_importance
from .metrics import compute_metrics
from .model import HAFusionNet, build_model
from .q3_reporting import export_attachment4_deliverables
from .utils import move_to_device, resolve_device, save_json, seed_everything


OUTPUT_KEYS = (
    "class_logits",
    "class_probabilities",
    "regression",
    "regression_raw",
    "regression_bias",
    "class_bias",
    "component_gates",
    "modality_gates",
    "temporal_weights",
    "pair_temporal_weights",
    "conflict_scores",
    "pair_reliability",
    "modality_regression_evidence",
    "modality_classification_evidence",
    "pair_regression_evidence",
    "pair_classification_evidence",
    "unary_regression_contributions",
    "unary_classification_contributions",
    "pair_regression_contributions",
    "pair_classification_contributions",
    "regression_contributions",
    "classification_contributions",
    "unary_local_regression_contributions",
    "unary_local_classification_contributions",
    "pair_local_regression_contributions",
    "pair_local_classification_contributions",
    "local_regression_contributions",
    "local_classification_contributions",
    "class_probability_std",
    "regression_std",
)


def load_models(
    checkpoint_paths: Sequence[str], device: torch.device
) -> tuple[List[HAFusionNet], List[Mapping[str, Any]]]:
    models, checkpoints = [], []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model(checkpoint["config"]["model"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        models.append(model)
        checkpoints.append(checkpoint)
    reference = checkpoints[0]["config"]["model"]
    if any(checkpoint["config"]["model"] != reference for checkpoint in checkpoints[1:]):
        raise ValueError("All ensemble checkpoints must use the same model configuration")
    return models, checkpoints


@torch.no_grad()
def ensemble_forward(
    models: Sequence[HAFusionNet], batch: Mapping[str, Any], amp_enabled: bool
) -> Dict[str, torch.Tensor]:
    members: List[Mapping[str, torch.Tensor]] = []
    with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
        for model in models:
            members.append(model(batch))
    averaged = {
        key: torch.stack([member[key] for member in members], dim=0).mean(dim=0)
        for key in OUTPUT_KEYS
        if key not in {"class_probability_std", "regression_std"}
    }
    # Ensemble class probabilities should be the mean of member probabilities.
    averaged["class_probabilities"] = torch.stack(
        [member["class_probabilities"] for member in members], dim=0
    ).mean(dim=0)
    averaged["regression"] = torch.stack(
        [member["regression"] for member in members], dim=0
    ).mean(dim=0)
    averaged["class_probability_std"] = torch.stack(
        [member["class_probabilities"] for member in members], dim=0
    ).std(dim=0, unbiased=False)
    averaged["regression_std"] = torch.stack(
        [member["regression"] for member in members], dim=0
    ).std(dim=0, unbiased=False)
    return averaged


def concatenate_outputs(chunks: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.cat([chunk[key].cpu() for chunk in chunks], dim=0) for key in OUTPUT_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference and evidence export")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--data",
        required=True,
        help="Monolithic feature PKL or attachment-4 directory containing 01.pkl, 02.pkl, ...",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--timestamps-csv", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--plot-limit", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    models, checkpoints = load_models(args.checkpoints, device)
    cfg = checkpoints[0]["config"]
    seed_everything(int(cfg.get("seed", 20260924)))

    raw = load_feature_source(args.data, split_name=args.split)
    if args.split not in raw and all(modality in raw for modality in MODALITIES):
        raw = {args.split: raw}
    arrays = parse_split(
        raw,
        args.split,
        mask_strategy=cfg["data"].get("mask_strategy", "text_shared"),
        require_labels=False,
    )
    normalizer = FeatureNormalizer.from_state_dict(checkpoints[0]["normalizer"])
    arrays = normalizer.transform(arrays)
    video_root = args.video_root
    automatic_video_root = Path(args.data) / "videos"
    if video_root is None and Path(args.data).is_dir() and automatic_video_root.is_dir():
        video_root = str(automatic_video_root)
    loader = DataLoader(
        MultimodalDataset(arrays),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    output_chunks: List[Mapping[str, torch.Tensor]] = []
    ablated_chunks: List[List[Mapping[str, torch.Tensor]]] = [[], [], []]
    mask_chunks, ids, raw_texts = [], [], []

    for cpu_batch in tqdm(loader, desc="inference"):
        batch = move_to_device(cpu_batch, device)
        full = ensemble_forward(models, batch, amp_enabled)
        output_chunks.append({key: value.detach().cpu() for key, value in full.items()})
        masks = torch.stack([batch[f"{m}_mask"] for m in MODALITIES], dim=1)
        mask_chunks.append(masks.cpu())
        ids.extend([str(x) for x in cpu_batch["id"]])
        raw_texts.extend([str(x) for x in cpu_batch["raw_text"]])
        if not args.skip_ablation:
            for m_idx, modality in enumerate(MODALITIES):
                counterfactual = dict(batch)
                counterfactual[modality] = torch.zeros_like(batch[modality])
                out = ensemble_forward(models, counterfactual, amp_enabled)
                ablated_chunks[m_idx].append(
                    {key: value.detach().cpu() for key, value in out.items()}
                )

    outputs = concatenate_outputs(output_chunks)
    masks = torch.cat(mask_chunks, dim=0).numpy()
    ablation_np = None
    if not args.skip_ablation:
        ablated = [concatenate_outputs(chunks) for chunks in ablated_chunks]
        ablation_np = ablation_modality_importance(outputs, ablated).numpy()

    timestamps = load_timestamps(args.timestamps_csv)
    summary, long_table = build_explanation_rows(
        ids=ids,
        raw_texts=raw_texts,
        masks=masks,
        outputs=outputs,
        explanation_cfg=cfg.get("explanation", {}),
        timestamps=timestamps,
        video_root=video_root,
        ablation_importance=ablation_np,
    )
    video_mapping = build_video_mapping(ids, video_root)
    video_mapping.to_csv(
        output_dir / "attachment4_video_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = summary.merge(video_mapping, on="id", how="left", sort=False, validate="one_to_one")
    summary = extract_vision_keyframes(
        summary, video_root, output_dir / "vision_keyframes"
    )
    deliverables = export_attachment4_deliverables(summary, output_dir)
    summary = deliverables["frame"]
    summary.to_csv(output_dir / "attachment4_predictions_and_explanations.csv", index=False, encoding="utf-8-sig")
    long_table.to_csv(output_dir / "attachment4_local_evidence.csv", index=False, encoding="utf-8-sig")
    plot_explanation_cards(
        summary,
        long_table,
        output_dir / "explanation_cards",
        limit=args.plot_limit,
    )

    metadata: Dict[str, Any] = {
        "samples": len(summary),
        "checkpoints": [str(Path(x).resolve()) for x in args.checkpoints],
        "split": args.split,
        "data_source": str(Path(args.data).resolve()),
        "video_root": None if video_root is None else str(Path(video_root).resolve()),
        "ablation_exported": not args.skip_ablation,
        "attachment4_descriptive_summary": deliverables["summary"],
        "problem3_output_manifest": "problem3_output_manifest.json",
        "video_mapping": {
            "file": "attachment4_video_mapping.csv",
            "found": int(video_mapping["video_found"].sum()),
            "missing": int((~video_mapping["video_found"]).sum()),
        },
    }
    if arrays.class_labels is not None and arrays.regression_labels is not None:
        probabilities = outputs["class_probabilities"].numpy()
        regression = outputs["regression"].numpy()
        metadata["metrics"] = compute_metrics(
            arrays.class_labels, probabilities, arrays.regression_labels, regression
        )
    save_json(metadata, output_dir / "inference_metadata.json")
    print(f"Exported {len(summary)} samples to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
