from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    CLASS_NAMES,
    MODALITIES,
    FeatureNormalizer,
    MultimodalDataset,
    attach_labels_from_excel,
    find_sibling_label_excel,
    load_pickle,
    parse_split,
)
from .explain import build_explanation_rows, plot_explanation_cards
from .infer import concatenate_outputs, ensemble_forward
from .losses import ablation_modality_importance
from .metrics import compute_metrics
from .model import build_model
from .q3_reporting import export_labeled_prediction_views, modality_summary_table
from .utils import move_to_device, resolve_device, save_json, seed_everything


def export_split(
    split_name: str,
    arrays,
    model: HAFusionNet,
    cfg: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    include_ablation: bool,
    include_local_table: bool,
    plot_limit: int,
) -> Dict[str, Any]:
    loader = DataLoader(
        MultimodalDataset(arrays),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    output_chunks: List[Mapping[str, torch.Tensor]] = []
    ablated_chunks: List[List[Mapping[str, torch.Tensor]]] = [[], [], []]
    mask_chunks, ids, raw_texts = [], [], []
    for cpu_batch in tqdm(loader, desc=f"attachment2-{split_name}"):
        batch = move_to_device(cpu_batch, device)
        full = ensemble_forward([model], batch, amp_enabled)
        output_chunks.append({key: value.detach().cpu() for key, value in full.items()})
        mask_chunks.append(
            torch.stack([batch[f"{modality}_mask"] for modality in MODALITIES], dim=1).cpu()
        )
        ids.extend([str(value) for value in cpu_batch["id"]])
        raw_texts.extend([str(value) for value in cpu_batch["raw_text"]])
        if include_ablation:
            for modality_index, modality in enumerate(MODALITIES):
                counterfactual = dict(batch)
                counterfactual[modality] = torch.zeros_like(batch[modality])
                ablated = ensemble_forward([model], counterfactual, amp_enabled)
                ablated_chunks[modality_index].append(
                    {key: value.detach().cpu() for key, value in ablated.items()}
                )

    outputs = concatenate_outputs(output_chunks)
    masks = torch.cat(mask_chunks, dim=0).numpy()
    ablation_importance = None
    if include_ablation:
        ablated_outputs = [concatenate_outputs(chunks) for chunks in ablated_chunks]
        ablation_importance = ablation_modality_importance(
            outputs, ablated_outputs
        ).numpy()
    summary, local_table = build_explanation_rows(
        ids=ids,
        raw_texts=raw_texts,
        masks=masks,
        outputs=outputs,
        explanation_cfg=cfg.get("explanation", {}),
        ablation_importance=ablation_importance,
        include_local_table=include_local_table,
    )
    summary["true_label"] = [CLASS_NAMES[int(value)] for value in arrays.class_labels]
    summary["true_intensity"] = arrays.regression_labels.astype(float)
    summary = export_labeled_prediction_views(summary, output_dir, split_name)
    summary.to_csv(
        output_dir / f"{split_name}_predictions_and_explanations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if include_local_table:
        local_table.to_csv(
            output_dir / f"{split_name}_local_evidence.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_explanation_cards(
            summary,
            local_table,
            output_dir / f"{split_name}_explanation_cards",
            limit=plot_limit,
        )
    modality_table = modality_summary_table(summary)
    modality_table.to_csv(
        output_dir / f"{split_name}_modality_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = compute_metrics(
        arrays.class_labels,
        outputs["class_probabilities"].numpy(),
        arrays.regression_labels,
        outputs["regression"].numpy(),
    )
    save_json(metrics, output_dir / f"{split_name}_metrics.json")
    return {
        "samples": int(arrays.size),
        "metrics": metrics,
        "ablation_exported": include_ablation,
        "local_evidence_exported": include_local_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export classification, regression and explanation results for all attachment-2 splits"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument(
        "--ablation-splits", nargs="+", default=["valid", "test"]
    )
    parser.add_argument(
        "--local-evidence-splits", nargs="+", default=["valid", "test"]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plot-limit", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    seed_everything(int(cfg.get("seed", 20260924)))
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    raw = load_pickle(args.data)
    label_path = Path(args.labels) if args.labels else find_sibling_label_excel(args.data)
    if label_path is not None:
        raw = attach_labels_from_excel(raw, label_path)
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    reports: Dict[str, Any] = {}
    metric_rows = []
    for split_name in args.splits:
        if split_name not in raw:
            raise KeyError(f"Attachment 2 has no split named {split_name!r}")
        arrays = parse_split(
            raw,
            split_name,
            mask_strategy=cfg["data"].get("mask_strategy", "text_shared"),
            require_labels=True,
        )
        arrays = normalizer.transform(arrays)
        report = export_split(
            split_name=split_name,
            arrays=arrays,
            model=model,
            cfg=cfg,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            include_ablation=split_name in set(args.ablation_splits),
            include_local_table=split_name in set(args.local_evidence_splits),
            plot_limit=args.plot_limit,
        )
        reports[split_name] = report
        metric_row = {"split": split_name}
        for metric_name in (
            "accuracy",
            "f1",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "mae",
            "rmse",
            "regression_bias",
            "pearson",
        ):
            metric_row[metric_name] = report["metrics"].get(metric_name, np.nan)
        metric_rows.append(metric_row)

    pd.DataFrame(metric_rows).to_csv(
        output_dir / "attachment2_all_split_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "splits": reports,
        "note": (
            "Train, validation and test have labels and therefore include classification "
            "and regression metrics. Local evidence and counterfactual ablation are exported "
            "for the configured subsets to control output size."
        ),
    }
    save_json(manifest, output_dir / "attachment2_output_manifest.json")
    print(f"Attachment-2 complete results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
