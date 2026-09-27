from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MODALITIES, FeatureNormalizer, MultimodalDataset, load_pickle, parse_split
from .explain import local_importance
from .losses import ablation_modality_importance, predicted_modality_importance
from .model import build_model
from .utils import move_to_device, resolve_device, save_json, seed_everything


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    if np.std(x_rank) < 1e-12 or np.std(y_rank) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def selected_mask(
    importance: torch.Tensor, valid: torch.Tensor, ratio: float, random: bool = False
) -> torch.Tensor:
    b, m, length = importance.shape
    result = torch.zeros_like(valid, dtype=torch.bool)
    flat_importance = importance.reshape(b, -1)
    flat_valid = valid.reshape(b, -1)
    for row in range(b):
        valid_idx = torch.nonzero(flat_valid[row], as_tuple=False).squeeze(1)
        k = max(1, int(np.ceil(len(valid_idx) * ratio)))
        if random:
            chosen = valid_idx[torch.randperm(len(valid_idx), device=valid_idx.device)[:k]]
        else:
            scores = flat_importance[row, valid_idx]
            chosen = valid_idx[torch.topk(scores, k=k, largest=True).indices]
        result.reshape(b, -1)[row, chosen] = True
    return result


def mask_batch(
    batch: Mapping[str, Any], selected: torch.Tensor, keep_selected: bool
) -> Dict[str, Any]:
    result = dict(batch)
    for index, modality in enumerate(MODALITIES):
        selector = selected[:, index].unsqueeze(-1)
        result[modality] = torch.where(
            selector if keep_selected else ~selector,
            batch[modality],
            torch.zeros_like(batch[modality]),
        )
    return result


def evidence_change(full: Mapping[str, torch.Tensor], changed: Mapping[str, torch.Tensor]):
    pred = full["class_probabilities"].argmax(dim=1)
    rows = torch.arange(len(pred), device=pred.device)
    full_conf = full["class_probabilities"][rows, pred]
    changed_conf = changed["class_probabilities"][rows, pred]
    return {
        "classification_drop": (full_conf - changed_conf).mean().item(),
        "classification_abs_change": (full_conf - changed_conf).abs().mean().item(),
        "regression_abs_change": (
            full["regression"] - changed["regression"]
        ).abs().mean().item(),
    }


def weighted_add(total: Dict[str, float], values: Mapping[str, float], n: int) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value) * n


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative faithfulness audit")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2])
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    seed_everything(int(cfg.get("seed", 20260924)))
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    raw = load_pickle(args.data)
    arrays = parse_split(
        raw,
        args.split,
        cfg["data"].get("mask_strategy", "text_shared"),
        require_labels=True,
    )
    arrays = FeatureNormalizer.from_state_dict(checkpoint["normalizer"]).transform(arrays)
    loader = DataLoader(MultimodalDataset(arrays), batch_size=args.batch_size, shuffle=False)

    totals: Dict[str, Dict[str, float]] = {}
    predicted_importance, observed_importance = [], []
    sample_count = 0
    regression_conservation_total = 0.0
    classification_conservation_total = 0.0
    interaction_fraction_total = 0.0
    conflict_total = 0.0
    for cpu_batch in tqdm(loader, desc="faithfulness audit"):
        batch = move_to_device(cpu_batch, device)
        full = model(batch)
        ablated = []
        for modality in MODALITIES:
            changed = dict(batch)
            changed[modality] = torch.zeros_like(batch[modality])
            ablated.append(model(changed))
        pred_imp = predicted_modality_importance(full)
        obs_imp = ablation_modality_importance(full, ablated)
        predicted_importance.append(pred_imp.cpu().numpy())
        observed_importance.append(obs_imp.cpu().numpy())

        local = local_importance(full)
        valid = torch.stack([batch[f"{m}_mask"] for m in MODALITIES], dim=1)
        n = batch["text"].size(0)
        sample_count += n
        regression_error = (
            full["regression_raw"]
            - full["regression_bias"]
            - full["regression_contributions"].sum(dim=1)
        ).abs()
        classification_error = (
            full["class_logits"]
            - full["class_bias"]
            - full["classification_contributions"].sum(dim=1)
        ).abs().amax(dim=1)
        regression_conservation_total += float(regression_error.sum())
        classification_conservation_total += float(classification_error.sum())
        pair_abs = full["pair_regression_contributions"].abs().sum(dim=1)
        total_abs = pair_abs + full["unary_regression_contributions"].abs().sum(dim=1)
        interaction_fraction_total += float(
            (pair_abs / total_abs.clamp_min(1e-8)).sum()
        )
        conflict_total += float(full["conflict_scores"].mean(dim=(1, 2)).sum())
        for ratio in args.ratios:
            top = selected_mask(local, valid, ratio, random=False)
            random_choice = selected_mask(local, valid, ratio, random=True)
            deleted = model(mask_batch(batch, top, keep_selected=False))
            random_deleted = model(mask_batch(batch, random_choice, keep_selected=False))
            sufficient = model(mask_batch(batch, top, keep_selected=True))
            key = f"top_{int(round(ratio * 100))}pct_deletion"
            totals.setdefault(key, {})
            weighted_add(totals[key], evidence_change(full, deleted), n)
            random_key = f"random_{int(round(ratio * 100))}pct_deletion"
            totals.setdefault(random_key, {})
            weighted_add(totals[random_key], evidence_change(full, random_deleted), n)
            sufficient_key = f"top_{int(round(ratio * 100))}pct_sufficiency"
            totals.setdefault(sufficient_key, {})
            weighted_add(totals[sufficient_key], evidence_change(full, sufficient), n)

    for group in totals.values():
        for key in group:
            group[key] /= max(sample_count, 1)
    predicted_np = np.concatenate(predicted_importance)
    observed_np = np.concatenate(observed_importance)
    per_sample_rho = [
        rank_correlation(predicted_np[i], observed_np[i]) for i in range(len(predicted_np))
    ]
    report = {
        "split": args.split,
        "samples": sample_count,
        "modality_importance_spearman_global": rank_correlation(
            predicted_np.reshape(-1), observed_np.reshape(-1)
        ),
        "modality_importance_spearman_per_sample_mean": float(np.mean(per_sample_rho)),
        "modality_importance_mae": float(np.mean(np.abs(predicted_np - observed_np))),
        "regression_conservation_error_mean": regression_conservation_total
        / max(sample_count, 1),
        "classification_conservation_error_mean": classification_conservation_total
        / max(sample_count, 1),
        "pair_interaction_fraction_mean": interaction_fraction_total
        / max(sample_count, 1),
        "mean_conflict_score": conflict_total / max(sample_count, 1),
        "evidence_tests": totals,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(report, output / "explanation_audit.json")
    pd.DataFrame(
        np.concatenate([predicted_np, observed_np], axis=1),
        columns=[f"predicted_{m}" for m in MODALITIES]
        + [f"ablation_{m}" for m in MODALITIES],
    ).to_csv(output / "modality_faithfulness.csv", index=False, encoding="utf-8-sig")
    print(report)


if __name__ == "__main__":
    main()
