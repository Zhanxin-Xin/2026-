from __future__ import annotations

"""Bounded dialogue-graph Neutral residual over a strict OOF parent."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES, load_pickle, parse_split
from .metrics import compute_metrics
from .train_dialogue_graph_emotion_oof import (
    DirectedGraphMessage,
    build_directional_neighbors,
    class_weights,
    dialogue_key,
    masked_temporal_mean,
    subset_tensor,
)
from .utils import resolve_device, save_json, seed_everything


class ParentAnchoredDialogueGraphNet(nn.Module):
    def __init__(
        self,
        semantic_dimension: int,
        audio_dimension: int,
        vision_dimension: int,
        hidden_dimension: int = 128,
        window: int = 2,
        depth: int = 2,
        dropout: float = 0.20,
        maximum_neutral_shift: float = 1.0,
        maximum_regression_shift: float = 0.50,
    ) -> None:
        super().__init__()
        self.maximum_neutral_shift = float(maximum_neutral_shift)
        self.maximum_regression_shift = float(maximum_regression_shift)
        half = hidden_dimension // 2
        self.semantic = nn.Sequential(
            nn.LayerNorm(semantic_dimension),
            nn.Linear(semantic_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emotion = nn.Sequential(nn.LayerNorm(7), nn.Linear(7, half), nn.GELU())
        self.audio = nn.Sequential(
            nn.LayerNorm(audio_dimension), nn.Linear(audio_dimension, half), nn.GELU()
        )
        self.vision = nn.Sequential(
            nn.LayerNorm(vision_dimension), nn.Linear(vision_dimension, half), nn.GELU()
        )
        self.parent = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, half), nn.GELU())
        input_dimension = hidden_dimension + 4 * half
        self.node_fusion = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
        )
        self.graph = nn.ModuleList(
            [DirectedGraphMessage(hidden_dimension, window, dropout) for _ in range(depth)]
        )
        residual_dimension = 4 * hidden_dimension + 4
        self.neutral_residual = nn.Sequential(
            nn.LayerNorm(residual_dimension),
            nn.Linear(residual_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        self.regression_residual = nn.Sequential(
            nn.LayerNorm(2 * hidden_dimension + 1),
            nn.Linear(2 * hidden_dimension + 1, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        # Exact no-op at initialization: the architecture begins as strict7.
        nn.init.zeros_(self.neutral_residual[-1].weight)
        nn.init.zeros_(self.neutral_residual[-1].bias)
        nn.init.zeros_(self.regression_residual[-1].weight)
        nn.init.zeros_(self.regression_residual[-1].bias)

    def forward(
        self,
        emotion_logits: Tensor,
        semantic: Tensor,
        audio: Tensor,
        vision: Tensor,
        parent_probability: Tensor,
        parent_regression: Tensor,
        neighbor_index: Tensor,
        direction: Tensor,
        distance: Tensor,
    ) -> dict[str, Tensor]:
        parent_probability = parent_probability.float().clamp_min(1e-8)
        current = self.node_fusion(
            torch.cat(
                [
                    self.semantic(semantic),
                    self.emotion(emotion_logits),
                    self.audio(audio),
                    self.vision(vision),
                    self.parent(parent_probability),
                ],
                dim=-1,
            )
        )
        contextual = current
        gates = []
        for layer in self.graph:
            contextual, gate, _ = layer(
                contextual, neighbor_index, direction, distance
            )
            gates.append(gate)
        neutral_parent = parent_probability[:, 1].clamp(1e-6, 1.0 - 1e-6)
        polar_parent = torch.maximum(parent_probability[:, 0], parent_probability[:, 2])
        neutral_uncertainty = 4.0 * neutral_parent * (1.0 - neutral_parent)
        parent_uncertainty = 1.0 - parent_probability.max(dim=1).values
        parent_margin = (neutral_parent - polar_parent).abs()
        interaction = torch.cat(
            [
                current,
                contextual,
                torch.abs(current - contextual),
                current * contextual,
                parent_probability.to(current.dtype),
                parent_uncertainty.unsqueeze(-1).to(current.dtype),
            ],
            dim=-1,
        )
        raw_neutral_residual = self.maximum_neutral_shift * torch.tanh(
            self.neutral_residual(interaction).squeeze(-1).float()
        )
        # Structural trust region: context is most useful close to the parent's
        # Neutral boundary and is attenuated for already confident predictions.
        trust_region = neutral_uncertainty * (0.25 + 0.75 * parent_uncertainty)
        neutral_residual = raw_neutral_residual * trust_region
        neutral_logit = torch.logit(neutral_parent) + neutral_residual
        neutral_probability = torch.sigmoid(neutral_logit).clamp(1e-6, 1.0 - 1e-6)
        positive_given_polar = parent_probability[:, 2] / (
            parent_probability[:, 0] + parent_probability[:, 2]
        ).clamp_min(1e-8)
        probability = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_given_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_given_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probability = probability / probability.sum(dim=1, keepdim=True)
        raw_regression_residual = self.maximum_regression_shift * torch.tanh(
            self.regression_residual(
                torch.cat(
                    [current, contextual, parent_regression.unsqueeze(-1)], dim=-1
                )
            ).squeeze(-1).float()
        )
        regression = (parent_regression.float() + raw_regression_residual).clamp(-3.0, 3.0)
        graph_gate = torch.stack(gates, dim=1).mean(dim=1)
        return {
            "probabilities": probability,
            "regression": regression,
            "neutral_residual": neutral_residual,
            "neutral_trust_region": trust_region,
            "parent_margin": parent_margin,
            "regression_residual": raw_regression_residual,
            "graph_gate": graph_gate,
        }


def fit_fold(
    emotion_logits: np.ndarray,
    semantic: np.ndarray,
    audio: np.ndarray,
    vision: np.ndarray,
    parent_probability: np.ndarray,
    parent_regression: np.ndarray,
    ids: np.ndarray,
    target: np.ndarray,
    intensity: np.ndarray,
    train_index: np.ndarray,
    heldout_index: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed_everything(seed)
    model = ParentAnchoredDialogueGraphNet(
        semantic_dimension=semantic.shape[1],
        audio_dimension=audio.shape[1],
        vision_dimension=vision.shape[1],
        window=window,
    ).to(device)
    train_neighbors = build_directional_neighbors(ids[train_index], window)
    heldout_neighbors = build_directional_neighbors(ids[heldout_index], window)

    def make_input(index: np.ndarray, neighbors: tuple[np.ndarray, ...]) -> dict[str, Tensor]:
        return {
            "emotion_logits": subset_tensor(emotion_logits, index, device),
            "semantic": subset_tensor(semantic, index, device),
            "audio": subset_tensor(audio, index, device),
            "vision": subset_tensor(vision, index, device),
            "parent_probability": subset_tensor(parent_probability, index, device),
            "parent_regression": subset_tensor(parent_regression[:, None], index, device).squeeze(-1),
            "neighbor_index": torch.from_numpy(neighbors[0]).to(device),
            "direction": torch.from_numpy(neighbors[1]).to(device),
            "distance": torch.from_numpy(neighbors[2]).to(device),
        }

    train_input = make_input(train_index, train_neighbors)
    heldout_input = make_input(heldout_index, heldout_neighbors)
    train_target = torch.from_numpy(target[train_index]).long().to(device)
    train_intensity = torch.from_numpy(intensity[train_index]).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2)
    weight = class_weights(target[train_index], device)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**train_input)
        classification = F.nll_loss(
            output["probabilities"].log(), train_target, weight=weight
        )
        regression = F.smooth_l1_loss(output["regression"], train_intensity)
        loss = classification + 0.20 * regression
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite parent-graph loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach()),
                    "classification": float(classification.detach()),
                    "regression": float(regression.detach()),
                }
            )
    model.eval()
    with torch.no_grad():
        output = model(**heldout_input)
    diagnostics = {
        name: {
            "mean": float(output[name].mean().cpu()),
            "std": float(output[name].std(unbiased=False).cpu()),
            "min": float(output[name].min().cpu()),
            "max": float(output[name].max().cpu()),
        }
        for name in (
            "neutral_residual",
            "neutral_trust_region",
            "parent_margin",
            "regression_residual",
            "graph_gate",
        )
    }
    return (
        output["probabilities"].cpu().numpy().astype(np.float64),
        output["regression"].cpu().numpy().astype(np.float64),
        {"history": history, "diagnostics": diagnostics},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.window < 1:
        raise ValueError("epochs and window must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    arrays = parse_split(
        {"train": load_pickle(args.data)["train"]},
        "train",
        "text_shared",
        require_labels=False,
    )
    parent = load_aligned([args.parent_oof], require_fold=True)[0]
    indexed = parent.set_index(parent["id"].astype(str), drop=False)
    ids = arrays.ids.astype(str)
    if set(indexed.index) != set(ids):
        raise ValueError("Parent IDs do not match train data")
    parent = indexed.loc[ids].reset_index(drop=True)
    cache = np.load(args.feature_cache, allow_pickle=False)
    if not np.array_equal(cache["ids"].astype(str), ids):
        raise ValueError("Frozen feature cache IDs do not match train data")
    frozen = cache["features"].astype(np.float32)
    if frozen.shape != (arrays.size, 1031):
        raise ValueError(f"Unexpected frozen feature shape: {frozen.shape}")
    emotion_logits, semantic = frozen[:, :7], frozen[:, 7:]
    audio = masked_temporal_mean(arrays.features["audio"], arrays.masks["audio"])
    vision = masked_temporal_mean(arrays.features["vision"], arrays.masks["vision"])
    parent_probability = parent.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float32)
    parent_probability /= parent_probability.sum(axis=1, keepdims=True).clip(min=1e-8)
    parent_regression = parent["predicted_intensity"].to_numpy(np.float32)
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    target = parent["true_label"].map(label_map).to_numpy(np.int64)
    intensity = parent["true_intensity"].to_numpy(np.float32)
    folds = parent["fold"].to_numpy(np.int64)
    oof_probability = np.zeros_like(parent_probability, dtype=np.float64)
    oof_regression = np.zeros(arrays.size, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    started = time.time()
    for fold in sorted(np.unique(folds).tolist()):
        train_index = np.flatnonzero(folds != fold)
        heldout_index = np.flatnonzero(folds == fold)
        train_groups = {dialogue_key(value)[0] for value in ids[train_index]}
        heldout_groups = {dialogue_key(value)[0] for value in ids[heldout_index]}
        if train_groups.intersection(heldout_groups):
            raise RuntimeError(f"Dialogue leakage in fold {fold}")
        probability, regression, detail = fit_fold(
            emotion_logits,
            semantic,
            audio,
            vision,
            parent_probability,
            parent_regression,
            ids,
            target,
            intensity,
            train_index,
            heldout_index,
            device,
            args.seed + fold,
            args.epochs,
            args.window,
        )
        oof_probability[heldout_index] = probability
        oof_regression[heldout_index] = regression
        metrics = compute_metrics(
            target[heldout_index], probability, intensity[heldout_index], regression
        )
        report = {
            "fold": int(fold),
            "seed": args.seed + fold,
            "train_samples": int(len(train_index)),
            "heldout_samples": int(len(heldout_index)),
            "train_groups": int(len(train_groups)),
            "heldout_groups": int(len(heldout_groups)),
            "group_overlap": 0,
            "metrics": metrics,
            **detail,
        }
        fold_reports.append(report)
        save_json(report, output / f"fold_{fold}_report.json")
        print(
            f"fold={fold} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
    metrics = compute_metrics(target, oof_probability, intensity, oof_regression)
    parent_metrics = compute_metrics(
        target, parent_probability, intensity, parent_regression
    )
    prediction = pd.DataFrame(
        {
            "id": ids,
            "fold": folds,
            "predicted_label": [
                CLASS_NAMES[index] for index in oof_probability.argmax(axis=1)
            ],
            "negative_probability": oof_probability[:, 0],
            "neutral_probability": oof_probability[:, 1],
            "positive_probability": oof_probability[:, 2],
            "predicted_intensity": oof_regression,
            "true_label": parent["true_label"].astype(str),
            "true_intensity": intensity,
        }
    )
    prediction.to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "scope": "train_only_grouped_oof_parent_anchored_no_valid_or_test_access",
        "architecture": "strict7_anchored_directed_dialogue_graph_neutral_residual",
        "seed": args.seed,
        "epochs": args.epochs,
        "window": args.window,
        "samples": arrays.size,
        "groups": int(len({dialogue_key(value)[0] for value in ids})),
        "checkpoint_selection_on_heldout_fold": False,
        "parent_oof": parent_metrics,
        "oof_metrics": metrics,
        "oof_delta": {
            "accuracy": metrics["accuracy"] - parent_metrics["accuracy"],
            "macro_f1": metrics["macro_f1"] - parent_metrics["macro_f1"],
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
        "fold_reports": fold_reports,
        "artifact": "train_oof_predictions.csv",
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
