from __future__ import annotations

"""Grouped-OOF dialogue graph over frozen EmoBERTa and multimodal nodes."""

import argparse
import json
import math
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
from .utils import resolve_device, save_json, seed_everything


def dialogue_key(identifier: str) -> tuple[str, int]:
    text = str(identifier)
    try:
        video, clip = text.rsplit("$_$", 1)
        return video, int(clip)
    except (TypeError, ValueError):
        return text, 0


def build_directional_neighbors(
    ids: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window < 1:
        raise ValueError("window must be positive")
    size = len(ids)
    width = 2 * window
    neighbors = np.full((size, width), -1, dtype=np.int64)
    directions = np.zeros((size, width), dtype=np.int64)
    distances = np.zeros((size, width), dtype=np.int64)
    groups: dict[str, list[tuple[int, int]]] = {}
    for index, identifier in enumerate(ids):
        video, order = dialogue_key(str(identifier))
        groups.setdefault(video, []).append((order, index))
    for group in groups.values():
        ordered = [index for _, index in sorted(group)]
        for position, index in enumerate(ordered):
            previous = ordered[max(0, position - window) : position]
            following = ordered[position + 1 : position + 1 + window]
            for slot, neighbor in enumerate(reversed(previous)):
                neighbors[index, slot] = neighbor
                directions[index, slot] = 0
                distances[index, slot] = slot
            for offset, neighbor in enumerate(following):
                slot = window + offset
                neighbors[index, slot] = neighbor
                directions[index, slot] = 1
                distances[index, slot] = offset
    return neighbors, directions, distances


def masked_temporal_mean(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weight = mask.astype(np.float32)[..., None]
    numerator = (features.astype(np.float32) * weight).sum(axis=1)
    denominator = weight.sum(axis=1).clip(min=1.0)
    result = numerator / denominator
    if not np.isfinite(result).all():
        raise ValueError("Non-finite pooled multimodal feature")
    return result.astype(np.float32)


class DirectedGraphMessage(nn.Module):
    def __init__(self, dimension: int, window: int, dropout: float) -> None:
        super().__init__()
        self.direction_embedding = nn.Embedding(2, dimension)
        self.distance_embedding = nn.Embedding(window, dimension)
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        interaction_dimension = 4 * dimension
        self.candidate = nn.Sequential(
            nn.LayerNorm(interaction_dimension),
            nn.Linear(interaction_dimension, 2 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension, dimension),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(interaction_dimension),
            nn.Linear(interaction_dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -1.0)
        self.norm = nn.LayerNorm(dimension)

    def forward(
        self,
        state: Tensor,
        neighbor_index: Tensor,
        direction: Tensor,
        distance: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        available = neighbor_index.ge(0)
        safe_index = neighbor_index.clamp_min(0)
        neighbor = state[safe_index]
        neighbor = (
            neighbor
            + self.direction_embedding(direction)
            + self.distance_embedding(distance)
        )
        score = (
            self.query(state).unsqueeze(1) * self.key(neighbor)
        ).sum(dim=-1) / math.sqrt(state.size(-1))
        score = score.masked_fill(~available, -1e4)
        attention = torch.softmax(score.float(), dim=1) * available.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        message = (attention.unsqueeze(-1).to(state.dtype) * self.value(neighbor)).sum(
            dim=1
        )
        interaction = torch.cat(
            [state, message, torch.abs(state - message), state * message], dim=-1
        )
        has_neighbor = available.any(dim=1).to(state.dtype)
        gate = torch.sigmoid(self.gate(interaction).squeeze(-1)) * has_neighbor
        updated = self.norm(
            state + gate.unsqueeze(-1) * self.candidate(interaction)
        )
        return updated, gate, attention


class DialogueGraphEmotionNet(nn.Module):
    def __init__(
        self,
        semantic_dimension: int,
        audio_dimension: int,
        vision_dimension: int,
        hidden_dimension: int = 160,
        window: int = 2,
        depth: int = 2,
        dropout: float = 0.20,
        polarity_adapter_maximum: float = 1.0,
    ) -> None:
        super().__init__()
        self.polarity_adapter_maximum = float(polarity_adapter_maximum)
        self.semantic = nn.Sequential(
            nn.LayerNorm(semantic_dimension),
            nn.Linear(semantic_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emotion = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, hidden_dimension // 2),
            nn.GELU(),
        )
        self.audio = nn.Sequential(
            nn.LayerNorm(audio_dimension),
            nn.Linear(audio_dimension, hidden_dimension // 2),
            nn.GELU(),
        )
        self.vision = nn.Sequential(
            nn.LayerNorm(vision_dimension),
            nn.Linear(vision_dimension, hidden_dimension // 2),
            nn.GELU(),
        )
        input_dimension = hidden_dimension + 3 * (hidden_dimension // 2)
        self.node_fusion = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
        )
        self.graph = nn.ModuleList(
            [
                DirectedGraphMessage(hidden_dimension, window, dropout)
                for _ in range(depth)
            ]
        )
        interaction_dimension = 4 * hidden_dimension
        self.neutral_head = nn.Sequential(
            nn.LayerNorm(interaction_dimension),
            nn.Linear(interaction_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        self.polarity_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension // 2, 1),
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dimension),
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(
        self,
        emotion_logits: Tensor,
        semantic: Tensor,
        audio: Tensor,
        vision: Tensor,
        neighbor_index: Tensor,
        direction: Tensor,
        distance: Tensor,
    ) -> dict[str, Tensor]:
        current = self.node_fusion(
            torch.cat(
                [
                    self.semantic(semantic),
                    self.emotion(emotion_logits),
                    self.audio(audio),
                    self.vision(vision),
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
        interaction = torch.cat(
            [
                current,
                contextual,
                torch.abs(current - contextual),
                current * contextual,
            ],
            dim=-1,
        )
        neutral_logit = self.neutral_head(interaction).squeeze(-1).float()
        raw_emotion_probability = torch.softmax(emotion_logits.float(), dim=-1)
        negative = raw_emotion_probability[:, [3, 4, 5, 6]].sum(dim=1)
        positive = raw_emotion_probability[:, 1]
        polarity_prior = positive.clamp_min(1e-8).log() - negative.clamp_min(1e-8).log()
        polarity_adapter = self.polarity_adapter_maximum * torch.tanh(
            self.polarity_adapter(current).squeeze(-1).float()
        )
        polarity_logit = polarity_prior + polarity_adapter
        neutral_probability = torch.sigmoid(neutral_logit).clamp(1e-6, 1.0 - 1e-6)
        positive_given_polar = torch.sigmoid(polarity_logit)
        probability = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_given_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_given_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probability = probability / probability.sum(dim=1, keepdim=True)
        regression = 3.0 * torch.tanh(
            self.regression_head(torch.cat([current, contextual], dim=-1)).squeeze(-1)
            / 3.0
        )
        graph_gate = torch.stack(gates, dim=1).mean(dim=1)
        return {
            "probabilities": probability,
            "regression": regression.float(),
            "neutral_probability": neutral_probability,
            "polarity_prior": polarity_prior,
            "polarity_adapter": polarity_adapter,
            "graph_gate": graph_gate,
        }


def subset_tensor(value: np.ndarray, index: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(value[index].astype(np.float32)).to(device)


def class_weights(target: np.ndarray, device: torch.device) -> Tensor:
    count = np.bincount(target, minlength=len(CLASS_NAMES)).astype(np.float64)
    weight = np.power(count.mean() / count.clip(min=1.0), 0.5)
    weight /= weight.mean()
    return torch.tensor(weight, dtype=torch.float32, device=device)


def fit_fold(
    emotion_logits: np.ndarray,
    semantic: np.ndarray,
    audio: np.ndarray,
    vision: np.ndarray,
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
    model = DialogueGraphEmotionNet(
        semantic_dimension=semantic.shape[1],
        audio_dimension=audio.shape[1],
        vision_dimension=vision.shape[1],
        window=window,
    ).to(device)
    train_neighbors = build_directional_neighbors(ids[train_index], window)
    heldout_neighbors = build_directional_neighbors(ids[heldout_index], window)

    train_input = {
        "emotion_logits": subset_tensor(emotion_logits, train_index, device),
        "semantic": subset_tensor(semantic, train_index, device),
        "audio": subset_tensor(audio, train_index, device),
        "vision": subset_tensor(vision, train_index, device),
        "neighbor_index": torch.from_numpy(train_neighbors[0]).to(device),
        "direction": torch.from_numpy(train_neighbors[1]).to(device),
        "distance": torch.from_numpy(train_neighbors[2]).to(device),
    }
    heldout_input = {
        "emotion_logits": subset_tensor(emotion_logits, heldout_index, device),
        "semantic": subset_tensor(semantic, heldout_index, device),
        "audio": subset_tensor(audio, heldout_index, device),
        "vision": subset_tensor(vision, heldout_index, device),
        "neighbor_index": torch.from_numpy(heldout_neighbors[0]).to(device),
        "direction": torch.from_numpy(heldout_neighbors[1]).to(device),
        "distance": torch.from_numpy(heldout_neighbors[2]).to(device),
    }
    train_target = torch.from_numpy(target[train_index]).long().to(device)
    train_intensity = torch.from_numpy(intensity[train_index]).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
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
            raise FloatingPointError(f"Non-finite graph loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % 20 == 0:
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
    probability = output["probabilities"].cpu().numpy().astype(np.float64)
    regression = output["regression"].cpu().numpy().astype(np.float64)
    diagnostics = {
        name: {
            "mean": float(output[name].mean().cpu()),
            "std": float(output[name].std(unbiased=False).cpu()),
            "min": float(output[name].min().cpu()),
            "max": float(output[name].max().cpu()),
        }
        for name in (
            "neutral_probability",
            "polarity_prior",
            "polarity_adapter",
            "graph_gate",
        )
    }
    return probability, regression, {"history": history, "diagnostics": diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.window < 1:
        raise ValueError("epochs and window must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    loaded = load_pickle(args.data)
    arrays = parse_split(
        {"train": loaded["train"]}, "train", "text_shared", require_labels=False
    )
    parent = load_aligned([args.parent_oof], require_fold=True)[0]
    parent_index = parent.set_index(parent["id"].astype(str), drop=False)
    expected_ids = arrays.ids.astype(str)
    if set(parent_index.index) != set(expected_ids):
        raise ValueError("Parent OOF IDs do not match train data")
    parent = parent_index.loc[expected_ids].reset_index(drop=True)
    cache = np.load(args.feature_cache, allow_pickle=False)
    if not np.array_equal(cache["ids"].astype(str), expected_ids):
        raise ValueError("Frozen feature cache IDs do not match train data")
    frozen = cache["features"].astype(np.float32)
    if frozen.shape != (arrays.size, 1031):
        raise ValueError(f"Unexpected frozen feature shape: {frozen.shape}")
    emotion_logits, semantic = frozen[:, :7], frozen[:, 7:]
    audio = masked_temporal_mean(arrays.features["audio"], arrays.masks["audio"])
    vision = masked_temporal_mean(arrays.features["vision"], arrays.masks["vision"])
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    target = parent["true_label"].map(label_map).to_numpy(np.int64)
    intensity = parent["true_intensity"].to_numpy(np.float32)
    folds = parent["fold"].to_numpy(np.int64)
    oof_probability = np.zeros((arrays.size, len(CLASS_NAMES)), dtype=np.float64)
    oof_regression = np.zeros(arrays.size, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    started = time.time()
    for fold in sorted(np.unique(folds).tolist()):
        train_index = np.flatnonzero(folds != fold)
        heldout_index = np.flatnonzero(folds == fold)
        train_groups = {dialogue_key(value)[0] for value in expected_ids[train_index]}
        heldout_groups = {dialogue_key(value)[0] for value in expected_ids[heldout_index]}
        if train_groups.intersection(heldout_groups):
            raise RuntimeError(f"Dialogue leakage in fold {fold}")
        probability, regression, detail = fit_fold(
            emotion_logits,
            semantic,
            audio,
            vision,
            expected_ids,
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
            target[heldout_index],
            probability,
            intensity[heldout_index],
            regression,
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
    prediction = pd.DataFrame(
        {
            "id": expected_ids,
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
        "scope": "train_only_grouped_oof_no_valid_or_test_access",
        "architecture": "frozen_emoberta_large_directed_dialogue_graph_valence_boundary",
        "seed": args.seed,
        "epochs": args.epochs,
        "window": args.window,
        "folds": int(len(np.unique(folds))),
        "samples": arrays.size,
        "groups": int(len({dialogue_key(value)[0] for value in expected_ids})),
        "feature_cache": str(Path(args.feature_cache)),
        "checkpoint_selection_on_heldout_fold": False,
        "elapsed_minutes": (time.time() - started) / 60.0,
        "oof_metrics": metrics,
        "fold_reports": fold_reports,
        "artifact": "train_oof_predictions.csv",
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
