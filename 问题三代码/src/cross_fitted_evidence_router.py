from __future__ import annotations

"""Cross-fitted dynamic evidence routing with an explicit parent fallback.

The architecture adapts the sample-wise modality policy idea from AdaMML
(ICCV 2021) to sentiment experts.  Unlike a free stacking head, every routed
prediction remains a convex expert mixture plus a bounded residual, and the
first route is an explicit reject/fallback to the task-trained parent.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASS_NAMES
from .metrics import compute_metrics
from .utils import atomic_torch_save, save_json, seed_everything


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)
POSITIVE_EMOTIONS = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}
NEGATIVE_EMOTIONS = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}
COGNITIVE_EMOTIONS = {"confusion", "curiosity", "realization", "surprise"}


def _groups(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.rsplit("$_$", 1)[0] if "$_$" in value else value for value in ids],
        dtype=object,
    )


def _load_aligned(path: str | Path, expected_ids: np.ndarray | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "true_label", *PROBABILITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate ids")
    if expected_ids is not None:
        indexed = frame.set_index("id", drop=False)
        if set(indexed.index) != set(expected_ids):
            raise ValueError(f"{path} ids do not align with the parent evidence")
        frame = indexed.loc[expected_ids].reset_index(drop=True)
    probability = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError(f"{path} has invalid probabilities")
    probability /= probability.sum(axis=1, keepdims=True).clip(min=1e-12)
    frame.loc[:, PROBABILITY_COLUMNS] = probability
    return frame


def _ontology_probability(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prefix = "emotion_probability_"
    names = sorted(column[len(prefix) :] for column in frame if column.startswith(prefix))
    if "neutral" not in names:
        raise ValueError("Ontology evidence has no neutral probability")

    def values(selected: set[str]) -> np.ndarray:
        columns = [f"{prefix}{name}" for name in sorted(selected) if f"{prefix}{name}" in frame]
        if not columns:
            return np.zeros((len(frame), 1), dtype=np.float64)
        return frame.loc[:, columns].to_numpy(np.float64)

    positive = values(POSITIVE_EMOTIONS)
    negative = values(NEGATIVE_EMOTIONS)
    cognitive = values(COGNITIVE_EMOTIONS)
    neutral = frame[f"{prefix}neutral"].to_numpy(np.float64)
    positive_score = positive.max(axis=1) + 0.25 * positive.mean(axis=1)
    negative_score = negative.max(axis=1) + 0.25 * negative.mean(axis=1)
    score = np.column_stack([negative_score, neutral, positive_score]).clip(min=1e-8)
    probability = score / score.sum(axis=1, keepdims=True)
    all_non_neutral = frame.loc[
        :, [f"{prefix}{name}" for name in names if name != "neutral"]
    ].to_numpy(np.float64)
    semantic = np.column_stack(
        [
            neutral,
            positive.max(axis=1),
            positive.mean(axis=1),
            negative.max(axis=1),
            negative.mean(axis=1),
            cognitive.max(axis=1),
            all_non_neutral.max(axis=1),
            neutral - all_non_neutral.max(axis=1),
        ]
    )
    semantic_names = [
        "ontology_neutral",
        "ontology_positive_max",
        "ontology_positive_mean",
        "ontology_negative_max",
        "ontology_negative_mean",
        "ontology_cognitive_max",
        "ontology_non_neutral_max",
        "ontology_neutral_gap",
    ]
    return probability, semantic, semantic_names


def _diagnostics(probability: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    ordered = np.sort(probability, axis=1)
    entropy = -(
        probability.clip(min=1e-8) * np.log(probability.clip(min=1e-8))
    ).sum(axis=1) / math.log(probability.shape[1])
    matrix = np.column_stack(
        [probability.max(axis=1), ordered[:, -1] - ordered[:, -2], entropy]
    )
    return matrix, [f"{prefix}_confidence", f"{prefix}_margin", f"{prefix}_entropy"]


def build_evidence(
    parent_path: str | Path,
    sentiment_path: str | Path,
    ontology_path: str | Path,
) -> dict[str, Any]:
    parent = _load_aligned(parent_path)
    ids = parent["id"].astype(str).to_numpy()
    sentiment = _load_aligned(sentiment_path, ids)
    ontology = pd.read_csv(ontology_path)
    ontology["id"] = ontology["id"].astype(str)
    if ontology["id"].duplicated().any():
        raise ValueError(f"{ontology_path} contains duplicate ids")
    ontology = ontology.set_index("id", drop=False)
    if set(ontology.index) != set(ids):
        raise ValueError("Ontology ids do not align with the parent evidence")
    ontology = ontology.loc[ids].reset_index(drop=True)

    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    targets = parent["true_label"].map(label_map)
    if targets.isna().any():
        raise ValueError("Parent evidence contains unknown labels")
    for name, frame in (("sentiment", sentiment), ("ontology", ontology)):
        if not np.array_equal(
            frame["true_label"].astype(str).to_numpy(),
            parent["true_label"].astype(str).to_numpy(),
        ):
            raise ValueError(f"{name} labels do not align with the parent evidence")

    parent_probability = parent.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    sentiment_probability = sentiment.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    ontology_probability, semantic, semantic_names = _ontology_probability(ontology)
    expert_probability = np.stack(
        [parent_probability, sentiment_probability, ontology_probability], axis=1
    )

    feature_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    for name, probability in zip(
        ("parent", "sentiment", "ontology"),
        (parent_probability, sentiment_probability, ontology_probability),
    ):
        logit = np.log(probability.clip(min=1e-8))
        centered_logit = logit - logit.mean(axis=1, keepdims=True)
        feature_parts.extend([probability, centered_logit])
        feature_names.extend([f"{name}_{label}_probability" for label in CLASS_NAMES])
        feature_names.extend([f"{name}_{label}_centered_logit" for label in CLASS_NAMES])
        diagnostic, diagnostic_names = _diagnostics(probability, name)
        feature_parts.append(diagnostic)
        feature_names.extend(diagnostic_names)
    feature_parts.append(semantic)
    feature_names.extend(semantic_names)
    disagreement = np.column_stack(
        [
            parent_probability.argmax(axis=1)
            != sentiment_probability.argmax(axis=1),
            parent_probability.argmax(axis=1)
            != ontology_probability.argmax(axis=1),
            sentiment_probability.argmax(axis=1)
            != ontology_probability.argmax(axis=1),
        ]
    ).astype(np.float64)
    feature_parts.append(disagreement)
    feature_names.extend(
        [
            "parent_sentiment_disagreement",
            "parent_ontology_disagreement",
            "sentiment_ontology_disagreement",
        ]
    )
    for name, frame in (("parent", parent), ("sentiment", sentiment)):
        if "predicted_intensity" in frame:
            feature_parts.append(
                frame["predicted_intensity"].to_numpy(np.float64)[:, None]
            )
            feature_names.append(f"{name}_predicted_intensity")
    features = np.concatenate(feature_parts, axis=1)
    if not np.isfinite(features).all():
        raise ValueError("Router features contain NaN/Inf")
    return {
        "ids": ids,
        "targets": targets.to_numpy(np.int64),
        "true_intensity": parent["true_intensity"].to_numpy(np.float64),
        "parent_intensity": parent["predicted_intensity"].to_numpy(np.float64),
        "features": features,
        "feature_names": feature_names,
        "experts": expert_probability,
        "expert_names": ["parent", "sentiment", "ontology"],
    }


class CrossFittedEvidenceRouter(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 64,
        expert_count: int = 3,
        dropout: float = 0.15,
        parent_bias: float = 2.2,
        maximum_logit_residual: float = 0.75,
    ) -> None:
        super().__init__()
        self.maximum_logit_residual = float(maximum_logit_residual)
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.route = nn.Linear(hidden_dimension, expert_count)
        self.residual = nn.Linear(hidden_dimension, 3)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        with torch.no_grad():
            self.route.bias[0] = parent_bias

    def forward(self, features: Tensor, experts: Tensor) -> dict[str, Tensor]:
        hidden = self.network(features)
        route_weights = torch.softmax(self.route(hidden).float(), dim=-1)
        mixture = torch.sum(route_weights.unsqueeze(-1) * experts.float(), dim=1)
        mixture = mixture.clamp_min(1e-8)
        correction = self.maximum_logit_residual * torch.tanh(
            self.residual(hidden).float()
        )
        correction = correction - correction.mean(dim=-1, keepdim=True)
        # Residual calibration is disabled when the router rejects to the parent.
        accept = 1.0 - route_weights[:, :1]
        bounded_correction = accept * correction
        logits = mixture.log() + bounded_correction
        probability = torch.softmax(logits, dim=-1)
        return {
            "probabilities": probability,
            "route_weights": route_weights,
            "accept_probability": accept.squeeze(-1),
            "bounded_correction": bounded_correction,
        }


def _class_weights(targets: np.ndarray, device: torch.device) -> Tensor:
    counts = np.bincount(targets, minlength=3).astype(np.float64)
    weights = np.sqrt(len(targets) / (3.0 * counts.clip(min=1.0)))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def fit_router(
    features: np.ndarray,
    experts: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    hidden_dimension: int,
    batch_size: int,
) -> tuple[CrossFittedEvidenceRouter, list[dict[str, float]]]:
    seed_everything(seed)
    model = CrossFittedEvidenceRouter(
        features.shape[1], hidden_dimension=hidden_dimension
    ).to(device)
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(experts, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weights = _class_weights(targets, device)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"loss": 0.0, "classification": 0.0, "selective": 0.0, "accept": 0.0}
        samples = 0
        for cpu_features, cpu_experts, cpu_targets in loader:
            batch_features = cpu_features.to(device)
            batch_experts = cpu_experts.to(device)
            batch_targets = cpu_targets.to(device)
            output = model(batch_features, batch_experts)
            probability = output["probabilities"].clamp_min(1e-8)
            classification = F.nll_loss(
                probability.log(), batch_targets, weight=weights
            )
            row = torch.arange(len(batch_targets), device=device)
            expert_nll = -batch_experts[row, :, batch_targets].clamp_min(1e-8).log()
            route_target = (
                expert_nll[:, 1:].min(dim=1).values + 0.05 < expert_nll[:, 0]
            ).float()
            selective = F.binary_cross_entropy(
                output["accept_probability"].clamp(1e-6, 1.0 - 1e-6),
                route_target,
            )
            accept_cost = output["accept_probability"].mean()
            residual_cost = output["bounded_correction"].square().mean()
            loss = (
                classification
                + 0.15 * selective
                + 0.025 * accept_cost
                + 0.01 * residual_cost
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(batch_targets)
            samples += size
            sums["loss"] += float(loss.detach()) * size
            sums["classification"] += float(classification.detach()) * size
            sums["selective"] += float(selective.detach()) * size
            sums["accept"] += float(accept_cost.detach()) * size
        scheduler.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    **{name: value / samples for name, value in sums.items()},
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    return model, history


@torch.inference_mode()
def predict_router(
    model: CrossFittedEvidenceRouter,
    features: np.ndarray,
    experts: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = model(
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(experts, dtype=torch.float32, device=device),
    )
    return (
        output["probabilities"].cpu().numpy(),
        output["route_weights"].cpu().numpy(),
        output["bounded_correction"].cpu().numpy(),
    )


def _prediction_frame(
    evidence: dict[str, Any],
    probability: np.ndarray,
    routes: np.ndarray,
    correction: np.ndarray,
    folds: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": evidence["ids"],
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": evidence["parent_intensity"],
            "true_label": [CLASS_NAMES[index] for index in evidence["targets"]],
            "true_intensity": evidence["true_intensity"],
            "parent_route_weight": routes[:, 0],
            "sentiment_route_weight": routes[:, 1],
            "ontology_route_weight": routes[:, 2],
            "residual_l2": np.linalg.norm(correction, axis=1),
        }
    )
    if folds is not None:
        frame.insert(1, "router_fold", folds)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted dynamic evidence router")
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--parent-valid", required=True)
    parser.add_argument("--sentiment-train", required=True)
    parser.add_argument("--sentiment-valid", required=True)
    parser.add_argument("--ontology-train", required=True)
    parser.add_argument("--ontology-valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.folds < 3:
        raise ValueError("At least three router folds are required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Phase A: only train evidence is loaded while architecture parameters fit.
    train = build_evidence(
        args.parent_oof, args.sentiment_train, args.ontology_train
    )
    groups = _groups(train["ids"])
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    folds = list(
        splitter.split(np.zeros(len(train["ids"])), train["targets"], groups)
    )
    oof_probability = np.zeros((len(train["ids"]), 3), dtype=np.float64)
    oof_routes = np.zeros((len(train["ids"]), 3), dtype=np.float64)
    oof_correction = np.zeros((len(train["ids"]), 3), dtype=np.float64)
    fold_ids = np.full(len(train["ids"]), -1, dtype=np.int64)
    fold_reports: list[dict[str, Any]] = []
    for fold, (fit_index, heldout_index) in enumerate(folds):
        fit_groups = set(groups[fit_index])
        heldout_groups = set(groups[heldout_index])
        if fit_groups.intersection(heldout_groups):
            raise RuntimeError(f"Router fold {fold} has video-group leakage")
        scaler = StandardScaler().fit(train["features"][fit_index])
        model, history = fit_router(
            scaler.transform(train["features"][fit_index]),
            train["experts"][fit_index],
            train["targets"][fit_index],
            device,
            args.seed + 100 + fold,
            args.epochs,
            args.hidden_dimension,
            args.batch_size,
        )
        probability, routes, correction = predict_router(
            model,
            scaler.transform(train["features"][heldout_index]),
            train["experts"][heldout_index],
            device,
        )
        oof_probability[heldout_index] = probability
        oof_routes[heldout_index] = routes
        oof_correction[heldout_index] = correction
        fold_ids[heldout_index] = fold
        metrics = compute_metrics(
            train["targets"][heldout_index],
            probability,
            train["true_intensity"][heldout_index],
            train["parent_intensity"][heldout_index],
        )
        fold_reports.append(
            {
                "fold": fold,
                "fit_samples": int(len(fit_index)),
                "heldout_samples": int(len(heldout_index)),
                "group_overlap": 0,
                "metrics": metrics,
                "history": history,
            }
        )
    if (fold_ids < 0).any():
        raise RuntimeError("Router OOF did not cover every train sample")
    router_oof_metrics = compute_metrics(
        train["targets"],
        oof_probability,
        train["true_intensity"],
        train["parent_intensity"],
    )
    parent_oof_metrics = compute_metrics(
        train["targets"],
        train["experts"][:, 0],
        train["true_intensity"],
        train["parent_intensity"],
    )
    _prediction_frame(
        train, oof_probability, oof_routes, oof_correction, fold_ids
    ).to_csv(
        output / "train_oof_router_predictions.csv", index=False, encoding="utf-8-sig"
    )

    final_scaler = StandardScaler().fit(train["features"])
    final_model, final_history = fit_router(
        final_scaler.transform(train["features"]),
        train["experts"],
        train["targets"],
        device,
        args.seed,
        args.epochs,
        args.hidden_dimension,
        args.batch_size,
    )
    joblib.dump(final_scaler, output / "feature_scaler.joblib")
    atomic_torch_save(
        {
            "model_state": final_model.state_dict(),
            "input_dimension": int(train["features"].shape[1]),
            "hidden_dimension": args.hidden_dimension,
            "feature_names": train["feature_names"],
            "expert_names": train["expert_names"],
            "seed": args.seed,
            "epochs": args.epochs,
        },
        output / "router.pt",
    )

    # Phase B: official valid evidence is loaded only after the final router locks.
    valid = build_evidence(
        args.parent_valid, args.sentiment_valid, args.ontology_valid
    )
    if valid["feature_names"] != train["feature_names"]:
        raise RuntimeError("Train/valid router feature schemas differ")
    valid_probability, valid_routes, valid_correction = predict_router(
        final_model,
        final_scaler.transform(valid["features"]),
        valid["experts"],
        device,
    )
    valid_metrics = compute_metrics(
        valid["targets"],
        valid_probability,
        valid["true_intensity"],
        valid["parent_intensity"],
    )
    parent_valid_metrics = compute_metrics(
        valid["targets"],
        valid["experts"][:, 0],
        valid["true_intensity"],
        valid["parent_intensity"],
    )
    _prediction_frame(
        valid, valid_probability, valid_routes, valid_correction
    ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "scope": "train_grouped_oof_selection_then_single_official_validation_no_test_access",
        "architecture": "dynamic_expert_mixture_with_parent_reject_and_bounded_residual",
        "research_basis": {
            "AdaMML_ICCV_2021": "https://github.com/IBM/AdaMML",
            "MISA_ACM_MM_2020": "https://github.com/declare-lab/MISA",
            "implementation_note": "conceptual adaptation; no external source copied",
        },
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "hidden_dimension": args.hidden_dimension,
        "train_samples": int(len(train["ids"])),
        "train_groups": int(len(set(groups))),
        "feature_dimension": int(train["features"].shape[1]),
        "feature_names": train["feature_names"],
        "expert_names": train["expert_names"],
        "parent_oof_metrics": parent_oof_metrics,
        "router_oof_metrics": router_oof_metrics,
        "fold_reports": fold_reports,
        "final_training_history": final_history,
        "parent_valid_metrics": parent_valid_metrics,
        "valid": valid_metrics,
        "valid_route_weight_mean": {
            name: float(valid_routes[:, index].mean())
            for index, name in enumerate(train["expert_names"])
        },
        "valid_route_weight_std": {
            name: float(valid_routes[:, index].std())
            for index, name in enumerate(train["expert_names"])
        },
        "valid_changed_decisions": int(
            np.sum(valid_probability.argmax(1) != valid["experts"][:, 0].argmax(1))
        ),
        "artifacts": {
            "router": "router.pt",
            "scaler": "feature_scaler.joblib",
            "train_oof_predictions": "train_oof_router_predictions.csv",
            "valid_predictions": "valid_predictions.csv",
        },
    }
    save_json(report, output / "final_metrics.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
