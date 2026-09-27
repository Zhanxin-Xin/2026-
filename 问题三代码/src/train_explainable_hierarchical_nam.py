"""Leakage-safe hierarchical Neural Additive Model over frozen experts.

EXP184 keeps the vote/probability consensus as an exact parent anchor and
learns two small, intrinsically interpretable residual axes:

1. Neutral versus Polar evidence.
2. Positive versus Negative evidence inside the Polar mass.

Every scalar concept owns an independent one-dimensional neural shape
function.  A contribution is ``f(x) - f(0)`` and is bounded independently,
so setting one standardized concept to zero removes exactly that concept's
logit contribution.  Zero-initialized output layers make the untrained model
identical to the frozen parent for every sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import yaml

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics


LABELS = np.asarray(CLASS_NAMES)
LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}
EPSILON = 1e-7


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20260924
    epochs: int = 220
    learning_rate: float = 0.01
    weight_decay: float = 0.001
    hidden_dim: int = 8
    feature_dropout: float = 0.10
    neutral_feature_bound: float = 0.08
    polarity_feature_bound: float = 0.06
    neutral_loss_weight: float = 0.25
    polarity_loss_weight: float = 0.15
    contribution_l1_weight: float = 0.015
    residual_l2_weight: float = 0.005
    class_balance_power: float = 0.50
    gradient_clip_norm: float = 5.0
    # The NAM is trained as a residual teacher at full strength.  Deployment
    # can then place that residual inside a fixed trust region without changing
    # the parent anchor or the exact additive explanation identity.
    neutral_inference_trust: float = 1.0
    polarity_inference_trust: float = 1.0


@dataclass(frozen=True)
class ConceptBundle:
    values: np.ndarray
    names: list[str]
    groups: list[str]
    parent_probability: np.ndarray
    mean_probability: np.ndarray
    regression: np.ndarray


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _normalise_probability(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-8, None)
    return values / values.sum(axis=-1, keepdims=True).clip(min=1e-12)


def build_concepts(frames: list[pd.DataFrame]) -> ConceptBundle:
    """Build named scalar concepts without using labels."""
    probability = np.stack(
        [
            _normalise_probability(
                frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
            )
            for frame in frames
        ],
        axis=1,
    )
    intensity = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    mean_probability = probability.mean(axis=1)
    vote_probability = np.eye(3, dtype=np.float64)[
        probability.argmax(axis=-1)
    ].mean(axis=1)
    parent = _normalise_probability(0.5 * (mean_probability + vote_probability))

    parts: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []

    def add(values: np.ndarray, name: str, group: str) -> None:
        parts.append(np.asarray(values, dtype=np.float64).reshape(-1, 1))
        names.append(name)
        groups.append(group)

    for expert in range(probability.shape[1]):
        p = probability[:, expert, :]
        ordered = np.sort(p, axis=1)
        add(
            np.log(p[:, 1]) - np.log(p[:, 0] + p[:, 2]),
            f"expert_{expert + 1}_neutral_log_odds",
            f"expert_{expert + 1}",
        )
        add(
            np.log(p[:, 2]) - np.log(p[:, 0]),
            f"expert_{expert + 1}_positive_negative_log_odds",
            f"expert_{expert + 1}",
        )
        add(
            -(p * np.log(p)).sum(axis=1) / np.log(3.0),
            f"expert_{expert + 1}_entropy",
            f"expert_{expert + 1}",
        )
        add(p.max(axis=1), f"expert_{expert + 1}_confidence", f"expert_{expert + 1}")
        add(
            ordered[:, -1] - ordered[:, -2],
            f"expert_{expert + 1}_margin",
            f"expert_{expert + 1}",
        )
        add(
            intensity[:, expert],
            f"expert_{expert + 1}_predicted_intensity",
            f"expert_{expert + 1}",
        )

    probability_statistics = {
        "mean": mean_probability,
        "std": probability.std(axis=1),
        "range": probability.max(axis=1) - probability.min(axis=1),
        "vote": vote_probability,
        "parent": parent,
    }
    for statistic, values in probability_statistics.items():
        for class_index, class_name in enumerate(CLASS_NAMES):
            add(
                values[:, class_index],
                f"consensus_{statistic}_{class_name.lower()}",
                f"consensus_{statistic}",
            )
    add(
        np.abs(mean_probability - vote_probability).sum(axis=1),
        "consensus_probability_vote_l1_disagreement",
        "consensus_disagreement",
    )
    add(
        -(vote_probability.clip(min=1e-8) * np.log(vote_probability.clip(min=1e-8))).sum(axis=1)
        / np.log(3.0),
        "consensus_vote_entropy",
        "consensus_disagreement",
    )
    add(
        -(parent * np.log(parent)).sum(axis=1) / np.log(3.0),
        "consensus_parent_entropy",
        "consensus_disagreement",
    )
    add(intensity.mean(axis=1), "consensus_intensity_mean", "intensity_consensus")
    add(intensity.std(axis=1), "consensus_intensity_std", "intensity_consensus")

    values = np.concatenate(parts, axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite concept value detected")
    return ConceptBundle(
        values=values,
        names=names,
        groups=groups,
        parent_probability=parent,
        mean_probability=mean_probability,
        regression=intensity.mean(axis=1),
    )


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=np.float64).mean(axis=0)
    scale = np.asarray(values, dtype=np.float64).std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


def apply_standardizer(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float64) - mean) / scale).astype(np.float32)


class ScalarShapeFunction(nn.Module):
    """One concept, one subnet; no cross-concept interaction is possible."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.output(self.hidden(value))


class AdditiveAxis(nn.Module):
    def __init__(self, feature_count: int, hidden_dim: int, feature_bound: float) -> None:
        super().__init__()
        self.shapes = nn.ModuleList(
            [ScalarShapeFunction(hidden_dim) for _ in range(feature_count)]
        )
        self.feature_bound = float(feature_bound)

    def forward(self, concepts: Tensor) -> tuple[Tensor, Tensor]:
        zero = torch.zeros_like(concepts[:, :1])
        raw = torch.cat(
            [
                shape(concepts[:, index : index + 1]) - shape(zero)
                for index, shape in enumerate(self.shapes)
            ],
            dim=1,
        )
        contributions = self.feature_bound * torch.tanh(
            raw / max(self.feature_bound, 1e-8)
        )
        return contributions.sum(dim=1), contributions


class HierarchicalExplainableNAM(nn.Module):
    def __init__(
        self,
        feature_count: int,
        hidden_dim: int = 8,
        neutral_feature_bound: float = 0.08,
        polarity_feature_bound: float = 0.06,
    ) -> None:
        super().__init__()
        self.neutral_axis = AdditiveAxis(
            feature_count, hidden_dim, neutral_feature_bound
        )
        self.polarity_axis = AdditiveAxis(
            feature_count, hidden_dim, polarity_feature_bound
        )

    @staticmethod
    def parent_axes(parent_probability: Tensor) -> tuple[Tensor, Tensor]:
        parent = parent_probability.clamp(min=EPSILON)
        parent = parent / parent.sum(dim=1, keepdim=True)
        neutral = parent[:, 1].clamp(min=EPSILON, max=1.0 - EPSILON)
        polar_mass = (parent[:, 0] + parent[:, 2]).clamp(min=EPSILON)
        positive_given_polar = (parent[:, 2] / polar_mass).clamp(
            min=EPSILON, max=1.0 - EPSILON
        )
        return torch.logit(neutral), torch.logit(positive_given_polar)

    def forward(
        self, concepts: Tensor, parent_probability: Tensor
    ) -> dict[str, Tensor]:
        parent_neutral_logit, parent_polarity_logit = self.parent_axes(
            parent_probability
        )
        neutral_residual, neutral_contributions = self.neutral_axis(concepts)
        polarity_residual, polarity_contributions = self.polarity_axis(concepts)
        neutral_logit = parent_neutral_logit + neutral_residual
        polarity_logit = parent_polarity_logit + polarity_residual
        neutral_probability = torch.sigmoid(neutral_logit)
        positive_given_polar = torch.sigmoid(polarity_logit)
        polar_mass = 1.0 - neutral_probability
        probability = torch.stack(
            [
                polar_mass * (1.0 - positive_given_polar),
                neutral_probability,
                polar_mass * positive_given_polar,
            ],
            dim=1,
        )
        return {
            "probability": probability,
            "neutral_logit": neutral_logit,
            "polarity_logit": polarity_logit,
            "neutral_residual": neutral_residual,
            "polarity_residual": polarity_residual,
            "neutral_contributions": neutral_contributions,
            "polarity_contributions": polarity_contributions,
        }


def class_sample_weights(targets: np.ndarray, power: float) -> np.ndarray:
    counts = np.bincount(targets, minlength=3).astype(np.float64)
    weights = (len(targets) / (3.0 * counts.clip(min=1.0))) ** float(power)
    weights /= weights.mean()
    return weights[targets].astype(np.float32)


def objective(
    output: dict[str, Tensor],
    targets: Tensor,
    sample_weights: Tensor,
    config: TrainConfig,
) -> tuple[Tensor, dict[str, float]]:
    probability = output["probability"].clamp(min=1e-7, max=1.0)
    selected = probability.gather(1, targets[:, None]).squeeze(1)
    cross_entropy = (-torch.log(selected) * sample_weights).mean()
    neutral_target = (targets == 1).float()
    neutral_loss = F.binary_cross_entropy(
        probability[:, 1], neutral_target, weight=sample_weights
    )
    polar = targets != 1
    if bool(polar.any()):
        polar_target = (targets[polar] == 2).float()
        positive_given_polar = probability[polar, 2] / (
            probability[polar, 0] + probability[polar, 2]
        ).clamp(min=1e-7)
        polarity_loss = F.binary_cross_entropy(
            positive_given_polar,
            polar_target,
            weight=sample_weights[polar],
        )
    else:
        polarity_loss = probability.sum() * 0.0
    contributions = torch.cat(
        [output["neutral_contributions"], output["polarity_contributions"]],
        dim=1,
    )
    contribution_l1 = contributions.abs().mean()
    residual_l2 = (
        output["neutral_residual"].square()
        + output["polarity_residual"].square()
    ).mean()
    total = (
        cross_entropy
        + config.neutral_loss_weight * neutral_loss
        + config.polarity_loss_weight * polarity_loss
        + config.contribution_l1_weight * contribution_l1
        + config.residual_l2_weight * residual_l2
    )
    components = {
        "total": float(total.detach().cpu()),
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "neutral_bce": float(neutral_loss.detach().cpu()),
        "polarity_bce": float(polarity_loss.detach().cpu()),
        "contribution_l1": float(contribution_l1.detach().cpu()),
        "residual_l2": float(residual_l2.detach().cpu()),
    }
    return total, components


def train_model(
    concepts: np.ndarray,
    parent_probability: np.ndarray,
    targets: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    seed: int,
) -> tuple[HierarchicalExplainableNAM, list[dict[str, float]]]:
    set_deterministic(seed)
    model = HierarchicalExplainableNAM(
        feature_count=concepts.shape[1],
        hidden_dim=config.hidden_dim,
        neutral_feature_bound=config.neutral_feature_bound,
        polarity_feature_bound=config.polarity_feature_bound,
    ).to(device)
    x = torch.as_tensor(concepts, dtype=torch.float32, device=device)
    parent = torch.as_tensor(parent_probability, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.long, device=device)
    weights = torch.as_tensor(
        class_sample_weights(targets, config.class_balance_power),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + 104729)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        if config.feature_dropout > 0.0:
            keep = torch.rand(
                x.shape, generator=generator, device=device, dtype=x.dtype
            ) >= config.feature_dropout
            epoch_x = x * keep
        else:
            epoch_x = x
        output = model(epoch_x, parent)
        loss, components = objective(output, y, weights, config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        optimizer.step()
        if epoch in {0, config.epochs - 1} or (epoch + 1) % 25 == 0:
            history.append(
                {
                    "epoch": int(epoch + 1),
                    **components,
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                }
            )
    return model, history


@torch.no_grad()
def predict(
    model: HierarchicalExplainableNAM,
    concepts: np.ndarray,
    parent_probability: np.ndarray,
    device: torch.device,
    neutral_trust: float = 1.0,
    polarity_trust: float = 1.0,
) -> dict[str, np.ndarray]:
    model.eval()
    parent = torch.as_tensor(parent_probability, dtype=torch.float32, device=device)
    output = model(torch.as_tensor(concepts, dtype=torch.float32, device=device), parent)
    if neutral_trust != 1.0 or polarity_trust != 1.0:
        parent_neutral_logit, parent_polarity_logit = model.parent_axes(parent)
        output["neutral_contributions"] = (
            output["neutral_contributions"] * float(neutral_trust)
        )
        output["polarity_contributions"] = (
            output["polarity_contributions"] * float(polarity_trust)
        )
        output["neutral_residual"] = output["neutral_contributions"].sum(dim=1)
        output["polarity_residual"] = output["polarity_contributions"].sum(dim=1)
        output["neutral_logit"] = parent_neutral_logit + output["neutral_residual"]
        output["polarity_logit"] = parent_polarity_logit + output["polarity_residual"]
        neutral_probability = torch.sigmoid(output["neutral_logit"])
        positive_given_polar = torch.sigmoid(output["polarity_logit"])
        polar_mass = 1.0 - neutral_probability
        output["probability"] = torch.stack(
            [
                polar_mass * (1.0 - positive_given_polar),
                neutral_probability,
                polar_mass * positive_given_polar,
            ],
            dim=1,
        )
    return {name: value.detach().cpu().numpy() for name, value in output.items()}


@torch.no_grad()
def deletion_faithfulness(
    model: HierarchicalExplainableNAM,
    concepts: np.ndarray,
    parent_probability: np.ndarray,
    device: torch.device,
    max_samples: int = 64,
    neutral_trust: float = 1.0,
    polarity_trust: float = 1.0,
) -> dict[str, float]:
    x = np.asarray(concepts[:max_samples], dtype=np.float32)
    parent = np.asarray(parent_probability[:max_samples], dtype=np.float32)
    original = predict(
        model, x, parent, device, neutral_trust, polarity_trust
    )
    expected_neutral: list[np.ndarray] = []
    observed_neutral: list[np.ndarray] = []
    expected_polarity: list[np.ndarray] = []
    observed_polarity: list[np.ndarray] = []
    for feature in range(x.shape[1]):
        deleted = x.copy()
        deleted[:, feature] = 0.0
        counterfactual = predict(
            model, deleted, parent, device, neutral_trust, polarity_trust
        )
        expected_neutral.append(original["neutral_contributions"][:, feature])
        observed_neutral.append(
            original["neutral_logit"] - counterfactual["neutral_logit"]
        )
        expected_polarity.append(original["polarity_contributions"][:, feature])
        observed_polarity.append(
            original["polarity_logit"] - counterfactual["polarity_logit"]
        )

    def statistics(expected_parts: Iterable[np.ndarray], observed_parts: Iterable[np.ndarray]) -> tuple[float, float]:
        expected = np.concatenate(list(expected_parts)).astype(np.float64)
        observed = np.concatenate(list(observed_parts)).astype(np.float64)
        max_error = float(np.max(np.abs(expected - observed)))
        if np.std(expected) < 1e-12 or np.std(observed) < 1e-12:
            correlation = 1.0 if max_error < 1e-6 else 0.0
        else:
            correlation = float(np.corrcoef(expected, observed)[0, 1])
        return max_error, correlation

    neutral_error, neutral_correlation = statistics(
        expected_neutral, observed_neutral
    )
    polarity_error, polarity_correlation = statistics(
        expected_polarity, observed_polarity
    )
    return {
        "samples_checked": int(len(x)),
        "concepts_checked": int(x.shape[1]),
        "neutral_max_absolute_error": neutral_error,
        "neutral_pearson": neutral_correlation,
        "polarity_max_absolute_error": polarity_error,
        "polarity_pearson": polarity_correlation,
    }


def prediction_frame(
    reference: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
    fold: np.ndarray | None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": LABELS[probability.argmax(axis=1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": reference["true_label"].astype(str),
            "true_intensity": reference["true_intensity"].to_numpy(np.float64),
        }
    )
    if fold is not None:
        output.insert(1, "meta_fold", fold)
    return output


def explanation_frame(
    reference: pd.DataFrame,
    names: list[str],
    result: dict[str, np.ndarray],
    fold: np.ndarray | None,
) -> pd.DataFrame:
    data: dict[str, Any] = {
        "id": reference["id"].astype(str).to_numpy(),
        "neutral_residual": result["neutral_residual"],
        "polarity_residual": result["polarity_residual"],
    }
    if fold is not None:
        data["meta_fold"] = fold
    for index, name in enumerate(names):
        data[f"neutral__{name}"] = result["neutral_contributions"][:, index]
        data[f"polarity__{name}"] = result["polarity_contributions"][:, index]
    return pd.DataFrame(data)


def feature_importance_frame(
    names: list[str], groups: list[str], result: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (name, group) in enumerate(zip(names, groups)):
        for axis in ("neutral", "polarity"):
            values = result[f"{axis}_contributions"][:, index]
            rows.append(
                {
                    "axis": axis,
                    "concept": name,
                    "concept_group": group,
                    "mean_absolute_contribution": float(np.mean(np.abs(values))),
                    "mean_signed_contribution": float(np.mean(values)),
                    "std_contribution": float(np.std(values)),
                    "maximum_absolute_contribution": float(np.max(np.abs(values))),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["axis", "mean_absolute_contribution"], ascending=[True, False]
    )


@torch.no_grad()
def shape_function_frame(
    model: HierarchicalExplainableNAM,
    names: list[str],
    groups: list[str],
    device: torch.device,
    points: int = 61,
    neutral_trust: float = 1.0,
    polarity_trust: float = 1.0,
) -> pd.DataFrame:
    model.eval()
    grid = torch.linspace(-3.0, 3.0, points, device=device)[:, None]
    zero = torch.zeros_like(grid)
    rows: list[dict[str, Any]] = []
    for index, (name, group) in enumerate(zip(names, groups)):
        for axis_name, axis in (
            ("neutral", model.neutral_axis),
            ("polarity", model.polarity_axis),
        ):
            raw = axis.shapes[index](grid) - axis.shapes[index](zero)
            contribution = axis.feature_bound * torch.tanh(
                raw / max(axis.feature_bound, 1e-8)
            )
            contribution = contribution * (
                float(neutral_trust) if axis_name == "neutral" else float(polarity_trust)
            )
            for standard_value, value in zip(
                grid[:, 0].detach().cpu().numpy(),
                contribution[:, 0].detach().cpu().numpy(),
            ):
                rows.append(
                    {
                        "axis": axis_name,
                        "concept": name,
                        "concept_group": group,
                        "standardized_value": float(standard_value),
                        "logit_contribution": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _metrics(
    reference: pd.DataFrame, probability: np.ndarray, regression: np.ndarray
) -> dict[str, Any]:
    targets = reference["true_label"].map(LABEL_MAP).to_numpy(np.int64)
    return compute_metrics(
        targets,
        probability,
        reference["true_intensity"].to_numpy(np.float64),
        regression,
    )


def _save_checkpoint(
    path: Path,
    model: HierarchicalExplainableNAM,
    mean: np.ndarray,
    scale: np.ndarray,
    names: list[str],
    groups: list[str],
    config: TrainConfig,
) -> None:
    torch.save(
        {
            "architecture": "hierarchical_explainable_neural_additive_model",
            "state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_scale": scale,
            "feature_names": names,
            "feature_groups": groups,
            "train_config": asdict(config),
        },
        path,
    )


def _load_config(path: Path) -> tuple[dict[str, Any], TrainConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    train = TrainConfig(**raw.get("training", {}))
    return raw, train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical explainable NAM with strict grouped meta-OOF"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--evaluate-valid",
        action="store_true",
        help="Load locked valid only after the configured OOF gate passes",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw_config, train_config = _load_config(config_path)
    oof_sources = [str(Path(value)) for value in raw_config["oof_sources"]]
    valid_sources = [str(Path(value)) for value in raw_config.get("valid_sources", [])]
    if len(oof_sources) < 2:
        raise ValueError("At least two OOF sources are required")
    if args.evaluate_valid and len(valid_sources) != len(oof_sources):
        raise ValueError("Every OOF source needs a matching valid source")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_deterministic(train_config.seed)

    # Locked validation files are intentionally not opened before the gate.
    oof_frames = load_aligned(oof_sources, require_fold=True)
    reference = oof_frames[0]
    concepts = build_concepts(oof_frames)
    folds = reference["fold"].to_numpy(np.int64)
    unique_folds = np.unique(folds)
    if len(unique_folds) != int(raw_config.get("meta_folds", 5)):
        raise RuntimeError(
            f"Expected {raw_config.get('meta_folds', 5)} aligned folds, got {unique_folds.tolist()}"
        )
    targets = reference["true_label"].map(LABEL_MAP).to_numpy(np.int64)
    oof_probability = np.zeros((len(reference), 3), dtype=np.float64)
    oof_neutral_contributions = np.zeros_like(concepts.values, dtype=np.float64)
    oof_polarity_contributions = np.zeros_like(concepts.values, dtype=np.float64)
    oof_neutral_residual = np.zeros(len(reference), dtype=np.float64)
    oof_polarity_residual = np.zeros(len(reference), dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    for fold in unique_folds:
        fit = folds != fold
        heldout = folds == fold
        fit_ids = reference.loc[fit, "id"].astype(str).to_numpy()
        heldout_ids = reference.loc[heldout, "id"].astype(str).to_numpy()
        fit_groups = {value.rsplit("$_$", 1)[0] for value in fit_ids}
        heldout_groups = {value.rsplit("$_$", 1)[0] for value in heldout_ids}
        overlap = fit_groups.intersection(heldout_groups)
        if overlap:
            raise RuntimeError(f"Meta fold {fold} has {len(overlap)} leaking groups")
        mean, scale = fit_standardizer(concepts.values[fit])
        fit_x = apply_standardizer(concepts.values[fit], mean, scale)
        heldout_x = apply_standardizer(concepts.values[heldout], mean, scale)
        model, history = train_model(
            fit_x,
            concepts.parent_probability[fit],
            targets[fit],
            train_config,
            device,
            train_config.seed + int(fold) * 1009,
        )
        result = predict(
            model,
            heldout_x,
            concepts.parent_probability[heldout],
            device,
            train_config.neutral_inference_trust,
            train_config.polarity_inference_trust,
        )
        oof_probability[heldout] = result["probability"]
        oof_neutral_contributions[heldout] = result["neutral_contributions"]
        oof_polarity_contributions[heldout] = result["polarity_contributions"]
        oof_neutral_residual[heldout] = result["neutral_residual"]
        oof_polarity_residual[heldout] = result["polarity_residual"]
        faithfulness = deletion_faithfulness(
            model,
            heldout_x,
            concepts.parent_probability[heldout],
            device,
            neutral_trust=train_config.neutral_inference_trust,
            polarity_trust=train_config.polarity_inference_trust,
        )
        fold_metrics = compute_metrics(
            targets[heldout],
            result["probability"],
            reference.loc[heldout, "true_intensity"].to_numpy(np.float64),
            concepts.regression[heldout],
        )
        _save_checkpoint(
            output / f"meta_fold_{int(fold)}.pt",
            model,
            mean,
            scale,
            concepts.names,
            concepts.groups,
            train_config,
        )
        fold_reports.append(
            {
                "fold": int(fold),
                "fit_samples": int(fit.sum()),
                "heldout_samples": int(heldout.sum()),
                "fit_groups": int(len(fit_groups)),
                "heldout_groups": int(len(heldout_groups)),
                "group_overlap": 0,
                "metrics": fold_metrics,
                "faithfulness": faithfulness,
                "training_history": history,
            }
        )

    parent_metrics = _metrics(
        reference, concepts.parent_probability, concepts.regression
    )
    oof_metrics = _metrics(reference, oof_probability, concepts.regression)
    oof_result = {
        "probability": oof_probability,
        "neutral_contributions": oof_neutral_contributions,
        "polarity_contributions": oof_polarity_contributions,
        "neutral_residual": oof_neutral_residual,
        "polarity_residual": oof_polarity_residual,
    }
    prediction_frame(reference, oof_probability, concepts.regression, folds).to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    explanation_frame(reference, concepts.names, oof_result, folds).to_csv(
        output / "concept_contributions_oof.csv", index=False, encoding="utf-8-sig"
    )
    feature_importance_frame(concepts.names, concepts.groups, oof_result).to_csv(
        output / "global_concept_importance_oof.csv", index=False, encoding="utf-8-sig"
    )

    full_mean, full_scale = fit_standardizer(concepts.values)
    full_x = apply_standardizer(concepts.values, full_mean, full_scale)
    deployment_model, deployment_history = train_model(
        full_x,
        concepts.parent_probability,
        targets,
        train_config,
        device,
        train_config.seed + 99991,
    )
    _save_checkpoint(
        output / "deployment_model.pt",
        deployment_model,
        full_mean,
        full_scale,
        concepts.names,
        concepts.groups,
        train_config,
    )
    shape_function_frame(
        deployment_model,
        concepts.names,
        concepts.groups,
        device,
        neutral_trust=train_config.neutral_inference_trust,
        polarity_trust=train_config.polarity_inference_trust,
    ).to_csv(output / "global_shape_functions.csv", index=False, encoding="utf-8-sig")

    gate = raw_config.get("oof_gate", {})
    gate_passed = bool(
        oof_metrics["accuracy"] >= float(gate.get("minimum_accuracy", 0.0))
        and oof_metrics["macro_f1"] >= float(gate.get("minimum_macro_f1", 0.0))
    )
    report: dict[str, Any] = {
        "experiment_id": raw_config.get("experiment_id", "EXP184"),
        "scope": "strict_grouped_meta_oof_no_test_access",
        "architecture": "parent_anchored_hierarchical_explainable_neural_additive_model",
        "innovation": [
            "two_stage_neutral_then_polar_decomposition",
            "one_scalar_concept_per_neural_shape_function",
            "exact_parent_identity_at_initialization",
            "bounded_additive_logit_residuals",
            "exact_concept_deletion_faithfulness",
        ],
        "selection_protocol": {
            "test_used_for_selection": False,
            "valid_loaded": False,
            "fixed_epoch_no_heldout_early_stopping": True,
            "meta_fold_source": "aligned_video_group_fold_column",
        },
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training": asdict(train_config),
        "feature_count": int(concepts.values.shape[1]),
        "feature_names": concepts.names,
        "feature_groups": concepts.groups,
        "oof_sources": oof_sources,
        "valid_sources": None,
        "parent_oof": parent_metrics,
        "oof": oof_metrics,
        "fold_reports": fold_reports,
        "oof_gate": {
            **gate,
            "passed": gate_passed,
            "accuracy_margin": float(
                oof_metrics["accuracy"] - float(gate.get("minimum_accuracy", 0.0))
            ),
            "macro_f1_margin": float(
                oof_metrics["macro_f1"] - float(gate.get("minimum_macro_f1", 0.0))
            ),
        },
        "deployment_training_history": deployment_history,
        "valid": None,
        "test": None,
        "evaluation_limitation": (
            "EXP183 test labels were previously viewed; EXP184 never uses test for "
            "architecture, threshold, or hyperparameter selection."
        ),
    }

    if args.evaluate_valid:
        if not gate_passed:
            raise RuntimeError(
                "OOF gate failed; locked validation was not opened. "
                f"OOF={oof_metrics['accuracy']:.6f}/{oof_metrics['macro_f1']:.6f}"
            )
        valid_frames = load_aligned(valid_sources, require_fold=False)
        valid_reference = valid_frames[0]
        valid_concepts = build_concepts(valid_frames)
        if valid_concepts.names != concepts.names:
            raise RuntimeError("OOF and valid concept schemas differ")
        valid_x = apply_standardizer(valid_concepts.values, full_mean, full_scale)
        valid_result = predict(
            deployment_model,
            valid_x,
            valid_concepts.parent_probability,
            device,
            train_config.neutral_inference_trust,
            train_config.polarity_inference_trust,
        )
        valid_metrics = _metrics(
            valid_reference, valid_result["probability"], valid_concepts.regression
        )
        prediction_frame(
            valid_reference,
            valid_result["probability"],
            valid_concepts.regression,
            None,
        ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
        explanation_frame(
            valid_reference, concepts.names, valid_result, None
        ).to_csv(
            output / "concept_contributions_valid.csv",
            index=False,
            encoding="utf-8-sig",
        )
        report["scope"] = "strict_grouped_meta_oof_then_locked_valid_no_test_access"
        report["selection_protocol"]["valid_loaded"] = True
        report["valid_sources"] = valid_sources
        report["valid_parent"] = _metrics(
            valid_reference,
            valid_concepts.parent_probability,
            valid_concepts.regression,
        )
        report["valid"] = valid_metrics

    input_paths = [Path(value).resolve() for value in oof_sources]
    if args.evaluate_valid:
        input_paths.extend(Path(value).resolve() for value in valid_sources)
    report["provenance_sha256"] = {
        str(path): _sha256(path) for path in [config_path, Path(__file__).resolve(), *input_paths]
    }
    (output / "resolved_config.json").write_text(
        json.dumps(
            {**raw_config, "training": asdict(train_config)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
