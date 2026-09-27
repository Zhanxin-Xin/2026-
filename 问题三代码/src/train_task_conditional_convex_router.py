from __future__ import annotations

"""Leakage-safe task-conditional convex routing over task-trained experts.

The router adapts the task-specific softmax gates of MMoE (KDD 2018) and the
sample-dependent reliability modulation of MMTM (CVPR 2020).  It deliberately
cannot create free logits or residuals: every class score is a convex mixture
of expert probabilities, and the normalized mixture is blended with the
uniform parent through a bounded fallback gate.  With all experts eligible,
uniform initialization is an exact identity mapping to the parent ensemble;
class-wise sparse eligibility instead initializes from the uniform mixture of
the eligible experts.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics
from .utils import atomic_torch_save, save_json, seed_everything


def _groups(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.rsplit("$_$", 1)[0] if "$_$" in value else value for value in ids],
        dtype=object,
    )


def _expert_arrays(
    frames: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    experts = np.stack(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    experts /= experts.sum(axis=2, keepdims=True).clip(min=1e-12)
    parent = experts.mean(axis=1)

    ordered = np.sort(experts, axis=2)
    confidence = experts.max(axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    certainty = 1.0 + (
        experts.clip(min=1e-8) * np.log(experts.clip(min=1e-8))
    ).sum(axis=2) / math.log(experts.shape[2])
    agreement = 1.0 - 0.5 * np.abs(experts - parent[:, None, :]).sum(axis=2)
    diagnostics = np.stack([confidence, margin, certainty, agreement], axis=2)

    parent_ordered = np.sort(parent, axis=1)
    parent_entropy = -(
        parent.clip(min=1e-8) * np.log(parent.clip(min=1e-8))
    ).sum(axis=1) / math.log(parent.shape[1])
    decision_disagreement = np.mean(
        experts.argmax(axis=2) != parent.argmax(axis=1)[:, None], axis=1
    )
    expert_variance = experts.var(axis=1).mean(axis=1)
    parent_margin = parent_ordered[:, -1] - parent_ordered[:, -2]
    trust_features = np.column_stack(
        [parent_entropy, decision_disagreement, expert_variance, parent_margin]
    )
    if not all(
        np.isfinite(value).all()
        for value in (experts, parent, diagnostics, trust_features)
    ):
        raise ValueError("Router evidence contains NaN/Inf")
    return experts, diagnostics, trust_features


class TaskConditionalConvexRouter(nn.Module):
    """Low-capacity class-conditional expert selector with parent fallback."""

    def __init__(
        self,
        expert_count: int,
        class_count: int = 3,
        diagnostic_count: int = 4,
        trust_feature_count: int = 4,
        maximum_acceptance: float = 0.45,
        preserve_neutral_probability: bool = False,
        route_eligibility: np.ndarray | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.expert_count = int(expert_count)
        self.class_count = int(class_count)
        self.maximum_acceptance = float(maximum_acceptance)
        self.preserve_neutral_probability = bool(preserve_neutral_probability)
        if route_eligibility is None:
            eligibility = torch.ones(class_count, expert_count, dtype=torch.bool)
        else:
            eligibility = torch.as_tensor(route_eligibility, dtype=torch.bool)
            if eligibility.shape != (class_count, expert_count):
                raise ValueError("route_eligibility has the wrong shape")
            if not eligibility.any(dim=1).all():
                raise ValueError("Every class needs at least one eligible expert")
        self.register_buffer("route_eligibility", eligibility)
        self.route_bias = nn.Parameter(torch.zeros(class_count, expert_count))
        # Reliability effects are shared across experts, while every class has
        # its own coefficients and static expert preference.
        self.reliability_weight = nn.Parameter(
            torch.zeros(class_count, diagnostic_count)
        )
        self.trust_weight = nn.Parameter(torch.zeros(trust_feature_count))
        self.trust_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        experts: Tensor,
        diagnostics: Tensor,
        trust_features: Tensor,
    ) -> dict[str, Tensor]:
        experts = experts.float().clamp_min(1e-8)
        parent = experts.mean(dim=1)
        reliability = torch.einsum(
            "ned,cd->nce", diagnostics.float(), self.reliability_weight
        )
        route_logits = self.route_bias.unsqueeze(0) + reliability
        route_logits = route_logits.masked_fill(
            ~self.route_eligibility.unsqueeze(0), -1e9
        )
        route_weights = torch.softmax(route_logits, dim=-1)
        if self.preserve_neutral_probability:
            neutral_mask = torch.zeros(
                (1, self.class_count, 1),
                dtype=route_weights.dtype,
                device=route_weights.device,
            )
            neutral_mask[:, 1, :] = 1.0
            route_weights = (
                route_weights * (1.0 - neutral_mask)
                + neutral_mask / self.expert_count
            )
        # For each sentiment class, select only that class's probability from
        # a convex combination of the experts.
        selected_score = torch.einsum("nce,nec->nc", route_weights, experts)
        if self.preserve_neutral_probability:
            polar_mass = (parent[:, 0] + parent[:, 2]).clamp_min(1e-8)
            routed_polar = (selected_score[:, 0] + selected_score[:, 2]).clamp_min(
                1e-8
            )
            selected = torch.stack(
                [
                    polar_mass * selected_score[:, 0] / routed_polar,
                    parent[:, 1],
                    polar_mass * selected_score[:, 2] / routed_polar,
                ],
                dim=1,
            )
        else:
            selected = selected_score / selected_score.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
        trust = self.maximum_acceptance * torch.sigmoid(
            self.trust_bias + trust_features.float() @ self.trust_weight
        )
        probability = (1.0 - trust[:, None]) * parent + trust[:, None] * selected
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        return {
            "probabilities": probability,
            "parent": parent,
            "selected": selected,
            "route_weights": route_weights,
            "trust": trust,
        }


def _class_weights(targets: np.ndarray, device: torch.device) -> Tensor:
    counts = np.bincount(targets, minlength=3).astype(np.float64)
    weights = np.sqrt(len(targets) / (3.0 * counts.clip(min=1.0)))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def soft_per_class_f1(probability: Tensor, targets: Tensor) -> Tensor:
    """Differentiable class-wise F1 (Dice) computed from soft probabilities."""
    if probability.ndim != 2:
        raise ValueError("probability must have shape [samples, classes]")
    if targets.ndim != 1 or len(targets) != len(probability):
        raise ValueError("targets must have shape [samples]")
    one_hot = F.one_hot(targets, num_classes=probability.shape[1]).to(
        dtype=probability.dtype
    )
    true_positive = (probability * one_hot).sum(dim=0)
    predicted_support = probability.sum(dim=0)
    true_support = one_hot.sum(dim=0)
    return (2.0 * true_positive + 1e-6) / (
        predicted_support + true_support + 1e-6
    )


def soft_macro_f1_loss(
    probability: Tensor,
    targets: Tensor,
    neutral_multiplier: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return weighted soft Macro-F1 loss and the per-class soft F1 values."""
    per_class = soft_per_class_f1(probability, targets)
    weights = torch.ones_like(per_class)
    if len(weights) > 1:
        weights[1] = float(neutral_multiplier)
    loss = 1.0 - (per_class * weights).sum() / weights.sum()
    return loss, per_class


def _class_expert_f1(experts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predictions = experts.argmax(axis=2)
    result = np.zeros((experts.shape[2], experts.shape[1]), dtype=np.float64)
    for class_index in range(experts.shape[2]):
        truth = targets == class_index
        for expert_index in range(experts.shape[1]):
            predicted = predictions[:, expert_index] == class_index
            true_positive = np.sum(truth & predicted)
            false_positive = np.sum(~truth & predicted)
            false_negative = np.sum(truth & ~predicted)
            denominator = 2 * true_positive + false_positive + false_negative
            result[class_index, expert_index] = (
                0.0 if denominator == 0 else 2.0 * true_positive / denominator
            )
    return result


def _eligibility(
    experts: np.ndarray, targets: np.ndarray, top_k_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    quality = _class_expert_f1(experts, targets)
    if top_k_per_class <= 0 or top_k_per_class >= experts.shape[1]:
        return np.ones_like(quality, dtype=bool), quality
    mask = np.zeros_like(quality, dtype=bool)
    for class_index in range(quality.shape[0]):
        selected = np.argsort(-quality[class_index], kind="stable")[
            :top_k_per_class
        ]
        mask[class_index, selected] = True
    return mask, quality


def fit_router(
    experts: np.ndarray,
    diagnostics: np.ndarray,
    trust_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    maximum_acceptance: float,
    preserve_neutral_probability: bool,
    route_eligibility: np.ndarray | None,
    soft_macro_f1_weight: float,
    neutral_dice_multiplier: float,
) -> tuple[TaskConditionalConvexRouter, list[dict[str, float]]]:
    seed_everything(seed)
    model = TaskConditionalConvexRouter(
        expert_count=experts.shape[1],
        maximum_acceptance=maximum_acceptance,
        preserve_neutral_probability=preserve_neutral_probability,
        route_eligibility=route_eligibility,
    ).to(device)
    expert_tensor = torch.tensor(experts, dtype=torch.float32, device=device)
    diagnostic_tensor = torch.tensor(
        diagnostics, dtype=torch.float32, device=device
    )
    trust_tensor = torch.tensor(
        trust_features, dtype=torch.float32, device=device
    )
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
    class_weights = _class_weights(targets, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        output = model(expert_tensor, diagnostic_tensor, trust_tensor)
        probability = output["probabilities"].clamp_min(1e-8)
        classification = F.nll_loss(
            probability.log(), target_tensor, weight=class_weights
        )
        macro_f1_loss, soft_class_f1 = soft_macro_f1_loss(
            probability,
            target_tensor,
            neutral_multiplier=neutral_dice_multiplier,
        )
        parent_kl = F.kl_div(
            probability.log(), output["parent"], reduction="batchmean"
        )
        uniform = 1.0 / experts.shape[1]
        route_deviation = (output["route_weights"] - uniform).square().mean()
        accepted_change = (
            output["trust"][:, None]
            * (output["selected"] - output["parent"]).abs()
        ).mean()
        loss = (
            classification
            + soft_macro_f1_weight * macro_f1_loss
            + 0.10 * parent_kl
            + 0.025 * route_deviation
            + 0.025 * accepted_change
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach()),
                    "classification": float(classification.detach()),
                    "soft_macro_f1_loss": float(macro_f1_loss.detach()),
                    "soft_negative_f1": float(soft_class_f1[0].detach()),
                    "soft_neutral_f1": float(soft_class_f1[1].detach()),
                    "soft_positive_f1": float(soft_class_f1[2].detach()),
                    "parent_kl": float(parent_kl.detach()),
                    "route_deviation": float(route_deviation.detach()),
                    "accepted_change": float(accepted_change.detach()),
                    "mean_trust": float(output["trust"].mean().detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    return model, history


@torch.inference_mode()
def predict(
    model: TaskConditionalConvexRouter,
    experts: np.ndarray,
    diagnostics: np.ndarray,
    trust_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = model(
        torch.tensor(experts, dtype=torch.float32, device=device),
        torch.tensor(diagnostics, dtype=torch.float32, device=device),
        torch.tensor(trust_features, dtype=torch.float32, device=device),
    )
    return (
        output["probabilities"].cpu().numpy(),
        output["selected"].cpu().numpy(),
        output["route_weights"].cpu().numpy(),
        output["trust"].cpu().numpy(),
    )


def _targets(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    result = frame["true_label"].map(mapping)
    if result.isna().any():
        raise ValueError("Unknown target label")
    return result.to_numpy(np.int64)


def _regression(frames: list[pd.DataFrame]) -> np.ndarray:
    return np.mean(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=0,
    )


def _metrics(
    frame: pd.DataFrame, probability: np.ndarray, regression: np.ndarray
) -> dict[str, Any]:
    return compute_metrics(
        _targets(frame),
        probability,
        frame["true_intensity"].to_numpy(np.float64),
        regression,
    )


def _prediction_frame(
    reference: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
    routes: np.ndarray,
    trust: np.ndarray,
    fold: np.ndarray | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": reference["true_label"],
            "true_intensity": reference["true_intensity"],
            "router_trust": trust,
        }
    )
    for class_index, class_name in enumerate(CLASS_NAMES):
        for expert_index in range(routes.shape[2]):
            output[f"{class_name.lower()}_expert_{expert_index}_weight"] = routes[
                :, class_index, expert_index
            ]
    if fold is not None:
        output.insert(1, "router_fold", fold)
    return output


def _routing_summary(routes: np.ndarray, trust: np.ndarray) -> dict[str, Any]:
    return {
        "mean_trust": float(trust.mean()),
        "std_trust": float(trust.std()),
        "min_trust": float(trust.min()),
        "max_trust": float(trust.max()),
        "mean_class_expert_weights": {
            class_name: {
                f"expert_{expert}": float(routes[:, class_index, expert].mean())
                for expert in range(routes.shape[2])
            }
            for class_index, class_name in enumerate(CLASS_NAMES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fitted task-conditional convex expert router"
    )
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--valid", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--maximum-acceptance", type=float, default=0.45)
    parser.add_argument("--minimum-macro-delta", type=float, default=0.0)
    parser.add_argument("--minimum-accuracy-delta", type=float, default=-0.002)
    parser.add_argument("--preserve-neutral-probability", action="store_true")
    parser.add_argument("--top-k-per-class", type=int, default=0)
    parser.add_argument("--soft-macro-f1-weight", type=float, default=0.0)
    parser.add_argument("--neutral-dice-multiplier", type=float, default=1.0)
    parser.add_argument("--oof-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.oof) != len(args.valid) or len(args.oof) < 2:
        raise ValueError("Need at least two matched OOF/valid expert sources")
    if not 0.0 < args.maximum_acceptance <= 1.0:
        raise ValueError("maximum-acceptance must be in (0, 1]")
    if args.top_k_per_class < 0:
        raise ValueError("top-k-per-class cannot be negative")
    if args.soft_macro_f1_weight < 0.0:
        raise ValueError("soft-macro-f1-weight cannot be negative")
    if args.neutral_dice_multiplier <= 0.0:
        raise ValueError("neutral-dice-multiplier must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Phase A: valid files are intentionally not loaded until the OOF gate passes.
    oof_frames = load_aligned(args.oof, require_fold=True)
    reference = oof_frames[0]
    fold_ids = reference["fold"].to_numpy(np.int64)
    unique_folds = sorted(np.unique(fold_ids).tolist())
    if len(unique_folds) < 3:
        raise ValueError("At least three base OOF folds are required")
    groups = _groups(reference["id"].astype(str).to_numpy())
    experts, diagnostics, trust_features = _expert_arrays(oof_frames)
    targets = _targets(reference)
    regression = _regression(oof_frames)
    parent = experts.mean(axis=1)
    oof_probability = np.zeros_like(parent)
    oof_routes = np.zeros(
        (len(reference), len(CLASS_NAMES), len(oof_frames)), dtype=np.float64
    )
    oof_trust = np.zeros(len(reference), dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in unique_folds:
        heldout_index = np.flatnonzero(fold_ids == fold)
        fit_index = np.flatnonzero(fold_ids != fold)
        overlap = set(groups[fit_index]).intersection(groups[heldout_index])
        if overlap:
            raise RuntimeError(f"Router fold {fold} has video-group leakage")
        eligibility, fit_class_expert_f1 = _eligibility(
            experts[fit_index], targets[fit_index], args.top_k_per_class
        )
        model, history = fit_router(
            experts[fit_index],
            diagnostics[fit_index],
            trust_features[fit_index],
            targets[fit_index],
            device,
            args.seed + 300 + fold,
            args.epochs,
            args.maximum_acceptance,
            args.preserve_neutral_probability,
            eligibility,
            args.soft_macro_f1_weight,
            args.neutral_dice_multiplier,
        )
        probability, _selected, routes, trust = predict(
            model,
            experts[heldout_index],
            diagnostics[heldout_index],
            trust_features[heldout_index],
            device,
        )
        oof_probability[heldout_index] = probability
        oof_routes[heldout_index] = routes
        oof_trust[heldout_index] = trust
        fold_reports.append(
            {
                "fold": int(fold),
                "fit_samples": int(len(fit_index)),
                "heldout_samples": int(len(heldout_index)),
                "group_overlap": 0,
                "parent_metrics": _metrics(
                    reference.iloc[heldout_index].reset_index(drop=True),
                    parent[heldout_index],
                    regression[heldout_index],
                ),
                "router_metrics": _metrics(
                    reference.iloc[heldout_index].reset_index(drop=True),
                    probability,
                    regression[heldout_index],
                ),
                "routing": _routing_summary(routes, trust),
                "fit_class_expert_f1": fit_class_expert_f1.tolist(),
                "route_eligibility": eligibility.tolist(),
                "history": history,
            }
        )
    parent_oof = _metrics(reference, parent, regression)
    router_oof = _metrics(reference, oof_probability, regression)
    macro_delta = router_oof["macro_f1"] - parent_oof["macro_f1"]
    accuracy_delta = router_oof["accuracy"] - parent_oof["accuracy"]
    oof_gate_passed = bool(
        macro_delta > args.minimum_macro_delta
        and accuracy_delta >= args.minimum_accuracy_delta
    )
    _prediction_frame(
        reference,
        oof_probability,
        regression,
        oof_routes,
        oof_trust,
        fold_ids,
    ).to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")

    report: dict[str, Any] = {
        "scope": "base_grouped_oof_selection_before_single_locked_validation_no_test_access",
        "architecture": "task_conditional_convex_expert_router_with_bounded_parent_fallback",
        "research_basis": {
            "MMoE_KDD_2018": "https://github.com/drawbridge/keras-mmoe",
            "MMTM_CVPR_2020": "https://github.com/haamoon/mmtm",
            "implementation_note": "conceptual adaptation; no external source copied",
        },
        "invariants": {
            "uniform_initialization_equals_parent": bool(
                args.top_k_per_class <= 0
                or args.top_k_per_class >= len(oof_frames)
            ),
            "initialization_reference": (
                "full_uniform_parent"
                if args.top_k_per_class <= 0
                or args.top_k_per_class >= len(oof_frames)
                else "fold_local_uniform_over_eligible_experts"
            ),
            "per_class_expert_weights_are_convex": True,
            "no_free_logit_or_probability_residual": True,
            "maximum_parent_departure_gate": args.maximum_acceptance,
            "neutral_probability_preserved": args.preserve_neutral_probability,
            "classwise_sparse_top_k": args.top_k_per_class,
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "expert_count": len(oof_frames),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "training_objective": {
            "weighted_nll": 1.0,
            "soft_macro_f1_dice": args.soft_macro_f1_weight,
            "neutral_dice_multiplier": args.neutral_dice_multiplier,
            "parent_kl": 0.10,
            "route_deviation": 0.025,
            "accepted_change": 0.025,
        },
        "oof_sources": args.oof,
        "parent_oof": parent_oof,
        "router_oof": router_oof,
        "oof_delta": {
            "accuracy": accuracy_delta,
            "macro_f1": macro_delta,
        },
        "oof_gate": {
            "minimum_accuracy_delta": args.minimum_accuracy_delta,
            "minimum_macro_delta": args.minimum_macro_delta,
            "passed": oof_gate_passed,
        },
        "oof_routing": _routing_summary(oof_routes, oof_trust),
        "fold_reports": fold_reports,
    }
    if not oof_gate_passed or args.oof_only:
        report["valid"] = None
        report["decision"] = (
            "oof_only_completed_without_loading_official_valid"
            if args.oof_only
            else "closed_before_loading_official_valid"
        )
        save_json(report, output / "final_metrics.json")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    final_eligibility, full_class_expert_f1 = _eligibility(
        experts, targets, args.top_k_per_class
    )
    final_model, final_history = fit_router(
        experts,
        diagnostics,
        trust_features,
        targets,
        device,
        args.seed,
        args.epochs,
        args.maximum_acceptance,
        args.preserve_neutral_probability,
        final_eligibility,
        args.soft_macro_f1_weight,
        args.neutral_dice_multiplier,
    )
    atomic_torch_save(
        {
            "model_state": final_model.state_dict(),
            "expert_count": len(oof_frames),
            "maximum_acceptance": args.maximum_acceptance,
            "preserve_neutral_probability": args.preserve_neutral_probability,
            "top_k_per_class": args.top_k_per_class,
            "soft_macro_f1_weight": args.soft_macro_f1_weight,
            "neutral_dice_multiplier": args.neutral_dice_multiplier,
            "route_eligibility": final_eligibility,
            "class_expert_f1": full_class_expert_f1,
            "seed": args.seed,
            "epochs": args.epochs,
            "oof_sources": args.oof,
        },
        output / "router.pt",
    )

    # Phase B: the official valid predictions are read once, after OOF selection.
    valid_frames = load_aligned(args.valid, require_fold=False)
    valid_experts, valid_diagnostics, valid_trust_features = _expert_arrays(
        valid_frames
    )
    valid_probability, _selected, valid_routes, valid_trust = predict(
        final_model,
        valid_experts,
        valid_diagnostics,
        valid_trust_features,
        device,
    )
    valid_regression = _regression(valid_frames)
    valid_reference = valid_frames[0]
    parent_valid_probability = valid_experts.mean(axis=1)
    _prediction_frame(
        valid_reference,
        valid_probability,
        valid_regression,
        valid_routes,
        valid_trust,
    ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    report.update(
        {
            "final_training_history": final_history,
            "full_oof_class_expert_f1": full_class_expert_f1.tolist(),
            "final_route_eligibility": final_eligibility.tolist(),
            "parent_valid": _metrics(
                valid_reference, parent_valid_probability, valid_regression
            ),
            "valid": _metrics(
                valid_reference, valid_probability, valid_regression
            ),
            "valid_routing": _routing_summary(valid_routes, valid_trust),
            "valid_changed_decisions": int(
                np.sum(
                    valid_probability.argmax(1)
                    != parent_valid_probability.argmax(1)
                )
            ),
            "decision": "oof_gate_passed_and_official_valid_evaluated_once",
        }
    )
    save_json(report, output / "final_metrics.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
