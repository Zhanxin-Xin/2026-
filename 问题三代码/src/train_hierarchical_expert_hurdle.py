from __future__ import annotations

"""Cross-fitted hierarchical hurdle pooling over a set of expert predictions.

The two learned queries are an original, compact adaptation of Pooling by
Multihead Attention from Set Transformer (ICML 2019): one query aggregates
Neutral-vs-Polar evidence and the other aggregates Positive-vs-Negative
evidence.  The model emits only bounded residuals around the uniform expert
parent, so zero initialization is an exact identity and free class logits are
not possible.
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
from torch.utils.data import DataLoader, TensorDataset

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics
from .utils import atomic_torch_save, save_json, seed_everything


def _groups(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.rsplit("$_$", 1)[0] if "$_$" in value else value for value in ids],
        dtype=object,
    )


def _targets(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    result = frame["true_label"].map(mapping)
    if result.isna().any():
        raise ValueError("Unknown target label")
    return result.to_numpy(np.int64)


def _expert_evidence(
    frames: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.stack(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    probabilities /= probabilities.sum(axis=2, keepdims=True).clip(min=1e-12)
    intensities = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    parent = probabilities.mean(axis=1)
    centered_log = np.log(probabilities.clip(min=1e-8))
    centered_log -= centered_log.mean(axis=2, keepdims=True)
    ordered = np.sort(probabilities, axis=2)
    confidence = probabilities.max(axis=2, keepdims=True)
    margin = (ordered[:, :, -1] - ordered[:, :, -2])[:, :, None]
    entropy = -(
        probabilities.clip(min=1e-8) * np.log(probabilities.clip(min=1e-8))
    ).sum(axis=2, keepdims=True) / math.log(probabilities.shape[2])
    tokens = np.concatenate(
        [
            probabilities,
            centered_log,
            confidence,
            margin,
            entropy,
            intensities[:, :, None],
            np.abs(intensities)[:, :, None],
        ],
        axis=2,
    )

    parent_ordered = np.sort(parent, axis=1)
    parent_entropy = -(
        parent.clip(min=1e-8) * np.log(parent.clip(min=1e-8))
    ).sum(axis=1) / math.log(parent.shape[1])
    parent_margin = parent_ordered[:, -1] - parent_ordered[:, -2]
    decision_disagreement = np.mean(
        probabilities.argmax(axis=2) != parent.argmax(axis=1)[:, None], axis=1
    )
    probability_variance = probabilities.var(axis=1).mean(axis=1)
    global_features = np.column_stack(
        [
            parent,
            parent_entropy,
            parent_margin,
            decision_disagreement,
            probability_variance,
            intensities.mean(axis=1),
            intensities.std(axis=1),
            np.abs(intensities).mean(axis=1),
        ]
    )
    if not all(
        np.isfinite(value).all()
        for value in (probabilities, intensities, tokens, global_features)
    ):
        raise ValueError("Expert hurdle evidence contains NaN/Inf")
    return probabilities, intensities, tokens, global_features


class HierarchicalExpertHurdle(nn.Module):
    """Two-query expert-set pooling with bounded hierarchical log-odds shifts."""

    def __init__(
        self,
        expert_count: int,
        token_dimension: int,
        global_dimension: int,
        hidden_dimension: int = 24,
        maximum_log_odds_residual: float = 0.75,
        uncertainty_envelope: bool = False,
    ) -> None:
        super().__init__()
        self.expert_count = int(expert_count)
        self.maximum_log_odds_residual = float(maximum_log_odds_residual)
        self.uncertainty_envelope = bool(uncertainty_envelope)
        self.token_projection = nn.Sequential(
            nn.Linear(token_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
        )
        self.expert_embedding = nn.Parameter(
            torch.empty(expert_count, hidden_dimension)
        )
        self.task_queries = nn.Parameter(torch.empty(2, hidden_dimension))
        self.global_projection = nn.Sequential(
            nn.Linear(global_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
        )
        self.neutral_head = nn.Sequential(
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.polarity_head = nn.Sequential(
            nn.Linear(2 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.normal_(self.expert_embedding, std=0.02)
        nn.init.xavier_uniform_(self.task_queries)
        for head in (self.neutral_head, self.polarity_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        probabilities: Tensor,
        tokens: Tensor,
        global_features: Tensor,
    ) -> dict[str, Tensor]:
        probabilities = probabilities.float().clamp_min(1e-8)
        parent = probabilities.mean(dim=1)
        token_hidden = self.token_projection(tokens.float())
        token_hidden = token_hidden + self.expert_embedding.unsqueeze(0)
        attention_logits = torch.einsum(
            "ned,qd->nqe", token_hidden, self.task_queries
        ) / math.sqrt(token_hidden.shape[-1])
        attention = torch.softmax(attention_logits, dim=-1)
        context = torch.einsum("nqe,ned->nqd", attention, token_hidden)
        global_hidden = self.global_projection(global_features.float())
        neutral_input = torch.cat([context[:, 0], global_hidden], dim=1)
        polarity_input = torch.cat([context[:, 1], global_hidden], dim=1)
        raw_residual = self.maximum_log_odds_residual * torch.tanh(
            torch.cat(
                [self.neutral_head(neutral_input), self.polarity_head(polarity_input)],
                dim=1,
            )
        )

        parent_neutral = parent[:, 1].clamp(1e-7, 1.0 - 1e-7)
        positive_given_parent_polar = parent[:, 2] / (
            parent[:, 0] + parent[:, 2]
        ).clamp_min(1e-7)
        if self.uncertainty_envelope:
            envelope = torch.stack(
                [
                    4.0 * parent_neutral * (1.0 - parent_neutral),
                    4.0
                    * positive_given_parent_polar
                    * (1.0 - positive_given_parent_polar),
                ],
                dim=1,
            )
            residual = raw_residual * envelope
        else:
            envelope = torch.ones_like(raw_residual)
            residual = raw_residual
        base_neutral_logit = torch.logit(parent_neutral)
        base_polarity_logit = torch.log(
            parent[:, 2].clamp_min(1e-7) / parent[:, 0].clamp_min(1e-7)
        )
        neutral_logit = base_neutral_logit + residual[:, 0]
        polarity_logit = base_polarity_logit + residual[:, 1]
        neutral = torch.sigmoid(neutral_logit)
        positive_given_polar = torch.sigmoid(polarity_logit)
        polar_mass = 1.0 - neutral
        probability = torch.stack(
            [
                polar_mass * (1.0 - positive_given_polar),
                neutral,
                polar_mass * positive_given_polar,
            ],
            dim=1,
        )
        return {
            "probabilities": probability,
            "parent": parent,
            "attention": attention,
            "residual": residual,
            "raw_residual": raw_residual,
            "uncertainty_envelope": envelope,
            "neutral_logit": neutral_logit,
            "polarity_logit": polarity_logit,
        }


def _class_weights(targets: np.ndarray, device: torch.device) -> Tensor:
    counts = np.bincount(targets, minlength=3).astype(np.float64)
    weights = np.sqrt(len(targets) / (3.0 * counts.clip(min=1.0)))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def fit_model(
    probabilities: np.ndarray,
    tokens: np.ndarray,
    global_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    hidden_dimension: int,
    maximum_log_odds_residual: float,
    uncertainty_envelope: bool,
    batch_size: int,
) -> tuple[HierarchicalExpertHurdle, list[dict[str, float]]]:
    seed_everything(seed)
    model = HierarchicalExpertHurdle(
        expert_count=probabilities.shape[1],
        token_dimension=tokens.shape[2],
        global_dimension=global_features.shape[1],
        hidden_dimension=hidden_dimension,
        maximum_log_odds_residual=maximum_log_odds_residual,
        uncertainty_envelope=uncertainty_envelope,
    ).to(device)
    dataset = TensorDataset(
        torch.tensor(probabilities, dtype=torch.float32),
        torch.tensor(tokens, dtype=torch.float32),
        torch.tensor(global_features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_weights = _class_weights(targets, device)
    neutral_count = max(1, int(np.sum(targets == 1)))
    polar_count = max(1, int(np.sum(targets != 1)))
    neutral_pos_weight = torch.tensor(
        polar_count / neutral_count, dtype=torch.float32, device=device
    )
    negative_count = max(1, int(np.sum(targets == 0)))
    positive_count = max(1, int(np.sum(targets == 2)))
    polarity_pos_weight = torch.tensor(
        negative_count / positive_count, dtype=torch.float32, device=device
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "neutral_hurdle": 0.0,
            "polarity_hurdle": 0.0,
            "parent_kl": 0.0,
            "residual_cost": 0.0,
        }
        sample_count = 0
        for cpu_probability, cpu_tokens, cpu_global, cpu_targets in loader:
            expert_probability = cpu_probability.to(device)
            batch_targets = cpu_targets.to(device)
            output = model(
                expert_probability,
                cpu_tokens.to(device),
                cpu_global.to(device),
            )
            probability = output["probabilities"].clamp_min(1e-8)
            classification = F.nll_loss(
                probability.log(), batch_targets, weight=class_weights
            )
            neutral_target = (batch_targets == 1).float()
            neutral_hurdle = F.binary_cross_entropy_with_logits(
                output["neutral_logit"],
                neutral_target,
                pos_weight=neutral_pos_weight,
            )
            polar_mask = batch_targets != 1
            polarity_hurdle = F.binary_cross_entropy_with_logits(
                output["polarity_logit"][polar_mask],
                (batch_targets[polar_mask] == 2).float(),
                pos_weight=polarity_pos_weight,
            )
            parent_kl = F.kl_div(
                probability.log(), output["parent"], reduction="batchmean"
            )
            residual_cost = output["residual"].square().mean()
            loss = (
                classification
                + 0.15 * neutral_hurdle
                + 0.05 * polarity_hurdle
                + 0.10 * parent_kl
                + 0.02 * residual_cost
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(batch_targets)
            sample_count += size
            for name, value in (
                ("loss", loss),
                ("classification", classification),
                ("neutral_hurdle", neutral_hurdle),
                ("polarity_hurdle", polarity_hurdle),
                ("parent_kl", parent_kl),
                ("residual_cost", residual_cost),
            ):
                totals[name] += float(value.detach()) * size
        scheduler.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    **{name: value / sample_count for name, value in totals.items()},
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    return model, history


@torch.inference_mode()
def predict(
    model: HierarchicalExpertHurdle,
    probabilities: np.ndarray,
    tokens: np.ndarray,
    global_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    output = model(
        torch.tensor(probabilities, dtype=torch.float32, device=device),
        torch.tensor(tokens, dtype=torch.float32, device=device),
        torch.tensor(global_features, dtype=torch.float32, device=device),
    )
    return (
        output["probabilities"].cpu().numpy(),
        output["attention"].cpu().numpy(),
        output["residual"].cpu().numpy(),
    )


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
    attention: np.ndarray,
    residual: np.ndarray,
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
            "neutral_log_odds_residual": residual[:, 0],
            "polarity_log_odds_residual": residual[:, 1],
        }
    )
    for task_index, task_name in enumerate(("neutral", "polarity")):
        for expert_index in range(attention.shape[2]):
            output[f"{task_name}_expert_{expert_index}_attention"] = attention[
                :, task_index, expert_index
            ]
    if fold is not None:
        output.insert(1, "hurdle_fold", fold)
    return output


def _pooling_summary(attention: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    return {
        "mean_absolute_neutral_residual": float(np.abs(residual[:, 0]).mean()),
        "mean_absolute_polarity_residual": float(np.abs(residual[:, 1]).mean()),
        "mean_attention": {
            task_name: {
                f"expert_{expert}": float(attention[:, task, expert].mean())
                for expert in range(attention.shape[2])
            }
            for task, task_name in enumerate(("neutral", "polarity"))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fitted two-query hierarchical expert hurdle"
    )
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--valid", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dimension", type=int, default=24)
    parser.add_argument("--maximum-log-odds-residual", type=float, default=0.75)
    parser.add_argument("--uncertainty-envelope", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minimum-macro-delta", type=float, default=0.0)
    parser.add_argument("--minimum-accuracy-delta", type=float, default=0.0)
    parser.add_argument("--oof-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.oof) != len(args.valid) or len(args.oof) < 2:
        raise ValueError("Need at least two matched OOF/valid expert sources")
    if args.epochs <= 0 or args.hidden_dimension <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, hidden-dimension, and batch-size must be positive")
    if not 0.0 < args.maximum_log_odds_residual <= 2.0:
        raise ValueError("maximum-log-odds-residual must be in (0, 2]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Phase A: valid sources remain untouched until the grouped-OOF gate passes.
    oof_frames = load_aligned(args.oof, require_fold=True)
    reference = oof_frames[0]
    fold_ids = reference["fold"].to_numpy(np.int64)
    unique_folds = sorted(np.unique(fold_ids).tolist())
    if len(unique_folds) < 3:
        raise ValueError("At least three OOF folds are required")
    groups = _groups(reference["id"].astype(str).to_numpy())
    probabilities, _intensities, tokens, global_features = _expert_evidence(
        oof_frames
    )
    targets = _targets(reference)
    regression = _regression(oof_frames)
    parent_probability = probabilities.mean(axis=1)
    oof_probability = np.zeros_like(parent_probability)
    oof_attention = np.zeros(
        (len(reference), 2, len(oof_frames)), dtype=np.float64
    )
    oof_residual = np.zeros((len(reference), 2), dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in unique_folds:
        heldout_index = np.flatnonzero(fold_ids == fold)
        fit_index = np.flatnonzero(fold_ids != fold)
        if set(groups[fit_index]).intersection(groups[heldout_index]):
            raise RuntimeError(f"Hurdle fold {fold} has video-group leakage")
        model, history = fit_model(
            probabilities[fit_index],
            tokens[fit_index],
            global_features[fit_index],
            targets[fit_index],
            device,
            args.seed + 500 + fold,
            args.epochs,
            args.hidden_dimension,
            args.maximum_log_odds_residual,
            args.uncertainty_envelope,
            args.batch_size,
        )
        probability, attention, residual = predict(
            model,
            probabilities[heldout_index],
            tokens[heldout_index],
            global_features[heldout_index],
            device,
        )
        oof_probability[heldout_index] = probability
        oof_attention[heldout_index] = attention
        oof_residual[heldout_index] = residual
        fold_frame = reference.iloc[heldout_index].reset_index(drop=True)
        fold_reports.append(
            {
                "fold": int(fold),
                "fit_samples": int(len(fit_index)),
                "heldout_samples": int(len(heldout_index)),
                "group_overlap": 0,
                "parent_metrics": _metrics(
                    fold_frame,
                    parent_probability[heldout_index],
                    regression[heldout_index],
                ),
                "hurdle_metrics": _metrics(
                    fold_frame, probability, regression[heldout_index]
                ),
                "pooling": _pooling_summary(attention, residual),
                "history": history,
            }
        )
    parent_oof = _metrics(reference, parent_probability, regression)
    hurdle_oof = _metrics(reference, oof_probability, regression)
    accuracy_delta = hurdle_oof["accuracy"] - parent_oof["accuracy"]
    macro_delta = hurdle_oof["macro_f1"] - parent_oof["macro_f1"]
    gate_passed = bool(
        accuracy_delta >= args.minimum_accuracy_delta
        and macro_delta > args.minimum_macro_delta
    )
    _prediction_frame(
        reference,
        oof_probability,
        regression,
        oof_attention,
        oof_residual,
        fold_ids,
    ).to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    report: dict[str, Any] = {
        "scope": "grouped_oof_architecture_selection_before_locked_valid_no_test_access",
        "architecture": "two_query_set_pooled_hierarchical_expert_hurdle",
        "research_basis": {
            "Set_Transformer_ICML_2019": {
                "repository": "https://github.com/juho-lee/set_transformer",
                "commit": "73432c640ac78140496d6738416c54d32c686d65",
                "license": "MIT",
                "adaptation": "learned task queries pool expert tokens; implementation is original",
            },
            "DORN_CVPR_2018": "paper-level hierarchical/ordinal decomposition; no source copied",
        },
        "invariants": {
            "zero_initialization_equals_uniform_parent": True,
            "maximum_log_odds_residual": args.maximum_log_odds_residual,
            "uncertainty_envelope": args.uncertainty_envelope,
            "neutral_and_polarity_factorization": True,
            "no_free_three_class_logit_head": True,
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dimension": args.hidden_dimension,
        "expert_count": len(oof_frames),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "oof_sources": args.oof,
        "parent_oof": parent_oof,
        "hurdle_oof": hurdle_oof,
        "oof_delta": {"accuracy": accuracy_delta, "macro_f1": macro_delta},
        "oof_gate": {
            "minimum_accuracy_delta": args.minimum_accuracy_delta,
            "minimum_macro_delta": args.minimum_macro_delta,
            "passed": gate_passed,
        },
        "oof_pooling": _pooling_summary(oof_attention, oof_residual),
        "fold_reports": fold_reports,
    }
    if not gate_passed or args.oof_only:
        report["valid"] = None
        report["decision"] = (
            "oof_only_completed_without_loading_official_valid"
            if args.oof_only
            else "closed_before_loading_official_valid"
        )
        save_json(report, output / "final_metrics.json")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    final_model, final_history = fit_model(
        probabilities,
        tokens,
        global_features,
        targets,
        device,
        args.seed,
        args.epochs,
        args.hidden_dimension,
        args.maximum_log_odds_residual,
        args.uncertainty_envelope,
        args.batch_size,
    )
    atomic_torch_save(
        {
            "model_state": final_model.state_dict(),
            "expert_count": len(oof_frames),
            "token_dimension": tokens.shape[2],
            "global_dimension": global_features.shape[1],
            "hidden_dimension": args.hidden_dimension,
            "maximum_log_odds_residual": args.maximum_log_odds_residual,
            "uncertainty_envelope": args.uncertainty_envelope,
            "seed": args.seed,
            "epochs": args.epochs,
            "oof_sources": args.oof,
        },
        output / "hierarchical_hurdle.pt",
    )

    valid_frames = load_aligned(args.valid, require_fold=False)
    valid_probabilities, _valid_intensity, valid_tokens, valid_global = (
        _expert_evidence(valid_frames)
    )
    valid_probability, valid_attention, valid_residual = predict(
        final_model,
        valid_probabilities,
        valid_tokens,
        valid_global,
        device,
    )
    valid_regression = _regression(valid_frames)
    valid_reference = valid_frames[0]
    _prediction_frame(
        valid_reference,
        valid_probability,
        valid_regression,
        valid_attention,
        valid_residual,
    ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    report.update(
        {
            "final_training_history": final_history,
            "parent_valid": _metrics(
                valid_reference,
                valid_probabilities.mean(axis=1),
                valid_regression,
            ),
            "valid": _metrics(valid_reference, valid_probability, valid_regression),
            "valid_pooling": _pooling_summary(valid_attention, valid_residual),
            "valid_sources": args.valid,
            "decision": "oof_gate_passed_then_single_locked_validation",
        }
    )
    save_json(report, output / "final_metrics.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
