from __future__ import annotations

"""Grouped cross-fitted Neutral-only residual correction.

This expert can move probability mass only between Neutral and the combined
polar mass.  Negative-vs-Positive odds are therefore an exact invariant.  It
is intentionally narrower than the failed free three-way router.
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

from .cross_fitted_evidence_router import build_evidence
from .data import CLASS_NAMES
from .metrics import compute_metrics
from .utils import atomic_torch_save, save_json, seed_everything


class NeutralOnlyResidual(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int = 48,
        dropout: float = 0.10,
        maximum_logit_shift: float = 1.0,
    ) -> None:
        super().__init__()
        self.maximum_logit_shift = float(maximum_logit_shift)
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.shift = nn.Linear(hidden_dimension, 1)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, features: Tensor, parent_probability: Tensor) -> dict[str, Tensor]:
        parent = parent_probability.float().clamp_min(1e-8)
        parent = parent / parent.sum(dim=-1, keepdim=True)
        neutral = parent[:, 1].clamp(1e-7, 1.0 - 1e-7)
        raw_shift = self.maximum_logit_shift * torch.tanh(
            self.shift(self.network(features)).squeeze(-1).float()
        )
        # High-confidence parents receive a smaller trust region.
        uncertainty = 4.0 * neutral * (1.0 - neutral)
        bounded_shift = uncertainty * raw_shift
        neutral_logit = torch.log(neutral / (1.0 - neutral)) + bounded_shift
        corrected_neutral = torch.sigmoid(neutral_logit)
        polar_mass = 1.0 - corrected_neutral
        old_polar_mass = (parent[:, 0] + parent[:, 2]).clamp_min(1e-8)
        probability = torch.stack(
            [
                polar_mass * parent[:, 0] / old_polar_mass,
                corrected_neutral,
                polar_mass * parent[:, 2] / old_polar_mass,
            ],
            dim=-1,
        )
        return {
            "probabilities": probability,
            "neutral_logit_shift": bounded_shift,
            "uncertainty": uncertainty,
        }


def _groups(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.rsplit("$_$", 1)[0] if "$_$" in value else value for value in ids],
        dtype=object,
    )


def _weights(targets: np.ndarray, device: torch.device) -> Tensor:
    counts = np.bincount(targets, minlength=3).astype(np.float64)
    weights = np.sqrt(len(targets) / (3.0 * counts.clip(min=1.0)))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def fit_model(
    features: np.ndarray,
    parent_probability: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    hidden_dimension: int,
    batch_size: int,
) -> tuple[NeutralOnlyResidual, list[dict[str, float]]]:
    seed_everything(seed)
    model = NeutralOnlyResidual(
        features.shape[1], hidden_dimension=hidden_dimension
    ).to(device)
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(parent_probability, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_weights = _weights(targets, device)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "shift_l2": 0.0}
        samples = 0
        for cpu_features, cpu_parent, cpu_targets in loader:
            batch_features = cpu_features.to(device)
            batch_parent = cpu_parent.to(device)
            batch_targets = cpu_targets.to(device)
            output = model(batch_features, batch_parent)
            classification = F.nll_loss(
                output["probabilities"].clamp_min(1e-8).log(),
                batch_targets,
                weight=class_weights,
            )
            shift_l2 = output["neutral_logit_shift"].square().mean()
            loss = classification + 0.02 * shift_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(batch_targets)
            samples += size
            totals["loss"] += float(loss.detach()) * size
            totals["classification"] += float(classification.detach()) * size
            totals["shift_l2"] += float(shift_l2.detach()) * size
        scheduler.step()
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    **{name: value / samples for name, value in totals.items()},
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    return model, history


@torch.inference_mode()
def predict(
    model: NeutralOnlyResidual,
    features: np.ndarray,
    parent_probability: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    result = model(
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(parent_probability, dtype=torch.float32, device=device),
    )
    return (
        result["probabilities"].cpu().numpy(),
        result["neutral_logit_shift"].cpu().numpy(),
    )


def _frame(
    evidence: dict[str, Any],
    probability: np.ndarray,
    shift: np.ndarray,
    fold: np.ndarray | None = None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "id": evidence["ids"],
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": evidence["parent_intensity"],
            "true_label": [CLASS_NAMES[index] for index in evidence["targets"]],
            "true_intensity": evidence["true_intensity"],
            "neutral_logit_shift": shift,
        }
    )
    if fold is not None:
        result.insert(1, "router_fold", fold)
    return result


def _metrics(evidence: dict[str, Any], probability: np.ndarray) -> dict[str, Any]:
    return compute_metrics(
        evidence["targets"],
        probability,
        evidence["true_intensity"],
        evidence["parent_intensity"],
    )


def _evaluate_locked(
    name: str,
    parent_path: str,
    sentiment_path: str,
    ontology_path: str,
    model: NeutralOnlyResidual,
    scaler: StandardScaler,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    evidence = build_evidence(parent_path, sentiment_path, ontology_path)
    probability, shift = predict(
        model, scaler.transform(evidence["features"]), evidence["experts"][:, 0], device
    )
    _frame(evidence, probability, shift).to_csv(
        output / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    parent_metrics = _metrics(evidence, evidence["experts"][:, 0])
    metrics = _metrics(evidence, probability)
    parent_decision = evidence["experts"][:, 0].argmax(1)
    decision = probability.argmax(1)
    changed = decision != parent_decision
    fixed = changed & (decision == evidence["targets"]) & (
        parent_decision != evidence["targets"]
    )
    broken = changed & (decision != evidence["targets"]) & (
        parent_decision == evidence["targets"]
    )
    return {
        "parent": parent_metrics,
        "corrected": metrics,
        "changed_decisions": int(changed.sum()),
        "fixed_errors": int(fixed.sum()),
        "broken_correct": int(broken.sum()),
        "mean_shift": float(shift.mean()),
        "std_shift": float(shift.std()),
        "max_abs_shift": float(np.abs(shift).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted Neutral-only residual")
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--parent-valid", required=True)
    parser.add_argument("--deployment-parent-valid", default=None)
    parser.add_argument("--sentiment-train", required=True)
    parser.add_argument("--sentiment-valid", required=True)
    parser.add_argument("--ontology-train", required=True)
    parser.add_argument("--ontology-valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dimension", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train = build_evidence(
        args.parent_oof, args.sentiment_train, args.ontology_train
    )
    groups = _groups(train["ids"])
    folds = list(
        StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed
        ).split(np.zeros(len(groups)), train["targets"], groups)
    )
    oof_probability = np.zeros((len(groups), 3), dtype=np.float64)
    oof_shift = np.zeros(len(groups), dtype=np.float64)
    fold_ids = np.full(len(groups), -1, dtype=np.int64)
    fold_reports: list[dict[str, Any]] = []
    for fold, (fit_index, heldout_index) in enumerate(folds):
        if set(groups[fit_index]).intersection(groups[heldout_index]):
            raise RuntimeError(f"Fold {fold} has group leakage")
        scaler = StandardScaler().fit(train["features"][fit_index])
        model, history = fit_model(
            scaler.transform(train["features"][fit_index]),
            train["experts"][fit_index, 0],
            train["targets"][fit_index],
            device,
            args.seed + 200 + fold,
            args.epochs,
            args.hidden_dimension,
            args.batch_size,
        )
        probability, shift = predict(
            model,
            scaler.transform(train["features"][heldout_index]),
            train["experts"][heldout_index, 0],
            device,
        )
        oof_probability[heldout_index] = probability
        oof_shift[heldout_index] = shift
        fold_ids[heldout_index] = fold
        fold_reports.append(
            {
                "fold": fold,
                "fit_samples": int(len(fit_index)),
                "heldout_samples": int(len(heldout_index)),
                "group_overlap": 0,
                "metrics": _metrics(
                    {key: value[heldout_index] if isinstance(value, np.ndarray) else value for key, value in train.items()},
                    probability,
                ),
                "history": history,
            }
        )
    if (fold_ids < 0).any():
        raise RuntimeError("Neutral residual OOF did not cover all samples")
    _frame(train, oof_probability, oof_shift, fold_ids).to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    parent_oof = _metrics(train, train["experts"][:, 0])
    corrected_oof = _metrics(train, oof_probability)

    final_scaler = StandardScaler().fit(train["features"])
    final_model, history = fit_model(
        final_scaler.transform(train["features"]),
        train["experts"][:, 0],
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
            "maximum_logit_shift": final_model.maximum_logit_shift,
            "feature_names": train["feature_names"],
            "seed": args.seed,
            "epochs": args.epochs,
        },
        output / "neutral_residual.pt",
    )
    valid_result = _evaluate_locked(
        "valid",
        args.parent_valid,
        args.sentiment_valid,
        args.ontology_valid,
        final_model,
        final_scaler,
        device,
        output,
    )
    deployment_result = None
    if args.deployment_parent_valid:
        deployment_result = _evaluate_locked(
            "deployment_valid",
            args.deployment_parent_valid,
            args.sentiment_valid,
            args.ontology_valid,
            final_model,
            final_scaler,
            device,
            output,
        )
    report = {
        "scope": "train_grouped_oof_then_locked_validation_no_test_access",
        "architecture": "uncertainty_bounded_neutral_only_evidence_residual",
        "invariants": {
            "negative_positive_odds_preserved": True,
            "maximum_raw_logit_shift": 1.0,
            "zero_shift_identity": True,
        },
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "hidden_dimension": args.hidden_dimension,
        "parent_oof": parent_oof,
        "corrected_oof": corrected_oof,
        "fold_reports": fold_reports,
        "final_training_history": history,
        "matching_parent_valid": valid_result,
        "deployment_parent_valid": deployment_result,
    }
    save_json(report, output / "final_metrics.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
