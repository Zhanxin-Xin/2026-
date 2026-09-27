from __future__ import annotations

"""Semantic selective routing over aligned OOF experts.

The architecture combines task-specific gates from MMoE (KDD 2018) with a
SelectiveNet-style bounded acceptance head (ICML 2019).  A frozen sentence
encoder supplies sample semantics, while each expert contributes only its OOF
posterior, regression estimate, and label-free uncertainty diagnostics.  The
router cannot invent unrestricted logits: it returns a bounded interpolation
between the uniform parent and a class-wise convex expert mixture.
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
from transformers import AutoModel, AutoTokenizer

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES, attach_labels_from_excel, load_pickle, parse_split
from .metrics import compute_metrics
from .train_frozen_minilm_prototype_expert import encode_texts
from .train_task_conditional_convex_router import soft_macro_f1_loss
from .utils import atomic_torch_save, save_json, seed_everything


def _groups(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.rsplit("$_$", 1)[0] if "$_$" in value else value for value in ids],
        dtype=object,
    )


def _targets(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    values = frame["true_label"].map(mapping)
    if values.isna().any():
        raise ValueError("Unknown class label in expert predictions")
    return values.to_numpy(np.int64)


def _text_lookup(ids: np.ndarray, texts: np.ndarray) -> dict[str, str]:
    result = {str(sample_id): str(text) for sample_id, text in zip(ids, texts)}
    if len(result) != len(ids):
        raise ValueError("Dataset IDs are not unique")
    return result


def _expert_evidence(
    frames: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    experts = np.stack(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    experts /= experts.sum(axis=2, keepdims=True).clip(min=1e-12)
    regression = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    parent = experts.mean(axis=1)
    centered_log = np.log(experts.clip(min=1e-8))
    centered_log -= centered_log.mean(axis=2, keepdims=True)
    ordered = np.sort(experts, axis=2)
    confidence = experts.max(axis=2)
    margin = ordered[:, :, -1] - ordered[:, :, -2]
    certainty = 1.0 + (
        experts.clip(min=1e-8) * np.log(experts.clip(min=1e-8))
    ).sum(axis=2) / math.log(experts.shape[2])
    agreement = 1.0 - 0.5 * np.abs(experts - parent[:, None, :]).sum(axis=2)
    parent_regression = regression.mean(axis=1, keepdims=True)
    diagnostic = np.concatenate(
        [
            experts,
            centered_log,
            confidence[..., None],
            margin[..., None],
            certainty[..., None],
            agreement[..., None],
            (regression / 3.0)[..., None],
            ((regression - parent_regression) / 3.0)[..., None],
        ],
        axis=2,
    ).astype(np.float32)
    parent_ordered = np.sort(parent, axis=1)
    parent_entropy = -(
        parent.clip(min=1e-8) * np.log(parent.clip(min=1e-8))
    ).sum(axis=1) / math.log(parent.shape[1])
    global_features = np.column_stack(
        [
            parent,
            parent_entropy,
            parent_ordered[:, -1] - parent_ordered[:, -2],
            np.mean(experts.argmax(axis=2) != parent.argmax(axis=1)[:, None], axis=1),
            experts.var(axis=1).mean(axis=1),
            parent_regression[:, 0] / 3.0,
            regression.std(axis=1) / 3.0,
        ]
    ).astype(np.float32)
    if not all(
        np.isfinite(value).all()
        for value in (experts, regression, diagnostic, global_features)
    ):
        raise ValueError("Expert evidence contains NaN/Inf")
    return experts.astype(np.float32), regression.astype(np.float32), diagnostic, global_features


class SemanticSelectiveExpertRouter(nn.Module):
    """Text-conditioned class-wise expert attention with bounded fallback."""

    def __init__(
        self,
        text_dimension: int,
        diagnostic_dimension: int,
        global_dimension: int,
        expert_count: int,
        hidden_dimension: int = 64,
        dropout: float = 0.20,
        maximum_acceptance: float = 0.35,
    ) -> None:
        super().__init__()
        if not 0.0 < maximum_acceptance <= 1.0:
            raise ValueError("maximum_acceptance must lie in (0,1]")
        self.expert_count = int(expert_count)
        self.maximum_acceptance = float(maximum_acceptance)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dimension),
            nn.Linear(text_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(diagnostic_dimension),
            nn.Linear(diagnostic_dimension, hidden_dimension),
            nn.GELU(),
        )
        self.expert_type = nn.Parameter(torch.zeros(expert_count, hidden_dimension))
        nn.init.normal_(self.expert_type, std=0.02)
        self.joint_norm = nn.LayerNorm(4 * hidden_dimension)
        self.joint = nn.Sequential(
            nn.Linear(4 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.route_head = nn.Linear(hidden_dimension, len(CLASS_NAMES))
        self.correctness_head = nn.Linear(hidden_dimension, 1)
        self.trust = nn.Sequential(
            nn.LayerNorm(hidden_dimension + global_dimension),
            nn.Linear(hidden_dimension + global_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        # Uniform class-wise routes reproduce the parent exactly at start.
        nn.init.zeros_(self.route_head.weight)
        nn.init.zeros_(self.route_head.bias)
        nn.init.zeros_(self.correctness_head.bias)
        nn.init.zeros_(self.trust[-1].weight)
        nn.init.constant_(self.trust[-1].bias, -2.0)

    def forward(
        self,
        text_embedding: Tensor,
        experts: Tensor,
        diagnostic: Tensor,
        global_features: Tensor,
    ) -> dict[str, Tensor]:
        text = self.text_projection(text_embedding.float())
        evidence = self.evidence_projection(diagnostic.float())
        typed = evidence + self.expert_type.unsqueeze(0)
        query = text.unsqueeze(1).expand_as(typed)
        joint = self.joint(
            self.joint_norm(
                torch.cat([query, typed, query * typed, (query - typed).abs()], dim=-1)
            )
        )
        route_logits = self.route_head(joint).permute(0, 2, 1)
        route_weights = torch.softmax(route_logits, dim=-1)
        selected_score = torch.einsum("nce,nec->nc", route_weights, experts.float())
        selected = selected_score / selected_score.sum(dim=1, keepdim=True).clamp_min(1e-8)
        parent = experts.float().mean(dim=1)
        trust_fraction = torch.sigmoid(
            self.trust(torch.cat([text, global_features.float()], dim=1)).squeeze(-1)
        )
        trust = self.maximum_acceptance * trust_fraction
        probability = (1.0 - trust[:, None]) * parent + trust[:, None] * selected
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return {
            "probabilities": probability,
            "parent": parent,
            "selected": selected,
            "route_weights": route_weights,
            "correctness_logits": self.correctness_head(joint).squeeze(-1),
            "trust": trust,
            "trust_fraction": trust_fraction,
        }


def _class_weights(targets: np.ndarray, device: torch.device) -> Tensor:
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = np.sqrt(len(targets) / (len(CLASS_NAMES) * counts.clip(min=1.0)))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def fit_router(
    embeddings: np.ndarray,
    experts: np.ndarray,
    diagnostic: np.ndarray,
    global_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    hidden_dimension: int,
    maximum_acceptance: float,
) -> tuple[SemanticSelectiveExpertRouter, list[dict[str, float]]]:
    seed_everything(seed)
    model = SemanticSelectiveExpertRouter(
        text_dimension=embeddings.shape[1],
        diagnostic_dimension=diagnostic.shape[2],
        global_dimension=global_features.shape[1],
        expert_count=experts.shape[1],
        hidden_dimension=hidden_dimension,
        maximum_acceptance=maximum_acceptance,
    ).to(device)
    dataset = TensorDataset(
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(experts, dtype=torch.float32),
        torch.tensor(diagnostic, dtype=torch.float32),
        torch.tensor(global_features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=generator)
    class_weights = _class_weights(targets, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = np.zeros(7, dtype=np.float64)
        seen = 0
        for cpu_embedding, cpu_experts, cpu_diag, cpu_global, cpu_target in loader:
            embedding = cpu_embedding.to(device)
            expert = cpu_experts.to(device)
            diag = cpu_diag.to(device)
            global_value = cpu_global.to(device)
            target = cpu_target.to(device)
            output = model(embedding, expert, diag, global_value)
            probability = output["probabilities"].clamp_min(1e-8)
            classification = F.nll_loss(probability.log(), target, weight=class_weights)
            macro_loss, _ = soft_macro_f1_loss(
                probability, target, neutral_multiplier=1.25
            )
            rows = torch.arange(len(target), device=device)
            true_probability = expert[rows, :, target]
            oracle_route = torch.softmax(true_probability.detach() / 0.10, dim=1)
            true_class_routes = output["route_weights"][rows, target]
            route_supervision = F.kl_div(
                true_class_routes.clamp_min(1e-8).log(),
                oracle_route,
                reduction="batchmean",
            )
            correctness = expert.argmax(dim=2).eq(target[:, None]).float()
            correctness_loss = F.binary_cross_entropy_with_logits(
                output["correctness_logits"], correctness
            )
            parent_kl = F.kl_div(
                probability.log(), output["parent"].detach(), reduction="batchmean"
            )
            parent_true = output["parent"][rows, target]
            opportunity = (true_probability.max(dim=1).values > parent_true + 0.04).float()
            selective = F.binary_cross_entropy(output["trust_fraction"], opportunity)
            loss = (
                classification
                + 0.25 * macro_loss
                + 0.15 * route_supervision
                + 0.10 * correctness_loss
                + 0.10 * parent_kl
                + 0.08 * selective
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = len(target)
            totals += batch_size * np.asarray(
                [
                    float(loss.detach()),
                    float(classification.detach()),
                    float(macro_loss.detach()),
                    float(route_supervision.detach()),
                    float(correctness_loss.detach()),
                    float(parent_kl.detach()),
                    float(output["trust"].mean().detach()),
                ]
            )
            seen += batch_size
        scheduler.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            values = totals / max(seen, 1)
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(values[0]),
                    "classification": float(values[1]),
                    "soft_macro_f1_loss": float(values[2]),
                    "route_supervision": float(values[3]),
                    "correctness": float(values[4]),
                    "parent_kl": float(values[5]),
                    "mean_trust": float(values[6]),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    return model, history


@torch.inference_mode()
def predict(
    model: SemanticSelectiveExpertRouter,
    embeddings: np.ndarray,
    experts: np.ndarray,
    diagnostic: np.ndarray,
    global_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for start in range(0, len(embeddings), 512):
        end = start + 512
        output = model(
            torch.tensor(embeddings[start:end], dtype=torch.float32, device=device),
            torch.tensor(experts[start:end], dtype=torch.float32, device=device),
            torch.tensor(diagnostic[start:end], dtype=torch.float32, device=device),
            torch.tensor(global_features[start:end], dtype=torch.float32, device=device),
        )
        chunks.append(
            (
                output["probabilities"].cpu().numpy(),
                output["route_weights"].cpu().numpy(),
                output["trust"].cpu().numpy(),
            )
        )
    return tuple(np.concatenate([value[index] for value in chunks]) for index in range(3))  # type: ignore[return-value]


def _metrics(
    reference: pd.DataFrame, probability: np.ndarray, regression: np.ndarray
) -> dict[str, Any]:
    return compute_metrics(
        _targets(reference),
        probability,
        reference["true_intensity"].to_numpy(np.float64),
        regression,
    )


def _prediction_frame(
    reference: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
    trust: np.ndarray,
    fold: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
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
    if fold is not None:
        frame.insert(1, "fold", fold)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic selective expert router")
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--valid", action="append", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pretrained-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--maximum-acceptance", type=float, default=0.35)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--oof-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.oof) != len(args.valid) or len(args.oof) < 2:
        raise ValueError("Need at least two matched OOF/valid expert sources")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    oof_frames = load_aligned(args.oof, require_fold=True)
    reference = oof_frames[0]
    fold_ids = reference["fold"].to_numpy(np.int64)
    experts, regression_experts, diagnostic, global_features = _expert_evidence(
        oof_frames
    )
    regression = regression_experts.mean(axis=1)
    parent = experts.mean(axis=1)
    targets = _targets(reference)
    loaded = load_pickle(args.data)
    train_raw = attach_labels_from_excel({"train": loaded["train"]}, args.labels)
    train_arrays = parse_split(train_raw, "train", "text_shared", require_labels=True)
    lookup = _text_lookup(train_arrays.ids.astype(str), train_arrays.raw_text)
    train_text = np.asarray([lookup[value] for value in reference["id"].astype(str)])
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    encoder = AutoModel.from_pretrained(args.pretrained_model).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    embeddings = encode_texts(
        train_text,
        tokenizer,
        encoder,
        device,
        args.max_length,
        args.encode_batch_size,
    )

    oof_probability = np.zeros_like(parent)
    oof_routes = np.zeros(
        (len(reference), len(CLASS_NAMES), len(oof_frames)), dtype=np.float32
    )
    oof_trust = np.zeros(len(reference), dtype=np.float32)
    fold_reports: list[dict[str, Any]] = []
    groups = _groups(reference["id"].astype(str).to_numpy())
    for fold in sorted(np.unique(fold_ids).tolist()):
        heldout = np.flatnonzero(fold_ids == fold)
        fit = np.flatnonzero(fold_ids != fold)
        if set(groups[fit]).intersection(groups[heldout]):
            raise RuntimeError(f"Router fold {fold} has group leakage")
        model, history = fit_router(
            embeddings[fit],
            experts[fit],
            diagnostic[fit],
            global_features[fit],
            targets[fit],
            device,
            args.seed + 400 + fold,
            args.epochs,
            args.hidden_dimension,
            args.maximum_acceptance,
        )
        probability, routes, trust = predict(
            model,
            embeddings[heldout],
            experts[heldout],
            diagnostic[heldout],
            global_features[heldout],
            device,
        )
        oof_probability[heldout] = probability
        oof_routes[heldout] = routes
        oof_trust[heldout] = trust
        fold_reports.append(
            {
                "fold": int(fold),
                "fit_samples": int(len(fit)),
                "heldout_samples": int(len(heldout)),
                "parent_metrics": _metrics(
                    reference.iloc[heldout].reset_index(drop=True),
                    parent[heldout],
                    regression[heldout],
                ),
                "router_metrics": _metrics(
                    reference.iloc[heldout].reset_index(drop=True),
                    probability,
                    regression[heldout],
                ),
                "mean_trust": float(trust.mean()),
                "mean_routes": routes.mean(axis=0).tolist(),
                "history": history,
            }
        )
    parent_oof = _metrics(reference, parent, regression)
    router_oof = _metrics(reference, oof_probability, regression)
    accuracy_delta = router_oof["accuracy"] - parent_oof["accuracy"]
    macro_delta = router_oof["macro_f1"] - parent_oof["macro_f1"]
    gate_passed = bool(accuracy_delta >= 0.0 and macro_delta > 0.0)
    _prediction_frame(
        reference, oof_probability, regression, oof_trust, fold_ids
    ).to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    report: dict[str, Any] = {
        "scope": "semantic_router_grouped_cross_fitted_oof_gate_before_locked_valid_no_test_access",
        "architecture": "frozen_sentence_semantic_classwise_mmoe_with_selective_parent_fallback",
        "research_basis": {
            "MMoE_KDD_2018_reference_commit": "2718f56e4313716fd3a86e9510bc2df0cc366238",
            "SelectiveNet_ICML_2019_reference_commit": "a6d0a8fd33dae61da910b61a2aae93102d2d4869",
            "Sentence_BERT_EMNLP_2019": args.pretrained_model,
            "implementation_note": "clean-room conceptual adaptation; no external source copied",
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "expert_count": len(oof_frames),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "maximum_acceptance": args.maximum_acceptance,
        "oof_sources": args.oof,
        "parent_oof": parent_oof,
        "router_oof": router_oof,
        "oof_delta": {"accuracy": accuracy_delta, "macro_f1": macro_delta},
        "oof_gate": {"accuracy_non_decrease": True, "macro_strict_improvement": True, "passed": gate_passed},
        "mean_oof_trust": float(oof_trust.mean()),
        "mean_oof_routes": oof_routes.mean(axis=0).tolist(),
        "fold_reports": fold_reports,
    }
    if not gate_passed or args.oof_only:
        report["valid"] = None
        report["decision"] = (
            "oof_only_completed_without_loading_valid"
            if args.oof_only
            else "closed_before_loading_valid"
        )
        save_json(report, output / "final_metrics.json")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    final_model, final_history = fit_router(
        embeddings,
        experts,
        diagnostic,
        global_features,
        targets,
        device,
        args.seed,
        args.epochs,
        args.hidden_dimension,
        args.maximum_acceptance,
    )
    atomic_torch_save(
        {
            "model_state": final_model.state_dict(),
            "pretrained_model": args.pretrained_model,
            "expert_count": len(oof_frames),
            "hidden_dimension": args.hidden_dimension,
            "maximum_acceptance": args.maximum_acceptance,
            "oof_sources": args.oof,
        },
        output / "router.pt",
    )
    valid_frames = load_aligned(args.valid, require_fold=False)
    valid_experts, valid_regression_experts, valid_diag, valid_global = _expert_evidence(
        valid_frames
    )
    valid_reference = valid_frames[0]
    valid_arrays = parse_split(
        attach_labels_from_excel({"valid": loaded["valid"]}, args.labels),
        "valid",
        "text_shared",
        require_labels=True,
    )
    valid_lookup = _text_lookup(valid_arrays.ids.astype(str), valid_arrays.raw_text)
    valid_text = np.asarray(
        [valid_lookup[value] for value in valid_reference["id"].astype(str)]
    )
    valid_embeddings = encode_texts(
        valid_text,
        tokenizer,
        encoder,
        device,
        args.max_length,
        args.encode_batch_size,
    )
    valid_probability, valid_routes, valid_trust = predict(
        final_model,
        valid_embeddings,
        valid_experts,
        valid_diag,
        valid_global,
        device,
    )
    valid_regression = valid_regression_experts.mean(axis=1)
    _prediction_frame(
        valid_reference, valid_probability, valid_regression, valid_trust
    ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    report.update(
        {
            "final_history": final_history,
            "parent_valid": _metrics(
                valid_reference, valid_experts.mean(axis=1), valid_regression
            ),
            "valid": _metrics(valid_reference, valid_probability, valid_regression),
            "mean_valid_trust": float(valid_trust.mean()),
            "mean_valid_routes": valid_routes.mean(axis=0).tolist(),
            "valid_changed_decisions": int(
                np.sum(
                    valid_probability.argmax(axis=1)
                    != valid_experts.mean(axis=1).argmax(axis=1)
                )
            ),
            "decision": "oof_gate_passed_then_single_locked_valid",
        }
    )
    save_json(report, output / "final_metrics.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
