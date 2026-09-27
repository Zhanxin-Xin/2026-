from __future__ import annotations

"""Grouped-OOF selective Neutral cascade with a bounded context residual."""

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES, SplitArrays, attach_labels_from_excel, load_pickle, parse_split
from .metrics import compute_metrics
from .train_heteroscedastic_ordinal_distribution import fast_classification_metrics
from .train_pretrained_fusion import supervised_contrastive_loss
from .train_pretrained_oof import subset_arrays, video_groups
from .utils import seed_everything


@dataclass(frozen=True)
class CascadeParameters:
    strength: float
    center_logit: float
    confidence_power: float


def cascade_grid() -> list[CascadeParameters]:
    return [
        CascadeParameters(strength, center, power)
        for strength in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
        for center in (-1.0, -0.5, 0.0, 0.5, 1.0)
        for power in (0.0, 1.0)
    ]


def _sequence_key(identifier: str) -> tuple[str, int]:
    if "$_$" not in identifier:
        return identifier, 0
    group, suffix = identifier.rsplit("$_$", 1)
    try:
        order = int(suffix)
    except ValueError:
        order = 0
    return group, order


def bidirectional_context(arrays: SplitArrays) -> tuple[list[str], np.ndarray]:
    ids = [str(value) for value in arrays.ids]
    text = [str(value) for value in arrays.raw_text]
    grouped: dict[str, list[tuple[int, int]]] = {}
    for index, identifier in enumerate(ids):
        group, order = _sequence_key(identifier)
        grouped.setdefault(group, []).append((order, index))
    contexts = [""] * len(ids)
    present = np.zeros(len(ids), dtype=np.float32)
    for values in grouped.values():
        values.sort()
        for position, (_, index) in enumerate(values):
            neighbours: list[str] = []
            if position > 0:
                neighbours.append(text[values[position - 1][1]])
            if position + 1 < len(values):
                neighbours.append(text[values[position + 1][1]])
            if neighbours:
                contexts[index] = " [SEP] ".join(neighbours)
                present[index] = 1.0
    return contexts, present


class EncodedNeutralDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        arrays: SplitArrays,
        current: dict[str, Tensor],
        context: dict[str, Tensor],
        context_present: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.current = current
        self.context = context
        self.context_present = torch.from_numpy(context_present.astype(np.float32))

    def __len__(self) -> int:
        return self.arrays.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "current_input_ids": self.current["input_ids"][index],
            "current_attention_mask": self.current["attention_mask"][index],
            "context_input_ids": self.context["input_ids"][index],
            "context_attention_mask": self.context["attention_mask"][index],
            "context_present": self.context_present[index],
            "target": torch.tensor(
                int(self.arrays.class_labels[index] == 1), dtype=torch.long
            ),
            "id": str(self.arrays.ids[index]),
        }


def encode_dataset(
    tokenizer: Any, arrays: SplitArrays, max_length: int
) -> EncodedNeutralDataset:
    contexts, present = bidirectional_context(arrays)
    options = {
        "padding": "max_length",
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    current = tokenizer([str(value) for value in arrays.raw_text], **options)
    context = tokenizer(contexts, **options)
    return EncodedNeutralDataset(arrays, current, context, present)


def masked_mean(hidden: Tensor, mask: Tensor) -> Tensor:
    weight = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class NeutralContextCascade(nn.Module):
    """Shared sentence encoder with a bounded, rejectable context shift."""

    def __init__(
        self,
        pretrained_model: str,
        revision: str | None,
        local_files_only: bool,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model,
            revision=revision,
            local_files_only=local_files_only,
        )
        source_dimension = int(self.encoder.config.hidden_size)
        self.current_projection = nn.Sequential(
            nn.Linear(source_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(source_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(4 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, 2 * hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dimension, hidden_dimension),
        )
        self.output_norm = nn.LayerNorm(hidden_dimension)
        self.direct_head = nn.Linear(hidden_dimension, 2)
        self.prototypes = nn.Parameter(torch.empty(2, hidden_dimension))
        self.prototype_scale_raw = nn.Parameter(torch.tensor(7.0))
        self.prototype_mix_logit = nn.Parameter(torch.tensor(-1.5))
        nn.init.normal_(self.prototypes, std=0.02)
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.constant_(self.context_gate[-1].bias, -1.5)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        current_ids = batch["current_input_ids"]
        context_ids = batch["context_input_ids"]
        input_ids = torch.cat([current_ids, context_ids], dim=0)
        attention_mask = torch.cat(
            [batch["current_attention_mask"], batch["context_attention_mask"]], dim=0
        )
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = masked_mean(encoded.last_hidden_state, attention_mask)
        current_raw, context_raw = pooled.chunk(2, dim=0)
        current = self.current_projection(current_raw)
        context = self.context_projection(context_raw)
        interaction = torch.cat(
            [current, context, torch.abs(current - context), current * context], dim=1
        )
        gate = torch.sigmoid(self.context_gate(interaction)).squeeze(1)
        gate = gate * batch["context_present"].float()
        fused = current + 0.50 * gate.unsqueeze(1) * (context - current)
        fused = self.output_norm(fused + self.residual(fused))
        embedding = F.normalize(fused.float(), dim=1)
        direct = torch.softmax(self.direct_head(fused).float(), dim=1)
        scale = 1.0 + F.softplus(self.prototype_scale_raw)
        prototype = torch.softmax(
            scale * embedding @ F.normalize(self.prototypes.float(), dim=1).T, dim=1
        )
        mix = torch.sigmoid(self.prototype_mix_logit)
        probability = (1.0 - mix) * direct + mix * prototype
        return {
            "probability": probability.clamp_min(1e-8),
            "embedding": embedding,
            "context_gate": gate,
            "prototype_mix": mix,
        }


def _loader(
    dataset: Dataset[dict[str, Any]], batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def predict_neutral(
    model: NeutralContextCascade, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, float]]:
    model.eval()
    probability: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    identifiers: list[str] = []
    gates: list[np.ndarray] = []
    mixes: list[float] = []
    for cpu_batch in loader:
        batch = _move(cpu_batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(batch)
        probability.append(output["probability"][:, 1].float().cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
        identifiers.extend([str(value) for value in cpu_batch["id"]])
        gates.append(output["context_gate"].float().cpu().numpy())
        mixes.append(float(output["prototype_mix"].detach()))
    return (
        np.concatenate(probability),
        np.concatenate(targets),
        identifiers,
        {
            "context_gate_mean": float(np.concatenate(gates).mean()),
            "context_gate_max": float(np.concatenate(gates).max()),
            "prototype_mix": float(np.mean(mixes)),
        },
    )


def train_expert(
    train_arrays: SplitArrays,
    query_arrays: SplitArrays,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, float]], dict[str, float], dict[str, Tensor]]:
    seed_everything(seed)
    train_dataset = encode_dataset(tokenizer, train_arrays, args.max_length)
    query_dataset = encode_dataset(tokenizer, query_arrays, args.max_length)
    train_loader = _loader(train_dataset, args.batch_size, True, seed)
    query_loader = _loader(query_dataset, args.eval_batch_size, False, seed)
    model = NeutralContextCascade(
        args.pretrained_model,
        args.revision,
        args.local_files_only,
        args.hidden_dimension,
        args.dropout,
    ).to(device)
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(value) for value in encoder_parameters}
    head_parameters = [value for value in model.parameters() if id(value) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    updates = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(updates * args.warmup_ratio),
        num_training_steps=max(1, updates),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    labels = (train_arrays.class_labels == 1).astype(np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = np.power(counts.sum() / counts.clip(min=1.0), 0.5)
    weights /= weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for cpu_batch in train_loader:
            batch = _move(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(batch)
                classification = F.nll_loss(
                    output["probability"].float().log(),
                    batch["target"],
                    weight=class_weights,
                )
                contrastive = supervised_contrastive_loss(
                    output["embedding"], batch["target"], temperature=0.10
                )
                loss = classification + args.contrastive_weight * contrastive
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach())
        history.append(
            {
                "epoch": float(epoch),
                "loss": loss_sum / max(1, len(train_loader)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
    probability, target, identifiers, diagnostics = predict_neutral(
        model, query_loader, device
    )
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    del model, optimizer, scheduler, scaler, train_loader, query_loader
    del train_dataset, query_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probability, target, identifiers, history, diagnostics, state


def cascade_probability(
    parent: np.ndarray, neutral_probability: np.ndarray, parameters: CascadeParameters
) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float64)
    neutral_probability = np.asarray(neutral_probability, dtype=np.float64).clip(1e-7, 1 - 1e-7)
    parent_neutral = parent[:, 1].clip(1e-7, 1 - 1e-7)
    expert_logit = np.log(neutral_probability / (1.0 - neutral_probability))
    confidence = np.power(2.0 * np.abs(neutral_probability - 0.5), parameters.confidence_power)
    base_logit = np.log(parent_neutral / (1.0 - parent_neutral))
    corrected_logit = base_logit + parameters.strength * confidence * (
        expert_logit - parameters.center_logit
    )
    corrected_neutral = 1.0 / (1.0 + np.exp(-corrected_logit))
    polar_mass = 1.0 - corrected_neutral
    polar_denominator = (parent[:, 0] + parent[:, 2]).clip(min=1e-12)
    output = np.column_stack(
        [
            polar_mass * parent[:, 0] / polar_denominator,
            corrected_neutral,
            polar_mass * parent[:, 2] / polar_denominator,
        ]
    )
    return output / output.sum(axis=1, keepdims=True).clip(min=1e-12)


def select_cascade(
    parent: np.ndarray,
    expert: np.ndarray,
    target: np.ndarray,
    candidates: list[CascadeParameters],
) -> tuple[CascadeParameters, dict[str, float]]:
    parent_accuracy, parent_macro = fast_classification_metrics(target, parent.argmax(1))
    best: tuple[tuple[float, float, float], CascadeParameters, float, float] | None = None
    for parameters in candidates:
        probability = cascade_probability(parent, expert, parameters)
        accuracy, macro_f1 = fast_classification_metrics(target, probability.argmax(1))
        if accuracy + 1e-12 < parent_accuracy:
            continue
        key = (macro_f1, accuracy, -parameters.strength)
        if best is None or key > best[0]:
            best = (key, parameters, accuracy, macro_f1)
    if best is None:
        raise RuntimeError("No cascade parameters satisfy the accuracy guard")
    return best[1], {
        "accuracy": best[2],
        "macro_f1": best[3],
        "parent_accuracy": parent_accuracy,
        "parent_macro_f1": parent_macro,
    }


def binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = probability >= 0.5
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=[0, 1], zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "neutral_precision": float(precision[1]),
        "neutral_recall": float(recall[1]),
        "neutral_f1": float(f1[1]),
        "neutral_support": int(support[1]),
    }


def output_frame(
    reference: pd.DataFrame,
    probability: np.ndarray,
    include_fold: bool,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": reference["predicted_intensity"].to_numpy(np.float64),
            "true_label": reference["true_label"].astype(str),
            "true_intensity": reference["true_intensity"].to_numpy(np.float64),
        }
    )
    if include_fold:
        result.insert(1, "fold", reference["fold"].to_numpy(np.int64))
    return result


def three_class_metrics(reference: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    target = reference["true_label"].map(mapping).to_numpy(np.int64)
    return compute_metrics(
        target,
        probability,
        reference["true_intensity"].to_numpy(np.float64),
        reference["predicted_intensity"].to_numpy(np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped-OOF Neutral context cascade")
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--parent-valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--pretrained-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--hidden-dimension", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parent_frame = load_aligned([args.parent_oof], require_fold=True)[0]

    loaded = load_pickle(args.data)
    raw_train = attach_labels_from_excel({"train": loaded["train"]}, args.labels)
    arrays = parse_split(raw_train, "train", "text_shared", require_labels=True)
    parent_by_id = parent_frame.set_index(parent_frame["id"].astype(str), drop=False)
    expected = arrays.ids.astype(str)
    if set(parent_by_id.index) != set(expected):
        raise ValueError("Parent OOF IDs do not align with train split")
    parent_frame = parent_by_id.loc[expected].reset_index(drop=True)
    folds = parent_frame["fold"].to_numpy(np.int64)
    groups = video_groups(arrays.ids)
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    candidates = cascade_grid()
    expert_oof = np.empty(arrays.size, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    started = time.time()
    for fold in sorted(np.unique(folds).tolist()):
        fit_index = np.flatnonzero(folds != fold)
        heldout_index = np.flatnonzero(folds == fold)
        if set(groups[fit_index]).intersection(groups[heldout_index]):
            raise RuntimeError(f"Fold {fold} contains video-group leakage")
        probability, target, ids, history, diagnostics, _ = train_expert(
            subset_arrays(arrays, fit_index),
            subset_arrays(arrays, heldout_index),
            tokenizer,
            args,
            device,
            args.seed + fold,
        )
        if ids != expected[heldout_index].tolist():
            raise RuntimeError(f"Fold {fold} prediction order mismatch")
        expert_oof[heldout_index] = probability
        fold_report = {
            "fold": int(fold),
            "fit_samples": int(len(fit_index)),
            "heldout_samples": int(len(heldout_index)),
            "group_overlap": 0,
            "binary_metrics": binary_metrics(target, probability),
            "diagnostics": diagnostics,
            "history": history,
        }
        fold_reports.append(fold_report)
        print(
            f"fold={fold} auc={fold_report['binary_metrics']['roc_auc']:.4f} "
            f"neutral_f1={fold_report['binary_metrics']['neutral_f1']:.4f}",
            flush=True,
        )

    parent_probability = parent_frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    class_target = arrays.class_labels.astype(np.int64)
    cross_fitted_probability = np.empty_like(parent_probability)
    cascade_reports: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds).tolist()):
        fit = folds != fold
        heldout = folds == fold
        parameters, fit_metrics = select_cascade(
            parent_probability[fit], expert_oof[fit], class_target[fit], candidates
        )
        cross_fitted_probability[heldout] = cascade_probability(
            parent_probability[heldout], expert_oof[heldout], parameters
        )
        heldout_metrics = fast_classification_metrics(
            class_target[heldout], cross_fitted_probability[heldout].argmax(1)
        )
        cascade_reports.append(
            {
                "fold": int(fold),
                "parameters": asdict(parameters),
                "fit_selection_metrics": fit_metrics,
                "heldout_accuracy": heldout_metrics[0],
                "heldout_macro_f1": heldout_metrics[1],
            }
        )
    full_parameters, full_selection = select_cascade(
        parent_probability, expert_oof, class_target, candidates
    )
    parent_metrics = three_class_metrics(parent_frame, parent_probability)
    oof_metrics = three_class_metrics(parent_frame, cross_fitted_probability)
    gate_passed = bool(
        oof_metrics["accuracy"] + 1e-12 >= parent_metrics["accuracy"]
        and oof_metrics["macro_f1"] > parent_metrics["macro_f1"] + 1e-12
    )
    output_frame(parent_frame, cross_fitted_probability, include_fold=True).to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "id": expected,
            "fold": folds,
            "neutral_probability": expert_oof,
            "true_neutral": (class_target == 1).astype(np.int64),
        }
    ).to_csv(output / "train_oof_neutral_expert.csv", index=False, encoding="utf-8-sig")
    report: dict[str, Any] = {
        "scope": "grouped_oof_neutral_cascade_gate_before_locked_valid_no_test_access",
        "architecture": "shared_minilm_bidirectional_context_residual_binary_prototype_cascade",
        "research_basis": {
            "DialogueRNN_AAAI_2019": {
                "idea": "separate current utterance and conversation state",
                "repository": "https://github.com/declare-lab/conv-emotion",
                "commit": "6128ca20e9c736605cce7e99d5d95db0356c35f5",
            },
            "SupCon_NeurIPS_2020": {
                "idea": "class-conditional contrastive compactness",
                "repository": "https://github.com/HobbitLong/SupContrast",
                "commit": "66a8fe53880d6a1084b2e4e0db0a019024d6d41a",
                "license": "BSD-2-Clause",
            },
            "implementation_note": "Original implementation; no external source code copied.",
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "pretrained_model": args.pretrained_model,
        "revision": args.revision,
        "elapsed_minutes_before_full_fit": (time.time() - started) / 60.0,
        "fold_reports": fold_reports,
        "cascade_reports": cascade_reports,
        "expert_oof_binary": binary_metrics((class_target == 1).astype(np.int64), expert_oof),
        "full_oof_parameters_for_valid": asdict(full_parameters),
        "full_oof_selection_metrics": full_selection,
        "parent_oof": parent_metrics,
        "oof": oof_metrics,
        "oof_delta": {
            "accuracy": oof_metrics["accuracy"] - parent_metrics["accuracy"],
            "macro_f1": oof_metrics["macro_f1"] - parent_metrics["macro_f1"],
        },
        "oof_gate_passed": gate_passed,
    }

    if not gate_passed:
        report["valid"] = None
        report["decision"] = "closed_before_loading_official_valid"
    else:
        valid_frame = load_aligned([args.parent_valid], require_fold=False)[0]
        raw_valid = attach_labels_from_excel({"valid": loaded["valid"]}, args.labels)
        valid_arrays = parse_split(raw_valid, "valid", "text_shared", require_labels=True)
        valid_by_id = valid_frame.set_index(valid_frame["id"].astype(str), drop=False)
        valid_ids = valid_arrays.ids.astype(str)
        if set(valid_by_id.index) != set(valid_ids):
            raise ValueError("Parent valid IDs do not align with valid split")
        valid_frame = valid_by_id.loc[valid_ids].reset_index(drop=True)
        valid_expert, valid_target, predicted_ids, full_history, full_diagnostics, state = train_expert(
            arrays,
            valid_arrays,
            tokenizer,
            args,
            device,
            args.seed,
        )
        if predicted_ids != valid_ids.tolist():
            raise RuntimeError("Valid prediction order mismatch")
        valid_parent = valid_frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
        valid_probability = cascade_probability(
            valid_parent, valid_expert, full_parameters
        )
        output_frame(valid_frame, valid_probability, include_fold=False).to_csv(
            output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "id": valid_ids,
                "neutral_probability": valid_expert,
                "true_neutral": valid_target,
            }
        ).to_csv(output / "valid_neutral_expert.csv", index=False, encoding="utf-8-sig")
        torch.save(
            {
                "model_state": state,
                "arguments": vars(args),
                "cascade_parameters": asdict(full_parameters),
            },
            output / "neutral_context_expert.pt",
        )
        report["full_fit_history"] = full_history
        report["full_fit_diagnostics"] = full_diagnostics
        report["valid_expert_binary"] = binary_metrics(valid_target, valid_expert)
        report["valid"] = three_class_metrics(valid_frame, valid_probability)
        report["decision"] = "promoted_after_oof_gate"
        report["elapsed_minutes_total"] = (time.time() - started) / 60.0
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
