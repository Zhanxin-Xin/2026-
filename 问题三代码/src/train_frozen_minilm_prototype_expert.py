from __future__ import annotations

"""Frozen Sentence-BERT prototype expert with grouped-OOF promotion gate."""

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

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS
from .data import (
    CLASS_NAMES,
    attach_labels_from_excel,
    compute_class_weights,
    load_pickle,
    parse_split,
)
from .metrics import compute_metrics
from .utils import atomic_torch_save, save_json, seed_everything


def _groups(ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.rsplit("$_$", 1)[0] if "$_$" in value else value for value in ids],
        dtype=object,
    )


def _load_parent(path: str | Path, expected_ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "id",
        "true_label",
        "true_intensity",
        "predicted_intensity",
        *PROBABILITY_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["id"] = frame["id"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} has duplicate IDs")
    indexed = frame.set_index("id", drop=False)
    expected = np.asarray(expected_ids, dtype=str)
    if set(indexed.index) != set(expected):
        raise ValueError(f"{path} IDs do not align with dataset")
    return indexed.loc[expected].reset_index(drop=True)


def _probability(frame: pd.DataFrame) -> np.ndarray:
    value = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    if not np.isfinite(value).all() or (value < 0.0).any():
        raise ValueError("Invalid parent probabilities")
    return value / value.sum(axis=1, keepdims=True).clip(min=1e-12)


@torch.inference_mode()
def encode_texts(
    texts: np.ndarray,
    tokenizer: Any,
    encoder: nn.Module,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    encoder.eval()
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_text = [str(value) for value in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            hidden = encoder(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        chunks.append(F.normalize(pooled, dim=1).cpu().numpy())
    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    if embeddings.shape[0] != len(texts) or not np.isfinite(embeddings).all():
        raise RuntimeError("Frozen text embedding extraction failed")
    return embeddings


def class_centroids(embeddings: np.ndarray, targets: np.ndarray) -> np.ndarray:
    centroids = []
    for class_index in range(len(CLASS_NAMES)):
        selected = embeddings[targets == class_index]
        if len(selected) == 0:
            raise ValueError(f"No samples for class {class_index}")
        centroid = selected.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        centroids.append(centroid)
    return np.stack(centroids).astype(np.float32)


class FrozenSentencePrototypeHead(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        centroids: np.ndarray | Tensor,
        hidden_dimension: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        centroid_tensor = torch.as_tensor(centroids, dtype=torch.float32)
        if centroid_tensor.shape != (len(CLASS_NAMES), embedding_dimension):
            raise ValueError("centroids have the wrong shape")
        self.register_buffer("centroids", F.normalize(centroid_tensor, dim=1))
        self.direct = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, len(CLASS_NAMES)),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension // 2, 1),
        )
        self.prototype_scale_raw = nn.Parameter(torch.tensor(math.log(8.0)))
        self.prototype_mix_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, embeddings: Tensor) -> dict[str, Tensor]:
        normalized = F.normalize(embeddings.float(), dim=1)
        direct_logits = self.direct(normalized).float()
        prototype_logits = self.prototype_scale_raw.exp().clamp(max=50.0) * (
            normalized @ self.centroids.T
        )
        mix = torch.sigmoid(self.prototype_mix_logit)
        probability = (
            (1.0 - mix) * torch.softmax(direct_logits, dim=1)
            + mix * torch.softmax(prototype_logits, dim=1)
        ).clamp_min(1e-8)
        regression = 3.0 * torch.tanh(
            self.regressor(normalized).squeeze(-1).float() / 3.0
        )
        return {
            "probabilities": probability,
            "regression": regression,
            "direct_logits": direct_logits,
            "prototype_logits": prototype_logits,
            "prototype_mix": mix,
            "embedding": normalized,
        }


def fit_head(
    embeddings: np.ndarray,
    targets: np.ndarray,
    intensities: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    hidden_dimension: int,
    batch_size: int,
) -> tuple[FrozenSentencePrototypeHead, list[dict[str, float]]]:
    seed_everything(seed)
    model = FrozenSentencePrototypeHead(
        embeddings.shape[1],
        class_centroids(embeddings, targets),
        hidden_dimension=hidden_dimension,
    ).to(device)
    dataset = TensorDataset(
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(intensities, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    class_weights = compute_class_weights(
        targets, power=0.5, max_weight=3.0
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "regression": 0.0}
        samples = 0
        for cpu_embedding, cpu_target, cpu_intensity in loader:
            batch_embedding = cpu_embedding.to(device)
            batch_target = cpu_target.to(device)
            batch_intensity = cpu_intensity.to(device)
            output = model(batch_embedding)
            classification = F.nll_loss(
                output["probabilities"].log(), batch_target, weight=class_weights
            )
            regression = F.smooth_l1_loss(
                output["regression"], batch_intensity, beta=0.5
            )
            loss = classification + 0.20 * regression
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(batch_target)
            samples += size
            totals["loss"] += float(loss.detach()) * size
            totals["classification"] += float(classification.detach()) * size
            totals["regression"] += float(regression.detach()) * size
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": float(epoch),
                    **{name: value / samples for name, value in totals.items()},
                    "prototype_mix": float(model.prototype_mix_logit.sigmoid().detach()),
                    "prototype_scale": float(
                        model.prototype_scale_raw.exp().clamp(max=50.0).detach()
                    ),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
    return model, history


@torch.inference_mode()
def predict_head(
    model: FrozenSentencePrototypeHead,
    embeddings: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    output = model(torch.tensor(embeddings, dtype=torch.float32, device=device))
    return (
        output["probabilities"].cpu().numpy(),
        output["regression"].cpu().numpy(),
        {
            "prototype_mix": float(output["prototype_mix"].cpu()),
            "prototype_scale": float(
                model.prototype_scale_raw.exp().clamp(max=50.0).cpu()
            ),
        },
    )


def _metrics(
    targets: np.ndarray,
    probability: np.ndarray,
    true_intensity: np.ndarray,
    regression: np.ndarray,
) -> dict[str, Any]:
    return compute_metrics(targets, probability, true_intensity, regression)


def _prediction_frame(
    ids: np.ndarray,
    targets: np.ndarray,
    true_intensity: np.ndarray,
    probability: np.ndarray,
    regression: np.ndarray,
    folds: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": ids.astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": [CLASS_NAMES[index] for index in targets],
            "true_intensity": true_intensity,
        }
    )
    if folds is not None:
        frame.insert(1, "fold", folds)
    return frame


def _combine(
    parent_probability: np.ndarray,
    parent_regression: np.ndarray,
    expert_probability: np.ndarray,
    expert_regression: np.ndarray,
    parent_member_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = float(parent_member_count + 1)
    return (
        (parent_member_count * parent_probability + expert_probability) / denominator,
        (parent_member_count * parent_regression + expert_regression) / denominator,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--parent-valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pretrained-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--revision", default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--parent-member-count", type=int, default=7)
    parser.add_argument("--oof-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(
        args.epochs,
        args.hidden_dimension,
        args.batch_size,
        args.encode_batch_size,
        args.max_length,
        args.parent_member_count,
    ) <= 0:
        raise ValueError("All count arguments must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    loaded = load_pickle(args.data)
    train_raw = attach_labels_from_excel({"train": loaded["train"]}, args.labels)
    train = parse_split(train_raw, "train", "text_shared", require_labels=True)
    parent_oof = _load_parent(args.parent_oof, train.ids)
    if "fold" not in parent_oof:
        raise ValueError("Parent OOF predictions do not contain fold IDs")
    fold_ids = parent_oof["fold"].to_numpy(np.int64)
    groups = _groups(train.ids)
    for fold in np.unique(fold_ids):
        if set(groups[fold_ids != fold]).intersection(groups[fold_ids == fold]):
            raise RuntimeError(f"Parent OOF fold {fold} has group leakage")

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        local_files_only=True,
    )
    encoder = AutoModel.from_pretrained(
        args.pretrained_model,
        revision=args.revision,
        local_files_only=True,
    ).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    train_embeddings = encode_texts(
        train.raw_text,
        tokenizer,
        encoder,
        device,
        args.max_length,
        args.encode_batch_size,
    )

    oof_probability = np.zeros((train.size, len(CLASS_NAMES)), dtype=np.float64)
    oof_regression = np.zeros(train.size, dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_ids).tolist()):
        fit = fold_ids != fold
        heldout = fold_ids == fold
        model, history = fit_head(
            train_embeddings[fit],
            train.class_labels[fit],
            train.regression_labels[fit],
            device,
            args.seed + 700 + fold,
            args.epochs,
            args.hidden_dimension,
            args.batch_size,
        )
        probability, regression, diagnostics = predict_head(
            model, train_embeddings[heldout], device
        )
        oof_probability[heldout] = probability
        oof_regression[heldout] = regression
        fold_reports.append(
            {
                "fold": int(fold),
                "fit_samples": int(fit.sum()),
                "heldout_samples": int(heldout.sum()),
                "group_overlap": 0,
                "metrics": _metrics(
                    train.class_labels[heldout],
                    probability,
                    train.regression_labels[heldout],
                    regression,
                ),
                "diagnostics": diagnostics,
                "history": history,
            }
        )
    expert_oof = _metrics(
        train.class_labels,
        oof_probability,
        train.regression_labels,
        oof_regression,
    )
    parent_oof_probability = _probability(parent_oof)
    parent_oof_regression = parent_oof["predicted_intensity"].to_numpy(np.float64)
    combined_oof_probability, combined_oof_regression = _combine(
        parent_oof_probability,
        parent_oof_regression,
        oof_probability,
        oof_regression,
        args.parent_member_count,
    )
    parent_oof_metrics = _metrics(
        train.class_labels,
        parent_oof_probability,
        train.regression_labels,
        parent_oof_regression,
    )
    combined_oof_metrics = _metrics(
        train.class_labels,
        combined_oof_probability,
        train.regression_labels,
        combined_oof_regression,
    )
    gate_passed = bool(
        combined_oof_metrics["accuracy"] >= parent_oof_metrics["accuracy"]
        and combined_oof_metrics["macro_f1"] > parent_oof_metrics["macro_f1"]
    )
    _prediction_frame(
        train.ids,
        train.class_labels,
        train.regression_labels,
        oof_probability,
        oof_regression,
        fold_ids,
    ).to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    report: dict[str, Any] = {
        "scope": "grouped_train_oof_promotion_before_locked_valid_no_test_access",
        "architecture": "frozen_minilm_sentence_embedding_centroid_prototype_expert",
        "research_basis": {
            "Sentence_BERT_EMNLP_2019": "frozen mean-pooled sentence representation",
            "model": args.pretrained_model,
            "revision": args.revision,
            "implementation_note": "maintained Hugging Face weights; original task head",
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_dimension": args.hidden_dimension,
        "parent_member_count": args.parent_member_count,
        "parent_oof_source": args.parent_oof,
        "expert_oof": expert_oof,
        "parent_oof": parent_oof_metrics,
        "combined_oof": combined_oof_metrics,
        "oof_delta": {
            "accuracy": combined_oof_metrics["accuracy"] - parent_oof_metrics["accuracy"],
            "macro_f1": combined_oof_metrics["macro_f1"] - parent_oof_metrics["macro_f1"],
        },
        "oof_gate_passed": gate_passed,
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

    full_model, full_history = fit_head(
        train_embeddings,
        train.class_labels,
        train.regression_labels,
        device,
        args.seed,
        args.epochs,
        args.hidden_dimension,
        args.batch_size,
    )
    valid_raw = attach_labels_from_excel({"valid": loaded["valid"]}, args.labels)
    valid = parse_split(valid_raw, "valid", "text_shared", require_labels=True)
    valid_embeddings = encode_texts(
        valid.raw_text,
        tokenizer,
        encoder,
        device,
        args.max_length,
        args.encode_batch_size,
    )
    valid_probability, valid_regression, valid_diagnostics = predict_head(
        full_model, valid_embeddings, device
    )
    parent_valid = _load_parent(args.parent_valid, valid.ids)
    parent_valid_probability = _probability(parent_valid)
    parent_valid_regression = parent_valid["predicted_intensity"].to_numpy(np.float64)
    combined_valid_probability, combined_valid_regression = _combine(
        parent_valid_probability,
        parent_valid_regression,
        valid_probability,
        valid_regression,
        args.parent_member_count,
    )
    _prediction_frame(
        valid.ids,
        valid.class_labels,
        valid.regression_labels,
        valid_probability,
        valid_regression,
    ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    _prediction_frame(
        valid.ids,
        valid.class_labels,
        valid.regression_labels,
        combined_valid_probability,
        combined_valid_regression,
    ).to_csv(
        output / "combined_valid_predictions.csv", index=False, encoding="utf-8-sig"
    )
    atomic_torch_save(
        {
            "model_state": full_model.state_dict(),
            "embedding_dimension": train_embeddings.shape[1],
            "hidden_dimension": args.hidden_dimension,
            "model": args.pretrained_model,
            "revision": args.revision,
            "seed": args.seed,
            "epochs": args.epochs,
        },
        output / "prototype_head.pt",
    )
    report.update(
        {
            "full_training_history": full_history,
            "valid_diagnostics": valid_diagnostics,
            "expert_valid": _metrics(
                valid.class_labels,
                valid_probability,
                valid.regression_labels,
                valid_regression,
            ),
            "parent_valid": _metrics(
                valid.class_labels,
                parent_valid_probability,
                valid.regression_labels,
                parent_valid_regression,
            ),
            "valid": _metrics(
                valid.class_labels,
                combined_valid_probability,
                valid.regression_labels,
                combined_valid_regression,
            ),
            "parent_valid_source": args.parent_valid,
            "decision": "oof_gate_passed_then_single_locked_validation",
        }
    )
    save_json(report, output / "final_metrics.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
