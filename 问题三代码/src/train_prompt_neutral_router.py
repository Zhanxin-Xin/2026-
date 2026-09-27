from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import attach_labels_from_excel, load_pickle, parse_split
from .diagnose_nli_sentiment import (
    _entailment_index,
    _load_reference_predictions,
)
from .metrics import CLASS_NAMES, compute_metrics
from .utils import (
    apply_overrides,
    atomic_torch_save,
    load_config,
    make_logger,
    resolve_device,
    save_json,
    seed_everything,
)


class PromptFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets == 1, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]


class PromptAttentionNeutralRouter(nn.Module):
    """Attend over multiple NLI verbalizers to isolate Neutral from Polar."""

    def __init__(self, prompt_count: int, hidden_dimension: int, dropout: float) -> None:
        super().__init__()
        self.prompt_count = prompt_count
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, hidden_dimension),
            nn.GELU(),
        )
        self.prompt_score = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.Tanh(),
            nn.Linear(hidden_dimension // 2, 1),
        )
        self.neutral_head = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1:] != (self.prompt_count, 3):
            raise ValueError(
                f"Expected [B,{self.prompt_count},3] prompt probabilities, "
                f"got {tuple(features.shape)}"
            )
        prompt = self.prompt_projection(features)
        weights = torch.softmax(self.prompt_score(prompt).squeeze(-1), dim=-1)
        pooled = (weights.unsqueeze(-1) * prompt).sum(dim=1)
        return self.neutral_head(pooled).squeeze(-1), weights


@torch.inference_mode()
def extract_prompt_features(
    model: AutoModelForSequenceClassification,
    tokenizer: Any,
    texts: list[str],
    prompt_sets: list[list[str]],
    entailment_index: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
    amp: bool,
) -> np.ndarray:
    flat_hypotheses = [hypothesis for prompts in prompt_sets for hypothesis in prompts]
    prompt_count = len(prompt_sets)
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        premises = [text for text in batch_texts for _ in flat_hypotheses]
        hypotheses = flat_hypotheses * len(batch_texts)
        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.amp.autocast(
            "cuda", enabled=amp and device.type == "cuda", dtype=torch.float16
        ):
            relation_logits = model(**encoded).logits
        entailment = relation_logits[:, entailment_index].float().reshape(
            len(batch_texts), prompt_count, 3
        )
        chunks.append(torch.softmax(entailment, dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = probability >= 0.5
    precision, recall, f1, support = precision_recall_fscore_support(
        target, prediction, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "neutral_f1": float(f1_score(target, prediction, zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(("Polar", "Neutral"))
        },
    }


def hurdle_probability(
    neutral_probability: np.ndarray,
    reference_probability: np.ndarray,
) -> np.ndarray:
    polar = reference_probability[:, [0, 2]]
    polar = polar / np.clip(polar.sum(axis=1, keepdims=True), 1e-8, None)
    remaining = 1.0 - neutral_probability
    return np.column_stack(
        [remaining * polar[:, 0], neutral_probability, remaining * polar[:, 1]]
    )


@torch.inference_mode()
def predict_router(
    model: PromptAttentionNeutralRouter,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    tensor = torch.as_tensor(features, dtype=torch.float32)
    for start in range(0, len(tensor), batch_size):
        logits, prompt_weights = model(tensor[start : start + batch_size].to(device))
        probability.append(torch.sigmoid(logits).cpu().numpy())
        weights.append(prompt_weights.cpu().numpy())
    return np.concatenate(probability), np.concatenate(weights)


def make_prediction_frame(
    ids: np.ndarray,
    targets: np.ndarray,
    regression_targets: np.ndarray,
    probabilities: np.ndarray,
    regression: np.ndarray,
    neutral_probability: np.ndarray,
    prompt_weights: np.ndarray,
) -> pd.DataFrame:
    prediction = probabilities.argmax(axis=1)
    data: dict[str, Any] = {
        "id": [str(value) for value in ids],
        "predicted_label": [CLASS_NAMES[index] for index in prediction],
        "negative_probability": probabilities[:, 0],
        "neutral_probability": probabilities[:, 1],
        "positive_probability": probabilities[:, 2],
        "predicted_intensity": regression,
        "true_label": [CLASS_NAMES[index] for index in targets],
        "true_intensity": regression_targets,
        "semantic_neutral_probability": neutral_probability,
    }
    for index in range(prompt_weights.shape[1]):
        data[f"prompt_{index}_weight"] = prompt_weights[:, index]
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a prompt-attention NLI Neutral router on train/valid only. "
            "The held-out test split is deliberately inaccessible."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed_everything(int(cfg["seed"]))
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "resolved_config.json")
    logger = make_logger(output, name="prompt_neutral_router")
    logger.info("Device: %s", device)

    loaded = load_pickle(args.data)
    raw = attach_labels_from_excel(
        {name: loaded[name] for name in ("train", "valid")}, args.labels
    )
    arrays = {
        name: parse_split(raw, name, "text_shared", require_labels=True)
        for name in ("train", "valid")
    }
    model_cfg = cfg["nli_model"]
    load_kwargs = {
        "revision": model_cfg.get("revision"),
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]), **load_kwargs
    )
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        str(model_cfg["pretrained_model"]), **load_kwargs
    ).to(device)
    entailment_index = _entailment_index(nli_model.config)
    prompt_sets = [
        [str(hypothesis) for hypothesis in prompts]
        for prompts in cfg["prompt_sets"]
    ]
    if not prompt_sets or any(len(prompts) != 3 for prompts in prompt_sets):
        raise ValueError("Each prompt set must contain Negative/Neutral/Positive hypotheses")
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    features = {
        name: extract_prompt_features(
            nli_model,
            tokenizer,
            [str(value) for value in split.raw_text],
            prompt_sets,
            entailment_index,
            device,
            int(cfg["data"].get("nli_batch_size", 4)),
            int(cfg["data"].get("max_length", 128)),
            amp,
        )
        for name, split in arrays.items()
    }
    del nli_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.savez_compressed(
        output / "prompt_features_train_valid.npz",
        train_ids=np.asarray(arrays["train"].ids, dtype=str),
        train_features=features["train"],
        valid_ids=np.asarray(arrays["valid"].ids, dtype=str),
        valid_features=features["valid"],
    )

    reference = {}
    for name, split in arrays.items():
        reference[name] = _load_reference_predictions(
            Path(args.reference) / f"{name}_predictions.csv",
            split.ids,
            split.class_labels,
        )
    training_cfg = cfg["training"]
    datasets = {
        name: PromptFeatureDataset(value, arrays[name].class_labels)
        for name, value in features.items()
    }
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(training_cfg.get("batch_size", 64)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    router = PromptAttentionNeutralRouter(
        prompt_count=len(prompt_sets),
        hidden_dimension=int(cfg["router"].get("hidden_dimension", 32)),
        dropout=float(cfg["router"].get("dropout", 0.15)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(training_cfg.get("learning_rate", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    neutral_count = int((arrays["train"].class_labels == 1).sum())
    polar_count = arrays["train"].size - neutral_count
    positive_weight = torch.tensor(
        polar_count / max(1, neutral_count), device=device, dtype=torch.float32
    )
    best_score = -float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output / "best_model.pt"
    started = time.time()
    for epoch in range(1, int(training_cfg.get("epochs", 80)) + 1):
        router.train()
        total_loss = 0.0
        for cpu_features, cpu_targets in train_loader:
            batch_features = cpu_features.to(device)
            batch_targets = cpu_targets.to(device)
            logits, _ = router(batch_features)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_targets, pos_weight=positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                router.parameters(), float(training_cfg.get("grad_clip", 1.0))
            )
            optimizer.step()
            total_loss += float(loss.detach())

        valid_neutral, valid_weights = predict_router(
            router,
            features["valid"],
            device,
            int(training_cfg.get("eval_batch_size", 256)),
        )
        valid_reference_probability, valid_reference_regression = reference["valid"]
        valid_probability = hurdle_probability(
            valid_neutral, valid_reference_probability
        )
        valid_metrics = compute_metrics(
            arrays["valid"].class_labels,
            valid_probability,
            arrays["valid"].regression_labels,
            valid_reference_regression,
        )
        neutral_metrics = binary_metrics(
            arrays["valid"].class_labels == 1, valid_neutral
        )
        score = 0.4 * valid_metrics["accuracy"] + 0.6 * valid_metrics["macro_f1"]
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            "valid": valid_metrics,
            "neutral_binary": neutral_metrics,
            "selection_score": score,
            "mean_prompt_weights": valid_weights.mean(axis=0).tolist(),
        }
        history.append(record)
        save_json(history, output / "history.json")
        logger.info(
            "Epoch %03d | loss %.4f | valid accuracy %.4f macro-F1 %.4f "
            "Neutral-F1 %.4f | score %.5f",
            epoch,
            record["train_loss"],
            valid_metrics["accuracy"],
            valid_metrics["macro_f1"],
            valid_metrics["per_class"]["Neutral"]["f1"],
            score,
        )
        if score > best_score + float(training_cfg.get("min_delta", 0.0001)):
            best_score = score
            patience = 0
            atomic_torch_save(
                {
                    "model_state": router.state_dict(),
                    "config": copy.deepcopy(cfg),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "neutral_binary": neutral_metrics,
                    "selection_score": score,
                    "resolved_commit_hash": getattr(
                        tokenizer, "_commit_hash", model_cfg.get("revision")
                    ),
                },
                checkpoint_path,
            )
        else:
            patience += 1
            if patience >= int(training_cfg.get("early_stopping_patience", 10)):
                logger.info("Early stopping at epoch %d", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    router.load_state_dict(checkpoint["model_state"], strict=True)
    final: dict[str, Any] = {
        "scope": "train_and_validation_only_no_test_access",
        "best_epoch": checkpoint["epoch"],
        "best_selection_score": checkpoint["selection_score"],
        "elapsed_minutes": (time.time() - started) / 60.0,
        "prompt_sets": prompt_sets,
    }
    for name, split in arrays.items():
        neutral_probability, prompt_weights = predict_router(
            router,
            features[name],
            device,
            int(training_cfg.get("eval_batch_size", 256)),
        )
        reference_probability, reference_regression = reference[name]
        probability = hurdle_probability(neutral_probability, reference_probability)
        final[name] = compute_metrics(
            split.class_labels,
            probability,
            split.regression_labels,
            reference_regression,
        )
        final[name]["neutral_binary"] = binary_metrics(
            split.class_labels == 1, neutral_probability
        )
        final[name]["mean_prompt_weights"] = prompt_weights.mean(axis=0).tolist()
        make_prediction_frame(
            split.ids,
            split.class_labels,
            split.regression_labels,
            probability,
            reference_regression,
            neutral_probability,
            prompt_weights,
        ).to_csv(
            output / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    save_json(final, output / "final_metrics.json")
    logger.info("Final metrics: %s", final)


if __name__ == "__main__":
    main()
