from __future__ import annotations

"""Frozen EmoBERTa conversation-emotion expert with a strict OOF gate."""

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES, load_pickle, parse_split
from .metrics import compute_metrics
from .utils import resolve_device


def _probability(frame: pd.DataFrame) -> np.ndarray:
    value = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Invalid parent probability")
    return value / value.sum(axis=1, keepdims=True).clip(min=1e-12)


def _target(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    target = frame["true_label"].map(mapping)
    if target.isna().any():
        raise ValueError("Unknown true label in parent predictions")
    return target.to_numpy(np.int64)


def _align_parent(path: str | Path, expected_ids: np.ndarray, require_fold: bool) -> pd.DataFrame:
    frame = load_aligned([path], require_fold=require_fold)[0]
    frame["id"] = frame["id"].astype(str)
    indexed = frame.set_index("id", drop=False)
    expected = np.asarray(expected_ids, dtype=str)
    if set(indexed.index) != set(expected):
        raise ValueError(f"{path} IDs do not align with the dataset split")
    return indexed.loc[expected].reset_index(drop=True)


@torch.inference_mode()
def extract_features(
    texts: np.ndarray,
    tokenizer: Any,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Return emotion-head logits plus normalized final-layer CLS state."""
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            [str(value) for value in texts[start : start + batch_size]],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device, non_blocking=True) for name, value in encoded.items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            sequence = model.base_model(**encoded, return_dict=True).last_hidden_state
            logits: Tensor = model.classifier(sequence)
            embedding = F.normalize(sequence[:, 0].float(), dim=1)
        feature = torch.cat([logits.float(), embedding], dim=1)
        chunks.append(feature.cpu().numpy().astype(np.float32))
    result = np.concatenate(chunks, axis=0)
    if result.shape[0] != len(texts) or not np.isfinite(result).all():
        raise RuntimeError("EmoBERTa feature extraction produced invalid output")
    return result


def make_discriminant(feature_dimension: int, sample_count: int, seed: int) -> Pipeline:
    components = min(128, feature_dimension, sample_count - len(CLASS_NAMES))
    if components < len(CLASS_NAMES):
        raise ValueError("Too few samples for the low-rank discriminant expert")
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "low_rank",
                PCA(
                    n_components=components,
                    whiten=True,
                    svd_solver="randomized",
                    random_state=seed,
                ),
            ),
            (
                "discriminant",
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ]
    )


def fit_predict(
    fit_features: np.ndarray,
    fit_target: np.ndarray,
    query_features: np.ndarray,
    seed: int,
) -> tuple[Pipeline, np.ndarray]:
    model = make_discriminant(fit_features.shape[1], len(fit_features), seed)
    model.fit(fit_features, fit_target)
    probability = model.predict_proba(query_features)
    if list(model.classes_) != list(range(len(CLASS_NAMES))):
        raise ValueError(f"Unexpected discriminant classes: {model.classes_}")
    return model, probability.astype(np.float64)


def _metrics(reference: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    return compute_metrics(
        _target(reference),
        probability,
        reference["true_intensity"].to_numpy(np.float64),
        reference["predicted_intensity"].to_numpy(np.float64),
    )


def combine(parent: np.ndarray, expert: np.ndarray, parent_member_count: int) -> np.ndarray:
    if parent_member_count < 1:
        raise ValueError("parent_member_count must be positive")
    result = (parent_member_count * parent + expert) / float(parent_member_count + 1)
    return result / result.sum(axis=1, keepdims=True).clip(min=1e-12)


def _prediction_frame(
    reference: pd.DataFrame, probability: np.ndarray, include_fold: bool
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(axis=1)],
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


def _cached_features(
    path: Path,
    ids: np.ndarray,
    texts: np.ndarray,
    tokenizer: Any,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    expected = np.asarray(ids, dtype=str)
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        cached_ids = cached["ids"].astype(str)
        if np.array_equal(cached_ids, expected):
            features = cached["features"].astype(np.float32)
            if len(features) == len(expected) and np.isfinite(features).all():
                return features
    features = extract_features(texts, tokenizer, model, device, batch_size, max_length)
    np.savez_compressed(path, ids=expected, features=features.astype(np.float16))
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--parent-valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="tae898/emoberta-large")
    parser.add_argument("--revision", default="8934b68e8b0d9fc3cd961cc7e7605533c7081e59")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--parent-member-count", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if min(args.parent_member_count, args.batch_size, args.max_length) < 1:
        raise ValueError("Count arguments must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    loaded = load_pickle(args.data)
    train = parse_split({"train": loaded["train"]}, "train", "text_shared", require_labels=False)
    parent_oof = _align_parent(args.parent_oof, train.ids, require_fold=True)
    target = _target(parent_oof)
    folds = parent_oof["fold"].to_numpy(np.int64)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    encoder = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    ).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    labels = {int(index): str(name) for index, name in dict(encoder.config.id2label).items()}
    if labels != {
        0: "neutral", 1: "joy", 2: "surprise", 3: "anger",
        4: "sadness", 5: "disgust", 6: "fear",
    }:
        raise ValueError(f"Unexpected EmoBERTa label contract: {labels}")
    train_features = _cached_features(
        output / "train_features.npz",
        train.ids,
        train.raw_text,
        tokenizer,
        encoder,
        device,
        args.batch_size,
        args.max_length,
    )

    oof_expert = np.zeros((train.size, len(CLASS_NAMES)), dtype=np.float64)
    fold_reports: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds).tolist()):
        fit = folds != fold
        heldout = folds == fold
        _, probability = fit_predict(
            train_features[fit], target[fit], train_features[heldout], args.seed + fold
        )
        oof_expert[heldout] = probability
        fold_reference = parent_oof.loc[heldout].reset_index(drop=True)
        fold_reports.append(
            {
                "fold": int(fold),
                "fit_samples": int(fit.sum()),
                "heldout_samples": int(heldout.sum()),
                "metrics": _metrics(fold_reference, probability),
            }
        )

    parent_probability = _probability(parent_oof)
    combined_oof = combine(parent_probability, oof_expert, args.parent_member_count)
    parent_metrics = _metrics(parent_oof, parent_probability)
    expert_metrics = _metrics(parent_oof, oof_expert)
    combined_metrics = _metrics(parent_oof, combined_oof)
    gate = bool(
        combined_metrics["accuracy"] + 1e-12 >= parent_metrics["accuracy"]
        and combined_metrics["macro_f1"] > parent_metrics["macro_f1"] + 1e-12
    )
    _prediction_frame(parent_oof, oof_expert, include_fold=True).to_csv(
        output / "train_expert_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    _prediction_frame(parent_oof, combined_oof, include_fold=True).to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report: dict[str, Any] = {
        "scope": "grouped_train_oof_gate_before_locked_valid_no_test_access",
        "architecture": "frozen_emoberta_large_low_rank_shrinkage_lda_eighth_expert",
        "research_basis": {
            "EmoBERTa": {
                "idea": "conversation-domain emotion pretraining with an explicit Neutral class",
                "repository": "https://github.com/tae898/erc",
                "commit": "faf25370d8805a1975bda768cb94288d5d72c490",
                "license": "MIT",
            },
            "model": args.model,
            "revision": args.revision,
            "training_domains": ["MELD", "IEMOCAP"],
            "implementation_note": "Original frozen-feature/low-rank LDA adapter; no repository source copied.",
        },
        "seed": args.seed,
        "parent_member_count": args.parent_member_count,
        "feature_dimension": int(train_features.shape[1]),
        "fold_reports": fold_reports,
        "expert_oof": expert_metrics,
        "parent_oof": parent_metrics,
        "oof": combined_metrics,
        "oof_delta": {
            "accuracy": combined_metrics["accuracy"] - parent_metrics["accuracy"],
            "macro_f1": combined_metrics["macro_f1"] - parent_metrics["macro_f1"],
        },
        "oof_gate_passed": gate,
        "valid": None,
        "decision": "closed_before_loading_official_valid",
    }
    if not gate:
        (output / "final_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    full_model, _ = fit_predict(train_features, target, train_features[:1], args.seed)
    joblib.dump(full_model, output / "emoberta_discriminant.joblib")
    valid = parse_split({"valid": loaded["valid"]}, "valid", "text_shared", require_labels=False)
    parent_valid = _align_parent(args.parent_valid, valid.ids, require_fold=False)
    valid_features = _cached_features(
        output / "valid_features.npz",
        valid.ids,
        valid.raw_text,
        tokenizer,
        encoder,
        device,
        args.batch_size,
        args.max_length,
    )
    valid_expert = full_model.predict_proba(valid_features).astype(np.float64)
    valid_combined = combine(_probability(parent_valid), valid_expert, args.parent_member_count)
    _prediction_frame(parent_valid, valid_expert, include_fold=False).to_csv(
        output / "valid_expert_predictions.csv", index=False, encoding="utf-8-sig"
    )
    _prediction_frame(parent_valid, valid_combined, include_fold=False).to_csv(
        output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report["valid"] = _metrics(parent_valid, valid_combined)
    report["decision"] = "promoted_after_grouped_oof_gate"
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
