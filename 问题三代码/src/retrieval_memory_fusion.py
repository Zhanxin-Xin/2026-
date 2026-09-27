from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from .data import CLASS_NAMES, MODALITIES, SplitArrays, load_pickle, parse_split
from .metrics import compute_metrics


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)


def video_groups(identifiers: Iterable[Any]) -> np.ndarray:
    return np.asarray(
        [
            value.rsplit("$_$", 1)[0] if "$_$" in value else value
            for value in map(str, identifiers)
        ],
        dtype=object,
    )


def masked_mean_features(arrays: SplitArrays) -> dict[str, np.ndarray]:
    pooled: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        mask = arrays.masks[modality][..., None].astype(np.float32)
        denominator = mask.sum(axis=1).clip(min=1.0)
        value = (arrays.features[modality] * mask).sum(axis=1) / denominator
        if not np.isfinite(value).all():
            raise ValueError(f"Pooled {modality} features contain NaN/Inf")
        pooled[modality] = value.astype(np.float32)
    return pooled


def standardized_cosine_similarity(
    memory: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Fit standardization on memory only, then return query-to-memory cosine."""
    memory = np.asarray(memory, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    mean = memory.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = memory.std(axis=0, dtype=np.float64).astype(np.float32).clip(min=1e-6)
    memory_normalized = (memory - mean) / std
    query_normalized = (query - mean) / std
    memory_normalized /= np.linalg.norm(
        memory_normalized, axis=1, keepdims=True
    ).clip(min=1e-8)
    query_normalized /= np.linalg.norm(
        query_normalized, axis=1, keepdims=True
    ).clip(min=1e-8)
    similarity = query_normalized @ memory_normalized.T
    if not np.isfinite(similarity).all():
        raise ValueError("Retrieval similarity contains NaN/Inf")
    return similarity.astype(np.float32)


def combined_similarity(
    memory_features: dict[str, np.ndarray],
    query_features: dict[str, np.ndarray],
    modalities: tuple[str, ...],
) -> np.ndarray:
    if not modalities:
        raise ValueError("At least one retrieval modality is required")
    unknown = set(modalities).difference(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown retrieval modalities: {sorted(unknown)}")
    matrices = [
        standardized_cosine_similarity(memory_features[name], query_features[name])
        for name in modalities
    ]
    return np.mean(matrices, axis=0, dtype=np.float32)


def class_balanced_posterior(
    similarity: np.ndarray,
    memory_labels: np.ndarray,
    memory_ids: np.ndarray,
    k: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve the same number per class with deterministic ID tie-breaking."""
    similarity = np.asarray(similarity, dtype=np.float64)
    memory_labels = np.asarray(memory_labels, dtype=np.int64)
    memory_ids = np.asarray(memory_ids, dtype=str)
    if similarity.ndim != 2 or similarity.shape[1] != len(memory_labels):
        raise ValueError("Similarity/memory label shape mismatch")
    if k < 1 or temperature <= 0.0:
        raise ValueError("k and temperature must be positive")
    top_scores: list[np.ndarray] = []
    top_indices: list[np.ndarray] = []
    effective_k: list[int] = []
    for class_index in range(3):
        candidates = np.flatnonzero(memory_labels == class_index)
        if not len(candidates):
            raise ValueError(f"Memory contains no samples for class {class_index}")
        # Sorting candidates by ID first and then using a stable score sort makes
        # equal-similarity ties resolve lexicographically by sample ID.
        candidates = candidates[np.argsort(memory_ids[candidates], kind="stable")]
        class_similarity = similarity[:, candidates]
        order = np.argsort(-class_similarity, axis=1, kind="stable")
        use_k = min(k, len(candidates))
        selected_order = order[:, :use_k]
        selected = candidates[selected_order]
        top_indices.append(selected)
        top_scores.append(np.take_along_axis(similarity, selected, axis=1))
        effective_k.append(use_k)
    width = max(effective_k)
    score_cube = np.full((similarity.shape[0], 3, width), -np.inf, dtype=np.float64)
    index_cube = np.full((similarity.shape[0], 3, width), -1, dtype=np.int64)
    for class_index, (scores, indices) in enumerate(zip(top_scores, top_indices)):
        score_cube[:, class_index, : scores.shape[1]] = scores
        index_cube[:, class_index, : indices.shape[1]] = indices
    scaled = score_cube / temperature
    global_max = np.max(scaled, axis=(1, 2), keepdims=True)
    evidence = np.exp(scaled - global_max)
    evidence[~np.isfinite(scaled)] = 0.0
    class_evidence = evidence.sum(axis=2)
    probability = class_evidence / class_evidence.sum(axis=1, keepdims=True).clip(min=1e-12)
    confidence = 1.0 - (
        -(probability.clip(min=1e-12) * np.log(probability.clip(min=1e-12))).sum(axis=1)
        / math.log(3.0)
    )
    return probability.astype(np.float64), index_cube, confidence.astype(np.float64)


def uncertainty_gated_fusion(
    anchor: np.ndarray,
    memory: np.ndarray,
    memory_confidence: np.ndarray,
    max_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    anchor = np.asarray(anchor, dtype=np.float64).clip(min=1e-12)
    memory = np.asarray(memory, dtype=np.float64).clip(min=1e-12)
    anchor /= anchor.sum(axis=1, keepdims=True)
    memory /= memory.sum(axis=1, keepdims=True)
    uncertainty = (1.0 - anchor.max(axis=1)) / (2.0 / 3.0)
    uncertainty = uncertainty.clip(0.0, 1.0)
    confidence = np.asarray(memory_confidence, dtype=np.float64).clip(0.0, 1.0)
    # Keep a small cache path even when its class posterior is diffuse; suppress
    # it on confident anchor decisions and strengthen it only with local evidence.
    weight = max_weight * uncertainty * (0.5 + 0.5 * confidence)
    fused = (1.0 - weight[:, None]) * anchor + weight[:, None] * memory
    fused /= fused.sum(axis=1, keepdims=True).clip(min=1e-12)
    return fused, weight


def _aligned_reference(path: str | Path, arrays: SplitArrays) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "true_label", "true_intensity", "predicted_intensity", *PROBABILITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate IDs")
    frame = frame.set_index(frame["id"].astype(str), drop=False)
    expected = [str(value) for value in arrays.ids]
    if set(frame.index) != set(expected):
        raise ValueError(f"Reference IDs do not align with split: {path}")
    aligned = frame.loc[expected].reset_index(drop=True)
    labels = np.asarray([CLASS_NAMES[index] for index in arrays.class_labels])
    if not np.array_equal(labels, aligned["true_label"].astype(str).to_numpy()):
        raise ValueError("Reference and feature labels disagree")
    return aligned


def _metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
    }


def _parse_modalities(value: str) -> tuple[str, ...]:
    return tuple(value.split("+"))


def select(args: argparse.Namespace) -> None:
    loaded = load_pickle(args.data)
    train = parse_split({"train": loaded["train"]}, "train", "text_shared", True)
    features = masked_mean_features(train)
    groups = video_groups(train.ids)
    folds = list(GroupKFold(args.folds).split(np.zeros(train.size), train.class_labels, groups))
    modality_sets = [
        ("text",),
        ("audio",),
        ("vision",),
        ("text", "audio"),
        ("text", "vision"),
        ("audio", "vision"),
        ("text", "audio", "vision"),
    ]
    k_values = [1, 2, 4, 8, 16, 32]
    temperatures = [0.02, 0.05, 0.10, 0.20, 0.40]
    oof = {
        (modalities, k, temperature): np.zeros((train.size, 3), dtype=np.float64)
        for modalities in modality_sets
        for k in k_values
        for temperature in temperatures
    }
    for train_index, valid_index in folds:
        similarities = {
            modality: standardized_cosine_similarity(
                features[modality][train_index], features[modality][valid_index]
            )
            for modality in MODALITIES
        }
        for modalities in modality_sets:
            similarity = np.mean(
                [similarities[modality] for modality in modalities],
                axis=0,
                dtype=np.float32,
            )
            for k in k_values:
                for temperature in temperatures:
                    probability, _, _ = class_balanced_posterior(
                        similarity,
                        train.class_labels[train_index],
                        train.ids[train_index],
                        k,
                        temperature,
                    )
                    oof[(modalities, k, temperature)][valid_index] = probability
    rows: list[dict[str, Any]] = []
    for (modalities, k, temperature), probability in oof.items():
        metrics = _metrics(train.class_labels, probability)
        rows.append(
            {
                "modalities": "+".join(modalities),
                "k_per_class": k,
                "temperature": temperature,
                "selection_score": 0.5 * metrics["accuracy"] + 0.5 * metrics["macro_f1"],
                **metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            row["selection_score"],
            row["macro_f1"],
            row["accuracy"],
            -row["k_per_class"],
        ),
        reverse=True,
    )
    selected = rows[0]
    key = (
        _parse_modalities(str(selected["modalities"])),
        int(selected["k_per_class"]),
        float(selected["temperature"]),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "train_cv_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "id": [str(value) for value in train.ids],
            "true_label": [CLASS_NAMES[index] for index in train.class_labels],
            "negative_probability": oof[key][:, 0],
            "neutral_probability": oof[key][:, 1],
            "positive_probability": oof[key][:, 2],
        }
    ).to_csv(output / "train_oof_memory_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "scope": "train_only_grouped_cv_no_valid_or_test_access",
        "data": str(args.data),
        "pooling": "mask_aware_mean_then_memory_fitted_standardization_and_l2",
        "grouping": "sample id prefix before $_$",
        "folds": args.folds,
        "samples": train.size,
        "groups": int(len(set(groups))),
        "selection_objective": "mean_accuracy_and_macro_f1",
        "selected": selected,
        "fixed_fusion": {
            "kind": "anchor_uncertainty_and_memory_confidence_gated_probability_residual",
            "max_weight": args.max_weight,
            "anchor": str(args.anchor),
        },
    }
    (output / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if args.split == "test" and not args.allow_test:
        raise ValueError("Test evaluation requires the explicit --allow-test gate")
    loaded = load_pickle(args.data)
    train = parse_split({"train": loaded["train"]}, "train", "text_shared", True)
    query = parse_split(
        {args.split: loaded[args.split]}, args.split, "text_shared", require_labels=True
    )
    selected = selection["selected"]
    modalities = _parse_modalities(str(selected["modalities"]))
    memory_features = masked_mean_features(train)
    query_features = masked_mean_features(query)
    similarity = combined_similarity(memory_features, query_features, modalities)
    memory_probability, neighbors, memory_confidence = class_balanced_posterior(
        similarity,
        train.class_labels,
        train.ids,
        int(selected["k_per_class"]),
        float(selected["temperature"]),
    )
    reference = _aligned_reference(args.reference_predictions, query)
    anchor = reference.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    fused, weight = uncertainty_gated_fusion(
        anchor,
        memory_probability,
        memory_confidence,
        float(selection["fixed_fusion"]["max_weight"]),
    )
    regression = reference["predicted_intensity"].to_numpy(np.float64)
    regression_target = reference["true_intensity"].to_numpy(np.float64)
    anchor_metrics = compute_metrics(
        query.class_labels, anchor, regression_target, regression
    )
    memory_metrics = compute_metrics(
        query.class_labels, memory_probability, regression_target, regression
    )
    fused_metrics = compute_metrics(
        query.class_labels, fused, regression_target, regression
    )
    anchor_correct = anchor.argmax(1) == query.class_labels
    fused_correct = fused.argmax(1) == query.class_labels
    report = {
        "scope": f"locked_train_selection_{args.split}_evaluation",
        "selection": str(output / "selection.json"),
        "reference": str(args.reference_predictions),
        "anchor": anchor_metrics,
        "memory": memory_metrics,
        "fused": fused_metrics,
        "diagnostics": {
            "weight_mean": float(weight.mean()),
            "weight_max": float(weight.max()),
            "memory_confidence_mean": float(memory_confidence.mean()),
            "changed_decisions": int((anchor.argmax(1) != fused.argmax(1)).sum()),
            "fixed_anchor_errors": int((~anchor_correct & fused_correct).sum()),
            "broken_anchor_correct": int((anchor_correct & ~fused_correct).sum()),
        },
    }
    pd.DataFrame(
        {
            "id": [str(value) for value in query.ids],
            "predicted_label": [CLASS_NAMES[index] for index in fused.argmax(1)],
            "negative_probability": fused[:, 0],
            "neutral_probability": fused[:, 1],
            "positive_probability": fused[:, 2],
            "predicted_intensity": regression,
            "true_label": [CLASS_NAMES[index] for index in query.class_labels],
            "true_intensity": regression_target,
            "memory_negative_probability": memory_probability[:, 0],
            "memory_neutral_probability": memory_probability[:, 1],
            "memory_positive_probability": memory_probability[:, 2],
            "memory_confidence": memory_confidence,
            "memory_weight": weight,
        }
    ).to_csv(output / f"{args.split}_predictions.csv", index=False, encoding="utf-8-sig")
    neighbor_rows: list[dict[str, Any]] = []
    for query_index, identifier in enumerate(query.ids):
        for class_index in range(3):
            for rank, memory_index in enumerate(neighbors[query_index, class_index]):
                if memory_index < 0:
                    continue
                neighbor_rows.append(
                    {
                        "query_id": str(identifier),
                        "class": CLASS_NAMES[class_index],
                        "rank": rank + 1,
                        "memory_id": str(train.ids[memory_index]),
                        "similarity": float(similarity[query_index, memory_index]),
                    }
                )
    pd.DataFrame(neighbor_rows).to_csv(
        output / f"{args.split}_neighbors.csv", index=False, encoding="utf-8-sig"
    )
    (output / f"{args.split}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only class-balanced retrieval memory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--data", required=True)
    select_parser.add_argument("--anchor", required=True)
    select_parser.add_argument("--output", required=True)
    select_parser.add_argument("--folds", type=int, default=5)
    select_parser.add_argument("--max-weight", type=float, default=0.25)
    select_parser.set_defaults(function=select)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--reference-predictions", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    evaluate_parser.add_argument("--allow-test", action="store_true")
    evaluate_parser.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
