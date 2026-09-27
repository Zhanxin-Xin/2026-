from __future__ import annotations

"""Prospective fixed retrieval recipe with leave-one-video-group-out memory."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_pickle, parse_split
from .metrics import compute_metrics
from .retrieval_memory_fusion import (
    PROBABILITY_COLUMNS,
    class_balanced_posterior,
    masked_mean_features,
    standardized_cosine_similarity,
    uncertainty_gated_fusion,
    video_groups,
)
from .train_cross_fitted_retrieval_memory import (
    aligned_reference,
    prediction_frame,
    targets,
)


def group_exclusive_memory(
    memory: np.ndarray,
    query: np.ndarray,
    memory_labels: np.ndarray,
    memory_ids: np.ndarray,
    memory_groups: np.ndarray,
    query_groups: np.ndarray,
    k: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.zeros((len(query), 3), dtype=np.float64)
    confidence = np.zeros(len(query), dtype=np.float64)
    for group in np.unique(query_groups):
        query_index = np.flatnonzero(query_groups == group)
        memory_index = np.flatnonzero(memory_groups != group)
        if not len(memory_index):
            raise ValueError(f"No group-exclusive memory remains for {group}")
        similarity = standardized_cosine_similarity(
            memory[memory_index], query[query_index]
        )
        posterior, _, posterior_confidence = class_balanced_posterior(
            similarity,
            memory_labels[memory_index],
            memory_ids[memory_index],
            k,
            temperature,
        )
        probability[query_index] = posterior
        confidence[query_index] = posterior_confidence
    return probability, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--oof-reference", required=True)
    parser.add_argument("--valid-reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Frozen prospectively from EXP132; no search occurs in this program.
    modality = "vision"
    k = 4
    temperature = 0.20
    maximum_weight = 0.30

    loaded = load_pickle(args.data)
    train = parse_split({"train": loaded["train"]}, "train", "text_shared", True)
    reference = aligned_reference(args.oof_reference, train.ids)
    target = targets(reference)
    if not np.array_equal(target, train.class_labels):
        raise ValueError("Reference labels disagree with training data")
    features = masked_mean_features(train)[modality]
    groups = video_groups(train.ids)
    memory_probability, memory_confidence = group_exclusive_memory(
        features,
        features,
        train.class_labels,
        train.ids.astype(str),
        groups,
        groups,
        k,
        temperature,
    )
    anchor = reference.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    fused, weight = uncertainty_gated_fusion(
        anchor, memory_probability, memory_confidence, maximum_weight
    )
    regression = reference["predicted_intensity"].to_numpy(np.float64)
    intensity = reference["true_intensity"].to_numpy(np.float64)
    parent_metrics = compute_metrics(target, anchor, intensity, regression)
    oof_metrics = compute_metrics(target, fused, intensity, regression)
    gate_passed = bool(
        oof_metrics["accuracy"] + 1e-12 >= parent_metrics["accuracy"]
        and oof_metrics["macro_f1"] > parent_metrics["macro_f1"] + 1e-12
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    folds = (
        reference["fold"].to_numpy(np.int64)
        if "fold" in reference.columns
        else None
    )
    prediction_frame(reference, fused, regression, folds).to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "scope": "fixed_leave_one_video_group_out_retrieval_gate_before_valid_no_test_access",
        "architecture": "fixed_vision_class_balanced_group_exclusive_memory",
        "source_experiment": "EXP132 prospective replication",
        "fixed_candidate": {
            "modality": modality,
            "k": k,
            "temperature": temperature,
            "maximum_weight": maximum_weight,
        },
        "group_exclusion_verified": True,
        "parent_oof": parent_metrics,
        "oof": oof_metrics,
        "oof_delta": {
            "accuracy": oof_metrics["accuracy"] - parent_metrics["accuracy"],
            "macro_f1": oof_metrics["macro_f1"] - parent_metrics["macro_f1"],
        },
        "diagnostics": {
            "memory_confidence_mean": float(memory_confidence.mean()),
            "weight_mean": float(weight.mean()),
            "weight_max": float(weight.max()),
        },
        "oof_gate_passed": gate_passed,
    }
    if gate_passed:
        valid = parse_split({"valid": loaded["valid"]}, "valid", "text_shared", True)
        valid_reference = aligned_reference(args.valid_reference, valid.ids)
        valid_features = masked_mean_features(valid)[modality]
        valid_probability, valid_confidence = group_exclusive_memory(
            features,
            valid_features,
            train.class_labels,
            train.ids.astype(str),
            groups,
            video_groups(valid.ids),
            k,
            temperature,
        )
        valid_anchor = valid_reference.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
        valid_fused, valid_weight = uncertainty_gated_fusion(
            valid_anchor,
            valid_probability,
            valid_confidence,
            maximum_weight,
        )
        valid_regression = valid_reference["predicted_intensity"].to_numpy(np.float64)
        valid_intensity = valid_reference["true_intensity"].to_numpy(np.float64)
        report["valid"] = compute_metrics(
            targets(valid_reference),
            valid_fused,
            valid_intensity,
            valid_regression,
        )
        report["valid_diagnostics"] = {
            "memory_confidence_mean": float(valid_confidence.mean()),
            "weight_mean": float(valid_weight.mean()),
            "weight_max": float(valid_weight.max()),
        }
        report["decision"] = "promoted_after_fixed_oof_replication_gate"
        prediction_frame(
            valid_reference, valid_fused, valid_regression, None
        ).to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    else:
        report["valid"] = None
        report["decision"] = "closed_before_loading_official_valid"
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
