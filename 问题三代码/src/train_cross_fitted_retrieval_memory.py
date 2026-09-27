from __future__ import annotations

"""Strict cross-fitted class-balanced retrieval memory over an OOF parent."""

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import CLASS_NAMES, MODALITIES, load_pickle, parse_split
from .metrics import compute_metrics
from .retrieval_memory_fusion import (
    PROBABILITY_COLUMNS,
    class_balanced_posterior,
    combined_similarity,
    masked_mean_features,
    standardized_cosine_similarity,
    uncertainty_gated_fusion,
)


@dataclass(frozen=True)
class Candidate:
    modalities: tuple[str, ...]
    k: int
    temperature: float
    maximum_weight: float


MODALITY_SETS = (
    ("text",),
    ("audio",),
    ("vision",),
    ("text", "audio"),
    ("text", "vision"),
    ("audio", "vision"),
    ("text", "audio", "vision"),
)
K_VALUES = (1, 2, 4, 8, 16, 32)
TEMPERATURES = (0.02, 0.05, 0.10, 0.20, 0.40)
MAXIMUM_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def aligned_reference(path: str | Path, identifiers: np.ndarray) -> pd.DataFrame:
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
        raise ValueError(f"{path} contains duplicate IDs")
    frame = frame.set_index("id", drop=False)
    expected = [str(value) for value in identifiers]
    if set(frame.index) != set(expected):
        raise ValueError(f"{path} IDs do not align with feature split")
    return frame.loc[expected].reset_index(drop=True)


def targets(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    result = frame["true_label"].map(mapping)
    if result.isna().any():
        raise ValueError("Unknown class label")
    return result.to_numpy(np.int64)


def fast_metrics(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    prediction = probability.argmax(axis=1)
    encoded = 3 * target.astype(np.int64) + prediction.astype(np.int64)
    matrix = np.bincount(encoded, minlength=9).reshape(3, 3)
    accuracy = float(np.trace(matrix) / max(1, matrix.sum()))
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(
        2.0 * np.diag(matrix),
        denominator,
        out=np.zeros(3, dtype=np.float64),
        where=denominator > 0,
    )
    return accuracy, float(f1.mean())


def candidate_grid() -> list[Candidate]:
    return [
        Candidate(modalities, k, temperature, maximum_weight)
        for modalities, k, temperature, maximum_weight in itertools.product(
            MODALITY_SETS, K_VALUES, TEMPERATURES, MAXIMUM_WEIGHTS
        )
    ]


def build_memory_oof(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    identifiers: np.ndarray,
    folds: np.ndarray,
) -> dict[tuple[tuple[str, ...], int, float], tuple[np.ndarray, np.ndarray]]:
    keys = list(itertools.product(MODALITY_SETS, K_VALUES, TEMPERATURES))
    predictions = {
        key: (
            np.zeros((len(labels), 3), dtype=np.float64),
            np.zeros(len(labels), dtype=np.float64),
        )
        for key in keys
    }
    for fold in sorted(np.unique(folds).tolist()):
        fit = folds != fold
        heldout = folds == fold
        similarity_by_modality = {
            modality: standardized_cosine_similarity(
                features[modality][fit], features[modality][heldout]
            )
            for modality in MODALITIES
        }
        for modalities in MODALITY_SETS:
            similarity = np.mean(
                [similarity_by_modality[name] for name in modalities],
                axis=0,
                dtype=np.float32,
            )
            for k in K_VALUES:
                for temperature in TEMPERATURES:
                    probability, _, confidence = class_balanced_posterior(
                        similarity,
                        labels[fit],
                        identifiers[fit],
                        k,
                        temperature,
                    )
                    predictions[(modalities, k, temperature)][0][heldout] = probability
                    predictions[(modalities, k, temperature)][1][heldout] = confidence
    return predictions


def fused_candidate(
    anchor: np.ndarray,
    memory: tuple[np.ndarray, np.ndarray],
    maximum_weight: float,
) -> np.ndarray:
    return uncertainty_gated_fusion(
        anchor, memory[0], memory[1], maximum_weight
    )[0]


def select_candidate(
    anchor: np.ndarray,
    memories: dict[tuple[tuple[str, ...], int, float], tuple[np.ndarray, np.ndarray]],
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[Candidate, dict[str, float]]:
    parent_accuracy, parent_macro_f1 = fast_metrics(target[mask], anchor[mask])
    selected: tuple[tuple[float, float, float], Candidate] | None = None
    selected_metrics: dict[str, float] | None = None
    for candidate in candidate_grid():
        memory = memories[
            (candidate.modalities, candidate.k, candidate.temperature)
        ]
        probability = fused_candidate(
            anchor[mask],
            (memory[0][mask], memory[1][mask]),
            candidate.maximum_weight,
        )
        accuracy, macro_f1 = fast_metrics(target[mask], probability)
        if accuracy + 1e-12 < parent_accuracy:
            continue
        key = (macro_f1, accuracy, -candidate.maximum_weight)
        if selected is None or key > selected[0]:
            selected = (key, candidate)
            selected_metrics = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "parent_accuracy": parent_accuracy,
                "parent_macro_f1": parent_macro_f1,
            }
    if selected is None or selected_metrics is None:
        raise RuntimeError("No retrieval candidate satisfied the accuracy guard")
    return selected[1], selected_metrics


def prediction_frame(
    reference: pd.DataFrame,
    probability: np.ndarray,
    regression: np.ndarray,
    folds: np.ndarray | None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": reference["true_label"].astype(str),
            "true_intensity": reference["true_intensity"].to_numpy(np.float64),
        }
    )
    if folds is not None:
        output.insert(1, "fold", folds)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--oof-reference", required=True)
    parser.add_argument("--valid-reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    loaded = load_pickle(args.data)
    train = parse_split({"train": loaded["train"]}, "train", "text_shared", True)
    reference = aligned_reference(args.oof_reference, train.ids)
    if "fold" not in reference.columns:
        raise ValueError("OOF reference must contain fold assignments")
    label = targets(reference)
    if not np.array_equal(label, train.class_labels):
        raise ValueError("OOF labels disagree with training features")
    folds = reference["fold"].to_numpy(np.int64)
    anchor = reference.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    regression = reference["predicted_intensity"].to_numpy(np.float64)
    features = masked_mean_features(train)
    memories = build_memory_oof(features, label, train.ids.astype(str), folds)

    cross_fitted = np.empty_like(anchor)
    fold_reports: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds).tolist()):
        fit = folds != fold
        heldout = folds == fold
        candidate, fit_metrics = select_candidate(anchor, memories, label, fit)
        memory = memories[(candidate.modalities, candidate.k, candidate.temperature)]
        cross_fitted[heldout] = fused_candidate(
            anchor[heldout],
            (memory[0][heldout], memory[1][heldout]),
            candidate.maximum_weight,
        )
        heldout_accuracy, heldout_macro_f1 = fast_metrics(
            label[heldout], cross_fitted[heldout]
        )
        fold_reports.append(
            {
                "fold": int(fold),
                "candidate": asdict(candidate),
                "fit_metrics": fit_metrics,
                "heldout_accuracy": heldout_accuracy,
                "heldout_macro_f1": heldout_macro_f1,
            }
        )

    all_rows = np.ones(len(label), dtype=bool)
    full_candidate, full_selection_metrics = select_candidate(
        anchor, memories, label, all_rows
    )
    intensity_target = reference["true_intensity"].to_numpy(np.float64)
    parent_metrics = compute_metrics(label, anchor, intensity_target, regression)
    oof_metrics = compute_metrics(label, cross_fitted, intensity_target, regression)
    gate_passed = bool(
        oof_metrics["accuracy"] + 1e-12 >= parent_metrics["accuracy"]
        and oof_metrics["macro_f1"] > parent_metrics["macro_f1"] + 1e-12
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame(reference, cross_fitted, regression, folds).to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report: dict[str, Any] = {
        "scope": "strict_grouped_cross_fitted_retrieval_gate_before_locked_valid_no_test_access",
        "architecture": "class_balanced_multimodal_retrieval_memory_with_uncertainty_gate",
        "research_basis": {
            "kNN_LM_ICLR_2020": "distance-weighted nonparametric memory",
            "Tip_Adapter_ECCV_2022": "cache keys and class values",
            "implementation": "original NumPy implementation; no external source copied",
        },
        "candidate_count": len(candidate_grid()),
        "fold_reports": fold_reports,
        "full_oof_candidate_for_valid": asdict(full_candidate),
        "full_oof_selection_metrics": full_selection_metrics,
        "parent_oof": parent_metrics,
        "oof": oof_metrics,
        "oof_delta": {
            "accuracy": oof_metrics["accuracy"] - parent_metrics["accuracy"],
            "macro_f1": oof_metrics["macro_f1"] - parent_metrics["macro_f1"],
        },
        "oof_gate_passed": gate_passed,
    }
    if gate_passed:
        valid = parse_split({"valid": loaded["valid"]}, "valid", "text_shared", True)
        valid_reference = aligned_reference(args.valid_reference, valid.ids)
        valid_features = masked_mean_features(valid)
        similarity = combined_similarity(
            features,
            valid_features,
            tuple(full_candidate.modalities),
        )
        memory_probability, _, memory_confidence = class_balanced_posterior(
            similarity,
            train.class_labels,
            train.ids.astype(str),
            full_candidate.k,
            full_candidate.temperature,
        )
        valid_anchor = valid_reference.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
        valid_probability = uncertainty_gated_fusion(
            valid_anchor,
            memory_probability,
            memory_confidence,
            full_candidate.maximum_weight,
        )[0]
        valid_regression = valid_reference["predicted_intensity"].to_numpy(np.float64)
        valid_target = targets(valid_reference)
        valid_intensity = valid_reference["true_intensity"].to_numpy(np.float64)
        report["valid"] = compute_metrics(
            valid_target, valid_probability, valid_intensity, valid_regression
        )
        report["decision"] = "promoted_after_oof_gate"
        prediction_frame(
            valid_reference, valid_probability, valid_regression, None
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
