from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASS_NAMES
from .metrics import compute_metrics


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)
POSITIVE_EMOTIONS = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}
NEGATIVE_EMOTIONS = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}
COGNITIVE_EMOTIONS = {"confusion", "curiosity", "realization", "surprise"}


def _video_groups(identifiers: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            value.rsplit("$_$", 1)[0] if "$_$" in value else value
            for value in identifiers.astype(str)
        ],
        dtype=object,
    )


def _aligned_reference(path: str | Path, identifiers: pd.Series) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "true_label", "true_intensity", "predicted_intensity", *PROBABILITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate ids")
    frame = frame.set_index(frame["id"].astype(str), drop=False)
    expected = identifiers.astype(str).tolist()
    if set(frame.index) != set(expected):
        raise ValueError(f"Reference ids do not align with ontology features: {path}")
    return frame.loc[expected].reset_index(drop=True)


def _emotion_names(frame: pd.DataFrame) -> list[str]:
    prefix = "emotion_logit_"
    names = sorted(column[len(prefix) :] for column in frame.columns if column.startswith(prefix))
    if "neutral" not in names:
        raise ValueError("Ontology table has no emotion_logit_neutral column")
    if len(names) < 3:
        raise ValueError("Ontology table contains too few emotion labels")
    return names


def _mean(frame: pd.DataFrame, prefix: str, names: set[str]) -> np.ndarray:
    selected = [f"{prefix}{name}" for name in sorted(names) if f"{prefix}{name}" in frame]
    if not selected:
        return np.zeros(len(frame), dtype=np.float64)
    return frame.loc[:, selected].to_numpy(np.float64).mean(axis=1)


def _max(frame: pd.DataFrame, prefix: str, names: set[str]) -> np.ndarray:
    selected = [f"{prefix}{name}" for name in sorted(names) if f"{prefix}{name}" in frame]
    if not selected:
        return np.zeros(len(frame), dtype=np.float64)
    return frame.loc[:, selected].to_numpy(np.float64).max(axis=1)


def build_features(frame: pd.DataFrame, mode: str, names: list[str]) -> np.ndarray:
    logits = frame.loc[:, [f"emotion_logit_{name}" for name in names]].to_numpy(np.float64)
    probability_prefix = "emotion_probability_"
    other_names = set(names).difference({"neutral"})
    semantic = np.column_stack(
        [
            frame["emotion_logit_neutral"].to_numpy(np.float64),
            frame["emotion_probability_neutral"].to_numpy(np.float64),
            _mean(frame, probability_prefix, POSITIVE_EMOTIONS),
            _max(frame, probability_prefix, POSITIVE_EMOTIONS),
            _mean(frame, probability_prefix, NEGATIVE_EMOTIONS),
            _max(frame, probability_prefix, NEGATIVE_EMOTIONS),
            _mean(frame, probability_prefix, COGNITIVE_EMOTIONS),
            _max(frame, probability_prefix, COGNITIVE_EMOTIONS),
            _max(frame, probability_prefix, other_names),
            frame["emotion_probability_neutral"].to_numpy(np.float64)
            - _max(frame, probability_prefix, other_names),
        ]
    )
    embedding_columns = sorted(
        column for column in frame.columns if column.startswith("emotion_embedding_")
    )
    embedding = (
        frame.loc[:, embedding_columns].to_numpy(np.float64)
        if embedding_columns
        else np.empty((len(frame), 0), dtype=np.float64)
    )
    if mode == "ontology":
        result = logits
    elif mode == "semantic":
        result = semantic
    elif mode == "hybrid":
        result = np.concatenate([logits, semantic], axis=1)
    elif mode == "embedding":
        if not embedding_columns:
            raise ValueError("Ontology table has no frozen emotion embeddings")
        result = embedding
    elif mode == "embedding_ontology":
        if not embedding_columns:
            raise ValueError("Ontology table has no frozen emotion embeddings")
        result = np.concatenate([embedding, logits, semantic], axis=1)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    if not np.isfinite(result).all():
        raise ValueError("Ontology features contain NaN/Inf")
    return result


def _correct_weighted_probability(probability: np.ndarray, positive_weight: float) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64).clip(1e-7, 1.0 - 1e-7)
    logit = np.log(probability / (1.0 - probability)) - math.log(positive_weight)
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))


def _binary_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(probability) >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, prediction, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "neutral_precision": float(precision[1]),
        "neutral_recall": float(recall[1]),
        "neutral_f1": float(f1[1]),
        "brier": float(np.mean(np.square(np.asarray(probability) - target))),
    }


def _make_model(c_value: float, positive_weight: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=c_value,
                    class_weight={0: 1.0, 1: positive_weight},
                    max_iter=2000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def neutral_residual_fusion(
    anchor_probability: np.ndarray,
    expert_neutral_probability: np.ndarray,
    prior: float,
    max_logit_residual: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded Neutral-vs-Polar update that leaves polar odds unchanged."""
    anchor = np.asarray(anchor_probability, dtype=np.float64).clip(1e-8, 1.0)
    anchor /= anchor.sum(axis=1, keepdims=True)
    neutral = anchor[:, 1].clip(1e-7, 1.0 - 1e-7)
    expert = np.asarray(expert_neutral_probability, dtype=np.float64).clip(1e-7, 1.0 - 1e-7)
    prior = float(np.clip(prior, 1e-7, 1.0 - 1e-7))
    evidence = np.log(expert / (1.0 - expert)) - math.log(prior / (1.0 - prior))
    evidence = np.clip(evidence, -max_logit_residual, max_logit_residual)
    uncertainty = 4.0 * neutral * (1.0 - neutral)
    correction = uncertainty * evidence
    anchor_logit = np.log(neutral / (1.0 - neutral))
    fused_neutral = 1.0 / (1.0 + np.exp(-np.clip(anchor_logit + correction, -30.0, 30.0)))
    polar_mass = 1.0 - fused_neutral
    old_polar = (anchor[:, 0] + anchor[:, 2]).clip(1e-8)
    fused = np.column_stack(
        [
            polar_mass * anchor[:, 0] / old_polar,
            fused_neutral,
            polar_mass * anchor[:, 2] / old_polar,
        ]
    )
    return fused, correction, uncertainty


def _load_targets(frame: pd.DataFrame) -> np.ndarray:
    if "true_label" not in frame:
        raise ValueError("Ontology table has no true_label column")
    mapped = frame["true_label"].map({name: index for index, name in enumerate(CLASS_NAMES)})
    if mapped.isna().any():
        raise ValueError("Ontology table contains an unknown true label")
    return mapped.to_numpy(np.int64)


def select(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.train_ontology)
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("Train ontology table contains duplicate ids")
    names = _emotion_names(frame)
    target_class = _load_targets(frame)
    target = (target_class == 1).astype(np.int64)
    groups = _video_groups(frame["id"])
    negative_to_neutral = float((target == 0).sum() / max(1, (target == 1).sum()))
    positive_weights = [1.0, math.sqrt(negative_to_neutral), negative_to_neutral]
    modes = ["ontology", "semantic", "hybrid"]
    if any(column.startswith("emotion_embedding_") for column in frame.columns):
        modes.extend(["embedding", "embedding_ontology"])
    c_values = [0.03, 0.10, 0.30, 1.00]
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_indices = list(splitter.split(np.zeros(len(frame)), target, groups))
    rows: list[dict[str, Any]] = []
    all_oof: dict[tuple[str, float, float], np.ndarray] = {}
    for mode in modes:
        features = build_features(frame, mode, names)
        for c_value in c_values:
            for positive_weight in positive_weights:
                oof = np.zeros(len(frame), dtype=np.float64)
                fold_id = np.full(len(frame), -1, dtype=np.int64)
                for index, (train_index, valid_index) in enumerate(fold_indices):
                    model = _make_model(c_value, positive_weight, args.seed + index)
                    model.fit(features[train_index], target[train_index])
                    # Retain the cost-sensitive posterior.  In this architecture
                    # 0.5 is the explicit Neutral-evidence boundary; undoing the
                    # class weight would require a different decision threshold
                    # and made the evidence semantics internally inconsistent.
                    oof[valid_index] = model.predict_proba(features[valid_index])[:, 1]
                    fold_id[valid_index] = index
                metrics = _binary_metrics(target, oof)
                score = 0.5 * metrics["balanced_accuracy"] + 0.5 * metrics["macro_f1"]
                key = (mode, c_value, positive_weight)
                all_oof[key] = oof
                rows.append(
                    {
                        "feature_mode": mode,
                        "c": c_value,
                        "positive_weight": positive_weight,
                        "selection_score": score,
                        **metrics,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["selection_score"],
            row["neutral_f1"],
            -row["brier"],
            -row["c"],
        ),
        reverse=True,
    )
    selected = rows[0]
    key = (
        str(selected["feature_mode"]),
        float(selected["c"]),
        float(selected["positive_weight"]),
    )
    features = build_features(frame, key[0], names)
    final_model = _make_model(key[1], key[2], args.seed)
    final_model.fit(features, target)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output / "calibrator.joblib")
    pd.DataFrame(rows).to_csv(output / "train_cv_table.csv", index=False, encoding="utf-8-sig")
    selected_fold_id = np.full(len(frame), -1, dtype=np.int64)
    for index, (_, valid_index) in enumerate(fold_indices):
        selected_fold_id[valid_index] = index
    if (selected_fold_id < 0).any():
        raise RuntimeError("Grouped CV did not assign every training sample exactly once")
    pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "fold": selected_fold_id,
            "true_neutral": target,
            "oof_neutral_probability": all_oof[key],
        }
    ).to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "scope": "train_only_grouped_cv_no_valid_or_test_access",
        "seed": args.seed,
        "folds": args.folds,
        "grouping": "sample id prefix before $_$",
        "samples": len(frame),
        "groups": int(len(set(groups))),
        "neutral_prior": float(target.mean()),
        "emotion_labels": names,
        "selected": selected,
        "fixed_fusion": {
            "kind": "bounded_neutral_log_odds_residual",
            "max_logit_residual": args.max_logit_residual,
            "expert_evidence_center": 0.5,
            "uncertainty": "4*p_anchor_neutral*(1-p_anchor_neutral)",
            "polar_odds_invariant": True,
        },
        "artifacts": {
            "calibrator": "calibrator.joblib",
            "cv_table": "train_cv_table.csv",
            "oof_predictions": "train_oof_predictions.csv",
        },
    }
    (output / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    model = joblib.load(output / "calibrator.joblib")
    frame = pd.read_csv(args.ontology)
    names = _emotion_names(frame)
    if names != list(selection["emotion_labels"]):
        raise ValueError("Emotion ontology labels differ from the locked train selection")
    selected = selection["selected"]
    features = build_features(frame, str(selected["feature_mode"]), names)
    expert = model.predict_proba(features)[:, 1]
    reference = _aligned_reference(args.reference_predictions, frame["id"])
    target = _load_targets(frame)
    expected = np.asarray([CLASS_NAMES[index] for index in target])
    if not np.array_equal(expected, reference["true_label"].astype(str).to_numpy()):
        raise ValueError("Ontology/reference true labels do not match")
    anchor = reference.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    fused, correction, uncertainty = neutral_residual_fusion(
        anchor,
        expert,
        float(selection["fixed_fusion"]["expert_evidence_center"]),
        float(selection["fixed_fusion"]["max_logit_residual"]),
    )
    regression_target = reference["true_intensity"].to_numpy(np.float64)
    regression = reference["predicted_intensity"].to_numpy(np.float64)
    metrics = compute_metrics(target, fused, regression_target, regression)
    anchor_metrics = compute_metrics(target, anchor, regression_target, regression)
    expert_metrics = _binary_metrics((target == 1).astype(np.int64), expert)
    polar_before = np.log(anchor[:, 2].clip(1e-12) / anchor[:, 0].clip(1e-12))
    polar_after = np.log(fused[:, 2].clip(1e-12) / fused[:, 0].clip(1e-12))
    report: dict[str, Any] = {
        "scope": f"locked_train_selection_{args.split}_evaluation_no_test_access"
        if args.split != "test"
        else "locked_train_selection_final_test_evaluation",
        "selection": str(output / "selection.json"),
        "reference": str(args.reference_predictions),
        "split": args.split,
        "anchor": anchor_metrics,
        "expert_neutral_binary": expert_metrics,
        "fused": metrics,
        "diagnostics": {
            "mean_expert_neutral_probability": float(expert.mean()),
            "mean_logit_correction": float(correction.mean()),
            "mean_anchor_uncertainty": float(uncertainty.mean()),
            "changed_decisions": int((anchor.argmax(1) != fused.argmax(1)).sum()),
            "polar_log_odds_max_abs_change": float(np.max(np.abs(polar_before - polar_after))),
        },
    }
    pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in fused.argmax(1)],
            "negative_probability": fused[:, 0],
            "neutral_probability": fused[:, 1],
            "positive_probability": fused[:, 2],
            "predicted_intensity": regression,
            "true_label": expected,
            "true_intensity": regression_target,
            "ontology_neutral_probability": expert,
            "neutral_logit_correction": correction,
            "anchor_uncertainty": uncertainty,
        }
    ).to_csv(output / f"{args.split}_predictions.csv", index=False, encoding="utf-8-sig")
    (output / f"{args.split}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train-only grouped-CV ontology Neutral calibrator and locked evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--train-ontology", required=True)
    select_parser.add_argument("--output", required=True)
    select_parser.add_argument("--folds", type=int, default=5)
    select_parser.add_argument("--seed", type=int, default=20260924)
    select_parser.add_argument("--max-logit-residual", type=float, default=1.5)
    select_parser.set_defaults(function=select)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--ontology", required=True)
    evaluate_parser.add_argument("--reference-predictions", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    evaluate_parser.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
