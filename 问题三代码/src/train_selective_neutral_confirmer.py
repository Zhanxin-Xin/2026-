from __future__ import annotations

"""Nested grouped-OOF selective Neutral confirmation over a locked parent.

The confirmer is deliberately unable to change Positive-versus-Negative odds.
It may only move a parent polar decision to Neutral when four independent
conditions agree: parent ambiguity, a cross-fitted Neutral calibrator, the
frozen ontology's explicit Neutral score, and low polar-emotion evidence.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics
from .train_ontology_neutral_calibrator import build_features, _emotion_names


POSITIVE_EMOTIONS = (
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
)
NEGATIVE_EMOTIONS = (
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
)


@dataclass(frozen=True)
class ConfirmationRule:
    enabled: bool
    maximum_parent_margin: float = -1.0
    minimum_calibrated_neutral: float = 2.0
    minimum_ontology_neutral: float = 2.0
    maximum_polar_emotion: float = -1.0


NO_OP = ConfirmationRule(enabled=False)


def candidate_rules() -> list[ConfirmationRule]:
    rules = [NO_OP]
    for margin in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30):
        for calibrated in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            for ontology in (0.0, 0.05, 0.10, 0.15, 0.20):
                for polar in (1.0, 0.50, 0.30, 0.20, 0.15, 0.10, 0.07):
                    rules.append(
                        ConfirmationRule(True, margin, calibrated, ontology, polar)
                    )
    return rules


def _target(frame: pd.DataFrame) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    result = frame["true_label"].map(mapping)
    if result.isna().any():
        raise ValueError("Parent predictions contain an unknown class label")
    return result.to_numpy(np.int64)


def _align(frame: pd.DataFrame, ids: pd.Series, description: str) -> pd.DataFrame:
    if "id" not in frame:
        raise ValueError(f"{description} has no id column")
    index = frame["id"].astype(str)
    if index.duplicated().any():
        raise ValueError(f"{description} contains duplicate ids")
    aligned = frame.set_index(index, drop=False)
    expected = ids.astype(str).tolist()
    if set(aligned.index) != set(expected):
        raise ValueError(f"{description} ids do not align with the parent")
    return aligned.loc[expected].reset_index(drop=True)


def load_semantic_evidence(path: str | Path, ids: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    names = set(POSITIVE_EMOTIONS + NEGATIVE_EMOTIONS)
    columns = {"id", "emotion_probability_neutral"}
    columns.update(f"emotion_probability_{name}" for name in names)
    frame = pd.read_csv(path, usecols=lambda column: column in columns)
    frame = _align(frame, ids, "ontology evidence")
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Ontology evidence is missing columns: {sorted(missing)}")
    neutral = frame["emotion_probability_neutral"].to_numpy(np.float64)
    polar = frame.loc[:, sorted(columns.difference({"id", "emotion_probability_neutral"}))]
    polar_maximum = polar.to_numpy(np.float64).max(axis=1)
    if not np.isfinite(neutral).all() or not np.isfinite(polar_maximum).all():
        raise ValueError("Ontology evidence contains NaN/Inf")
    return neutral, polar_maximum


def confirmation_mask(
    parent: np.ndarray,
    calibrated_neutral: np.ndarray,
    ontology_neutral: np.ndarray,
    polar_emotion_maximum: np.ndarray,
    rule: ConfirmationRule,
) -> np.ndarray:
    if not rule.enabled:
        return np.zeros(len(parent), dtype=bool)
    prediction = parent.argmax(axis=1)
    polar_maximum = np.maximum(parent[:, 0], parent[:, 2])
    margin = polar_maximum - parent[:, 1]
    return (
        (prediction != 1)
        & (margin <= rule.maximum_parent_margin)
        & (calibrated_neutral >= rule.minimum_calibrated_neutral)
        & (ontology_neutral >= rule.minimum_ontology_neutral)
        & (polar_emotion_maximum <= rule.maximum_polar_emotion)
    )


def minimal_neutral_projection(parent: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cross the Neutral decision face by the smallest polar-odds invariant move."""
    output = np.asarray(parent, dtype=np.float64).copy()
    output /= output.sum(axis=1, keepdims=True).clip(min=1e-12)
    if not np.any(mask):
        return output
    polar_mass = (output[mask, 0] + output[mask, 2]).clip(min=1e-12)
    polar_share = output[mask][:, [0, 2]] / polar_mass[:, None]
    boundary = polar_share.max(axis=1) / (1.0 + polar_share.max(axis=1))
    neutral = np.maximum(output[mask, 1], boundary + 1e-6)
    neutral = np.minimum(neutral, 1.0 - 1e-6)
    output[np.ix_(mask, [0, 2])] = (1.0 - neutral)[:, None] * polar_share
    output[mask, 1] = neutral
    return output / output.sum(axis=1, keepdims=True).clip(min=1e-12)


def apply_rule(
    parent: np.ndarray,
    calibrated_neutral: np.ndarray,
    ontology_neutral: np.ndarray,
    polar_emotion_maximum: np.ndarray,
    rule: ConfirmationRule,
) -> tuple[np.ndarray, np.ndarray]:
    mask = confirmation_mask(
        parent, calibrated_neutral, ontology_neutral, polar_emotion_maximum, rule
    )
    return minimal_neutral_projection(parent, mask), mask


def _fast_metrics(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    prediction = probability.argmax(axis=1)
    return (
        float(accuracy_score(target, prediction)),
        float(f1_score(target, prediction, average="macro", zero_division=0)),
    )


def select_rule(
    parent: np.ndarray,
    calibrated_neutral: np.ndarray,
    ontology_neutral: np.ndarray,
    polar_emotion_maximum: np.ndarray,
    target: np.ndarray,
    rules: list[ConfirmationRule],
) -> tuple[ConfirmationRule, dict[str, float], list[dict[str, Any]]]:
    parent_accuracy, parent_macro = _fast_metrics(target, parent)
    minimum_accuracy_gain = 1.0 / len(target)
    minimum_macro_gain = 5e-4
    rows: list[dict[str, Any]] = []
    eligible: list[tuple[tuple[float, float, float, int], ConfirmationRule, dict[str, float]]] = []
    for rule in rules:
        probability, changed = apply_rule(
            parent, calibrated_neutral, ontology_neutral, polar_emotion_maximum, rule
        )
        accuracy, macro_f1 = _fast_metrics(target, probability)
        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "accuracy_delta": accuracy - parent_accuracy,
            "macro_f1_delta": macro_f1 - parent_macro,
            "changed_decisions": int(changed.sum()),
        }
        rows.append({**asdict(rule), **metrics})
        if not rule.enabled:
            eligible.append(((0.0, 0.0, 0.0, 0), rule, metrics))
        elif (
            metrics["changed_decisions"] > 0
            and metrics["accuracy_delta"] + 1e-12 >= minimum_accuracy_gain
            and metrics["macro_f1_delta"] + 1e-12 >= minimum_macro_gain
        ):
            key = (
                min(metrics["accuracy_delta"], metrics["macro_f1_delta"]),
                metrics["macro_f1_delta"],
                metrics["accuracy_delta"],
                -metrics["changed_decisions"],
            )
            eligible.append((key, rule, metrics))
    selected = max(eligible, key=lambda item: item[0])
    return selected[1], selected[2], rows


def _full_metrics(reference: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    return compute_metrics(
        _target(reference),
        probability,
        reference["true_intensity"].to_numpy(np.float64),
        reference["predicted_intensity"].to_numpy(np.float64),
    )


def _output_frame(reference: pd.DataFrame, probability: np.ndarray, fold: bool) -> pd.DataFrame:
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
    if fold:
        result.insert(1, "fold", reference["fold"].to_numpy(np.int64))
    return result


def _valid_expert_probability(ontology_path: str | Path, selection_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    directory = Path(selection_dir)
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(ontology_path)
    names = _emotion_names(frame)
    if names != list(selection["emotion_labels"]):
        raise ValueError("Valid ontology labels differ from the locked train selection")
    selected = selection["selected"]
    features = build_features(frame, str(selected["feature_mode"]), names)
    model = joblib.load(directory / "calibrator.joblib")
    return frame, model.predict_proba(features)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested grouped-OOF selective Neutral confirmer")
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--train-ontology", required=True)
    parser.add_argument("--expert-oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-valid")
    parser.add_argument("--valid-ontology")
    parser.add_argument("--expert-selection-dir")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parent_frame = load_aligned([args.parent_oof], require_fold=True)[0]
    parent = parent_frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    target = _target(parent_frame)
    folds = parent_frame["fold"].to_numpy(np.int64)
    ontology_neutral, polar_emotion = load_semantic_evidence(
        args.train_ontology, parent_frame["id"]
    )
    expert_frame = _align(
        pd.read_csv(args.expert_oof), parent_frame["id"], "cross-fitted Neutral expert"
    )
    calibrated = expert_frame["oof_neutral_probability"].to_numpy(np.float64)
    rules = candidate_rules()

    cross_fitted = parent.copy()
    fold_reports: list[dict[str, Any]] = []
    for fold_id in sorted(np.unique(folds).tolist()):
        fit = folds != fold_id
        heldout = folds == fold_id
        rule, fit_metrics, _ = select_rule(
            parent[fit], calibrated[fit], ontology_neutral[fit], polar_emotion[fit],
            target[fit], rules,
        )
        heldout_probability, changed = apply_rule(
            parent[heldout], calibrated[heldout], ontology_neutral[heldout],
            polar_emotion[heldout], rule,
        )
        cross_fitted[heldout] = heldout_probability
        heldout_parent_accuracy, heldout_parent_macro = _fast_metrics(
            target[heldout], parent[heldout]
        )
        heldout_accuracy, heldout_macro = _fast_metrics(target[heldout], heldout_probability)
        fold_reports.append(
            {
                "fold": int(fold_id),
                "selected_rule": asdict(rule),
                "fit_selection_metrics": fit_metrics,
                "heldout_changed_decisions": int(changed.sum()),
                "heldout_accuracy": heldout_accuracy,
                "heldout_macro_f1": heldout_macro,
                "heldout_accuracy_delta": heldout_accuracy - heldout_parent_accuracy,
                "heldout_macro_f1_delta": heldout_macro - heldout_parent_macro,
            }
        )

    final_rule, full_selection, candidate_rows = select_rule(
        parent, calibrated, ontology_neutral, polar_emotion, target, rules
    )
    pd.DataFrame(candidate_rows).to_csv(
        output / "full_oof_candidate_table.csv", index=False, encoding="utf-8-sig"
    )
    parent_metrics = _full_metrics(parent_frame, parent)
    oof_metrics = _full_metrics(parent_frame, cross_fitted)
    gate = bool(
        oof_metrics["accuracy"] + 1e-12 >= parent_metrics["accuracy"]
        and oof_metrics["macro_f1"] > parent_metrics["macro_f1"] + 1e-12
    )
    _output_frame(parent_frame, cross_fitted, fold=True).to_csv(
        output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    report: dict[str, Any] = {
        "scope": "nested_grouped_oof_gate_before_locked_valid_no_test_access",
        "architecture": "four_condition_selective_neutral_confirmer_with_minimal_projection",
        "research_basis": {
            "SelectiveNet_ICML_2019": {
                "idea": "learn or select only where evidence is sufficiently reliable",
                "repository": "https://github.com/geifmany/selectivenet",
                "commit": "a6d0a8fd33dae61da910b61a2aae93102d2d4869",
                "license_note": "No root license was exposed; no source code was copied.",
            },
            "Deep_Gamblers_NeurIPS_2019": {
                "idea": "explicit reject option for uncertain decisions",
                "repository": "https://github.com/Z-T-WANG/NIPS2019DeepGamblers",
                "commit": "0d1b595611a8bc653fddfdf1419bd8dbde153532",
                "license": "MIT",
            },
            "GoEmotions_ACL_2020": {
                "idea": "frozen fine-grained emotion ontology as independent evidence",
                "repository": "https://github.com/google-research/google-research/tree/master/goemotions",
                "commit": "d36068b845da4c2b24927fee2cea1e6ef98dadda",
                "license": "Apache-2.0",
            },
            "implementation_note": "Original implementation; no external source code copied.",
        },
        "seed": args.seed,
        "fold_reports": fold_reports,
        "parent_oof": parent_metrics,
        "oof": oof_metrics,
        "oof_delta": {
            "accuracy": oof_metrics["accuracy"] - parent_metrics["accuracy"],
            "macro_f1": oof_metrics["macro_f1"] - parent_metrics["macro_f1"],
        },
        "oof_gate_passed": gate,
        "full_fit_rule_for_valid": asdict(final_rule),
        "full_fit_selection_metrics": full_selection,
        "valid": None,
        "decision": "closed_before_loading_official_valid",
    }

    if gate:
        required = (args.parent_valid, args.valid_ontology, args.expert_selection_dir)
        if any(value is None for value in required):
            raise ValueError("OOF gate passed; valid parent, ontology, and expert selection are required")
        valid_parent_frame = load_aligned([args.parent_valid], require_fold=False)[0]
        ontology_frame, valid_calibrated = _valid_expert_probability(
            args.valid_ontology, args.expert_selection_dir
        )
        ontology_frame = _align(ontology_frame, valid_parent_frame["id"], "valid ontology")
        valid_calibrated = pd.Series(
            valid_calibrated, index=pd.read_csv(args.valid_ontology, usecols=["id"])["id"].astype(str)
        ).loc[valid_parent_frame["id"].astype(str)].to_numpy(np.float64)
        valid_ontology_neutral, valid_polar_emotion = load_semantic_evidence(
            args.valid_ontology, valid_parent_frame["id"]
        )
        valid_parent = valid_parent_frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
        valid_probability, valid_changed = apply_rule(
            valid_parent, valid_calibrated, valid_ontology_neutral,
            valid_polar_emotion, final_rule,
        )
        _output_frame(valid_parent_frame, valid_probability, fold=False).to_csv(
            output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report["valid"] = _full_metrics(valid_parent_frame, valid_probability)
        report["valid_changed_decisions"] = int(valid_changed.sum())
        report["decision"] = "promoted_after_nested_oof_gate"

    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
