"""Fit and freeze the EXP186 sparse Neutral calibration head.

The official validation split is explicitly treated as calibration training
data.  Consequently, its reported number is a calibration-fit score, not an
independent validation estimate.  Test files are neither accepted nor opened
by this program; deployment evaluation is a separate command after the lock is
written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from .build_cross_fitted_ensemble import load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics
from .train_explainable_hierarchical_nam import ConceptBundle, build_concepts


LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}
LABELS = np.asarray(CLASS_NAMES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def apply_sparse_neutral_head(
    bundle: ConceptBundle,
    model: Any,
    neutral_logit_bias: float,
) -> tuple[np.ndarray, np.ndarray]:
    neutral_probability = np.clip(
        model.predict_proba(bundle.values)[:, 1], 1e-8, 1.0 - 1e-8
    )
    neutral_logit = np.log(neutral_probability / (1.0 - neutral_probability))
    neutral_probability = 1.0 / (
        1.0 + np.exp(-(neutral_logit + float(neutral_logit_bias)))
    )
    polar_odds = bundle.parent_probability[:, [0, 2]].copy()
    polar_odds /= polar_odds.sum(axis=1, keepdims=True).clip(min=1e-12)
    probability = np.column_stack(
        [
            (1.0 - neutral_probability) * polar_odds[:, 0],
            neutral_probability,
            (1.0 - neutral_probability) * polar_odds[:, 1],
        ]
    )
    return probability, neutral_logit + float(neutral_logit_bias)


def metrics(
    reference: pd.DataFrame, bundle: ConceptBundle, probability: np.ndarray
) -> dict[str, Any]:
    return compute_metrics(
        reference["true_label"].map(LABEL_MAP).to_numpy(np.int64),
        probability,
        reference["true_intensity"].to_numpy(np.float64),
        bundle.regression,
    )


def prediction_frame(
    reference: pd.DataFrame,
    bundle: ConceptBundle,
    probability: np.ndarray,
    neutral_logit: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": LABELS[probability.argmax(axis=1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": bundle.regression,
            "sparse_neutral_logit": neutral_logit,
            "true_label": reference["true_label"].astype(str),
            "true_intensity": reference["true_intensity"].to_numpy(np.float64),
        }
    )


def contribution_frame(
    reference: pd.DataFrame,
    bundle: ConceptBundle,
    model: Any,
    neutral_logit_bias: float,
) -> pd.DataFrame:
    scaler: StandardScaler = model.named_steps["standardscaler"]
    classifier: LogisticRegression = model.named_steps["logisticregression"]
    standardized = scaler.transform(bundle.values)
    contribution = standardized * classifier.coef_[0][None, :]
    output: dict[str, Any] = {
        "id": reference["id"].astype(str).to_numpy(),
        "intercept_plus_bias": np.full(
            len(reference),
            float(classifier.intercept_[0] + neutral_logit_bias),
        ),
    }
    for index, name in enumerate(bundle.names):
        output[f"neutral__{name}"] = contribution[:, index]
    output["reconstructed_neutral_logit"] = (
        contribution.sum(axis=1)
        + float(classifier.intercept_[0] + neutral_logit_bias)
    )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and lock EXP186")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    valid_sources = [str(Path(value)) for value in config["valid_sources"]]
    oof_sources = [str(Path(value)) for value in config["oof_sources"]]
    valid_frames = load_aligned(valid_sources, require_fold=False)
    valid_reference = valid_frames[0]
    valid_bundle = build_concepts(valid_frames)
    valid_target = (
        valid_reference["true_label"].map(LABEL_MAP).to_numpy(np.int64) == 1
    ).astype(np.int64)
    c_value = float(config["regularization_c"])
    neutral_bias = float(config["neutral_logit_bias"])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=20260924,
        ),
    )
    model.fit(valid_bundle.values, valid_target)
    valid_probability, valid_neutral_logit = apply_sparse_neutral_head(
        valid_bundle, model, neutral_bias
    )
    valid_metrics = metrics(valid_reference, valid_bundle, valid_probability)

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "sparse_neutral_head.joblib")
    prediction_frame(
        valid_reference,
        valid_bundle,
        valid_probability,
        valid_neutral_logit,
    ).to_csv(output / "valid_calibration_predictions.csv", index=False, encoding="utf-8-sig")
    contribution_frame(valid_reference, valid_bundle, model, neutral_bias).to_csv(
        output / "valid_concept_contributions.csv", index=False, encoding="utf-8-sig"
    )

    classifier: LogisticRegression = model.named_steps["logisticregression"]
    coefficient = classifier.coef_[0]
    coefficient_table = pd.DataFrame(
        {
            "concept": valid_bundle.names,
            "concept_group": valid_bundle.groups,
            "coefficient": coefficient,
            "absolute_coefficient": np.abs(coefficient),
            "selected": coefficient != 0.0,
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_table.to_csv(
        output / "global_sparse_coefficients.csv", index=False, encoding="utf-8-sig"
    )

    # Reverse-domain audit only: this model was fitted on validation, so train
    # OOF remains label-independent evaluation for the calibration head.  It is
    # diagnostic and is not used to tune the already fixed C or bias.
    oof_frames = load_aligned(oof_sources, require_fold=True)
    oof_reference = oof_frames[0]
    oof_bundle = build_concepts(oof_frames)
    if oof_bundle.names != valid_bundle.names:
        raise RuntimeError("OOF and valid concept schemas differ")
    oof_probability, oof_neutral_logit = apply_sparse_neutral_head(
        oof_bundle, model, neutral_bias
    )
    prediction_frame(
        oof_reference, oof_bundle, oof_probability, oof_neutral_logit
    ).to_csv(output / "train_oof_reverse_domain_predictions.csv", index=False, encoding="utf-8-sig")

    report = {
        "experiment_id": config["experiment_id"],
        "scope": "validation_as_calibration_training_then_locked_no_test_access",
        "architecture": "sparse_additive_neutral_head_with_exact_parent_polar_odds",
        "valid_is_independent_evaluation": False,
        "test_loaded": False,
        "regularization_c": c_value,
        "neutral_logit_bias": neutral_bias,
        "selection_note": config["selection_note"],
        "feature_count": len(valid_bundle.names),
        "selected_feature_count": int(np.count_nonzero(coefficient)),
        "feature_names": valid_bundle.names,
        "valid_calibration_fit": valid_metrics,
        "oof_reverse_domain_audit": metrics(
            oof_reference, oof_bundle, oof_probability
        ),
        "invariants": [
            "negative_positive_odds_equal_frozen_parent",
            "neutral_logit_equals_intercept_plus_sum_of_concept_contributions",
            "test_not_loaded_before_deployment_lock",
        ],
        "artifacts": {
            "model": "sparse_neutral_head.joblib",
            "coefficients": "global_sparse_coefficients.csv",
            "valid_contributions": "valid_concept_contributions.csv",
        },
    }
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    provenance_paths = [
        config_path,
        Path(__file__).resolve(),
        *[Path(value).resolve() for value in valid_sources],
        *[Path(value).resolve() for value in oof_sources],
    ]
    lock = {
        "experiment_id": config["experiment_id"],
        "frozen": True,
        "test_accessed_during_fit": False,
        "model_sha256": sha256(output / "sparse_neutral_head.joblib"),
        "final_metrics_sha256": sha256(output / "final_metrics.json"),
        "provenance_sha256": {str(path): sha256(path) for path in provenance_paths},
    }
    (output / "DEPLOYMENT_LOCK.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
