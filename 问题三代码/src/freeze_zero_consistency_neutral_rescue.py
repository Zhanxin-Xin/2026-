"""Freeze EXP187 without accepting or opening test inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .build_cross_fitted_ensemble import load_aligned
from .data import CLASS_NAMES
from .metrics import compute_metrics
from .train_explainable_hierarchical_nam import ConceptBundle, build_concepts
from .train_validation_sparse_neutral_refit import sha256


LABEL_MAP = {name: index for index, name in enumerate(CLASS_NAMES)}
LABELS = np.asarray(CLASS_NAMES)


def apply_rescue(
    bundle: ConceptBundle, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    probability = bundle.parent_probability.copy()
    probability[:, 1] *= np.exp(float(config["neutral_bias"]))
    probability /= probability.sum(axis=1, keepdims=True)
    polar_maximum = np.maximum(probability[:, 0], probability[:, 2])
    deficit = np.log(polar_maximum.clip(min=1e-12)) - np.log(
        probability[:, 1].clip(min=1e-12)
    )
    vote_index = bundle.names.index("consensus_vote_neutral")
    neutral_votes = np.rint(bundle.values[:, vote_index] * 7.0).astype(np.int64)
    rescued = (
        (probability.argmax(axis=1) != 1)
        & (np.abs(bundle.regression) <= float(config["maximum_absolute_intensity"]))
        & (neutral_votes >= int(config["minimum_neutral_votes"]))
        & (deficit <= float(config["maximum_neutral_logit_deficit"]))
    )
    probability[rescued, 1] = polar_maximum[rescued] * (1.0 + 1e-7)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, {
        "neutral_votes": neutral_votes,
        "absolute_intensity": np.abs(bundle.regression),
        "neutral_logit_deficit": deficit,
        "zero_consistency_rescued": rescued,
    }


def split_metrics(
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
    diagnostics: dict[str, np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": LABELS[probability.argmax(axis=1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": bundle.regression,
            **diagnostics,
            "true_label": reference["true_label"].astype(str),
            "true_intensity": reference["true_intensity"].to_numpy(np.float64),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze EXP187 before test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text("utf-8"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "experiment_id": config["experiment_id"],
        "scope": "oof_and_valid_frozen_zero_consistency_rule_no_test_access",
        "architecture": "zero_consistency_neutral_rescue_with_exact_parent_polar_odds",
        "parameters": {
            name: config[name]
            for name in (
                "neutral_bias",
                "maximum_absolute_intensity",
                "minimum_neutral_votes",
                "maximum_neutral_logit_deficit",
            )
        },
        "selection_note": config["selection_note"],
        "test_loaded": False,
        "invariants": [
            "negative_positive_odds_equal_frozen_parent",
            "rescue_requires_probability_vote_and_regression_consistency",
            "test_not_loaded_before_deployment_lock",
        ],
    }
    for split, paths, require_fold in (
        ("oof", config["oof_sources"], True),
        ("valid", config["valid_sources"], False),
    ):
        frames = load_aligned([str(Path(value)) for value in paths], require_fold)
        reference = frames[0]
        bundle = build_concepts(frames)
        probability, diagnostics = apply_rescue(bundle, config)
        prediction_frame(
            reference, bundle, probability, diagnostics
        ).to_csv(output / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
        report[split] = split_metrics(reference, bundle, probability)
        report[f"{split}_rescued_count"] = int(
            diagnostics["zero_consistency_rescued"].sum()
        )
    (output / "resolved_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lock = {
        "experiment_id": config["experiment_id"],
        "frozen": True,
        "test_accessed_during_freeze": False,
        "config_sha256": sha256(config_path),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "final_metrics_sha256": sha256(output / "final_metrics.json"),
    }
    (output / "DEPLOYMENT_LOCK.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
