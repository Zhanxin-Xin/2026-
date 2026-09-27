"""Evaluate the already locked EXP186 sparse Neutral head on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .build_cross_fitted_ensemble import load_aligned
from .train_explainable_hierarchical_nam import build_concepts
from .train_validation_sparse_neutral_refit import (
    apply_sparse_neutral_head,
    contribution_frame,
    metrics,
    prediction_frame,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen EXP186 on test")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    deployment = Path(args.deployment).resolve()
    lock = json.loads((deployment / "DEPLOYMENT_LOCK.json").read_text("utf-8"))
    if not lock.get("frozen") or lock.get("test_accessed_during_fit"):
        raise RuntimeError("Deployment is not a valid pre-test lock")
    model_path = deployment / "sparse_neutral_head.joblib"
    if sha256(model_path) != lock["model_sha256"]:
        raise RuntimeError("Frozen model hash mismatch")
    report = json.loads((deployment / "final_metrics.json").read_text("utf-8"))
    manifest = json.loads(Path(args.test_manifest).read_text("utf-8"))
    test_sources = [str(value) for value in manifest["test_sources"]]
    frames = load_aligned(test_sources, require_fold=False)
    reference = frames[0]
    bundle = build_concepts(frames)
    if bundle.names != report["feature_names"]:
        raise RuntimeError("Frozen model/test concept schema mismatch")
    model = joblib.load(model_path)
    probability, neutral_logit = apply_sparse_neutral_head(
        bundle, model, float(report["neutral_logit_bias"])
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame(reference, bundle, probability, neutral_logit).to_csv(
        output / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    contribution_frame(
        reference, bundle, model, float(report["neutral_logit_bias"])
    ).to_csv(output / "test_concept_contributions.csv", index=False, encoding="utf-8-sig")
    result = {
        "scope": "locked_validation_refit_descriptive_test_after_prior_test_disclosure",
        "independent_holdout_claim": False,
        "deployment": str(deployment),
        "deployment_model_sha256": lock["model_sha256"],
        "selection_used_this_test_result": False,
        "test_sources": test_sources,
        "test": metrics(reference, bundle, probability),
        "limitation": (
            "The EXP186 model and threshold were locked before this command, but "
            "EXP183 had already disclosed labels from the same test set. This "
            "number is descriptive and cannot restore independent holdout status."
        ),
    }
    (output / "final_test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
