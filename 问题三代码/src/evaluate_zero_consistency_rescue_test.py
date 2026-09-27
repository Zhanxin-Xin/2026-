"""Descriptively evaluate the locked EXP187 rule on the disclosed test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .build_cross_fitted_ensemble import load_aligned
from .freeze_zero_consistency_neutral_rescue import (
    apply_rescue,
    prediction_frame,
    split_metrics,
)
from .train_explainable_hierarchical_nam import build_concepts
from .train_validation_sparse_neutral_refit import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen EXP187")
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deployment = Path(args.deployment).resolve()
    lock = json.loads((deployment / "DEPLOYMENT_LOCK.json").read_text("utf-8"))
    if not lock.get("frozen") or lock.get("test_accessed_during_freeze"):
        raise RuntimeError("EXP187 was not locked before test evaluation")
    config_path = Path("configs/exp187_zero_consistency_neutral_rescue.yaml").resolve()
    if sha256(config_path) != lock["config_sha256"]:
        raise RuntimeError("Frozen EXP187 config hash mismatch")
    config = yaml.safe_load(config_path.read_text("utf-8"))
    manifest = json.loads(Path(args.test_manifest).read_text("utf-8"))
    test_sources = [str(value) for value in manifest["test_sources"]]
    frames = load_aligned(test_sources, require_fold=False)
    reference = frames[0]
    bundle = build_concepts(frames)
    probability, diagnostics = apply_rescue(bundle, config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame(reference, bundle, probability, diagnostics).to_csv(
        output / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "scope": "locked_rule_descriptive_test_after_prior_test_disclosure",
        "independent_holdout_claim": False,
        "selection_used_this_test_result": False,
        "deployment": str(deployment),
        "test_sources": test_sources,
        "test_rescued_count": int(diagnostics["zero_consistency_rescued"].sum()),
        "test": split_metrics(reference, bundle, probability),
        "limitation": (
            "EXP187 parameters were locked from validation before this command, "
            "but EXP183 had already disclosed this test set. The result is "
            "descriptive and not an independent holdout claim."
        ),
    }
    (output / "final_test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
