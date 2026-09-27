"""Descriptively evaluate a class-bias calibrator frozen before test access.

This utility does not fit or search any value.  It reconstructs the fixed
vote/probability parent on the supplied test sources and applies the exact
validation-fitted bias stored in ``calibration.json``.  Because this project's
test labels were already viewed by EXP183, the output is explicitly marked as
descriptive and must not be presented as a new independent holdout result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_cross_fitted_ensemble import load_aligned
from .build_vote_probability_consensus import consensus
from .calibrate_class_bias import _evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a historically frozen validation calibrator on test"
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    calibration_dir = Path(args.calibration)
    calibration = json.loads(
        (calibration_dir / "calibration.json").read_text(encoding="utf-8")
    )
    manifest_path = Path(args.test_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_sources = [str(value) for value in manifest["test_sources"]]
    parent_metrics, parent_frame = consensus(
        load_aligned(test_sources, require_fold=False), include_fold=False
    )
    bias = calibration["bias_negative_neutral_positive"]
    polarity_scale = float(calibration.get("polarity_scale", 0.0))

    import numpy as np

    calibrated_metrics, predictions = _evaluate(
        parent_frame, np.asarray(bias, dtype=np.float64), polarity_scale
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        output / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "scope": "historically_frozen_before_test_descriptive_recheck_after_test_disclosure",
        "independent_holdout_claim": False,
        "selection_used_this_test_result": False,
        "calibration": str(calibration_dir),
        "frozen_method": calibration["method"],
        "frozen_bias_negative_neutral_positive": bias,
        "frozen_polarity_scale": polarity_scale,
        "test_sources": test_sources,
        "test_parent": parent_metrics,
        "test": calibrated_metrics,
        "limitation": (
            "EXP178 was frozen using validation before EXP183 opened test, but the "
            "same test labels are now known; this is a descriptive historical "
            "recheck, not an independent model-selection result."
        ),
    }
    (output / "final_test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
