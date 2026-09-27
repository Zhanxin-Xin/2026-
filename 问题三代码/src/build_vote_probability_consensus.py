"""Build a fixed vote/probability consensus without fitted ensemble weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .metrics import compute_metrics


LABELS = np.asarray(["Negative", "Neutral", "Positive"])


def consensus(
    frames: list[pd.DataFrame], include_fold: bool
) -> tuple[dict[str, object], pd.DataFrame]:
    probabilities = np.stack(
        [frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64) for frame in frames],
        axis=1,
    )
    probability_channel = probabilities.mean(axis=1)
    votes = np.eye(3, dtype=np.float64)[probabilities.argmax(axis=-1)].mean(axis=1)
    combined = 0.5 * probability_channel + 0.5 * votes
    combined /= combined.sum(axis=1, keepdims=True)
    regression = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in frames],
        axis=1,
    ).mean(axis=1)
    reference = frames[0]
    true_label = reference["true_label"].map(
        {"Negative": 0, "Neutral": 1, "Positive": 2}
    ).to_numpy(np.int64)
    true_intensity = reference["true_intensity"].to_numpy(np.float64)
    metrics = compute_metrics(true_label, combined, true_intensity, regression)
    output = pd.DataFrame(
        {
            "id": reference["id"].astype(str),
            "predicted_label": LABELS[combined.argmax(axis=1)],
            "negative_probability": combined[:, 0],
            "neutral_probability": combined[:, 1],
            "positive_probability": combined[:, 2],
            "predicted_intensity": regression,
            "true_label": reference["true_label"].astype(str),
            "true_intensity": true_intensity,
        }
    )
    if include_fold:
        output.insert(1, "fold", reference["fold"].to_numpy(np.int64))
    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed equal consensus of expert votes and probabilities"
    )
    parser.add_argument("--oof", action="append", required=True)
    parser.add_argument("--valid", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--oof-only", action="store_true")
    parser.add_argument("--allow-independent-folds", action="store_true")
    args = parser.parse_args()
    if len(args.oof) < 2:
        raise ValueError("Need OOF sources for at least two experts")
    if not args.oof_only and len(args.oof) != len(args.valid):
        raise ValueError("Need matching OOF/valid sources")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    oof_metrics, oof = consensus(
        load_aligned(
            args.oof,
            require_fold=True,
            require_fold_alignment=not args.allow_independent_folds,
        ),
        True,
    )
    oof.to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    report = {
        "scope": (
            "fixed_oof_consensus_only_no_valid_or_test_access"
            if args.oof_only
            else "fixed_oof_consensus_then_locked_validation_no_test_access"
        ),
        "method": "equal_vote_and_probability_channels",
        "channel_weights": {"vote": 0.5, "probability": 0.5},
        "oof_sources": args.oof,
        "valid_sources": args.valid,
        "source_count": len(args.oof),
        "fold_alignment_verified": not args.allow_independent_folds,
        "independently_cross_fitted_sources": bool(args.allow_independent_folds),
        "oof": oof_metrics,
    }
    if args.oof_only:
        report["valid_sources"] = None
        report["valid"] = None
    else:
        # The caller must establish the OOF gate before invoking this path;
        # validation is never materialized during candidate screening.
        valid_metrics, valid = consensus(
            load_aligned(args.valid, require_fold=False), False
        )
        valid.to_csv(
            output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report["valid"] = valid_metrics
    (output / "final_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
