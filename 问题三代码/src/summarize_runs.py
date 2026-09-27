from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


METRICS = (
    "accuracy",
    "f1",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "mae",
    "rmse",
    "regression_bias",
    "pearson",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-seed final_metrics.json")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Explicit run directory; may be repeated",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    files = [Path(run) / "final_metrics.json" for run in args.run]
    if args.runs_root:
        files.extend(sorted(Path(args.runs_root).glob("seed_*/final_metrics.json")))
    files = list(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError("No final_metrics.json files were selected")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run metrics: {missing}")
    rows: List[Dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        row: Dict[str, Any] = {
            "run": path.parent.name,
            "best_epoch": report.get("best_epoch"),
            "selection_score": report.get("best_selection_score"),
        }
        for split in ("valid", "test"):
            for metric in METRICS:
                row[f"{split}_{metric}"] = report.get(split, {}).get(metric, np.nan)
        rows.append(row)
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "multi_seed_metrics.csv", index=False, encoding="utf-8-sig")

    summary: Dict[str, Any] = {"runs": len(frame), "metrics": {}}
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if len(values):
            summary["metrics"][column] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
    with (output / "multi_seed_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(frame.to_string(index=False))
    print(f"\nSummary written to {output.resolve()}")


if __name__ == "__main__":
    main()
