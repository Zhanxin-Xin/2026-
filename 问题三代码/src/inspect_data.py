from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .data import (
    CLASS_NAMES,
    MODALITIES,
    attach_labels_from_excel,
    find_sibling_label_excel,
    load_feature_source,
    parse_split,
)
from .utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a competition feature pickle")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", help="Used for a directory or direct-field PKL")
    parser.add_argument("--labels", default=None, help="Optional label.xlsx")
    parser.add_argument("--mask-strategy", default="text_shared")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    raw = load_feature_source(args.data, split_name=args.split)
    label_path = args.labels or find_sibling_label_excel(args.data)
    if label_path is not None and not Path(args.data).is_dir():
        raw = attach_labels_from_excel(raw, label_path)
    report: Dict[str, Any] = {"top_level_keys": list(raw.keys()), "splits": {}}
    split_names = [name for name in ("train", "valid", "test") if name in raw]
    if not split_names and all(m in raw for m in MODALITIES):
        raw = {args.split: raw}
        split_names = [args.split]
    for name in split_names:
        arrays = parse_split(raw, name, args.mask_strategy, require_labels=False)
        split_report: Dict[str, Any] = {
            "samples": arrays.size,
            "unique_ids": int(len(np.unique(arrays.ids.astype(str)))),
            "duplicate_ids": int(arrays.size - len(np.unique(arrays.ids.astype(str)))),
            "shapes": {m: list(arrays.features[m].shape) for m in MODALITIES},
            "valid_length": {
                m: {
                    "min": int(arrays.masks[m].sum(1).min()),
                    "median": float(np.median(arrays.masks[m].sum(1))),
                    "max": int(arrays.masks[m].sum(1).max()),
                }
                for m in MODALITIES
            },
        }
        if arrays.class_labels is not None:
            counts = np.bincount(arrays.class_labels, minlength=3)
            split_report["class_counts"] = {
                CLASS_NAMES[i]: int(counts[i]) for i in range(3)
            }
        if arrays.regression_labels is not None:
            split_report["regression"] = {
                "min": float(arrays.regression_labels.min()),
                "mean": float(arrays.regression_labels.mean()),
                "max": float(arrays.regression_labels.max()),
                "std": float(arrays.regression_labels.std()),
            }
        if arrays.class_labels is not None and arrays.regression_labels is not None:
            expected = np.where(
                arrays.regression_labels < 0.0,
                0,
                np.where(arrays.regression_labels > 0.0, 2, 1),
            )
            mismatches = np.flatnonzero(arrays.class_labels != expected)
            split_report["label_consistency"] = {
                "passed": bool(len(mismatches) == 0),
                "mismatches": int(len(mismatches)),
                "example_ids": [str(arrays.ids[index]) for index in mismatches[:5]],
            }
        report["splits"][name] = split_report
    print(report)
    if args.output:
        save_json(report, args.output)


if __name__ == "__main__":
    main()
