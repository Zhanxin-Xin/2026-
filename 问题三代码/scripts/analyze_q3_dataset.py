"""Generate a reproducible, read-only audit of the Problem-3 aligned dataset.

The script never rewrites the pickle or spreadsheet.  It records raw fields,
derived masks, label consistency, split leakage indicators and paper-ready
class/continuous-label tables under ``paper_results/q3/dataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CLASS_NAMES, MODALITIES, load_pickle, parse_split


EXPECTED_DIMS = {"text": 768, "audio": 74, "vision": 35}
EXPECTED_FIELDS = (
    "id",
    "raw_text",
    "text",
    "text_bert",
    "audio",
    "vision",
    "annotations",
    "classification_labels",
    "regression_labels",
    "length",
    "mask",
)
SPLITS = ("train", "valid", "test")


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def raw_field_row(split_name: str, name: str, value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "split": split_name,
        "field": name,
        "python_type": type(value).__name__,
        "shape": str(tuple(array.shape)),
        "dtype": str(array.dtype),
        "length": int(len(array)) if array.ndim else 1,
        "present_in_pickle": True,
    }


def text_mask(split: dict[str, Any], n: int, length: int) -> np.ndarray:
    bert = np.asarray(split["text_bert"])
    if bert.shape == (n, 3, length):
        return bert[:, 1, :].astype(bool)
    if bert.shape == (n, length, 3):
        return bert[:, :, 1].astype(bool)
    raise ValueError(f"Unexpected text_bert shape: {bert.shape}")


def sample_hash(features: dict[str, np.ndarray], masks: dict[str, np.ndarray], i: int) -> str:
    digest = hashlib.sha256()
    for modality in MODALITIES:
        digest.update(np.ascontiguousarray(features[modality][i]).view(np.uint8))
        digest.update(np.ascontiguousarray(masks[modality][i]).view(np.uint8))
    return digest.hexdigest()


def pair_overlaps(values: dict[str, set[str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    names = list(values)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            result[f"{left}__{right}"] = len(values[left] & values[right])
    return result


def build_excel_map(path: Path) -> tuple[pd.DataFrame, dict[str, tuple[float, str]]]:
    table = pd.read_excel(path)
    ids = table["video_id"].astype(str) + "$_$" + table["clip_id"].map(
        lambda value: str(int(value)) if isinstance(value, (float, np.floating)) and float(value).is_integer() else str(value)
    )
    result: dict[str, tuple[float, str]] = {}
    for identifier, score, annotation in zip(ids, table["label"], table["annotation"]):
        if identifier in result:
            raise ValueError(f"Duplicate label-sheet id: {identifier}")
        result[identifier] = (float(score), str(annotation).strip().lower())
    return table, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Problem-3 aligned data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-strategy", default="text_shared")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw = load_pickle(args.data)
    label_table, excel_map = build_excel_map(args.labels)
    field_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    intensity_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "data_path": str(args.data.resolve()),
        "labels_path": str(args.labels.resolve()),
        "mask_strategy": args.mask_strategy,
        "top_level_keys": list(raw),
        "expected_dimensions": EXPECTED_DIMS,
        "label_sheet": {
            "rows": int(len(label_table)),
            "columns": [str(column) for column in label_table.columns],
            "missing_cells": {str(column): int(label_table[column].isna().sum()) for column in label_table.columns},
            "duplicate_ids": int(len(label_table) - len(excel_map)),
        },
        "splits": {},
    }

    id_sets: dict[str, set[str]] = {}
    video_sets: dict[str, set[str]] = {}
    text_sets: dict[str, set[str]] = {}
    hash_sets: dict[str, set[str]] = {}
    all_class_ids: list[str] = []

    for split_name in SPLITS:
        if split_name not in raw:
            continue
        split = raw[split_name]
        for name, value in split.items():
            field_rows.append(raw_field_row(split_name, str(name), value))
        for name in EXPECTED_FIELDS:
            if name not in split:
                field_rows.append(
                    {
                        "split": split_name,
                        "field": name,
                        "python_type": "",
                        "shape": "",
                        "dtype": "",
                        "length": "",
                        "present_in_pickle": False,
                    }
                )

        arrays = parse_split(raw, split_name, args.mask_strategy, require_labels=True)
        n = arrays.size
        raw_mask = text_mask(split, n, 50)
        raw_mask_values = sorted(np.unique(np.asarray(split["text_bert"])[:, 1, :]).tolist())
        prefix_violations = int(((~raw_mask[:, :-1]) & raw_mask[:, 1:]).any(axis=1).sum())
        lengths = raw_mask.sum(axis=1)
        labels = np.asarray(arrays.class_labels, dtype=np.int64)
        scores = np.asarray(arrays.regression_labels, dtype=np.float64)
        expected_labels = np.where(scores < 0.0, 0, np.where(scores > 0.0, 2, 1))
        ids = np.asarray(arrays.ids, dtype=str)
        raw_texts = np.asarray(arrays.raw_text, dtype=str)
        unique_ids = set(ids.tolist())
        videos = {identifier.split("$_$", 1)[0] for identifier in ids}
        nonempty_texts = {value.strip() for value in raw_texts if value.strip()}
        hashes = [sample_hash(arrays.features, arrays.masks, index) for index in range(n)]

        excel_missing = 0
        excel_score_mismatch = 0
        excel_class_mismatch = 0
        for identifier, score, class_id in zip(ids, scores, labels):
            if identifier not in excel_map:
                excel_missing += 1
                continue
            sheet_score, annotation = excel_map[identifier]
            excel_score_mismatch += int(not np.isclose(score, sheet_score, atol=1e-6))
            annotation_id = {"negative": 0, "neutral": 1, "positive": 2}.get(annotation)
            excel_class_mismatch += int(annotation_id is None or annotation_id != class_id)

        feature_checks: dict[str, Any] = {}
        for modality in MODALITIES:
            values = np.asarray(split[modality])
            expected_shape = (n, 50, EXPECTED_DIMS[modality])
            finite = np.isfinite(values)
            padding = ~raw_mask
            padding_nonzero = (np.abs(values) > 1e-12) & padding[..., None]
            valid_nonzero = (np.abs(values) > 1e-12) & raw_mask[..., None]
            all_zero = np.all(np.abs(values) <= 1e-12, axis=(1, 2))
            valid_all_zero = ~valid_nonzero.any(axis=(1, 2))
            padding_samples = padding_nonzero.any(axis=(1, 2))
            feature_checks[modality] = {
                "actual_shape": list(values.shape),
                "expected_shape": list(expected_shape),
                "shape_ok": bool(values.shape == expected_shape),
                "nan_count": int(np.isnan(values).sum()),
                "inf_count": int(np.isinf(values).sum()),
                "finite_fraction": float(finite.mean()),
                "all_zero_samples": int(all_zero.sum()),
                "valid_region_all_zero_samples": int(valid_all_zero.sum()),
                "samples_with_nonzero_padding": int(padding_samples.sum()),
                "max_abs_padding_value": float(np.abs(values[padding]).max()) if padding.any() else 0.0,
            }

        split_report = {
            "samples": n,
            "raw_fields": [str(key) for key in split],
            "missing_expected_raw_fields": [field for field in EXPECTED_FIELDS if field not in split],
            "derived_fields": ["text_mask", "audio_mask", "vision_mask", "length_from_text_bert_attention_mask"],
            "unique_sample_ids": len(unique_ids),
            "duplicate_sample_ids": n - len(unique_ids),
            "unique_video_ids": len(videos),
            "empty_raw_texts": int(np.sum(np.char.strip(raw_texts) == "")),
            "duplicate_nonempty_raw_text_rows": int(len([value for value in raw_texts if value.strip()]) - len(nonempty_texts)),
            "unique_exact_feature_hashes": len(set(hashes)),
            "duplicate_exact_feature_rows": n - len(set(hashes)),
            "mask": {
                "source": "text_bert attention-mask channel",
                "raw_unique_values": raw_mask_values,
                "zero_length_samples": int((lengths == 0).sum()),
                "prefix_contiguity_violations": prefix_violations,
                "length_min": int(lengths.min()),
                "length_q1": float(np.quantile(lengths, 0.25)),
                "length_median": float(np.median(lengths)),
                "length_q3": float(np.quantile(lengths, 0.75)),
                "length_max": int(lengths.max()),
            },
            "labels": {
                "class_unique_values": sorted(np.unique(labels).tolist()),
                "class_counts": {CLASS_NAMES[index]: int((labels == index).sum()) for index in range(3)},
                "classification_sign_mismatches": int((labels != expected_labels).sum()),
                "regression_nan_count": int(np.isnan(scores).sum()),
                "regression_inf_count": int(np.isinf(scores).sum()),
                "regression_out_of_range_count": int(((scores < -3.0) | (scores > 3.0)).sum()),
                "exact_zero_count": int((scores == 0.0).sum()),
                "excel_missing_ids": excel_missing,
                "excel_regression_mismatches": excel_score_mismatch,
                "excel_classification_mismatches": excel_class_mismatch,
            },
            "features": feature_checks,
        }
        report["splits"][split_name] = split_report
        id_sets[split_name] = unique_ids
        video_sets[split_name] = videos
        text_sets[split_name] = nonempty_texts
        hash_sets[split_name] = set(hashes)
        all_class_ids.extend(ids.tolist())

        class_rows.append(
            {
                "split": split_name,
                "Negative": int((labels == 0).sum()),
                "Neutral": int((labels == 1).sum()),
                "Positive": int((labels == 2).sum()),
                "Total": n,
            }
        )
        intensity_rows.append(
            {
                "split": split_name,
                "Mean": float(scores.mean()),
                "Std": float(scores.std(ddof=1)),
                "Min": float(scores.min()),
                "Q1": float(np.quantile(scores, 0.25)),
                "Median": float(np.median(scores)),
                "Q3": float(np.quantile(scores, 0.75)),
                "Max": float(scores.max()),
            }
        )
        checks = {
            "duplicate_sample_ids": split_report["duplicate_sample_ids"],
            "label_sign_mismatches": split_report["labels"]["classification_sign_mismatches"],
            "label_sheet_regression_mismatches": excel_score_mismatch,
            "label_sheet_classification_mismatches": excel_class_mismatch,
            "mask_prefix_violations": prefix_violations,
            "zero_length_samples": split_report["mask"]["zero_length_samples"],
        }
        for name, value in checks.items():
            integrity_rows.append({"split": split_name, "check": name, "count": int(value), "pass": int(value) == 0})

    report["cross_split_overlap"] = {
        "sample_ids": pair_overlaps(id_sets),
        "video_ids": pair_overlaps(video_sets),
        "nonempty_raw_texts": pair_overlaps(text_sets),
        "exact_feature_hashes": pair_overlaps(hash_sets),
    }
    report["all_split_sample_id_duplicates"] = len(all_class_ids) - len(set(all_class_ids))

    pd.DataFrame(field_rows).sort_values(["split", "field"]).to_csv(
        args.output / "field_inventory.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(class_rows).to_csv(
        args.output / "class_distribution.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(intensity_rows).to_csv(
        args.output / "dataset_statistics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(integrity_rows).to_csv(
        args.output / "integrity_checks.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "dataset_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_value), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve()), "splits": class_rows, "cross_split_overlap": report["cross_split_overlap"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
