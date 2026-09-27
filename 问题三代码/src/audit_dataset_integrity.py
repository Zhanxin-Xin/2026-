from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from .data import (
    MODALITIES,
    attach_labels_from_excel,
    load_pickle,
    parse_split,
)
from .utils import save_json


SPLITS = ("train", "valid", "test")


def _pairs(values: Dict[str, set[str]]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    names = list(values)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result[f"{left}__{right}"] = len(values[left] & values[right])
    return result


def _video_id(sample_id: str) -> str:
    return sample_id.split("$_$", 1)[0]


def _sample_hash(features: Iterable[np.ndarray], masks: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array, mask in zip(features, masks):
        contiguous = np.ascontiguousarray(array)
        digest.update(memoryview(contiguous).cast("B"))
        digest.update(memoryview(np.ascontiguousarray(mask)).cast("B"))
    return digest.hexdigest()


def _feature_summary(array: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    valid = array[mask].reshape(-1)
    if valid.size == 0:
        raise ValueError("Cannot summarize an empty modality")
    # Quantiles over at most one million evenly spaced values avoid a second
    # very large allocation while remaining deterministic and representative.
    stride = max(1, valid.size // 1_000_000)
    sampled = valid[::stride][:1_000_000].astype(np.float64, copy=False)
    return {
        "valid_frames": int(mask.sum()),
        "scalar_values": int(valid.size),
        "mean": float(sampled.mean()),
        "std": float(sampled.std()),
        "mean_abs": float(np.abs(sampled).mean()),
        "min": float(sampled.min()),
        "q01": float(np.quantile(sampled, 0.01)),
        "median": float(np.median(sampled)),
        "q99": float(np.quantile(sampled, 0.99)),
        "max": float(sampled.max()),
        "zero_fraction": float(np.mean(sampled == 0.0)),
        "nan_count": int(np.isnan(valid).sum()),
        "inf_count": int(np.isinf(valid).sum()),
    }


def build_report(data_path: Path, labels_path: Path, mask_strategy: str) -> Dict[str, Any]:
    raw = attach_labels_from_excel(load_pickle(data_path), labels_path)
    arrays = {
        name: parse_split(raw, name, mask_strategy, require_labels=True)
        for name in SPLITS
        if name in raw
    }

    id_sets = {name: set(map(str, split.ids)) for name, split in arrays.items()}
    video_sets = {
        name: {_video_id(str(value)) for value in split.ids}
        for name, split in arrays.items()
    }
    text_sets = {
        name: {
            str(value).strip()
            for value in split.raw_text
            if str(value).strip()
        }
        for name, split in arrays.items()
    }
    feature_hashes: Dict[str, set[str]] = {}
    for name, split in arrays.items():
        feature_hashes[name] = {
            _sample_hash(
                (split.features[modality][index] for modality in MODALITIES),
                (split.masks[modality][index] for modality in MODALITIES),
            )
            for index in range(split.size)
        }

    label_table = pd.read_excel(labels_path)
    split_reports: Dict[str, Any] = {}
    for name, split in arrays.items():
        split_reports[name] = {
            "samples": split.size,
            "unique_sample_ids": len(id_sets[name]),
            "unique_video_ids": len(video_sets[name]),
            "unique_nonempty_raw_texts": len(text_sets[name]),
            "unique_exact_feature_hashes": len(feature_hashes[name]),
            "class_counts": np.bincount(split.class_labels, minlength=3).tolist(),
            "feature_scale": {
                modality: _feature_summary(
                    split.features[modality], split.masks[modality]
                )
                for modality in MODALITIES
            },
        }

    return {
        "data_path": str(data_path.resolve()),
        "labels_path": str(labels_path.resolve()),
        "mask_strategy": mask_strategy,
        "label_table": {
            "rows": int(len(label_table)),
            "columns": [str(column) for column in label_table.columns],
            "missing_cells_by_column": {
                str(column): int(label_table[column].isna().sum())
                for column in label_table.columns
            },
        },
        "cross_split_overlap": {
            "sample_ids": _pairs(id_sets),
            "video_ids": _pairs(video_sets),
            "nonempty_raw_texts": _pairs(text_sets),
            "exact_feature_hashes": _pairs(feature_hashes),
        },
        "splits": split_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit multimodal split integrity")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--mask-strategy", default="text_shared")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = build_report(args.data, args.labels, args.mask_strategy)
    save_json(report, args.output)
    print(f"Dataset integrity report written to {args.output}")


if __name__ == "__main__":
    main()
