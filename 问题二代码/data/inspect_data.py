#!/usr/bin/env python3
"""Inspect a multimodal pickle dataset without changing or reshaping its data."""

from __future__ import annotations

import argparse
import collections
import pickle
import re
import sys
from pathlib import Path
from typing import Any

# Permit a project-local dependency directory in minimal competition environments.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / ".python_deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np


SPLITS = ("train", "valid", "test")
MODALITIES = ("text", "audio", "vision")
LENGTH_PATTERN = re.compile(r"(length|lengths|len|mask|valid|attention)", re.I)


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(shape)
        except TypeError:
            return None
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None


def dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return str(dtype)
    if isinstance(value, (list, tuple)) and value:
        return f"sequence[{type(value[0]).__name__}]"
    return "n/a"


def preview(value: Any, limit: int = 3, max_chars: int = 600) -> str:
    try:
        if isinstance(value, np.ndarray):
            item = value[:limit].tolist() if value.ndim else value.item()
        elif isinstance(value, (list, tuple)):
            item = value[:limit]
        elif isinstance(value, dict):
            item = list(value.items())[:limit]
        else:
            item = value
        result = repr(item)
    except Exception as exc:  # inspection must continue for unusual objects
        result = f"<preview failed: {exc}>"
    return result if len(result) <= max_chars else result[:max_chars] + "..."


def numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return value
    return None


def finite_summary(name: str, value: Any, out: Reporter) -> None:
    arr = numeric_array(value)
    if arr is None:
        return
    if np.issubdtype(arr.dtype, np.inexact):
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
    else:
        nan_count = inf_count = 0
    out.write(f"    {name}: NaN={nan_count}, Inf={inf_count}")


def label_summary(name: str, value: Any, out: Reporter) -> None:
    out.write(f"  {name} format: type={type(value).__name__}, shape={shape_of(value)}, "
              f"dtype={dtype_of(value)}, preview={preview(value)}")
    arr = numeric_array(value)
    if arr is None or arr.size == 0:
        out.write("    Numeric statistics unavailable (not a non-empty numeric ndarray).")
        return
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.inexact) else arr.reshape(-1)
    if finite.size == 0:
        out.write("    No finite values.")
        return
    out.write(f"    range=[{finite.min()!r}, {finite.max()!r}]")
    if name == "classification_labels":
        values, counts = np.unique(finite, return_counts=True)
        if len(values) <= 100:
            distribution = {str(v.item() if hasattr(v, "item") else v): int(c)
                            for v, c in zip(values, counts)}
            out.write(f"    value distribution={distribution}")
        else:
            out.write(f"    unique values={len(values)} (distribution omitted: >100 values)")
    else:
        out.write(f"    mean={float(np.mean(finite)):.8g}")


def zero_row_summary(name: str, value: Any, out: Reporter) -> np.ndarray | None:
    arr = numeric_array(value)
    if arr is None or arr.ndim < 2 or arr.size == 0:
        out.write(f"  {name}: all-zero-row analysis unavailable")
        return None
    zero_rows = np.all(arr == 0, axis=-1)
    total = int(zero_rows.sum())
    out.write(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, all-zero feature rows="
              f"{total}/{zero_rows.size} ({100 * total / zero_rows.size:.4f}%)")
    if zero_rows.ndim != 2:
        out.write(f"    Resulting zero mask shape={zero_rows.shape}; padding inference requires [N,T,D].")
        return zero_rows

    counts = zero_rows.sum(axis=0).astype(int).tolist()
    out.write(f"    all-zero count at each time position={counts}")
    has_zero = zero_rows.any(axis=1)
    all_zero_samples = zero_rows.all(axis=1)
    boundary_only = 0
    interior = 0
    leading = 0
    inferred_lengths: list[int] = []
    for row in zero_rows:
        nonzero_positions = np.flatnonzero(~row)
        if nonzero_positions.size == 0:
            inferred_lengths.append(0)
            continue
        first = int(nonzero_positions[0])
        last = int(nonzero_positions[-1])
        inferred_lengths.append(last + 1)
        row_has_leading = bool(row[:first].any())
        row_has_interior = bool(row[first:last + 1].any())
        leading += int(row_has_leading)
        interior += int(row_has_interior)
        boundary_only += int(row.any() and not row_has_interior)
    length_counts = dict(sorted(collections.Counter(inferred_lengths).items()))
    out.write(f"    samples with zero rows={int(has_zero.sum())}/{len(zero_rows)}; "
              f"fully-zero samples={int(all_zero_samples.sum())}; boundary-only={boundary_only}; "
              f"leading-zero={leading}; interior-zero={interior}")
    out.write(f"    inferred last-nonzero length distribution={length_counts}")
    if total == 0:
        verdict = "No all-zero feature rows; zero padding is not visible."
    elif interior == 0:
        verdict = ("Every non-fully-zero sample has zeros only outside its first-to-last active "
                   "interval, consistent with boundary padding (left, right, or both).")
    else:
        verdict = ("Zero rows occur inside the active interval; boundary padding alone cannot explain "
                   "them, so genuine missing/corrupt positions are plausible.")
    if int(all_zero_samples.sum()):
        verdict += (f" Additionally, {int(all_zero_samples.sum())} fully-zero samples are not ordinary "
                    "partial sequence padding and require an explicit missing-modality policy.")
    out.write(f"    assessment: {verdict}")
    return zero_rows


def sample_count(split: dict[str, Any]) -> tuple[int | None, dict[str, int]]:
    sizes: dict[str, int] = {}
    for key, value in split.items():
        shape = shape_of(value)
        if shape:
            sizes[str(key)] = int(shape[0])
    if not sizes:
        return None, sizes
    counts = collections.Counter(sizes.values())
    return counts.most_common(1)[0][0], sizes


def inspect_split(split_name: str, split: Any, out: Reporter) -> None:
    out.write(f"\n=== split: {split_name} ===")
    if not isinstance(split, dict):
        out.write(f"Actual split type is {type(split).__name__}, not dict; preview={preview(split)}")
        return
    out.write(f"keys={list(split.keys())}")
    n, sizes = sample_count(split)
    out.write(f"sample count (modal first-dimension)={n}")
    if n is not None:
        mismatched = {k: v for k, v in sizes.items() if v != n}
        out.write(f"fields whose first dimension differs from {n}={mismatched or 'none'}")
    out.write("field inventory:")
    for key, value in split.items():
        out.write(f"  - {key}: type={type(value).__name__}, shape={shape_of(value)}, "
                  f"dtype={dtype_of(value)}")

    if "id" in split:
        out.write(f"first 3 id={preview(split['id'])}")
    else:
        out.write("first 3 id=UNAVAILABLE (no 'id' field)")

    for key in ("classification_labels", "regression_labels"):
        if key in split:
            label_summary(key, split[key], out)
        else:
            out.write(f"  {key}: MISSING")

    if "annotations" in split:
        value = split["annotations"]
        out.write(f"  annotations format: type={type(value).__name__}, shape={shape_of(value)}, "
                  f"dtype={dtype_of(value)}, preview={preview(value)}")
    else:
        out.write("  annotations: MISSING")

    candidates = [str(key) for key in split if LENGTH_PATTERN.search(str(key))]
    out.write(f"length/mask-like fields={candidates or 'none'}")
    for key in candidates:
        out.write(f"  {key}: type={type(split[key]).__name__}, shape={shape_of(split[key])}, "
                  f"dtype={dtype_of(split[key])}, preview={preview(split[key])}")

    out.write("NaN/Inf checks (all numeric ndarray fields):")
    for key, value in split.items():
        finite_summary(str(key), value, out)

    out.write("all-zero feature-row checks:")
    masks: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        if modality in split:
            mask = zero_row_summary(modality, split[modality], out)
            if mask is not None:
                masks[modality] = mask
        else:
            out.write(f"  {modality}: MISSING")
    for i, left in enumerate(MODALITIES):
        for right in MODALITIES[i + 1:]:
            if left in masks and right in masks and masks[left].shape == masks[right].shape:
                different = int(np.logical_xor(masks[left], masks[right]).sum())
                out.write(f"  zero-mask mismatch {left} vs {right}: {different}/{masks[left].size}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True,
                        help="Path to aligned_50.pkl or another pickle dataset")
    parser.add_argument("--report_path", type=Path,
                        default=Path("reports/data_inspection.txt"),
                        help="UTF-8 text report output path")
    args = parser.parse_args()

    out = Reporter()
    path = args.data_path.expanduser().resolve()
    out.write("CMU-MOSEI pickle data inspection")
    out.write(f"data path={path}")
    out.write(f"file size={path.stat().st_size} bytes")
    out.write("Loading pickle read-only; no data will be reshaped or rewritten...")
    with path.open("rb") as handle:
        data = pickle.load(handle)

    out.write(f"outer type={type(data).__name__}")
    if isinstance(data, dict):
        out.write(f"outer keys={list(data.keys())}")
    else:
        out.write(f"outer keys=UNAVAILABLE; preview={preview(data)}")
        out.save(args.report_path)
        return

    for split_name in SPLITS:
        if split_name in data:
            inspect_split(split_name, data[split_name], out)
        else:
            out.write(f"\n=== split: {split_name} MISSING ===")
    extra = [key for key in data if key not in SPLITS]
    out.write(f"\nextra outer keys={extra or 'none'}")
    out.write(f"report path={args.report_path.resolve()}")
    out.save(args.report_path)


if __name__ == "__main__":
    main()
