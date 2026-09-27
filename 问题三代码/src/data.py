from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("text", "audio", "vision")
CLASS_NAMES = ("Negative", "Neutral", "Positive")
PAIR_NAMES = ("text_audio", "text_vision", "audio_vision")
PAIR_MODALITY_INDICES = ((0, 1), (0, 2), (1, 2))


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _as_feature_array(value: Any, field: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Field '{field}' must be [N,L,D], got {arr.shape}")
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"Field '{field}' contains {bad} NaN/Inf values")
    return arr


def _parse_class_labels(split: Mapping[str, Any]) -> Optional[np.ndarray]:
    source = None
    for key in ("classification_labels", "classification_label", "annotations", "annotation"):
        if key in split:
            source = np.asarray(split[key], dtype=object)
            break
    if source is None:
        return None

    if source.ndim == 2 and source.shape[1] == 3:
        try:
            numeric_matrix = np.asarray(source, dtype=np.float64)
            return numeric_matrix.argmax(axis=1).astype(np.int64)
        except (TypeError, ValueError):
            pass
    source = source.reshape(-1)

    string_map = {"negative": 0, "neutral": 1, "positive": 2, "neg": 0, "neu": 1, "pos": 2}
    if any(isinstance(x, (str, bytes)) for x in source):
        parsed = []
        for item in source:
            key = _decode(item).strip().lower()
            if key not in string_map:
                raise ValueError(f"Unknown classification label: {item!r}")
            parsed.append(string_map[key])
        return np.asarray(parsed, dtype=np.int64)

    numeric = np.asarray(source, dtype=np.float64)
    unique = set(np.unique(numeric).tolist())
    if unique.issubset({-1.0, 0.0, 1.0}):
        return (numeric.astype(np.int64) + 1)
    if unique.issubset({0.0, 1.0, 2.0}):
        return numeric.astype(np.int64)
    raise ValueError(f"Unsupported numeric classification labels: {sorted(unique)}")


def _parse_regression_labels(split: Mapping[str, Any]) -> Optional[np.ndarray]:
    for key in ("regression_labels", "regression_label", "label", "labels"):
        if key not in split:
            continue
        try:
            arr = np.asarray(split[key], dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if np.isfinite(arr).all() and np.all((-3.0001 <= arr) & (arr <= 3.0001)):
            return arr
    return None


def _mask_from_lengths(lengths: Any, n: int, length: int) -> np.ndarray:
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if len(lengths) != n:
        raise ValueError(f"Length field has {len(lengths)} rows, expected {n}")
    lengths = np.clip(lengths, 0, length)
    return np.arange(length)[None, :] < lengths[:, None]


def _text_bert_attention_mask(split: Mapping[str, Any], n: int, length: int) -> Optional[np.ndarray]:
    if "text_bert" not in split:
        return None
    bert = np.asarray(split["text_bert"])
    if bert.ndim != 3 or bert.shape[0] != n:
        return None
    # Competition description uses [N,3,50], but accept [N,50,3].
    if bert.shape[1] == 3 and bert.shape[2] == length:
        mask = bert[:, 1, :]
    elif bert.shape[2] == 3 and bert.shape[1] == length:
        mask = bert[:, :, 1]
    else:
        return None
    return mask.astype(bool)


def _feature_mask(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.linalg.norm(features, axis=-1) > eps


@dataclass
class SplitArrays:
    features: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    ids: np.ndarray
    raw_text: np.ndarray
    class_labels: Optional[np.ndarray]
    regression_labels: Optional[np.ndarray]

    @property
    def size(self) -> int:
        return len(self.ids)


def load_pickle(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("The top-level pickle object must be a dictionary")
    return data


def _natural_path_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def _find_feature_mapping(obj: Any, split_name: str) -> Mapping[str, Any]:
    """Find a mapping containing text/audio/vision inside a per-sample pickle."""
    queue = [obj]
    visited = set()
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, Mapping):
            if all(modality in candidate for modality in MODALITIES):
                return candidate
            if split_name in candidate:
                queue.insert(0, candidate[split_name])
            for value in candidate.values():
                if isinstance(value, Mapping):
                    queue.append(value)
    raise ValueError(
        "Could not find a dictionary containing text/audio/vision in a sample pickle"
    )


def _ensure_batched_feature(value: Any, field: str, source: Path) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"{source.name}: field '{field}' must be [L,D] or [N,L,D], got {arr.shape}")
    return arr


def _ensure_batched_optional(value: Any, n: int, field: str, source: Path) -> np.ndarray:
    arr = np.asarray(value)
    if field == "text_bert" and arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), n)
    elif arr.shape[0] != n and n == 1:
        arr = arr[None, ...]
    if arr.shape[0] != n:
        raise ValueError(f"{source.name}: field '{field}' has {arr.shape[0]} rows, expected {n}")
    return arr


def load_pickle_directory(path: str | Path, split_name: str = "test") -> Dict[str, Any]:
    """Merge competition attachment-4 per-sample pickle files into one split.

    Files are naturally ordered (01, 02, ..., 10). For the expected one-sample-per-file
    layout, the filename stem is deliberately used as the public sample id so it also
    matches videos/01.mp4 and the result table can be checked against the source files.
    """
    directory = Path(path)
    files = sorted(directory.glob("*.pkl"), key=_natural_path_key)
    if not files:
        raise FileNotFoundError(f"No .pkl files found directly under {directory}")

    required_chunks: Dict[str, list[np.ndarray]] = {m: [] for m in MODALITIES}
    optional_fields = (
        "text_bert",
        "text_lengths",
        "audio_lengths",
        "vision_lengths",
        "classification_labels",
        "regression_labels",
        "annotations",
    )
    optional_chunks: Dict[str, list[np.ndarray]] = {field: [] for field in optional_fields}
    optional_complete = {field: True for field in optional_fields}
    ids: list[str] = []
    raw_texts: list[str] = []
    reference_shapes: Dict[str, tuple[int, int]] = {}

    for source in files:
        with source.open("rb") as f:
            obj = pickle.load(f)
        mapping = _find_feature_mapping(obj, split_name)
        features = {
            modality: _ensure_batched_feature(mapping[modality], modality, source)
            for modality in MODALITIES
        }
        n = features["text"].shape[0]
        if any(feature.shape[0] != n for feature in features.values()):
            raise ValueError(f"{source.name}: modalities have different sample counts")
        for modality, feature in features.items():
            shape = (feature.shape[1], feature.shape[2])
            if modality in reference_shapes and shape != reference_shapes[modality]:
                raise ValueError(
                    f"{source.name}: {modality} shape {shape} differs from {reference_shapes[modality]}"
                )
            reference_shapes.setdefault(modality, shape)
            required_chunks[modality].append(feature)

        if n == 1:
            file_ids = [source.stem]
        else:
            internal_ids = mapping.get("id", mapping.get("ids"))
            if internal_ids is not None and len(np.asarray(internal_ids).reshape(-1)) == n:
                file_ids = [_decode(x) for x in np.asarray(internal_ids).reshape(-1)]
            else:
                file_ids = [f"{source.stem}_{index:03d}" for index in range(n)]
        ids.extend(file_ids)

        text_value = mapping.get("raw_text", mapping.get("text_raw"))
        if text_value is None:
            raw_texts.extend([""] * n)
        else:
            text_array = np.asarray(text_value, dtype=object).reshape(-1)
            if len(text_array) == 1 and n > 1:
                text_array = np.repeat(text_array, n)
            if len(text_array) != n:
                raise ValueError(f"{source.name}: raw_text length does not match samples")
            raw_texts.extend([_decode(x) for x in text_array])

        for field in optional_fields:
            if field not in mapping:
                optional_complete[field] = False
                continue
            optional_chunks[field].append(
                _ensure_batched_optional(mapping[field], n, field, source)
            )

    merged: Dict[str, Any] = {
        modality: np.concatenate(chunks, axis=0)
        for modality, chunks in required_chunks.items()
    }
    merged["id"] = np.asarray(ids, dtype=object)
    merged["raw_text"] = np.asarray(raw_texts, dtype=object)
    for field in optional_fields:
        if optional_complete[field] and len(optional_chunks[field]) == len(files):
            merged[field] = np.concatenate(optional_chunks[field], axis=0)
    return {split_name: merged}


def load_feature_source(path: str | Path, split_name: str = "test") -> Dict[str, Any]:
    """Load either a monolithic attachment-2 pickle or an attachment-4 directory."""
    source = Path(path)
    if source.is_dir():
        return load_pickle_directory(source, split_name=split_name)
    if not source.is_file():
        raise FileNotFoundError(source)
    return load_pickle(source)


def _excel_identifier(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return _decode(value).strip()


def attach_labels_from_excel(
    data: Dict[str, Any],
    excel_path: str | Path,
) -> Dict[str, Any]:
    """Attach labels to monolithic attachment-2 splits by sample id.

    Supported identity layouts are either an `id` column or the pair
    `video_id` + `clip_id`. Classification labels may be explicit or are derived
    from the sign of the continuous label according to the competition rules.
    Existing labels inside the pickle are never overwritten.
    """
    import pandas as pd

    table = pd.read_excel(excel_path)
    normalized_columns = {
        str(column).strip().lower().replace(" ", "_"): column for column in table.columns
    }

    if "id" in normalized_columns:
        ids = table[normalized_columns["id"]].map(_excel_identifier)
    elif "video_id" in normalized_columns and "clip_id" in normalized_columns:
        video_ids = table[normalized_columns["video_id"]].map(_excel_identifier)
        clip_ids = table[normalized_columns["clip_id"]].map(_excel_identifier)
        ids = video_ids + "$_$" + clip_ids
    else:
        raise ValueError(
            f"{excel_path}: label sheet needs 'id' or both 'video_id' and 'clip_id' columns"
        )

    regression_aliases = (
        "regression_labels",
        "regression_label",
        "label",
        "sentiment_score",
        "score",
    )
    classification_aliases = (
        "classification_labels",
        "classification_label",
        "annotations",
        "annotation",
        "sentiment_class",
        "class",
    )
    regression_column = next(
        (normalized_columns[name] for name in regression_aliases if name in normalized_columns),
        None,
    )
    classification_column = next(
        (
            normalized_columns[name]
            for name in classification_aliases
            if name in normalized_columns
        ),
        None,
    )
    if regression_column is None:
        raise ValueError(f"{excel_path}: no continuous sentiment label column found")

    regression_values = pd.to_numeric(table[regression_column], errors="raise").astype(float)
    if not regression_values.between(-3, 3).all():
        raise ValueError(f"{excel_path}: continuous labels fall outside [-3,3]")
    if classification_column is None:
        classification_values = np.where(
            regression_values.to_numpy() < 0,
            "Negative",
            np.where(regression_values.to_numpy() > 0, "Positive", "Neutral"),
        )
    else:
        classification_values = table[classification_column].to_numpy()

    label_map: Dict[str, tuple[Any, float]] = {}
    for identifier, classification, regression in zip(
        ids, classification_values, regression_values.to_numpy()
    ):
        if identifier in label_map:
            raise ValueError(f"{excel_path}: duplicate id '{identifier}'")
        label_map[str(identifier)] = (classification, float(regression))

    for split_name in ("train", "valid", "test"):
        if split_name not in data or not isinstance(data[split_name], Mapping):
            continue
        split = data[split_name]
        has_classification = any(
            key in split
            for key in ("classification_labels", "classification_label", "annotations", "annotation")
        )
        has_regression = any(
            key in split
            for key in ("regression_labels", "regression_label", "label", "labels")
        )
        if has_classification and has_regression:
            continue
        if "id" not in split and "ids" not in split:
            raise ValueError(f"Split '{split_name}' has no id field for Excel label matching")
        split_ids = [
            _excel_identifier(value)
            for value in np.asarray(split.get("id", split.get("ids")), dtype=object).reshape(-1)
        ]
        missing = [identifier for identifier in split_ids if identifier not in label_map]
        if missing:
            raise ValueError(
                f"{excel_path}: {len(missing)} ids from split '{split_name}' are missing; "
                f"examples={missing[:5]}"
            )
        if not has_classification:
            split["classification_labels"] = np.asarray(
                [label_map[identifier][0] for identifier in split_ids], dtype=object
            )
        if not has_regression:
            split["regression_labels"] = np.asarray(
                [label_map[identifier][1] for identifier in split_ids], dtype=np.float32
            )
    return data


def find_sibling_label_excel(feature_path: str | Path) -> Optional[Path]:
    source = Path(feature_path)
    directory = source if source.is_dir() else source.parent
    candidates = [directory / "label.xlsx", directory / "labels.xlsx"]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def parse_split(
    data: Mapping[str, Any],
    split_name: str,
    mask_strategy: str = "text_shared",
    require_labels: bool = True,
) -> SplitArrays:
    if split_name not in data:
        raise KeyError(f"Split '{split_name}' not found. Available: {list(data)}")
    split = data[split_name]
    features = {m: _as_feature_array(split[m], m) for m in MODALITIES}
    n = features["text"].shape[0]
    if any(x.shape[0] != n for x in features.values()):
        raise ValueError("Modalities contain different numbers of samples")

    ids_source = split.get("id", split.get("ids", np.arange(n)))
    ids = np.asarray([_decode(x) for x in np.asarray(ids_source, dtype=object).reshape(-1)], dtype=object)
    text_source = split.get("raw_text", np.full(n, "", dtype=object))
    raw_text = np.asarray([_decode(x) for x in np.asarray(text_source, dtype=object).reshape(-1)], dtype=object)
    if len(ids) != n or len(raw_text) != n:
        raise ValueError("id/raw_text length does not match feature arrays")

    text_len = features["text"].shape[1]
    text_mask = None
    if "text_lengths" in split:
        text_mask = _mask_from_lengths(split["text_lengths"], n, text_len)
    if text_mask is None:
        text_mask = _text_bert_attention_mask(split, n, text_len)
    if text_mask is None:
        text_mask = _feature_mask(features["text"])

    masks: Dict[str, np.ndarray] = {"text": text_mask}
    aligned = all(x.shape[1] == text_len for x in features.values())
    if mask_strategy == "text_shared":
        if not aligned:
            raise ValueError("text_shared mask requires aligned modality sequence lengths")
        masks.update(audio=text_mask.copy(), vision=text_mask.copy())
    elif mask_strategy == "per_modality":
        for modality in ("audio", "vision"):
            length_key = f"{modality}_lengths"
            if length_key in split:
                masks[modality] = _mask_from_lengths(
                    split[length_key], n, features[modality].shape[1]
                )
            else:
                masks[modality] = _feature_mask(features[modality])
    else:
        raise ValueError("mask_strategy must be 'text_shared' or 'per_modality'")

    for modality in MODALITIES:
        empty = ~masks[modality].any(axis=1)
        if empty.any():
            # Keep one safe zero position so attention kernels never see an all-masked row.
            masks[modality][empty, 0] = True
            features[modality][empty, 0] = 0.0

    cls = _parse_class_labels(split)
    reg = _parse_regression_labels(split)
    if require_labels and (cls is None or reg is None):
        raise ValueError(f"Split '{split_name}' does not contain both target fields")
    if cls is not None and len(cls) != n:
        raise ValueError("Classification label length mismatch")
    if reg is not None and len(reg) != n:
        raise ValueError("Regression label length mismatch")
    return SplitArrays(features, masks, ids, raw_text, cls, reg)


class FeatureNormalizer:
    def __init__(
        self,
        normalize_text: bool = False,
        eps: float = 1e-6,
        clip_value: Optional[float] = 10.0,
    ) -> None:
        self.normalize_text = normalize_text
        self.eps = eps
        self.clip_value = clip_value
        self.stats: Dict[str, Dict[str, np.ndarray]] = {}

    def fit(self, split: SplitArrays) -> "FeatureNormalizer":
        for modality in MODALITIES:
            if modality == "text" and not self.normalize_text:
                continue
            valid = split.features[modality][split.masks[modality]]
            if valid.size == 0:
                raise ValueError(f"No valid values found for {modality}")
            mean = valid.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = valid.std(axis=0, dtype=np.float64).astype(np.float32)
            std = np.maximum(std, self.eps)
            self.stats[modality] = {"mean": mean, "std": std}
        return self

    def transform(self, split: SplitArrays) -> SplitArrays:
        features: Dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            x = split.features[modality].astype(np.float32, copy=True)
            if modality in self.stats:
                x = (x - self.stats[modality]["mean"]) / self.stats[modality]["std"]
                if self.clip_value is not None:
                    x = np.clip(x, -float(self.clip_value), float(self.clip_value))
            x *= split.masks[modality][..., None]
            if not np.isfinite(x).all():
                raise ValueError(f"Normalized '{modality}' features contain NaN/Inf")
            features[modality] = x
        return SplitArrays(
            features=features,
            masks={k: v.copy() for k, v in split.masks.items()},
            ids=split.ids.copy(),
            raw_text=split.raw_text.copy(),
            class_labels=None if split.class_labels is None else split.class_labels.copy(),
            regression_labels=None
            if split.regression_labels is None
            else split.regression_labels.copy(),
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "normalize_text": self.normalize_text,
            "eps": self.eps,
            "clip_value": self.clip_value,
            "stats": self.stats,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureNormalizer":
        # Older checkpoints did not clip normalized values.  Preserve their
        # exact inference behavior when clip_value is absent.
        obj = cls(
            bool(state.get("normalize_text", False)),
            float(state.get("eps", 1e-6)),
            state.get("clip_value", None),
        )
        obj.stats = {
            modality: {
                "mean": np.asarray(values["mean"], dtype=np.float32),
                "std": np.asarray(values["std"], dtype=np.float32),
            }
            for modality, values in state.get("stats", {}).items()
        }
        return obj


class MultimodalDataset(Dataset):
    def __init__(self, arrays: SplitArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays.size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            modality: torch.from_numpy(self.arrays.features[modality][index])
            for modality in MODALITIES
        }
        item.update(
            {
                f"{modality}_mask": torch.from_numpy(self.arrays.masks[modality][index])
                for modality in MODALITIES
            }
        )
        item["id"] = self.arrays.ids[index]
        item["raw_text"] = self.arrays.raw_text[index]
        if self.arrays.class_labels is not None:
            item["class_label"] = torch.tensor(
                self.arrays.class_labels[index], dtype=torch.long
            )
        if self.arrays.regression_labels is not None:
            item["regression_label"] = torch.tensor(
                self.arrays.regression_labels[index], dtype=torch.float32
            )
        return item


def compute_class_weights(
    labels: Iterable[int], power: float = 0.5, max_weight: Optional[float] = 3.0
) -> torch.Tensor:
    """Return tempered inverse-frequency weights for the three classes.

    Full inverse-frequency weighting is too aggressive when exact-neutral
    samples are rare (as in continuous MOSEI labels) and can collapse the
    classifier toward the neutral class.  Square-root inverse frequency keeps
    minority-class support without overwhelming the predictive objective.
    """
    labels = np.asarray(list(labels), dtype=np.int64)
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if not 0.0 <= float(power) <= 1.0:
        raise ValueError("class-weight power must lie in [0,1]")
    weights = np.power(len(labels) / (3.0 * np.maximum(counts, 1.0)), float(power))
    weights /= weights.mean()
    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))
        weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def augment_batch(
    batch: Dict[str, Any],
    span_probability: float,
    max_ratio: float,
    noise_std: float,
) -> Dict[str, Any]:
    """Mild stochastic augmentation; labels and metadata remain unchanged."""
    output = dict(batch)
    for modality in MODALITIES:
        x = batch[modality].clone()
        mask = batch[f"{modality}_mask"]
        if noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            x = x + noise * mask.unsqueeze(-1)
        if span_probability > 0 and max_ratio > 0:
            for row in range(x.size(0)):
                if torch.rand((), device=x.device).item() >= span_probability:
                    continue
                valid_len = int(mask[row].sum().item())
                if valid_len <= 2:
                    continue
                width = max(1, int(valid_len * max_ratio * torch.rand((), device=x.device).item()))
                start_max = max(1, valid_len - width + 1)
                start = int(torch.randint(0, start_max, (1,), device=x.device).item())
                x[row, start : start + width] = 0.0
        output[modality] = x
    return output
