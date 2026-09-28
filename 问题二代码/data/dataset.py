"""PyTorch Dataset for the inspected aligned_50.pkl structure."""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_FIELDS = {
    "id",
    "text",
    "audio",
    "vision",
    "text_bert",
    "classification_labels",
    "regression_labels",
}


@lru_cache(maxsize=1)
def _load_pickle(path: str) -> dict[str, Any]:
    """Load once per process so train/valid datasets do not duplicate the ~1 GB pickle."""
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"pickle root must be dict, got {type(data).__name__}")
    return data


class MOSEIAlignedDataset(Dataset):
    """Expose one split of the aligned MOSEI pickle as float32 tensors.

    The inspected ``text_bert`` layout is [input_ids, attention_mask,
    token_type_ids].  Valid temporal positions are content tokens selected by
    the attention mask, excluding BERT's [CLS] and [SEP] positions.  Crucially,
    feature values (including all-zero vision rows) are not used to infer
    padding, so pre-existing missing observations remain valid time positions.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str = "train",
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.data_path = Path(data_path).expanduser().resolve()
        self.split_name = split
        self.transform = transform

        data = _load_pickle(str(self.data_path))
        if not isinstance(data, dict) or split not in data:
            available = list(data) if isinstance(data, dict) else type(data).__name__
            raise KeyError(f"split {split!r} not found; available={available}")
        split_data = data[split]
        if not isinstance(split_data, dict):
            raise TypeError(f"data[{split!r}] must be dict, got {type(split_data).__name__}")
        missing = REQUIRED_FIELDS.difference(split_data)
        if missing:
            raise KeyError(f"data[{split!r}] lacks required fields: {sorted(missing)}")

        self.data = split_data
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n = len(self.data["id"])
        expected = {"text": (50, 768), "audio": (50, 74), "vision": (50, 35)}
        for name, tail in expected.items():
            value = self.data[name]
            if not isinstance(value, np.ndarray) or value.shape != (n, *tail):
                raise ValueError(f"{name} expected {(n, *tail)}, got {getattr(value, 'shape', None)}")
        if self.data["text_bert"].shape != (n, 3, 50):
            raise ValueError(f"text_bert expected {(n, 3, 50)}, got {self.data['text_bert'].shape}")
        for name in ("classification_labels", "regression_labels"):
            if self.data[name].shape != (n,):
                raise ValueError(f"{name} expected {(n,)}, got {self.data[name].shape}")

        attention = self.data["text_bert"][:, 1, :]
        if not np.isin(attention, (0, 1)).all():
            raise ValueError("text_bert channel 1 is not a binary attention mask")

    def __len__(self) -> int:
        return len(self.data["id"])

    def _valid_mask(self, index: int) -> torch.Tensor:
        input_ids = self.data["text_bert"][index, 0]
        attention = self.data["text_bert"][index, 1].astype(bool, copy=False)
        # 101=[CLS], 102=[SEP]. They are model tokens, not aligned content time steps.
        content = attention & (input_ids != 101) & (input_ids != 102)
        return torch.from_numpy(content.copy())

    def __getitem__(self, index: int) -> dict[str, Any]:
        valid = self._valid_mask(index)
        sample: dict[str, Any] = {
            "id": self.data["id"][index],
            "text": torch.as_tensor(self.data["text"][index], dtype=torch.float32),
            "audio": torch.as_tensor(self.data["audio"][index], dtype=torch.float32),
            "vision": torch.as_tensor(self.data["vision"][index], dtype=torch.float32),
            "valid_mask_text": valid.clone(),
            "valid_mask_audio": valid.clone(),
            "valid_mask_vision": valid.clone(),
            "classification_label": torch.tensor(
                int(self.data["classification_labels"][index]), dtype=torch.long
            ),
            "regression_label": torch.tensor(
                float(self.data["regression_labels"][index]), dtype=torch.float32
            ),
        }
        return self.transform(sample) if self.transform is not None else sample
