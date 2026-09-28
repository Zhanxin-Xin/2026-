"""Multi-view continuous missing augmentation for TASP-MSA training."""
from __future__ import annotations

from typing import Any

from data.missing_augmentation import ContinuousMissingAugmentation


class MultiViewMissingTransform:
    """Return full data plus independent single- and double-missing views."""

    def __init__(self, ratio_range=(0.1, 0.6), double_interval_mode="random") -> None:
        self.single = ContinuousMissingAugmentation(
            probabilities=(0.0, 1.0, 0.0),
            ratio_range=ratio_range,
            double_interval_mode=double_interval_mode,
        )
        self.double = ContinuousMissingAugmentation(
            probabilities=(0.0, 0.0, 1.0),
            ratio_range=ratio_range,
            double_interval_mode=double_interval_mode,
        )

    @staticmethod
    def _view(augmented: dict[str, Any]) -> dict[str, Any]:
        keys = [
            key for key in augmented
            if key.startswith(("masked_", "missing_mask_", "missing_ratio_", "missing_interval_"))
        ]
        result = {key: augmented[key] for key in keys}
        for key in ("missing_type", "missing_modalities", "double_interval_mode"):
            result[key] = augmented[key]
        return result

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        result["single_view"] = self._view(self.single(sample, missing_type="single"))
        result["double_view"] = self._view(self.double(sample, missing_type="double"))
        return result
