"""Continuous, padding-aware modality-missing augmentation."""

from __future__ import annotations

import random
import math
import hashlib
from typing import Any, Literal, Sequence

import torch


MODALITIES = ("text", "audio", "vision")
MISSING_TYPES = ("none", "single", "double")


class ContinuousMissingAugmentation:
    """Zero continuous valid intervals while retaining separate missing masks."""

    def __init__(
        self,
        probabilities: Sequence[float] = (0.20, 0.50, 0.30),
        ratio_range: tuple[float, float] = (0.10, 0.60),
        double_interval_mode: Literal["same", "different", "random"] = "random",
        seed: int | None = None,
    ) -> None:
        if len(probabilities) != 3 or any(p < 0 for p in probabilities):
            raise ValueError("probabilities must be three non-negative values")
        if abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError("probabilities must sum to 1")
        low, high = ratio_range
        if not (0 < low <= high <= 1):
            raise ValueError("ratio_range must satisfy 0 < low <= high <= 1")
        if double_interval_mode not in {"same", "different", "random"}:
            raise ValueError("double_interval_mode must be same, different, or random")
        self.probabilities = tuple(float(p) for p in probabilities)
        self.ratio_range = (float(low), float(high))
        self.double_interval_mode = double_interval_mode
        self._rng = random.Random(seed) if seed is not None else None

    @property
    def rng(self) -> random.Random:
        # DataLoader seeds Python's module RNG independently in each worker.
        return self._rng if self._rng is not None else random  # type: ignore[return-value]

    @staticmethod
    def _runs(mask: torch.Tensor) -> list[tuple[int, int]]:
        """Return half-open contiguous True runs."""
        values = mask.to(dtype=torch.bool, device="cpu").tolist()
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for i, value in enumerate(values + [False]):
            if value and start is None:
                start = i
            elif not value and start is not None:
                runs.append((start, i))
                start = None
        return runs

    def _choose_interval(self, valid_mask: torch.Tensor, ratio: float) -> tuple[int, int]:
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return (-1, -1)
        # Ceil keeps the realized discrete ratio from falling below the sampled ratio.
        requested = max(1, min(valid_count, math.ceil(valid_count * ratio)))
        runs = self._runs(valid_mask)
        max_run = max(end - start for start, end in runs)
        length = min(requested, max_run)
        starts = [
            position
            for run_start, run_end in runs
            for position in range(run_start, run_end - length + 1)
        ]
        start = self.rng.choice(starts)
        return (start, start + length)

    def _choose_shared_interval(
        self, left: torch.Tensor, right: torch.Tensor, ratio: float
    ) -> tuple[int, int]:
        return self._choose_interval(left.bool() & right.bool(), ratio)

    def __call__(
        self,
        sample: dict[str, Any],
        missing_type: Literal["none", "single", "double"] | None = None,
    ) -> dict[str, Any]:
        result = dict(sample)
        for modality in MODALITIES:
            result[f"masked_{modality}"] = sample[modality].clone()
            result[f"missing_mask_{modality}"] = torch.ones(
                sample[modality].shape[0], dtype=torch.bool
            )
            result[f"missing_ratio_{modality}"] = torch.tensor(0.0, dtype=torch.float32)
            result[f"missing_interval_{modality}"] = torch.tensor([-1, -1], dtype=torch.long)

        chosen_type = missing_type or self.rng.choices(MISSING_TYPES, self.probabilities, k=1)[0]
        if chosen_type not in MISSING_TYPES:
            raise ValueError(f"unknown missing_type: {chosen_type}")
        result["missing_type"] = chosen_type
        result["missing_modalities"] = ""
        result["double_interval_mode"] = "none"
        if chosen_type == "none":
            return result

        count = 1 if chosen_type == "single" else 2
        selected = self.rng.sample(MODALITIES, count)
        result["missing_modalities"] = "+".join(selected)
        sampled_ratio = self.rng.uniform(*self.ratio_range)

        intervals: dict[str, tuple[int, int]] = {}
        mode = self.double_interval_mode
        if chosen_type == "double" and mode == "random":
            mode = self.rng.choice(("same", "different"))
        if chosen_type == "double" and mode == "same":
            left, right = selected
            interval = self._choose_shared_interval(
                sample[f"valid_mask_{left}"], sample[f"valid_mask_{right}"], sampled_ratio
            )
            intervals[left] = intervals[right] = interval
            result["double_interval_mode"] = "same"
        else:
            for modality in selected:
                intervals[modality] = self._choose_interval(
                    sample[f"valid_mask_{modality}"], sampled_ratio
                )
            if chosen_type == "double":
                result["double_interval_mode"] = "different"

        for modality, (start, end) in intervals.items():
            if start < 0:  # no valid positions
                continue
            valid = sample[f"valid_mask_{modality}"].bool()
            interval_mask = torch.zeros_like(valid)
            interval_mask[start:end] = True
            if not torch.all(valid[interval_mask]):
                raise RuntimeError("internal error: attempted to mask padding")
            result[f"masked_{modality}"][start:end] = 0
            result[f"missing_mask_{modality}"][start:end] = False
            actual = end - start
            result[f"missing_ratio_{modality}"] = torch.tensor(
                actual / int(valid.sum().item()), dtype=torch.float32
            )
            result[f"missing_interval_{modality}"] = torch.tensor([start, end])
        return result


class DeterministicMissingAugmentation:
    """Repeat the same augmentation for a sample id on every validation pass."""

    def __init__(
        self,
        probabilities: Sequence[float] = (0.20, 0.50, 0.30),
        ratio_range: tuple[float, float] = (0.10, 0.60),
        double_interval_mode: Literal["same", "different", "random"] = "random",
        seed: int = 42,
    ) -> None:
        self.probabilities = tuple(probabilities)
        self.ratio_range = ratio_range
        self.double_interval_mode = double_interval_mode
        self.seed = seed

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        identity = f"{self.seed}:{sample['id']}".encode("utf-8")
        sample_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        augmentation = ContinuousMissingAugmentation(
            probabilities=self.probabilities,
            ratio_range=self.ratio_range,
            double_interval_mode=self.double_interval_mode,
            seed=sample_seed,
        )
        return augmentation(sample)
