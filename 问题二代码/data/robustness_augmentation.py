"""Deterministic continuous missing patterns for robustness evaluation."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Literal, Sequence

import torch


MODALITIES = ("text", "audio", "vision")
Position = Literal["beginning", "middle", "end", "random"]


class RobustnessMissingTransform:
    """Apply one controlled, repeatable continuous interval to selected modalities."""

    def __init__(
        self,
        modalities: Sequence[str] = (),
        missing_ratio: float = 0.0,
        position: Position = "random",
        seed: int = 42,
    ) -> None:
        unknown = set(modalities).difference(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities: {sorted(unknown)}")
        if not 0 <= missing_ratio <= 1:
            raise ValueError("missing_ratio must be in [0, 1]")
        if position not in {"beginning", "middle", "end", "random"}:
            raise ValueError(f"unknown position: {position}")
        self.modalities = tuple(modalities)
        self.missing_ratio = float(missing_ratio)
        self.position = position
        self.seed = int(seed)

    @staticmethod
    def _runs(mask: torch.Tensor) -> list[tuple[int, int]]:
        values = mask.bool().tolist() + [False]
        runs: list[tuple[int, int]] = []
        start = None
        for index, value in enumerate(values):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None
        return runs

    def _interval(self, valid: torch.Tensor, sample_id: str) -> tuple[int, int]:
        valid_count = int(valid.sum())
        if valid_count == 0 or self.missing_ratio == 0:
            return (-1, -1)
        requested = max(1, min(valid_count, math.ceil(valid_count * self.missing_ratio)))
        runs = self._runs(valid)
        max_run = max(end - start for start, end in runs)
        length = min(requested, max_run)
        eligible = [(start, end) for start, end in runs if end - start >= length]
        if self.position == "beginning":
            start = eligible[0][0]
        elif self.position == "end":
            start = eligible[-1][1] - length
        elif self.position == "middle":
            run_start, run_end = max(eligible, key=lambda item: item[1] - item[0])
            start = run_start + ((run_end - run_start - length) // 2)
        else:
            candidates = [
                index
                for run_start, run_end in eligible
                for index in range(run_start, run_end - length + 1)
            ]
            digest = hashlib.sha256(
                f"{self.seed}:{sample_id}:{self.modalities}:{self.missing_ratio}".encode()
            ).digest()
            start = random.Random(int.from_bytes(digest[:8], "big")).choice(candidates)
        return start, start + length

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        result = dict(sample)
        for name in MODALITIES:
            result[f"masked_{name}"] = sample[name].clone()
            result[f"missing_mask_{name}"] = torch.ones(50, dtype=torch.bool)
        if not self.modalities or self.missing_ratio == 0:
            return result

        shared_valid = torch.ones(50, dtype=torch.bool)
        for name in self.modalities:
            shared_valid &= sample[f"valid_mask_{name}"].bool()
        start, end = self._interval(shared_valid, str(sample["id"]))
        if start < 0:
            return result
        for name in self.modalities:
            result[f"masked_{name}"][start:end] = 0
            result[f"missing_mask_{name}"][start:end] = False
        return result
