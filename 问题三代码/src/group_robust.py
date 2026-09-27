from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor


def video_group_name(identifier: object) -> str:
    """Return the video prefix used by every grouped OOF split."""
    value = str(identifier)
    return value.rsplit("$_$", 1)[0] if "$_$" in value else value


@dataclass(frozen=True)
class VideoGroupMetadata:
    names: tuple[str, ...]
    index: np.ndarray
    size: np.ndarray

    @property
    def group_count(self) -> int:
        return len(self.names)


def build_video_group_metadata(identifiers: Iterable[object]) -> VideoGroupMetadata:
    groups = [video_group_name(value) for value in identifiers]
    names = tuple(sorted(set(groups)))
    lookup = {name: index for index, name in enumerate(names)}
    group_index = np.asarray([lookup[name] for name in groups], dtype=np.int64)
    counts = np.bincount(group_index, minlength=len(names)).astype(np.float32)
    return VideoGroupMetadata(
        names=names,
        index=group_index,
        size=counts[group_index],
    )


class VideoGroupDRO:
    """Online group-DRO over real video identities with equal-video sampling risk.

    Each sample receives q_g / n_g weight, so a video contributes the same total
    mass regardless of its utterance count.  At epoch end q is exponentiated by
    the observed mean loss for that video.  No held-out labels or group metrics
    enter the update.
    """

    def __init__(
        self,
        group_count: int,
        sample_count: int,
        eta: float,
        device: torch.device,
    ) -> None:
        if group_count < 2:
            raise ValueError("video group DRO requires at least two groups")
        if sample_count < group_count:
            raise ValueError("sample count cannot be smaller than video group count")
        if eta <= 0.0:
            raise ValueError("video group DRO eta must be positive")
        self.group_count = int(group_count)
        self.sample_count = int(sample_count)
        self.eta = float(eta)
        self.device = device
        self.weights = torch.full(
            (self.group_count,),
            1.0 / self.group_count,
            dtype=torch.float32,
            device=device,
        )
        self._risk_sum = torch.zeros_like(self.weights)
        self._risk_count = torch.zeros_like(self.weights)

    def loss(
        self,
        per_sample_loss: Tensor,
        group_index: Tensor,
        group_size: Tensor,
    ) -> Tensor:
        per_sample_loss = per_sample_loss.float().reshape(-1)
        group_index = group_index.long().reshape(-1)
        group_size = group_size.float().reshape(-1)
        if not (
            per_sample_loss.numel()
            == group_index.numel()
            == group_size.numel()
        ):
            raise ValueError("group DRO tensors must share the batch dimension")
        if bool((group_index < 0).any()) or bool(
            (group_index >= self.group_count).any()
        ):
            raise ValueError("group index lies outside the configured group set")
        if bool((group_size <= 0).any()):
            raise ValueError("video group sizes must be positive")

        # Uniformly shuffled minibatches estimate
        #   sum_g q_g * mean_{i in g}(loss_i).
        # The fixed N/B factor makes this unbiased. Batch-local normalization
        # would be a ratio estimator and silently reintroduce large-video bias.
        sample_weight = self.weights[group_index].detach() / group_size
        robust = (
            float(self.sample_count)
            / float(per_sample_loss.numel())
            * (sample_weight * per_sample_loss).sum()
        )
        with torch.no_grad():
            self._risk_sum.scatter_add_(0, group_index, per_sample_loss.detach())
            self._risk_count.scatter_add_(
                0, group_index, torch.ones_like(per_sample_loss)
            )
        return robust

    @torch.no_grad()
    def finish_epoch(self) -> dict[str, float]:
        observed = self._risk_count > 0
        if not bool(observed.all()):
            missing = int((~observed).sum().item())
            raise RuntimeError(
                f"group-DRO epoch did not observe {missing} training video groups"
            )
        mean_risk = self._risk_sum / self._risk_count.clamp_min(1.0)
        centered = mean_risk - mean_risk.mean()
        log_weight = self.weights.clamp_min(1e-30).log() + self.eta * centered
        self.weights.copy_(torch.softmax(log_weight, dim=0))

        entropy = -(
            self.weights * self.weights.clamp_min(1e-30).log()
        ).sum()
        sorted_risk = torch.sort(mean_risk).values
        tail_count = max(1, int(math.ceil(0.10 * self.group_count)))
        diagnostics = {
            "video_group_mean_risk": float(mean_risk.mean().item()),
            "video_group_worst_decile_risk": float(
                sorted_risk[-tail_count:].mean().item()
            ),
            "video_group_weight_max": float(self.weights.max().item()),
            "video_group_weight_min": float(self.weights.min().item()),
            "video_group_effective_count": float(torch.exp(entropy).item()),
        }
        self._risk_sum.zero_()
        self._risk_count.zero_()
        return diagnostics
