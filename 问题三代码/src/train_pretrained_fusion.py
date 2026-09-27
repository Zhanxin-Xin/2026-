from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import (
    CLASS_NAMES,
    FeatureNormalizer,
    attach_labels_from_excel,
    compute_class_weights,
    load_pickle,
    parse_split,
)
from .losses import pearson_loss
from .metrics import compute_metrics
from .pretrained_fusion import PretrainedTextFusionNet
from .group_robust import VideoGroupDRO, build_video_group_metadata
from .utils import (
    apply_overrides,
    atomic_torch_save,
    load_config,
    make_logger,
    resolve_device,
    save_json,
    seed_everything,
)


class EncodedTextDataset(Dataset):
    def __init__(
        self,
        arrays: Any,
        tokenized: Mapping[str, Tensor],
        *,
        include_video_pair_supervision: bool = False,
        cross_video_references: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        self.arrays = arrays
        self.tokenized = tokenized
        self.video_groups = build_video_group_metadata(arrays.ids)
        self.video_references = {
            modality: self._leave_one_out_video_reference(modality)
            for modality in ("text", "audio", "vision")
        }
        self.video_pair_supervision = (
            self._video_pair_supervision()
            if include_video_pair_supervision
            else None
        )
        self.cross_video_references = cross_video_references
        if cross_video_references is not None:
            required = {
                "semantic",
                "audio",
                "vision",
                "similarity",
                "mask",
                "bank_indices",
            }
            missing = required.difference(cross_video_references)
            if missing:
                raise ValueError(
                    f"cross-video references are missing fields: {sorted(missing)}"
                )
            for name in required:
                if len(cross_video_references[name]) != arrays.size:
                    raise ValueError(
                        f"cross-video reference field '{name}' has the wrong length"
                    )

    def _leave_one_out_video_reference(self, modality: str) -> np.ndarray:
        """Pool every *other* utterance in the same video without labels."""
        features = np.asarray(self.arrays.features[modality], dtype=np.float32)
        mask = np.asarray(self.arrays.masks[modality], dtype=np.float32)
        pooled = (features * mask[..., None]).sum(axis=1)
        pooled /= np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        group_sum = np.zeros(
            (self.video_group_count, pooled.shape[-1]), dtype=np.float32
        )
        np.add.at(group_sum, self.video_groups.index, pooled)
        denominator = self.video_groups.size - 1.0
        reference = np.zeros_like(pooled)
        available = denominator > 0.0
        reference[available] = (
            group_sum[self.video_groups.index[available]] - pooled[available]
        ) / denominator[available, None]
        return reference

    def _video_pair_supervision(self) -> dict[str, np.ndarray]:
        """Aggregate every ordered, label-discordant Neutral/Polar pair.

        This method is enabled only for a training dataset.  For sample ``i``,
        ``y_i - mean(y_peer)`` is the signed mean of all same-video ordered
        pair comparisons, where Neutral=1 and Polar=0.  Its magnitude is the
        fraction of informative (discordant) peers, so identical-label pairs
        and singleton videos contribute exactly zero loss.
        """
        if self.arrays.class_labels is None:
            raise ValueError("video pair supervision requires training labels")
        neutral = (
            np.asarray(self.arrays.class_labels, dtype=np.int64) == 1
        ).astype(np.float32)
        group_neutral_sum = np.zeros(self.video_group_count, dtype=np.float32)
        np.add.at(group_neutral_sum, self.video_groups.index, neutral)
        peer_count = self.video_groups.size.astype(np.float32) - 1.0
        peer_neutral = np.zeros_like(neutral)
        available = peer_count > 0.0
        peer_neutral[available] = (
            group_neutral_sum[self.video_groups.index[available]]
            - neutral[available]
        ) / peer_count[available]
        signed_difference = neutral - peer_neutral
        weight = np.abs(signed_difference).astype(np.float32)
        direction = (signed_difference > 0.0).astype(np.float32)
        return {
            "direction": direction,
            "weight": weight,
            "signed_difference": signed_difference.astype(np.float32),
        }

    @property
    def video_group_count(self) -> int:
        return self.video_groups.group_count

    def __len__(self) -> int:
        return self.arrays.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "input_ids": self.tokenized["input_ids"][index],
            "attention_mask": self.tokenized["attention_mask"][index],
            "token_type_ids": self.tokenized["token_type_ids"][index],
            "context_available": self.tokenized["context_available"][index],
            "legacy_text": torch.from_numpy(
                self.arrays.features["text"][index]
            ),
            "legacy_text_mask": torch.from_numpy(
                self.arrays.masks["text"][index]
            ),
            "audio": torch.from_numpy(self.arrays.features["audio"][index]),
            "audio_mask": torch.from_numpy(self.arrays.masks["audio"][index]),
            "vision": torch.from_numpy(self.arrays.features["vision"][index]),
            "vision_mask": torch.from_numpy(self.arrays.masks["vision"][index]),
            "id": str(self.arrays.ids[index]),
            "video_group_index": torch.tensor(
                self.video_groups.index[index], dtype=torch.long
            ),
            "video_group_size": torch.tensor(
                self.video_groups.size[index], dtype=torch.float32
            ),
            "video_reference_available": torch.tensor(
                bool(self.video_groups.size[index] > 1.0), dtype=torch.bool
            ),
            "video_group_log_size": torch.tensor(
                np.log1p(self.video_groups.size[index]), dtype=torch.float32
            ),
            "video_audio_reference": torch.from_numpy(
                self.video_references["audio"][index]
            ),
            "video_vision_reference": torch.from_numpy(
                self.video_references["vision"][index]
            ),
            "video_text_reference": torch.from_numpy(
                self.video_references["text"][index]
            ),
        }
        if self.arrays.class_labels is not None:
            item["class_label"] = torch.tensor(
                self.arrays.class_labels[index], dtype=torch.long
            )
        if self.arrays.regression_labels is not None:
            item["regression_label"] = torch.tensor(
                self.arrays.regression_labels[index], dtype=torch.float32
            )
        for key in (
            "current_input_ids",
            "current_attention_mask",
            "previous_input_ids",
            "previous_attention_mask",
            "previous_context_available",
            "following_input_ids",
            "following_attention_mask",
            "following_context_available",
        ):
            if key in self.tokenized:
                item[key] = self.tokenized[key][index]
        for key in (
            "emotion_input_ids",
            "emotion_attention_mask",
            "emotion_token_type_ids",
        ):
            if key in self.tokenized:
                item[key] = self.tokenized[key][index]
        if self.video_pair_supervision is not None:
            item["video_pair_direction"] = torch.tensor(
                self.video_pair_supervision["direction"][index],
                dtype=torch.float32,
            )
            item["video_pair_weight"] = torch.tensor(
                self.video_pair_supervision["weight"][index],
                dtype=torch.float32,
            )
            item["video_pair_signed_difference"] = torch.tensor(
                self.video_pair_supervision["signed_difference"][index],
                dtype=torch.float32,
            )
        if self.cross_video_references is not None:
            item["cross_video_semantic_references"] = torch.from_numpy(
                self.cross_video_references["semantic"][index]
            )
            item["cross_video_audio_references"] = torch.from_numpy(
                self.cross_video_references["audio"][index]
            )
            item["cross_video_vision_references"] = torch.from_numpy(
                self.cross_video_references["vision"][index]
            )
            item["cross_video_retrieval_similarity"] = torch.from_numpy(
                self.cross_video_references["similarity"][index]
            )
            item["cross_video_reference_mask"] = torch.from_numpy(
                self.cross_video_references["mask"][index]
            )
        return item


def _pool_split_modality(arrays: Any, modality: str) -> np.ndarray:
    """Return a mask-aware utterance vector for relation retrieval."""

    features = np.asarray(arrays.features[modality], dtype=np.float32)
    mask = np.asarray(arrays.masks[modality], dtype=np.float32)
    pooled = (features * mask[..., None]).sum(axis=1)
    pooled /= np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return pooled.astype(np.float32, copy=False)


def build_cross_video_relation_references(
    bank_arrays: Any,
    query_arrays: Any,
    references_per_class: int,
    *,
    block_size: int = 512,
) -> dict[str, np.ndarray]:
    """Retrieve a balanced labelled reference set from *other* train videos.

    The bank must be an outer-fold training split. Retrieval uses only cosine
    similarity of the legacy semantic embedding. Labels are used solely to
    partition that training bank into three equally sized candidate pools.
    Query labels are never read. For training queries every utterance from the
    same video is masked, which prevents both self-reference and local-video
    leakage; held-out OOF groups are already disjoint but pass the same check.
    """

    k = int(references_per_class)
    if k <= 0:
        raise ValueError("references_per_class must be positive")
    if bank_arrays.class_labels is None:
        raise ValueError("cross-video relation bank requires training labels")

    bank_vectors = {
        modality: _pool_split_modality(bank_arrays, modality)
        for modality in ("text", "audio", "vision")
    }
    query_vectors = {
        modality: _pool_split_modality(query_arrays, modality)
        for modality in ("text", "audio", "vision")
    }
    bank_semantic = bank_vectors["text"]
    query_semantic = query_vectors["text"]
    bank_unit = bank_semantic / np.maximum(
        np.linalg.norm(bank_semantic, axis=1, keepdims=True), 1e-6
    )
    query_unit = query_semantic / np.maximum(
        np.linalg.norm(query_semantic, axis=1, keepdims=True), 1e-6
    )

    def group_names(identifiers: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                value.rsplit("$_$", 1)[0] if "$_$" in value else value
                for value in map(str, identifiers)
            ],
            dtype=object,
        )

    bank_groups = group_names(bank_arrays.ids)
    query_groups = group_names(query_arrays.ids)
    labels = np.asarray(bank_arrays.class_labels, dtype=np.int64)
    sample_count = int(query_arrays.size)
    selected = np.full((sample_count, 3, k), -1, dtype=np.int64)
    similarity = np.full((sample_count, 3, k), -1.0, dtype=np.float32)
    available = np.zeros((sample_count, 3, k), dtype=np.bool_)

    for class_index in range(3):
        candidates = np.flatnonzero(labels == class_index)
        if candidates.size == 0:
            raise ValueError(f"relation bank has no examples for class {class_index}")
        class_unit = bank_unit[candidates]
        candidate_groups = bank_groups[candidates]
        take = min(k, int(candidates.size))
        for start in range(0, sample_count, int(block_size)):
            stop = min(sample_count, start + int(block_size))
            scores = query_unit[start:stop] @ class_unit.T
            same_video = (
                query_groups[start:stop, None] == candidate_groups[None, :]
            )
            scores[same_video] = -np.inf
            local = np.argpartition(
                scores, kth=scores.shape[1] - take, axis=1
            )[:, -take:]
            local_scores = np.take_along_axis(scores, local, axis=1)
            order = np.argsort(local_scores, axis=1)[:, ::-1]
            local = np.take_along_axis(local, order, axis=1)
            local_scores = np.take_along_axis(local_scores, order, axis=1)
            for row in range(stop - start):
                valid = np.isfinite(local_scores[row])
                count = min(k, int(valid.sum()))
                if count == 0:
                    continue
                chosen_local = local[row, valid][:count]
                chosen = candidates[chosen_local]
                selected[start + row, class_index, :count] = chosen
                similarity[start + row, class_index, :count] = local_scores[
                    row, valid
                ][:count]
                available[start + row, class_index, :count] = True

    safe_selected = np.maximum(selected, 0)
    references = {
        "semantic": bank_vectors["text"][safe_selected].astype(
            np.float32, copy=False
        ),
        "audio": bank_vectors["audio"][safe_selected].astype(np.float32, copy=False),
        "vision": bank_vectors["vision"][safe_selected].astype(
            np.float32, copy=False
        ),
        "similarity": similarity,
        "mask": available,
        "bank_indices": selected,
    }
    for modality in ("semantic", "audio", "vision"):
        references[modality] *= available[..., None]
    return references


def build_directional_contexts(
    arrays: Any, window: int
) -> tuple[list[str], list[str]]:
    """Return split-local previous and following utterances as separate views."""

    previous_contexts = [""] * arrays.size
    following_contexts = [""] * arrays.size
    groups: dict[str, list[tuple[int, int]]] = {}
    for index, identifier in enumerate(arrays.ids):
        text_id = str(identifier)
        try:
            video_id, clip_id = text_id.rsplit("$_$", 1)
            order = int(clip_id)
        except (ValueError, TypeError):
            video_id, order = text_id, index
        groups.setdefault(video_id, []).append((order, index))
    for group in groups.values():
        ordered = [index for _, index in sorted(group)]
        for position, index in enumerate(ordered):
            previous = ordered[max(0, position - window) : position]
            following = ordered[position + 1 : position + 1 + window]
            previous_contexts[index] = " ".join(
                str(arrays.raw_text[item]) for item in previous
            ).strip()
            following_contexts[index] = " ".join(
                str(arrays.raw_text[item]) for item in following
            ).strip()
    return previous_contexts, following_contexts


def build_dialogue_contexts(
    arrays: Any, window: int, mode: str = "previous"
) -> list[str]:
    if mode not in {"previous", "bidirectional"}:
        raise ValueError("context_mode must be 'previous' or 'bidirectional'")
    previous_contexts, following_contexts = build_directional_contexts(arrays, window)
    if mode == "previous":
        return previous_contexts
    contexts = []
    for previous_text, following_text in zip(
        previous_contexts, following_contexts
    ):
        parts = []
        if previous_text:
            parts.append(f"Previous utterances: {previous_text}")
        if following_text:
            parts.append(f"Following utterances: {following_text}")
        contexts.append("\n".join(parts))
    return contexts


def build_previous_contexts(arrays: Any, window: int) -> list[str]:
    """Backward-compatible wrapper for the original causal context view."""

    return build_dialogue_contexts(arrays, window, mode="previous")


def encode_split(
    tokenizer: Any,
    arrays: Any,
    max_length: int,
    context_window: int = 0,
    context_mode: str = "previous",
) -> dict[str, Tensor]:
    texts = [str(value) for value in arrays.raw_text]
    if context_window > 0 and context_mode == "directional":
        previous, following = build_directional_contexts(arrays, context_window)
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        previous_encoded = tokenizer(
            previous,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        following_encoded = tokenizer(
            following,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "token_type_ids" not in encoded:
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
        encoded["context_available"] = torch.tensor(
            [bool(left.strip() or right.strip()) for left, right in zip(previous, following)],
            dtype=torch.bool,
        )
        encoded["previous_input_ids"] = previous_encoded["input_ids"]
        encoded["previous_attention_mask"] = previous_encoded["attention_mask"]
        encoded["previous_context_available"] = torch.tensor(
            [bool(value.strip()) for value in previous], dtype=torch.bool
        )
        encoded["following_input_ids"] = following_encoded["input_ids"]
        encoded["following_attention_mask"] = following_encoded["attention_mask"]
        encoded["following_context_available"] = torch.tensor(
            [bool(value.strip()) for value in following], dtype=torch.bool
        )
        return encoded
    if context_mode not in {"previous", "bidirectional"}:
        raise ValueError(
            "context_mode must be 'previous', 'bidirectional', or 'directional'"
        )
    contexts = (
        build_dialogue_contexts(arrays, context_window, context_mode)
        if context_window > 0
        else None
    )
    encoded = tokenizer(
        texts,
        text_pair=contexts,
        padding="max_length",
        truncation="longest_first",
        max_length=max_length,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    encoded["context_available"] = torch.tensor(
        [bool(value.strip()) for value in contexts]
        if contexts is not None
        else [False] * len(texts),
        dtype=torch.bool,
    )
    if contexts is not None:
        # Keep a current-only view for architectures that isolate polar
        # direction from contextual Neutral-boundary evidence.
        current_encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["current_input_ids"] = current_encoded["input_ids"]
        encoded["current_attention_mask"] = current_encoded["attention_mask"]
    return encoded


def encode_emotion_view(
    tokenizer: Any,
    arrays: Any,
    max_length: int,
) -> dict[str, Tensor]:
    """Encode an independent current-utterance view for a frozen emotion model."""

    encoded = tokenizer(
        [str(value) for value in arrays.raw_text],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if "token_type_ids" not in encoded:
        encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])
    return {
        "emotion_input_ids": encoded["input_ids"],
        "emotion_attention_mask": encoded["attention_mask"],
        "emotion_token_type_ids": encoded["token_type_ids"],
    }


def attach_emotion_view(
    tokenized: dict[str, Tensor],
    tokenizer: Any | None,
    arrays: Any,
    max_length: int,
) -> dict[str, Tensor]:
    if tokenizer is not None:
        tokenized.update(encode_emotion_view(tokenizer, arrays, max_length))
    return tokenized


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def supervised_contrastive_loss(
    embedding: Tensor, target: Tensor, temperature: float = 0.10
) -> Tensor:
    """Single-view supervised contrastive objective with safe singleton handling."""
    if embedding.size(0) < 2:
        return embedding.sum() * 0.0
    features = torch.nn.functional.normalize(embedding.float(), dim=-1)
    logits = features @ features.transpose(0, 1) / float(temperature)
    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, -torch.inf)
    positive_mask = target[:, None].eq(target[None, :]) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embedding.sum() * 0.0
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(
        log_probability.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_count.clamp_min(1)
    )
    return per_anchor[valid].mean()


def _symmetric_kl_per_sample(
    first_probability: Tensor,
    second_probability: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Return Jeffreys/R-Drop divergence for each categorical sample."""
    first = first_probability.float().clamp_min(epsilon)
    second = second_probability.float().clamp_min(epsilon)
    first = first / first.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    second = second / second.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    first_log = first.log()
    second_log = second.log()
    forward = (first * (first_log - second_log)).sum(dim=-1)
    reverse = (second * (second_log - first_log)).sum(dim=-1)
    return 0.5 * (forward + reverse)


def hierarchical_rdrop_consistency(
    first_outputs: Mapping[str, Tensor],
    second_outputs: Mapping[str, Tensor],
    cfg: Mapping[str, Any] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Factor R-Drop over Neutral detection and conditional polar sentiment.

    The three-way posterior is represented as two decisions: Neutral versus
    Polar, then Negative versus Positive conditional on Polar. This prevents
    the majority polar classes from hiding an unstable Neutral boundary in a
    single three-class KL term. The conditional polarity term is weighted by
    the two views' detached polar mass, so confidently Neutral examples do not
    inject arbitrary polarity gradients. Inference remains a single forward.
    """
    cfg = {} if cfg is None else cfg
    first = first_outputs["class_probabilities"].float()
    second = second_outputs["class_probabilities"].float()
    if first.shape != second.shape or first.ndim != 2 or first.size(-1) != 3:
        raise ValueError(
            "hierarchical R-Drop requires matching [batch, 3] probabilities"
        )

    first_neutral = first[:, 1]
    second_neutral = second[:, 1]
    first_polar_mass = (first[:, 0] + first[:, 2]).clamp_min(1e-6)
    second_polar_mass = (second[:, 0] + second[:, 2]).clamp_min(1e-6)

    neutral_first = torch.stack((first_neutral, first_polar_mass), dim=-1)
    neutral_second = torch.stack((second_neutral, second_polar_mass), dim=-1)
    neutral = _symmetric_kl_per_sample(neutral_first, neutral_second).mean()

    polar_first = torch.stack((first[:, 0], first[:, 2]), dim=-1)
    polar_second = torch.stack((second[:, 0], second[:, 2]), dim=-1)
    polar_first = polar_first / first_polar_mass.unsqueeze(-1)
    polar_second = polar_second / second_polar_mass.unsqueeze(-1)
    polar_weight = 0.5 * (first_polar_mass + second_polar_mass).detach()
    polarity = (
        _symmetric_kl_per_sample(polar_first, polar_second) * polar_weight
    ).mean()

    regression = torch.nn.functional.smooth_l1_loss(
        first_outputs["regression"].float(),
        second_outputs["regression"].float(),
        beta=0.25,
    )
    components = {
        "hierarchical_rdrop_neutral": neutral,
        "hierarchical_rdrop_polarity": polarity,
        "hierarchical_rdrop_regression": regression,
    }
    total = (
        float(cfg.get("hierarchical_rdrop_neutral_weight", 1.0)) * neutral
        + float(cfg.get("hierarchical_rdrop_polarity_weight", 1.0)) * polarity
        + float(cfg.get("hierarchical_rdrop_regression_weight", 0.0))
        * regression
    )
    return total, components


def _directed_kl_per_sample(student: Tensor, teacher: Tensor) -> Tensor:
    """KL(teacher || student) with a stop-gradient teacher distribution."""
    student_probability = student.float().clamp_min(1e-6)
    teacher_probability = teacher.float().detach().clamp_min(1e-6)
    student_probability = student_probability / student_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    teacher_probability = teacher_probability / teacher_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)
    return (
        teacher_probability
        * (teacher_probability.log() - student_probability.log())
    ).sum(dim=-1)


def hierarchical_best_view_consistency(
    first_outputs: Mapping[str, Tensor],
    second_outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    cfg: Mapping[str, Any] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Label-aligned, one-way co-training for two stochastic task views.

    At each hierarchy level, the view assigning more probability to the true
    train label is a detached teacher for the weaker view. Neutral examples
    never participate in the conditional polarity term. This deliberately
    differs from symmetric R-Drop: a correct Neutral view cannot be pulled
    toward a less label-aligned Polar view.
    """
    cfg = {} if cfg is None else cfg
    first = first_outputs["class_probabilities"].float()
    second = second_outputs["class_probabilities"].float()
    target = batch["class_label"].long()
    if first.shape != second.shape or first.ndim != 2 or first.size(-1) != 3:
        raise ValueError(
            "hierarchical best-view training requires matching [batch, 3] probabilities"
        )
    if target.shape != first.shape[:1]:
        raise ValueError("class_label must contain one target per probability row")

    first_polar_mass = first[:, 0] + first[:, 2]
    second_polar_mass = second[:, 0] + second[:, 2]
    first_boundary = torch.stack((first_polar_mass, first[:, 1]), dim=-1)
    second_boundary = torch.stack((second_polar_mass, second[:, 1]), dim=-1)
    boundary_target = target.eq(1).long()
    first_boundary_score = first_boundary.gather(
        1, boundary_target.unsqueeze(-1)
    ).squeeze(-1).detach()
    second_boundary_score = second_boundary.gather(
        1, boundary_target.unsqueeze(-1)
    ).squeeze(-1).detach()
    boundary_teacher_first = first_boundary_score >= second_boundary_score
    boundary_teacher = torch.where(
        boundary_teacher_first.unsqueeze(-1), first_boundary, second_boundary
    )
    boundary_student = torch.where(
        boundary_teacher_first.unsqueeze(-1), second_boundary, first_boundary
    )
    neutral = _directed_kl_per_sample(boundary_student, boundary_teacher).mean()

    polar_mask = target.ne(1)
    if polar_mask.any():
        first_polar = torch.stack((first[:, 0], first[:, 2]), dim=-1)
        second_polar = torch.stack((second[:, 0], second[:, 2]), dim=-1)
        first_polar = first_polar / first_polar_mass.clamp_min(1e-6).unsqueeze(-1)
        second_polar = second_polar / second_polar_mass.clamp_min(1e-6).unsqueeze(-1)
        polarity_target = target[polar_mask].eq(2).long()
        first_polar_score = first_polar[polar_mask].gather(
            1, polarity_target.unsqueeze(-1)
        ).squeeze(-1).detach()
        second_polar_score = second_polar[polar_mask].gather(
            1, polarity_target.unsqueeze(-1)
        ).squeeze(-1).detach()
        polarity_teacher_first = first_polar_score >= second_polar_score
        polarity_teacher = torch.where(
            polarity_teacher_first.unsqueeze(-1),
            first_polar[polar_mask],
            second_polar[polar_mask],
        )
        polarity_student = torch.where(
            polarity_teacher_first.unsqueeze(-1),
            second_polar[polar_mask],
            first_polar[polar_mask],
        )
        polarity = _directed_kl_per_sample(
            polarity_student, polarity_teacher
        ).mean()
        polarity_teacher_first_fraction = polarity_teacher_first.float().mean()
    else:
        polarity = (first.sum() + second.sum()) * 0.0
        polarity_teacher_first_fraction = first.new_zeros(())

    regression_target = batch["regression_label"].float()
    first_regression = first_outputs["regression"].float()
    second_regression = second_outputs["regression"].float()
    regression_teacher_first = (
        (first_regression.detach() - regression_target).abs()
        <= (second_regression.detach() - regression_target).abs()
    )
    regression_teacher = torch.where(
        regression_teacher_first, first_regression, second_regression
    ).detach()
    regression_student = torch.where(
        regression_teacher_first, second_regression, first_regression
    )
    regression = torch.nn.functional.smooth_l1_loss(
        regression_student, regression_teacher, beta=0.25
    )

    components = {
        "hierarchical_best_view_neutral": neutral,
        "hierarchical_best_view_polarity": polarity,
        "hierarchical_best_view_regression": regression,
        "hierarchical_best_view_boundary_teacher_first_fraction": (
            boundary_teacher_first.float().mean()
        ),
        "hierarchical_best_view_polarity_teacher_first_fraction": (
            polarity_teacher_first_fraction
        ),
        "hierarchical_best_view_regression_teacher_first_fraction": (
            regression_teacher_first.float().mean()
        ),
    }
    total = (
        float(cfg.get("hierarchical_best_view_neutral_weight", 1.0)) * neutral
        + float(cfg.get("hierarchical_best_view_polarity_weight", 1.0))
        * polarity
        + float(cfg.get("hierarchical_best_view_regression_weight", 0.0))
        * regression
    )
    return total, components


def resolve_loss_config(
    cfg: Mapping[str, Any], class_labels: np.ndarray
) -> dict[str, Any]:
    """Resolve fold-local statistics required by train-only loss terms."""
    resolved = dict(cfg)
    if bool(resolved.get("neutral_energy_prior_balanced", False)):
        labels = np.asarray(class_labels, dtype=np.int64)
        neutral_count = int(np.sum(labels == 1))
        polar_count = int(np.sum(labels != 1))
        if neutral_count <= 0 or polar_count <= 0:
            raise ValueError(
                "prior-balanced Neutral evidence requires both Neutral and Polar labels"
            )
        resolved["_neutral_energy_positive_weight"] = (
            float(polar_count) / float(neutral_count)
        )
    if float(resolved.get("cross_video_bilateral_relation", 0.0)) > 0.0:
        labels = np.asarray(class_labels, dtype=np.int64)
        counts = [int(np.sum(labels == index)) for index in range(3)]
        if min(counts) <= 0:
            raise ValueError(
                "cross-video bilateral relation requires all three classes"
            )
        resolved["_cross_video_left_positive_weight"] = (
            float(counts[0]) / float(counts[1])
        )
        resolved["_cross_video_right_positive_weight"] = (
            float(counts[2]) / float(counts[1])
        )
    return resolved


def compute_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    class_weights: Tensor,
    cfg: Mapping[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    target = batch["class_label"]
    regression_target = batch["regression_label"].float()
    classification = torch.nn.functional.cross_entropy(
        outputs["class_logits"].float(),
        target,
        weight=class_weights.float(),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    regression = torch.nn.functional.smooth_l1_loss(
        outputs["regression"].float(), regression_target, beta=0.5
    )
    correlation = pearson_loss(outputs["regression"], regression_target)
    text_auxiliary = torch.nn.functional.cross_entropy(
        outputs["text_auxiliary_logits"].float(),
        target,
        weight=class_weights.float(),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    neutral_target = (target == 1).float()
    neutral_auxiliary = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["neutral_logit"].float(), neutral_target
    )
    polar = target != 1
    if polar.any():
        polarity_target = (target[polar] == 2).float()
        polarity_auxiliary = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["polarity_logit"].float()[polar], polarity_target
        )
    else:
        polarity_auxiliary = classification.new_zeros(())
    hurdle_regression = torch.nn.functional.smooth_l1_loss(
        outputs["hurdle_regression"].float(), regression_target, beta=0.5
    )
    prototype_classification = torch.nn.functional.cross_entropy(
        outputs["prototype_logits"].float(),
        target,
        weight=class_weights.float(),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    subcenter_classification = torch.nn.functional.cross_entropy(
        outputs["subcenter_logits"].float(),
        target,
        weight=class_weights.float(),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    subcenter_diversity = outputs["subcenter_diversity"].float()
    contrastive = supervised_contrastive_loss(
        outputs["contrastive_embedding"],
        target,
        temperature=float(cfg.get("contrastive_temperature", 0.10)),
    )
    external_classification = torch.nn.functional.cross_entropy(
        outputs["external_logits"].float(),
        target,
        weight=class_weights.float(),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    emotion_auxiliary = torch.nn.functional.cross_entropy(
        outputs["emotion_auxiliary_logits"].float(),
        target,
        weight=class_weights.float(),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    ordinal_classification = torch.nn.functional.nll_loss(
        outputs["ordinal_probabilities"].float().clamp_min(1e-8).log(),
        target,
        weight=class_weights.float(),
    )
    ordinal_target = torch.stack([target >= 1, target >= 2], dim=-1).float()
    ordinal_threshold = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["ordinal_threshold_logits"].float(), ordinal_target
    )
    decomposition_consensus_logits = outputs.get(
        "decomposition_consensus_logits"
    )
    if decomposition_consensus_logits is not None:
        decomposition_classification = torch.nn.functional.cross_entropy(
            decomposition_consensus_logits.float(),
            target,
            weight=class_weights.float(),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        )
    else:
        decomposition_classification = classification.new_zeros(())

    decomposition_modality_logits = outputs.get(
        "decomposition_modality_logits"
    )
    if decomposition_modality_logits is not None:
        modality_target = target.unsqueeze(1).expand(-1, 3).reshape(-1)
        unimodal_classification = torch.nn.functional.cross_entropy(
            decomposition_modality_logits.float().reshape(-1, 3),
            modality_target,
            weight=class_weights.float(),
            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        )
        parent_probability = outputs["decomposition_parent_probability"].float()
        parent_probability = parent_probability.detach().clamp_min(1e-8)
        modality_log_probability = torch.log_softmax(
            decomposition_modality_logits.float(), dim=-1
        )
        per_modality_distillation = torch.nn.functional.kl_div(
            modality_log_probability,
            parent_probability.unsqueeze(1).expand_as(modality_log_probability),
            reduction="none",
        ).sum(dim=-1)
        modality_reliability = outputs[
            "decomposition_modality_reliability"
        ].float()
        modality_reliability = modality_reliability / modality_reliability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        unimodal_distillation = (
            per_modality_distillation * modality_reliability
        ).sum(dim=-1).mean()
    else:
        unimodal_classification = classification.new_zeros(())
        unimodal_distillation = classification.new_zeros(())

    decomposition_shared_embeddings = outputs.get(
        "decomposition_shared_embeddings"
    )
    if decomposition_shared_embeddings is not None:
        crossmodal_target = target.unsqueeze(1).expand(-1, 3).reshape(-1)
        crossmodal_contrastive = supervised_contrastive_loss(
            decomposition_shared_embeddings.float().reshape(
                -1, decomposition_shared_embeddings.size(-1)
            ),
            crossmodal_target,
            temperature=float(
                cfg.get(
                    "crossmodal_contrastive_temperature",
                    cfg.get("contrastive_temperature", 0.10),
                )
            ),
        )
    else:
        crossmodal_contrastive = classification.new_zeros(())
    decomposition_orthogonality = outputs.get("decomposition_orthogonality")
    orthogonality = (
        decomposition_orthogonality.float().mean()
        if decomposition_orthogonality is not None
        else classification.new_zeros(())
    )
    innovation_classification = torch.nn.functional.cross_entropy(
        outputs["innovation_logits"].float(),
        target,
        weight=class_weights.float(),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    innovation_neutral = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["innovation_neutral_logit"].float(), neutral_target
    )
    audio_reconstruction_per_sample = torch.nn.functional.smooth_l1_loss(
        outputs["innovation_predicted_audio"].float(),
        outputs["innovation_audio_target"].float(),
        beta=0.5,
        reduction="none",
    ).mean(dim=-1)
    vision_reconstruction_per_sample = torch.nn.functional.smooth_l1_loss(
        outputs["innovation_predicted_vision"].float(),
        outputs["innovation_vision_target"].float(),
        beta=0.5,
        reduction="none",
    ).mean(dim=-1)
    audio_reconstruction = (
        audio_reconstruction_per_sample
        * (0.25 + 0.75 * outputs["audio_reliability"].float().detach())
    ).mean()
    vision_reconstruction = (
        vision_reconstruction_per_sample
        * (0.25 + 0.75 * outputs["visual_reliability"].float().detach())
    ).mean()
    energy_modality_logits = outputs["neutral_energy_modality_logits"].float()
    energy_neutral_target = neutral_target.unsqueeze(-1).expand_as(
        energy_modality_logits
    )
    positive_weight_value = cfg.get("_neutral_energy_positive_weight")
    if positive_weight_value is not None:
        positive_weight = energy_modality_logits.new_tensor(
            float(positive_weight_value)
        )
        neutral_energy_classification = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                energy_modality_logits,
                energy_neutral_target,
                pos_weight=positive_weight,
                reduction="none",
            ).mean()
        )
    else:
        neutral_energy_classification = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                energy_modality_logits,
                energy_neutral_target,
                reduction="none",
            ).mean(dim=-1)
            * class_weights[target].float()
        ).mean()
    energy_reliability = outputs["neutral_energy_reliability"].float().clamp(
        1e-6, 1.0 - 1e-6
    )
    with torch.no_grad():
        evidence_target = 1.0 - torch.abs(
            torch.sigmoid(energy_modality_logits) - energy_neutral_target
        )
    neutral_evidence_calibration = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(energy_reliability), evidence_target
        )
    )
    neutral_energy_consistency = outputs[
        "neutral_energy_disagreement"
    ].float().mean()
    neutral_energy_orthogonality = outputs[
        "neutral_energy_orthogonality"
    ].float().mean()
    neutral_energy_sparsity = outputs[
        "neutral_energy_contributions"
    ].float().abs().sum(dim=-1).mean()
    energy_margin = (
        outputs["neutral_energy_values"].float()
        - outputs["polar_energy_values"].float()
    )
    energy_sign = neutral_target.mul(2.0).sub(1.0).unsqueeze(-1)
    separation_per_sample = torch.nn.functional.softplus(
        -energy_sign * energy_margin
    ).mean(dim=-1)
    if positive_weight_value is not None:
        separation_weight = torch.where(
            neutral_target.bool(),
            separation_per_sample.new_tensor(float(positive_weight_value)),
            separation_per_sample.new_ones(()),
        )
        neutral_energy_separation = (
            separation_per_sample * separation_weight
        ).sum() / separation_weight.sum().clamp_min(1e-6)
    else:
        neutral_energy_separation = separation_per_sample.mean()
    pairwise_relative_logit = outputs.get("pairwise_video_relative_logit")
    pairwise_direction = batch.get("video_pair_direction")
    pairwise_weight = batch.get("video_pair_weight")
    if (
        pairwise_relative_logit is not None
        and pairwise_direction is not None
        and pairwise_weight is not None
    ):
        informative_weight = (
            pairwise_weight.float() * class_weights[target].float()
        )
        pairwise_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
            pairwise_relative_logit.float(),
            pairwise_direction.float(),
            reduction="none",
        )
        pairwise_video_relative = (
            pairwise_per_sample * informative_weight
        ).sum() / informative_weight.sum().clamp_min(1e-6)
    else:
        pairwise_video_relative = classification.new_zeros(())
    pairwise_neutral_logit = outputs.get("pairwise_video_neutral_logit")
    if pairwise_neutral_logit is not None:
        pairwise_video_anchor = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                pairwise_neutral_logit.float(), neutral_target, reduction="none"
            )
            * class_weights[target].float()
        ).mean()
    else:
        pairwise_video_anchor = classification.new_zeros(())
    cross_video_left_logit = outputs.get("cross_video_left_logit")
    cross_video_right_logit = outputs.get("cross_video_right_logit")
    if cross_video_left_logit is not None and cross_video_right_logit is not None:
        neutral_target = (target == 1).float()
        left_mask = target != 2
        right_mask = target != 0
        left_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            cross_video_left_logit.float(),
            neutral_target,
            pos_weight=cross_video_left_logit.new_tensor(
                float(cfg.get("_cross_video_left_positive_weight", 1.0))
            ),
            reduction="none",
        )
        right_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            cross_video_right_logit.float(),
            neutral_target,
            pos_weight=cross_video_right_logit.new_tensor(
                float(cfg.get("_cross_video_right_positive_weight", 1.0))
            ),
            reduction="none",
        )
        cross_video_bilateral_relation = 0.5 * (
            (left_loss * left_mask.float()).sum()
            / left_mask.float().sum().clamp_min(1.0)
            + (right_loss * right_mask.float()).sum()
            / right_mask.float().sum().clamp_min(1.0)
        )
    else:
        cross_video_bilateral_relation = classification.new_zeros(())
    components = {
        "classification": classification,
        "regression": regression,
        "pearson": correlation,
        "text_auxiliary": text_auxiliary,
        "neutral_auxiliary": neutral_auxiliary,
        "polarity_auxiliary": polarity_auxiliary,
        "hurdle_regression": hurdle_regression,
        "prototype_classification": prototype_classification,
        "subcenter_classification": subcenter_classification,
        "subcenter_diversity": subcenter_diversity,
        "contrastive": contrastive,
        "external_classification": external_classification,
        "emotion_auxiliary": emotion_auxiliary,
        "ordinal_classification": ordinal_classification,
        "ordinal_threshold": ordinal_threshold,
        "decomposition_classification": decomposition_classification,
        "unimodal_classification": unimodal_classification,
        "unimodal_distillation": unimodal_distillation,
        "crossmodal_contrastive": crossmodal_contrastive,
        "orthogonality": orthogonality,
        "innovation_classification": innovation_classification,
        "innovation_neutral": innovation_neutral,
        "audio_reconstruction": audio_reconstruction,
        "vision_reconstruction": vision_reconstruction,
        "neutral_energy_classification": neutral_energy_classification,
        "neutral_evidence_calibration": neutral_evidence_calibration,
        "neutral_energy_consistency": neutral_energy_consistency,
        "neutral_energy_orthogonality": neutral_energy_orthogonality,
        "neutral_energy_sparsity": neutral_energy_sparsity,
        "neutral_energy_separation": neutral_energy_separation,
        "pairwise_video_relative": pairwise_video_relative,
        "pairwise_video_anchor": pairwise_video_anchor,
        "cross_video_bilateral_relation": cross_video_bilateral_relation,
    }
    total = sum(float(cfg.get(name, 0.0)) * value for name, value in components.items())
    return total, components


def build_video_group_dro(
    training_cfg: Mapping[str, Any],
    loss_cfg: Mapping[str, Any],
    dataset: Dataset,
    device: torch.device,
) -> tuple[VideoGroupDRO | None, float]:
    """Build the train-only real-video DRO objective declared by a config."""
    coefficient = float(loss_cfg.get("video_group_dro_classification", 0.0))
    if coefficient <= 0.0:
        return None, 0.0
    if float(loss_cfg.get("classification", 0.0)) != 0.0:
        raise ValueError(
            "loss.classification must be zero when video Group-DRO replaces it"
        )
    if not isinstance(dataset, EncodedTextDataset):
        raise TypeError("video Group-DRO requires EncodedTextDataset metadata")
    controller = VideoGroupDRO(
        group_count=dataset.video_group_count,
        sample_count=len(dataset),
        eta=float(training_cfg.get("video_group_dro_eta", 0.25)),
        device=device,
    )
    return controller, coefficient


def add_video_group_dro_loss(
    loss: Tensor,
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    class_weights: Tensor,
    loss_cfg: Mapping[str, Any],
    controller: VideoGroupDRO | None,
    coefficient: float,
) -> tuple[Tensor, Tensor | None]:
    """Add class-weighted, label-smoothed equal-video robust risk."""
    if controller is None:
        return loss, None
    per_sample = torch.nn.functional.cross_entropy(
        outputs["class_logits"].float(),
        batch["class_label"],
        weight=class_weights.float(),
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        reduction="none",
    )
    robust = controller.loss(
        per_sample,
        batch["video_group_index"],
        batch["video_group_size"],
    )
    return loss + float(coefficient) * robust, robust


@torch.no_grad()
def evaluate(
    model: PretrainedTextFusionNet,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    probabilities, regression, targets, regression_targets, identifiers = [], [], [], [], []
    diagnostic_names = (
        "fusion_scale",
        "audio_reliability",
        "visual_reliability",
        "audio_background_gate",
        "vision_background_gate",
        "audio_background_delta_norm",
        "vision_background_delta_norm",
        "audio_background_similarity",
        "vision_background_similarity",
        "video_background_anchor_uncertainty",
        "video_relative_neutral_shift",
        "video_relative_neutral_trust",
        "video_relative_crossmodal_agreement",
        "video_relative_audio_similarity",
        "video_relative_vision_similarity",
        "pairwise_video_relative_shift",
        "pairwise_video_relative_trust",
        "pairwise_video_relative_logit",
        "pairwise_video_audio_evidence",
        "pairwise_video_vision_evidence",
        "pairwise_video_crossmodal_agreement",
        "pairwise_video_audio_route",
        "pairwise_video_vision_route",
        "pairwise_video_correction_scale",
        "pairwise_video_semantic_evidence",
        "pairwise_video_semantic_route",
        "pairwise_video_anchor_logit",
        "pairwise_video_anchor_shift",
        "pairwise_video_neutral_logit",
        "cross_video_neutral_shift",
        "cross_video_relation_trust",
        "cross_video_left_logit",
        "cross_video_right_logit",
        "cross_video_left_residual",
        "cross_video_right_residual",
        "cross_video_semantic_left_margin",
        "cross_video_semantic_right_margin",
        "cross_video_audio_left_margin",
        "cross_video_audio_right_margin",
        "cross_video_vision_left_margin",
        "cross_video_vision_right_margin",
        "cross_video_crossmodal_agreement",
        "cross_video_negative_affinity",
        "cross_video_neutral_affinity",
        "cross_video_positive_affinity",
        "cross_video_neutral_logit",
        "cross_video_relation_scale",
        "context_reliability",
        "context_reject_weight",
        "previous_context_weight",
        "following_context_weight",
        "context_transition_scale",
        "context_transition_norm",
        "hyper_reliability",
        "layer_mix_scale",
        "neutral_boundary_probability",
        "neutral_boundary_mix",
        "neutral_evidence_shift",
        "neutral_distance_scale",
        "polar_uncertainty",
        "polarity_anchor_logit",
        "polarity_prior_logit",
        "polarity_prior_residual",
        "polarity_residual",
        "polarity_trust",
        "low_rank_interaction_mix",
        "low_rank_interaction_norm",
        "spectral_audio_gate",
        "spectral_vision_gate",
        "spectral_audio_velocity",
        "spectral_vision_velocity",
        "spectral_audio_acceleration",
        "spectral_vision_acceleration",
        "spectral_audio_low_energy",
        "spectral_vision_low_energy",
        "spectral_audio_high_energy",
        "spectral_vision_high_energy",
        "emotion_representation_reliability",
        "emotion_decision_reliability",
        "emotion_neutral_residual",
        "emotion_polarity_residual",
        "emotion_agreement",
        "subcenter_mix",
        "subcenter_scale",
        "conflict_neutral_shift",
        "conflict_mass",
        "ignorance_mass",
        "aligned_neutral_shift",
        "aligned_polarity_shift",
        "aligned_attention_entropy",
        "decomposition_residual_norm",
        "innovation_audio_error",
        "innovation_vision_error",
        "innovation_neutral_shift",
        "innovation_polarity_shift",
        "neutral_energy_poe_residual",
        "neutral_energy_disagreement",
        "neutral_energy_agreement",
        "neutral_energy_coherence",
        "neutral_energy_unanimous",
        "neutral_energy_boundary_gate",
        "neutral_energy_shift",
        "neutral_energy_text_probability",
        "neutral_energy_audio_probability",
        "neutral_energy_vision_probability",
        "neutral_energy_text_reliability",
        "neutral_energy_audio_reliability",
        "neutral_energy_vision_reliability",
        "neutral_energy_text_weight",
        "neutral_energy_audio_weight",
        "neutral_energy_vision_weight",
        "neutral_energy_text_contribution",
        "neutral_energy_audio_contribution",
        "neutral_energy_vision_contribution",
        "neutral_energy_text_neutral_energy",
        "neutral_energy_audio_neutral_energy",
        "neutral_energy_vision_neutral_energy",
        "neutral_energy_text_polar_energy",
        "neutral_energy_audio_polar_energy",
        "neutral_energy_vision_polar_energy",
    )
    diagnostics: dict[str, list[np.ndarray]] = {name: [] for name in diagnostic_names}
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            outputs = model(batch)
        probabilities.append(outputs["class_probabilities"].float().cpu().numpy())
        regression.append(outputs["regression"].float().cpu().numpy())
        targets.append(batch["class_label"].cpu().numpy())
        regression_targets.append(batch["regression_label"].float().cpu().numpy())
        identifiers.extend([str(value) for value in cpu_batch["id"]])
        batch_size = batch["class_label"].size(0)
        for name in diagnostic_names:
            value = outputs.get(name)
            if value is None:
                continue
            value = value.detach().float()
            if value.ndim == 0:
                value = value.expand(batch_size)
            if value.ndim == 1 and value.size(0) == batch_size:
                diagnostics[name].append(value.cpu().numpy())
    probability = np.concatenate(probabilities)
    regression_prediction = np.concatenate(regression)
    target = np.concatenate(targets)
    regression_target = np.concatenate(regression_targets)
    metrics = compute_metrics(target, probability, regression_target, regression_prediction)
    frame = pd.DataFrame(
        {
            "id": identifiers,
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression_prediction,
            "true_label": [CLASS_NAMES[index] for index in target],
            "true_intensity": regression_target,
        }
    )
    model_diagnostics: dict[str, dict[str, float]] = {}
    for name, chunks in diagnostics.items():
        if not chunks:
            continue
        values = np.concatenate(chunks)
        frame[name] = values
        model_diagnostics[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    metrics["model_diagnostics"] = model_diagnostics
    return metrics, frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained text anchor with optional visual fusion"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Warm-start matching parameters while allowing new architecture modules",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "resolved_config.json")
    logger = make_logger(output)
    seed_everything(int(cfg["seed"]))
    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    loaded = load_pickle(args.data)
    split_names = ["train", "valid"] + (["test"] if args.evaluate_test else [])
    # Unless final evaluation is explicitly requested, remove test before attaching
    # labels so training and model selection cannot materialize held-out targets.
    raw = attach_labels_from_excel(
        {name: loaded[name] for name in split_names}, args.labels
    )
    train_arrays = parse_split(raw, "train", "text_shared", require_labels=True)
    valid_arrays = parse_split(raw, "valid", "text_shared", require_labels=True)
    arrays = {"train": train_arrays, "valid": valid_arrays}
    if args.evaluate_test:
        arrays["test"] = parse_split(
            raw, "test", "text_shared", require_labels=True
        )
    normalizer = FeatureNormalizer(normalize_text=False, clip_value=10.0).fit(train_arrays)
    arrays = {name: normalizer.transform(value) for name, value in arrays.items()}
    train_arrays = arrays["train"]
    valid_arrays = arrays["valid"]

    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    emotion_tokenizer = None
    if bool(model_cfg.get("use_emotion_evidence", False)):
        emotion_tokenizer = AutoTokenizer.from_pretrained(
            str(model_cfg["emotion_pretrained_model"]),
            revision=model_cfg.get("emotion_revision"),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
        )
    max_length = int(cfg["data"].get("max_length", 128))
    context_window = int(cfg["data"].get("context_window", 0))
    context_mode = str(cfg["data"].get("context_mode", "previous"))
    tokenized = {
        name: attach_emotion_view(
            encode_split(
                tokenizer,
                split,
                max_length,
                context_window,
                context_mode=context_mode,
            ),
            emotion_tokenizer,
            split,
            int(model_cfg.get("emotion_max_length", max_length)),
        )
        for name, split in arrays.items()
    }
    datasets = {
        name: EncodedTextDataset(
            split,
            tokenized[name],
            include_video_pair_supervision=name == "train",
        )
        for name, split in arrays.items()
    }
    training_cfg = cfg["training"]
    train_generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        generator=train_generator,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    eval_batch_size = int(training_cfg.get("eval_batch_size", training_cfg["batch_size"]))
    eval_loaders = {
        name: DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=int(cfg["data"].get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    model = PretrainedTextFusionNet(model_cfg).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        current_state = model.state_dict()
        initial_model_name = str(
            initial.get("config", {}).get("model", {}).get("pretrained_model", "")
        )
        current_model_name = str(model_cfg["pretrained_model"])
        skip_encoder = bool(initial_model_name and initial_model_name != current_model_name)
        compatible_state = {
            key: value
            for key, value in initial["model_state"].items()
            if key in current_state
            and current_state[key].shape == value.shape
            and not (skip_encoder and key.startswith("text_encoder."))
        }
        skipped_shape = [
            key
            for key, value in initial["model_state"].items()
            if key in current_state and current_state[key].shape != value.shape
        ]
        incompatible = model.load_state_dict(compatible_state, strict=False)
        logger.info(
            "Warm start: %s | loaded=%d | skip_encoder=%s | missing=%s | skipped_shape=%s | unexpected=%s",
            args.init_checkpoint,
            len(compatible_state),
            skip_encoder,
            incompatible.missing_keys,
            skipped_shape,
            incompatible.unexpected_keys,
        )
    trainable_module_prefixes = tuple(
        str(value).strip()
        for value in training_cfg.get("trainable_module_prefixes", [])
        if str(value).strip()
    )
    if trainable_module_prefixes:
        for prefix in trainable_module_prefixes:
            model.get_submodule(prefix)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in trainable_module_prefixes
                )
            )
        logger.info(
            "Isolated trainable modules: %s",
            ", ".join(trainable_module_prefixes),
        )
    logger.info("Trainable parameters: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    class_weights = compute_class_weights(
        train_arrays.class_labels,
        power=float(training_cfg.get("class_weight_power", 0.5)),
        max_weight=float(training_cfg.get("class_weight_max", 3.0)),
    ).to(device)
    all_encoder_parameters = list(model.text_encoder.parameters())
    encoder_parameters = [
        parameter for parameter in all_encoder_parameters if parameter.requires_grad
    ]
    encoder_ids = {id(parameter) for parameter in all_encoder_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(training_cfg["encoder_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(training_cfg["head_learning_rate"]),
            },
        ],
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    accumulation = int(training_cfg.get("gradient_accumulation", 1))
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    total_updates = int(training_cfg["epochs"]) * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * float(training_cfg.get("warmup_ratio", 0.1))),
        num_training_steps=total_updates,
    )
    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_cfg = resolve_loss_config(cfg["loss"], train_arrays.class_labels)
    video_group_dro, video_group_dro_coefficient = build_video_group_dro(
        training_cfg,
        loss_cfg,
        datasets["train"],
        device,
    )
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_macro_f1 = -float("inf")
    patience = 0
    started = time.time()
    checkpoint_path = output / "best_model.pt"
    macro_checkpoint_path = output / "best_macro_model.pt"

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        if trainable_module_prefixes:
            encoder_frozen = True
            model.eval()
            for prefix in trainable_module_prefixes:
                model.get_submodule(prefix).train()
        else:
            encoder_frozen = epoch <= int(
                training_cfg.get("freeze_encoder_epochs", 0)
            )
            model.set_text_encoder_trainable(not encoder_frozen)
            model.train()
        logger.info("Epoch %02d | text encoder frozen=%s", epoch, encoder_frozen)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"pretrained {epoch:02d}", leave=False)
        for step, cpu_batch in enumerate(progress, start=1):
            batch = move_batch(cpu_batch, device)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(batch)
                loss, _ = compute_loss(outputs, batch, class_weights, loss_cfg)
                rdrop_coefficient = float(
                    loss_cfg.get("hierarchical_rdrop_consistency", 0.0)
                )
                best_view_coefficient = float(
                    loss_cfg.get("hierarchical_best_view_consistency", 0.0)
                )
                if rdrop_coefficient > 0.0 and best_view_coefficient > 0.0:
                    raise ValueError(
                        "symmetric R-Drop and best-view consistency are mutually exclusive"
                    )
                if rdrop_coefficient > 0.0 or best_view_coefficient > 0.0:
                    second_outputs = model(batch)
                    second_loss, _ = compute_loss(
                        second_outputs, batch, class_weights, loss_cfg
                    )
                    loss = 0.5 * (loss + second_loss)
                    if rdrop_coefficient > 0.0:
                        consistency_loss, _ = hierarchical_rdrop_consistency(
                            outputs, second_outputs, loss_cfg
                        )
                        consistency_coefficient = rdrop_coefficient
                    else:
                        consistency_loss, _ = hierarchical_best_view_consistency(
                            outputs, second_outputs, batch, loss_cfg
                        )
                        consistency_coefficient = best_view_coefficient
                    loss = loss + consistency_coefficient * consistency_loss
                    group_outputs = dict(outputs)
                    group_outputs["class_logits"] = 0.5 * (
                        outputs["class_logits"] + second_outputs["class_logits"]
                    )
                else:
                    group_outputs = outputs
                loss, _ = add_video_group_dro_loss(
                    loss,
                    group_outputs,
                    batch,
                    class_weights,
                    loss_cfg,
                    video_group_dro,
                    video_group_dro_coefficient,
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_update = step % accumulation == 0 or step == len(train_loader)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_cfg.get("grad_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += float(loss.detach())
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        valid_metrics, valid_predictions = evaluate(
            model, eval_loaders["valid"], device, amp
        )
        score = (
            float(cfg["selection"].get("accuracy", 0.4)) * valid_metrics["accuracy"]
            + float(cfg["selection"].get("macro_f1", 0.6)) * valid_metrics["macro_f1"]
        )
        video_group_diagnostics = (
            video_group_dro.finish_epoch()
            if video_group_dro is not None
            else {}
        )
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "valid": valid_metrics,
            "selection_score": score,
            **video_group_diagnostics,
        }
        history.append(record)
        save_json(history, output / "history.json")
        logger.info(
            "Epoch %02d | loss %.4f | valid accuracy %.4f macro-F1 %.4f MAE %.4f Pearson %.4f | score %.5f",
            epoch,
            record["train_loss"],
            valid_metrics["accuracy"],
            valid_metrics["macro_f1"],
            valid_metrics["mae"],
            valid_metrics["pearson"],
            score,
        )
        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(valid_metrics["macro_f1"])
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "config": copy.deepcopy(cfg),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "selection_score": score,
                    "selection_metric": "macro_f1",
                },
                macro_checkpoint_path,
            )
            valid_predictions.to_csv(
                output / "best_macro_valid_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if score > best_score + float(training_cfg.get("min_delta", 0.0)):
            best_score = score
            patience = 0
            atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "config": copy.deepcopy(cfg),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "selection_score": score,
                },
                checkpoint_path,
            )
            valid_predictions.to_csv(
                output / "best_valid_predictions.csv", index=False, encoding="utf-8-sig"
            )
        else:
            patience += 1
            if patience >= int(training_cfg.get("early_stopping_patience", 2)):
                logger.info("Early stopping at epoch %d", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    final: dict[str, Any] = {
        "scope": (
            "train_validation_and_explicit_final_test"
            if args.evaluate_test
            else "train_and_validation_only_no_test_access"
        ),
        "best_epoch": checkpoint["epoch"],
        "best_selection_score": checkpoint["selection_score"],
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    splits = ["train", "valid"] + (["test"] if args.evaluate_test else [])
    for split in splits:
        metrics, predictions = evaluate(model, eval_loaders[split], device, amp)
        final[split] = metrics
        predictions.to_csv(
            output / f"{split}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    save_json(final, output / "final_metrics.json")
    logger.info("Final metrics: %s", final)


if __name__ == "__main__":
    main()
