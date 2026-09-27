"""Reliability-calibrated hierarchical fusion for Problem 3.

The implementation adapts, rather than copies, four published ideas to the
competition's aligned pre-extracted features and additive explanation contract:

* CICA (CVPR 2026): intrinsic-structure attention and confidence/uncertainty
  coupling (CVF Open Access paper by Jiang et al.).
* CMAD and G2D (ICCV 2025): missing-modality learning, correlation preservation,
  and protection of weak modality gradients.
* MIDAS (IEEE TPAMI 2026, doi:10.1109/TPAMI.2026.3713694): shared/exclusive
  factorization and posterior-uncertainty-modulated attention.
* EUAR (ACM MM 2024, doi:10.1145/3664647.3680949): conditional routing under
  heterogeneous modality noise.

All public output keys used by training, inference, and explanation remain
backward-compatible.  Checkpoint parameters are intentionally not compatible
with earlier model versions and therefore require retraining.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import MODALITIES, PAIR_MODALITY_INDICES, PAIR_NAMES


def masked_sparsemax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Sparsemax with padding support (Martins & Astudillo, 2016)."""
    original_dtype = logits.dtype
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    mask = mask.bool()
    very_negative = torch.finfo(logits.dtype).min / 2
    z = logits.masked_fill(~mask, very_negative)
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted = torch.sort(z, dim=dim, descending=True).values
    z_cumsum = z_sorted.cumsum(dim)
    size = logits.size(dim)
    ranks_shape = [1] * logits.ndim
    ranks_shape[dim] = size
    ranks = torch.arange(1, size + 1, device=logits.device, dtype=logits.dtype).view(
        ranks_shape
    )
    support = (1 + ranks * z_sorted) > z_cumsum
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau_sum = z_cumsum.gather(dim, (k - 1).long())
    tau = (tau_sum - 1) / k.to(logits.dtype)
    probs = torch.clamp(z - tau, min=0) * mask.to(logits.dtype)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return probs.to(original_dtype)


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class SinusoidalPositionEncoding(nn.Module):
    """Parameter-free temporal positions to improve generalization on 3.4k samples."""

    def __init__(self, d_model: int, max_length: int = 512) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_length = int(max_length)
        self.register_buffer(
            "encoding",
            self._build(self.max_length, self.d_model),
            persistent=False,
        )

    @staticmethod
    def _build(length: int, d_model: int) -> Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(1, length, d_model, dtype=torch.float32)
        encoding[0, :, 0::2] = torch.sin(position * frequency)
        if d_model > 1:
            encoding[0, :, 1::2] = torch.cos(
                position * frequency[: encoding[0, :, 1::2].shape[-1]]
            )
        return encoding

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        if length <= self.encoding.size(1):
            encoding = self.encoding[:, :length]
        else:
            encoding = self._build(length, self.d_model).to(x.device)
        return encoding.to(device=x.device, dtype=x.dtype)


class ReliabilityAwareComponentGate(nn.Module):
    """Score six evidence streams with a shared, low-parameter reliability rule."""

    def __init__(
        self,
        d_model: int,
        num_components: int,
        dropout: float,
        temperature: float = 1.0,
        gate_floor: float = 0.05,
        component_dropout: float = 0.10,
        text_anchor_prior: float = 0.80,
        reliability_logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("component gate temperature must be positive")
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError("component gate floor must lie in [0,1)")
        if not 0.0 <= component_dropout < 1.0:
            raise ValueError("component dropout must lie in [0,1)")
        if reliability_logit_scale <= 0.0:
            raise ValueError("reliability_logit_scale must be positive")
        self.temperature = float(temperature)
        self.gate_floor = float(gate_floor)
        self.component_dropout = float(component_dropout)
        self.type_embeddings = nn.Parameter(torch.zeros(num_components, d_model))
        nn.init.normal_(self.type_embeddings, std=0.02)
        self.scorer = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Text is the most stable stream for utterance-level sentiment.  This
        # is only a learnable initialization prior: non-verbal evidence can
        # override it whenever the sample supports doing so.
        initial_bias = torch.zeros(num_components)
        if num_components >= 1:
            initial_bias[0] = float(text_anchor_prior)
        if num_components > 3:
            initial_bias[3:] = -0.5 * float(text_anchor_prior)
        self.type_bias = nn.Parameter(initial_bias)
        self.reliability_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(reliability_logit_scale))))
        )

    def _effective_availability(self, available: Tensor) -> Tensor:
        if not self.training or self.component_dropout <= 0.0:
            return available
        keep = torch.rand(available.shape, device=available.device) >= self.component_dropout
        effective = available & keep
        empty = ~effective.any(dim=1)
        if empty.any():
            effective[empty] = available[empty]
        return effective

    def forward(
        self,
        components: Tensor,
        available: Tensor,
        reliability: Tensor | None = None,
    ) -> Tensor:
        available = available.bool()
        has_component = available.any(dim=1, keepdim=True)
        effective = self._effective_availability(available)
        empty = ~effective.any(dim=1)
        if empty.any():
            effective[empty, 0] = True
        available_f = effective.unsqueeze(-1).to(components.dtype)
        count = available_f.sum(dim=1).clamp_min(1.0)
        global_context = (components * available_f).sum(dim=1) / count
        global_expanded = global_context.unsqueeze(1).expand_as(components)
        typed_components = components + self.type_embeddings.unsqueeze(0).to(components.dtype)
        features = torch.cat(
            [
                typed_components,
                global_expanded,
                (typed_components - global_expanded).abs(),
                typed_components * global_expanded,
            ],
            dim=-1,
        )
        logits = self.scorer(features).squeeze(-1) + self.type_bias
        if reliability is not None:
            reliability_bias = torch.log(reliability.float().clamp(0.05, 4.0))
            reliability_bias = reliability_bias.to(logits.dtype)
            logits = logits + F.softplus(self.reliability_scale_raw) * reliability_bias
        logits = logits.masked_fill(~effective, torch.finfo(logits.dtype).min)
        gates = torch.softmax(logits / self.temperature, dim=-1)
        if self.gate_floor > 0.0:
            uniform = effective.to(gates.dtype)
            uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
            gates = (1.0 - self.gate_floor) * gates + self.gate_floor * uniform
        return gates * has_component.to(gates.dtype)


class ReliabilityModulatedAttentionLayer(nn.Module):
    """Self-attention whose keys are softened by component reliability."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        ff_multiplier: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        available: Tensor,
        reliability: Tensor,
    ) -> Tensor:
        batch_size, components, d_model = x.shape
        normalized = self.norm1(x)
        qkv = self.qkv(normalized).reshape(
            batch_size, components, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # MIDAS-style reliability modulation acts on keys.  The square root
        # avoids flattening the attention distribution and preserves gradients.
        key_scale = reliability.float().clamp(0.05, 4.0).sqrt()
        k = k * key_scale[:, None, :, None].to(k.dtype)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = available[:, None, None, :].bool()
        scores_fp32 = scores.float().masked_fill(~key_mask, -1e4)
        attention = torch.softmax(scores_fp32, dim=-1).to(scores.dtype)
        attention = self.attention_dropout(attention)
        update = torch.matmul(attention, v).transpose(1, 2).reshape(
            batch_size, components, d_model
        )
        x = x + self.residual_dropout(self.output(update))
        x = x + self.ffn(self.norm2(x))
        return x * available.unsqueeze(-1).to(x.dtype)


class ComponentInteractionMixer(nn.Module):
    """Reliability-modulated set transformer over six evidence tokens."""

    def __init__(
        self,
        d_model: int,
        num_components: int,
        n_heads: int,
        dropout: float,
        ff_multiplier: int = 2,
        n_layers: int = 1,
        initial_logit: float = -1.5,
    ) -> None:
        super().__init__()
        self.type_embeddings = nn.Parameter(torch.zeros(num_components, d_model))
        nn.init.normal_(self.type_embeddings, std=0.02)
        self.layers = nn.ModuleList(
            [
                ReliabilityModulatedAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    ff_multiplier=ff_multiplier,
                )
                for _ in range(n_layers)
            ]
        )
        self.residual_scale_logit = nn.Parameter(torch.tensor(float(initial_logit)))

    def forward(
        self,
        components: Tensor,
        available: Tensor,
        reliability: Tensor | None = None,
    ) -> Tensor:
        available = available.bool()
        safe_available = available.clone()
        empty = ~safe_available.any(dim=1)
        if empty.any():
            safe_available[empty, 0] = True
        available_f = available.unsqueeze(-1).to(components.dtype)
        tokens = (
            components + self.type_embeddings.unsqueeze(0).to(components.dtype)
        ) * available_f
        if reliability is None:
            reliability = torch.ones_like(available, dtype=components.dtype)
        reliability = torch.where(
            safe_available,
            reliability.to(components.dtype),
            torch.zeros_like(reliability, dtype=components.dtype),
        )
        encoded = tokens
        for layer in self.layers:
            encoded = layer(encoded, safe_available, reliability)
        delta = (encoded - tokens) * available_f
        scale = torch.sigmoid(self.residual_scale_logit)
        return (components + scale * delta) * available_f


class SharedExclusiveDecomposer(nn.Module):
    """Disentangle common sentiment semantics from modality-specific residue."""

    def __init__(self, d_model: int, dropout: float, initial_logit: float = -1.5) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
        )
        self.private = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model, bias=False),
                    nn.GELU(),
                )
                for _ in MODALITIES
            ]
        )
        self.unary_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in MODALITIES]
        )
        self.pair_adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pair_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in PAIR_NAMES])
        self.residual_scale_logit = nn.Parameter(torch.tensor(float(initial_logit)))

    def forward(
        self,
        components: Tensor,
        available: Tensor,
    ) -> tuple[Tensor, Tensor]:
        unary = components[:, : len(MODALITIES)]
        shared = self.shared(unary)
        private = torch.stack(
            [self.private[idx](unary[:, idx]) for idx in range(len(MODALITIES))],
            dim=1,
        )
        # Exact sample-wise orthogonalization is a stable, loss-free proxy for
        # shared/exclusive mutual-information disentanglement on small data.
        projection = (private * shared).sum(dim=-1, keepdim=True)
        projection = projection / shared.square().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        private_orthogonal = private - projection * shared
        scale = torch.sigmoid(self.residual_scale_logit)
        unary_out = torch.stack(
            [
                self.unary_norms[idx](
                    unary[:, idx]
                    + scale * (shared[:, idx] + private_orthogonal[:, idx])
                )
                for idx in range(len(MODALITIES))
            ],
            dim=1,
        )
        pair_out = []
        for pair_idx, (left, right) in enumerate(PAIR_MODALITY_INDICES):
            consensus = 0.5 * (shared[:, left] + shared[:, right])
            pair_out.append(
                self.pair_norms[pair_idx](
                    components[:, len(MODALITIES) + pair_idx]
                    + scale * self.pair_adapter(consensus)
                )
            )
        output = torch.cat([unary_out, torch.stack(pair_out, dim=1)], dim=1)
        output = output * available.unsqueeze(-1).to(output.dtype)
        orthogonality = F.cosine_similarity(
            shared.float(), private_orthogonal.float(), dim=-1, eps=1e-6
        ).abs()
        return output, orthogonality


class ContextConditionedDecision(nn.Module):
    """Predict additive evidence after every component has seen global context."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        initial_logit: float = -1.5,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 4),
        )
        # Start close to the proven additive evidence model and learn only the
        # cross-component correction justified by the data.
        self.residual_scale_logit = nn.Parameter(torch.tensor(float(initial_logit)))

    def forward(
        self,
        components: Tensor,
        gates: Tensor,
        available: Tensor,
    ) -> tuple[Tensor, Tensor]:
        global_context = torch.sum(gates.unsqueeze(-1) * components, dim=1)
        global_expanded = global_context.unsqueeze(1).expand_as(components)
        features = torch.cat(
            [
                components,
                global_expanded,
                (components - global_expanded).abs(),
                components * global_expanded,
            ],
            dim=-1,
        )
        correction = self.network(features)
        correction = correction * available.unsqueeze(-1).to(correction.dtype)
        scale = torch.sigmoid(self.residual_scale_logit)
        regression_delta = scale * 1.5 * torch.tanh(correction[..., 0])
        classification_delta = scale * correction[..., 1:]
        return regression_delta, classification_delta


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_multiplier: int,
        dropout: float,
        kernel_sizes: List[int],
        max_sequence_length: int = 512,
        input_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if not kernel_sizes or any(kernel <= 0 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("conv_kernel_sizes must contain positive odd integers")
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionEncoding(d_model, max_sequence_length)
        self.position_scale = nn.Parameter(torch.tensor(0.10))
        self.input_dropout = nn.Dropout(input_dropout)
        self.local_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=k,
                    padding=k // 2,
                    groups=d_model,
                )
                for k in kernel_sizes
            ]
        )
        self.local_mix = nn.Sequential(
            nn.LayerNorm(d_model * len(kernel_sizes)),
            nn.Linear(d_model * len(kernel_sizes), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # A small initial residual scale makes the local branch useful without
        # allowing it to dominate the pretrained input features at epoch one.
        self.local_scale_logit = nn.Parameter(torch.tensor(-1.50))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        h = self.projection(self.input_norm(x))
        h = self.input_dropout(h + self.position_scale * self.position(h)) * mask_f
        conv_input = h.transpose(1, 2)
        local = [F.gelu(conv(conv_input)).transpose(1, 2) for conv in self.local_convs]
        local_scale = torch.sigmoid(self.local_scale_logit)
        h = h + local_scale * self.local_mix(torch.cat(local, dim=-1)) * mask_f
        h = self.transformer(h, src_key_padding_mask=~mask.bool())
        return self.output_norm(self.dropout(h)) * mask_f


class ConflictAwarePairwiseContext(nn.Module):
    """Separate cross-modal consensus from disagreement at each aligned position."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        use_conflict: bool = True,
        context_initial_logit: float = -2.0,
    ) -> None:
        super().__init__()
        self.use_conflict = use_conflict
        self.common_projections = nn.ModuleDict(
            {m: nn.Linear(d_model, d_model, bias=False) for m in MODALITIES}
        )
        self.common_norms = nn.ModuleDict(
            {m: nn.LayerNorm(d_model) for m in MODALITIES}
        )
        self.conflict_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reliability = nn.Sequential(
            nn.Linear(3 * d_model + 1, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.pair_embeddings = nn.Parameter(torch.zeros(len(PAIR_NAMES), d_model))
        nn.init.normal_(self.pair_embeddings, std=0.02)
        self.pair_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in PAIR_NAMES])
        self.target_adapters = nn.ModuleDict(
            {m: nn.Linear(d_model, d_model, bias=False) for m in MODALITIES}
        )
        self.target_norms = nn.ModuleDict({m: nn.LayerNorm(d_model) for m in MODALITIES})
        self.context_scale_logits = nn.Parameter(
            torch.full((len(MODALITIES),), float(context_initial_logit))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        streams: Mapping[str, Tensor],
        masks: Mapping[str, Tensor],
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
        common = {
            m: self.common_norms[m](self.common_projections[m](streams[m]))
            * masks[m].unsqueeze(-1).to(streams[m].dtype)
            for m in MODALITIES
        }
        pair_features: Dict[str, Tensor] = {}
        conflict_scores: Dict[str, Tensor] = {}
        reliabilities: Dict[str, Tensor] = {}
        pair_masks: Dict[str, Tensor] = {}
        updates = {m: torch.zeros_like(streams[m]) for m in MODALITIES}
        update_counts = {m: 0 for m in MODALITIES}

        for pair_idx, (left_idx, right_idx) in enumerate(PAIR_MODALITY_INDICES):
            left, right = MODALITIES[left_idx], MODALITIES[right_idx]
            name = PAIR_NAMES[pair_idx]
            pair_mask = masks[left] & masks[right]
            similarity = F.cosine_similarity(common[left], common[right], dim=-1, eps=1e-8)
            similarity = similarity * pair_mask.to(similarity.dtype)
            agreement = ((similarity + 1.0) / 2.0).clamp(0.0, 1.0)
            conflict_strength = 1.0 - agreement
            consensus = agreement.unsqueeze(-1) * (common[left] + common[right]) / 2.0
            signed_difference = common[left] - common[right]
            if self.use_conflict:
                conflict = conflict_strength.unsqueeze(-1) * self.conflict_projection(
                    torch.cat([signed_difference.abs(), signed_difference], dim=-1)
                )
            else:
                conflict = torch.zeros_like(consensus)
            pair = self.pair_norm[pair_idx](
                consensus + conflict + self.pair_embeddings[pair_idx].view(1, 1, -1)
            )
            reliability_input = torch.cat(
                [streams[left], streams[right], pair, similarity.unsqueeze(-1)], dim=-1
            )
            reliability = torch.sigmoid(self.reliability(reliability_input)).squeeze(-1)
            reliability = reliability * pair_mask.to(reliability.dtype)
            pair = pair * reliability.unsqueeze(-1)
            pair = pair * pair_mask.unsqueeze(-1).to(pair.dtype)

            pair_features[name] = pair
            conflict_scores[name] = conflict_strength * pair_mask.to(conflict_strength.dtype)
            reliabilities[name] = reliability
            pair_masks[name] = pair_mask
            updates[left] = updates[left] + pair
            updates[right] = updates[right] + pair
            update_counts[left] += 1
            update_counts[right] += 1

        contextualized: Dict[str, Tensor] = {}
        for modality_idx, modality in enumerate(MODALITIES):
            update = updates[modality] / max(update_counts[modality], 1)
            mask_f = masks[modality].unsqueeze(-1).to(update.dtype)
            context_scale = torch.sigmoid(self.context_scale_logits[modality_idx])
            contextualized[modality] = self.target_norms[modality](
                streams[modality]
                + context_scale * self.dropout(self.target_adapters[modality](update))
            ) * mask_f
        return contextualized, pair_features, conflict_scores, reliabilities, pair_masks


class EvidenceHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        sparse_attention: bool,
        temperature: float,
        attention_soft_gradient: float = 0.15,
        structure_kernel_size: int = 5,
        structure_strength: float = 0.35,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("evidence_temperature must be positive")
        if not 0.0 <= attention_soft_gradient <= 1.0:
            raise ValueError("attention_soft_gradient must lie in [0,1]")
        if structure_kernel_size <= 0 or structure_kernel_size % 2 == 0:
            raise ValueError("structure_kernel_size must be a positive odd integer")
        if structure_strength <= 0.0:
            raise ValueError("structure_strength must be positive")
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.regression = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.classification = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )
        self.sparse_attention = sparse_attention
        self.temperature = temperature
        self.attention_soft_gradient = attention_soft_gradient
        self.local_structure = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=structure_kernel_size,
            padding=structure_kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.saliency = nn.Linear(d_model, 1)
        initial_structure_raw = math.log(math.expm1(float(structure_strength)))
        self.geometry_scale_raw = nn.Parameter(torch.tensor(initial_structure_raw))
        self.saliency_scale_raw = nn.Parameter(torch.tensor(initial_structure_raw))

    def forward(self, h: Tensor, mask: Tensor) -> Dict[str, Tensor]:
        mask_f = mask.unsqueeze(-1).to(h.dtype)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        query = F.normalize((h * mask_f).sum(dim=1) / count, dim=-1, eps=1e-6)
        local_context = self.local_structure((h * mask_f).transpose(1, 2)).transpose(1, 2)
        local_context = F.normalize(local_context, dim=-1, eps=1e-6)
        geometry = torch.sum(local_context * query.unsqueeze(1), dim=-1)
        saliency = torch.sigmoid(self.saliency(h).squeeze(-1)).clamp_min(1e-4)
        # CICA-inspired intrinsic structure modulation: attention sees both
        # content relevance and whether each temporal key is locally coherent.
        scores = (
            self.attention(h).squeeze(-1)
            + F.softplus(self.geometry_scale_raw) * geometry
            + F.softplus(self.saliency_scale_raw) * torch.log(saliency)
        ) / self.temperature
        if self.sparse_attention:
            sparse_weights = masked_sparsemax(scores, mask)
            if self.training and self.attention_soft_gradient > 0.0:
                # Exact sparse evidence in the forward pass, plus a small
                # smooth surrogate gradient that prevents early dead tokens.
                soft_weights = masked_softmax(scores, mask)
                weights = sparse_weights + self.attention_soft_gradient * (
                    soft_weights - soft_weights.detach()
                )
            else:
                weights = sparse_weights
        else:
            weights = masked_softmax(scores, mask)
        local_regression = 3.0 * torch.tanh(self.regression(h).squeeze(-1) / 3.0)
        local_classification = self.classification(h)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        modality_regression = torch.sum(weights * local_regression, dim=1)
        modality_classification = torch.sum(
            weights.unsqueeze(-1) * local_classification, dim=1
        )
        # A label-free predictive posterior: confidence comes from the pooled
        # class margin, while uncertainty measures disagreement among temporal
        # predictions.  Both are detached later before routing, as in a frozen
        # confidence perceiver, so the predictor cannot game its own gate.
        class_probability = torch.softmax(modality_classification.float(), dim=-1)
        entropy = -(
            class_probability * class_probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(3.0)
        confidence = (1.0 - entropy).clamp(0.0, 1.0)
        regression_variance = torch.sum(
            weights.float()
            * (local_regression.float() - modality_regression.float().unsqueeze(-1)).square(),
            dim=1,
        ) / 9.0
        local_probability = torch.softmax(local_classification.float(), dim=-1)
        classification_variance = torch.sum(
            weights.float().unsqueeze(-1)
            * (local_probability - class_probability.unsqueeze(1)).square(),
            dim=(1, 2),
        )
        uncertainty = (
            0.5 * regression_variance.clamp_min(0.0).sqrt()
            + 0.5 * classification_variance.clamp_min(0.0).sqrt()
        ).clamp(0.0, 1.0)
        return {
            "weights": weights,
            "local_regression": local_regression,
            "local_classification": local_classification,
            "pooled": pooled,
            "modality_regression": modality_regression,
            "modality_classification": modality_classification,
            "confidence": confidence.to(h.dtype),
            "uncertainty": uncertainty.to(h.dtype),
        }


class HAFusionNet(nn.Module):
    """C3-HAFusion: conflict-aware, conservative and counterfactually calibrated."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dims = cfg["input_dims"]
        d_model = int(cfg["d_model"])
        dropout = float(cfg.get("dropout", 0.2))
        self.encoders = nn.ModuleDict(
            {
                modality: MultiScaleTemporalEncoder(
                    input_dim=int(dims[modality]),
                    d_model=d_model,
                    n_heads=int(cfg["n_heads"]),
                    n_layers=int(cfg["n_layers"]),
                    ff_multiplier=int(cfg.get("ff_multiplier", 4)),
                    dropout=dropout,
                    kernel_sizes=list(cfg.get("conv_kernel_sizes", [3, 5])),
                    max_sequence_length=int(cfg.get("max_sequence_length", 512)),
                    input_dropout=float(cfg.get("input_dropout", 0.10)),
                )
                for modality in MODALITIES
            }
        )
        self.use_cross_context = bool(cfg.get("use_cross_context", True))
        self.use_pairwise_evidence = bool(cfg.get("use_pairwise_evidence", True))
        self.fixed_modality_gate = bool(cfg.get("fixed_modality_gate", False))
        enabled = tuple(cfg.get("enabled_modalities", MODALITIES))
        unknown = set(enabled) - set(MODALITIES)
        if unknown or not enabled:
            raise ValueError(f"Invalid enabled_modalities: {enabled}")
        self.enabled_modalities = enabled
        self.register_buffer(
            "enabled_mask",
            torch.tensor([m in enabled for m in MODALITIES], dtype=torch.bool),
            persistent=False,
        )
        pair_enabled = [
            self.use_pairwise_evidence
            and MODALITIES[left] in enabled
            and MODALITIES[right] in enabled
            for left, right in PAIR_MODALITY_INDICES
        ]
        self.register_buffer(
            "pair_enabled_mask", torch.tensor(pair_enabled, dtype=torch.bool), persistent=False
        )
        pair_allocation = torch.zeros(len(PAIR_NAMES), len(MODALITIES))
        for pair_idx, (left, right) in enumerate(PAIR_MODALITY_INDICES):
            pair_allocation[pair_idx, left] = 0.5
            pair_allocation[pair_idx, right] = 0.5
        self.register_buffer("pair_allocation", pair_allocation, persistent=False)
        self.conflict_context = ConflictAwarePairwiseContext(
            d_model=d_model,
            dropout=dropout,
            use_conflict=bool(cfg.get("use_conflict_context", True)),
            context_initial_logit=float(cfg.get("context_initial_logit", -2.0)),
        )
        self.evidence_heads = nn.ModuleDict(
            {
                modality: EvidenceHead(
                    d_model=d_model,
                    dropout=dropout,
                    sparse_attention=bool(cfg.get("sparse_attention", True)),
                    temperature=float(cfg.get("evidence_temperature", 1.0)),
                    attention_soft_gradient=float(
                        cfg.get("attention_soft_gradient", 0.15)
                    ),
                    structure_kernel_size=int(cfg.get("structure_kernel_size", 5)),
                    structure_strength=float(cfg.get("structure_strength", 0.35)),
                )
                for modality in MODALITIES
            }
        )
        self.pair_evidence_head = EvidenceHead(
            d_model=d_model,
            dropout=dropout,
            sparse_attention=bool(cfg.get("sparse_attention", True)),
            temperature=float(cfg.get("evidence_temperature", 1.0)),
            attention_soft_gradient=float(cfg.get("attention_soft_gradient", 0.15)),
            structure_kernel_size=int(cfg.get("structure_kernel_size", 5)),
            structure_strength=float(cfg.get("structure_strength", 0.35)),
        )
        num_components = len(MODALITIES) + len(PAIR_NAMES)
        self.shared_exclusive = SharedExclusiveDecomposer(
            d_model=d_model,
            dropout=dropout,
            initial_logit=float(cfg.get("disentangle_initial_logit", -1.5)),
        )
        self.uncertainty_temperature = float(cfg.get("uncertainty_temperature", 2.0))
        if self.uncertainty_temperature <= 0.0:
            raise ValueError("uncertainty_temperature must be positive")
        self.detach_reliability = bool(cfg.get("detach_reliability", True))
        self.component_mixer = ComponentInteractionMixer(
            d_model=d_model,
            num_components=num_components,
            n_heads=int(cfg["n_heads"]),
            dropout=dropout,
            ff_multiplier=int(cfg.get("component_ff_multiplier", 2)),
            n_layers=int(cfg.get("component_mixer_layers", 1)),
            initial_logit=float(cfg.get("component_mixer_initial_logit", -1.5)),
        )
        self.component_gate = ReliabilityAwareComponentGate(
            d_model=d_model,
            num_components=num_components,
            dropout=dropout,
            temperature=float(cfg.get("component_gate_temperature", 1.0)),
            gate_floor=float(cfg.get("component_gate_floor", 0.05)),
            component_dropout=float(cfg.get("component_dropout", 0.10)),
            text_anchor_prior=float(cfg.get("text_anchor_prior", 0.80)),
            reliability_logit_scale=float(cfg.get("reliability_logit_scale", 1.0)),
        )
        self.context_decision = ContextConditionedDecision(
            d_model=d_model,
            dropout=dropout,
            initial_logit=float(cfg.get("decision_initial_logit", -1.5)),
        )
        self.class_bias = nn.Parameter(torch.zeros(3))
        self.regression_bias = nn.Parameter(torch.zeros(()))
        self.class_from_regression_logit = nn.Parameter(
            torch.tensor(float(cfg.get("class_from_regression_logit", -1.0)))
        )
        self.regression_from_class_logit = nn.Parameter(
            torch.tensor(float(cfg.get("regression_from_class_logit", -1.5)))
        )
        self.register_buffer(
            "polarity_basis",
            torch.tensor([-1.0, 0.0, 1.0]),
            persistent=False,
        )

    def forward(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        declared_masks = {m: batch[f"{m}_mask"].bool() for m in MODALITIES}
        # Zeroing a modality is how the training pipeline computes a
        # counterfactual.  Treat zero feature rows as genuinely absent so the
        # projection bias and positional encoding cannot recreate a supposedly
        # ablated signal.
        masks = {
            m: declared_masks[m]
            & torch.isfinite(batch[m]).all(dim=-1)
            & (batch[m].detach().abs().amax(dim=-1) > 1e-12)
            for m in MODALITIES
        }
        encoder_masks: Dict[str, Tensor] = {}
        for modality in MODALITIES:
            safe_mask = masks[modality].clone()
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
            encoder_masks[modality] = safe_mask
        streams = {
            m: self.encoders[m](batch[m], encoder_masks[m])
            * masks[m].any(dim=1)[:, None, None].to(batch[m].dtype)
            for m in MODALITIES
        }
        streams = {
            m: streams[m] if m in self.enabled_modalities else torch.zeros_like(streams[m])
            for m in MODALITIES
        }
        context_masks = {
            m: masks[m] if m in self.enabled_modalities else torch.zeros_like(masks[m])
            for m in MODALITIES
        }
        (
            contextualized,
            pair_features,
            conflict_scores_dict,
            pair_reliability_dict,
            pair_masks_dict,
        ) = self.conflict_context(streams, context_masks)
        if self.use_cross_context:
            streams = contextualized
        evidence = {
            m: self.evidence_heads[m](streams[m], encoder_masks[m])
            for m in MODALITIES
        }

        pair_evidence: Dict[str, Dict[str, Tensor]] = {}
        pair_masks_safe: Dict[str, Tensor] = {}
        for name in PAIR_NAMES:
            safe_mask = pair_masks_dict[name].clone()
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
            pair_masks_safe[name] = safe_mask
            pair_evidence[name] = self.pair_evidence_head(pair_features[name], safe_mask)

        pooled_components = torch.stack(
            [evidence[m]["pooled"] for m in MODALITIES]
            + [pair_evidence[name]["pooled"] for name in PAIR_NAMES],
            dim=1,
        )
        base_modality_regression = torch.stack(
            [evidence[m]["modality_regression"] for m in MODALITIES], dim=1
        )
        base_modality_classification = torch.stack(
            [evidence[m]["modality_classification"] for m in MODALITIES], dim=1
        )
        base_pair_regression = torch.stack(
            [pair_evidence[name]["modality_regression"] for name in PAIR_NAMES], dim=1
        )
        base_pair_classification = torch.stack(
            [pair_evidence[name]["modality_classification"] for name in PAIR_NAMES], dim=1
        )
        base_component_regression = torch.cat(
            [base_modality_regression, base_pair_regression], dim=1
        )
        base_component_classification = torch.cat(
            [base_modality_classification, base_pair_classification], dim=1
        )
        unary_confidence = torch.stack(
            [evidence[m]["confidence"] for m in MODALITIES], dim=1
        )
        unary_uncertainty = torch.stack(
            [evidence[m]["uncertainty"] for m in MODALITIES], dim=1
        )
        pair_confidence = torch.stack(
            [pair_evidence[name]["confidence"] for name in PAIR_NAMES], dim=1
        )
        pair_uncertainty = torch.stack(
            [pair_evidence[name]["uncertainty"] for name in PAIR_NAMES], dim=1
        )
        unary_available = torch.stack(
            [
                masks[m].any(dim=1) & self.enabled_mask[idx]
                for idx, m in enumerate(MODALITIES)
            ],
            dim=1,
        )
        pair_available = torch.stack(
            [
                pair_masks_dict[name].any(dim=1) & self.pair_enabled_mask[idx]
                for idx, name in enumerate(PAIR_NAMES)
            ],
            dim=1,
        )
        component_available = torch.cat([unary_available, pair_available], dim=1)
        pair_quality = torch.stack(
            [
                pair_reliability_dict[name].sum(dim=1)
                / pair_masks_dict[name].sum(dim=1).clamp_min(1).to(
                    pair_reliability_dict[name].dtype
                )
                for name in PAIR_NAMES
            ],
            dim=1,
        )
        pair_confidence = 0.75 * pair_confidence + 0.25 * pair_quality
        pair_uncertainty = 0.75 * pair_uncertainty + 0.25 * (1.0 - pair_quality)
        component_confidence = torch.cat(
            [unary_confidence, pair_confidence], dim=1
        ).float()
        component_uncertainty = torch.cat(
            [unary_uncertainty, pair_uncertainty], dim=1
        ).float()
        available_f = component_available.to(component_uncertainty.dtype)
        uncertainty_center = (
            (component_uncertainty * available_f).sum(dim=1, keepdim=True)
            / available_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        midas_weight = torch.sigmoid(
            -self.uncertainty_temperature
            * (component_uncertainty - uncertainty_center)
        )
        cica_reliability = F.relu(
            1.0 + component_confidence - component_uncertainty
        )
        component_reliability = (
            cica_reliability * torch.sqrt((2.0 * midas_weight).clamp_min(1e-6))
        ).clamp(0.05, 2.5)
        component_reliability = component_reliability * available_f
        if self.detach_reliability:
            component_reliability = component_reliability.detach()
        pooled_components = (
            pooled_components
            * component_available.unsqueeze(-1).to(pooled_components.dtype)
        )
        disentangled_components, shared_private_orthogonality = self.shared_exclusive(
            pooled_components, component_available
        )
        mixed_components = self.component_mixer(
            disentangled_components,
            component_available,
            component_reliability,
        )
        if self.fixed_modality_gate:
            component_gates = component_available.to(pooled_components.dtype)
            component_gates = component_gates / component_gates.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        else:
            component_gates = self.component_gate(
                mixed_components,
                component_available,
                component_reliability,
            )
        unary_gates = component_gates[:, :3]
        pair_gates = component_gates[:, 3:]
        modality_gates = unary_gates + pair_gates @ self.pair_allocation

        regression_delta, classification_delta = self.context_decision(
            mixed_components, component_gates, component_available
        )
        uncoupled_component_regression = base_component_regression + regression_delta
        uncoupled_component_classification = (
            base_component_classification + classification_delta
        )
        class_from_regression = 0.5 * torch.sigmoid(
            self.class_from_regression_logit
        )
        regression_from_class = 0.25 * torch.sigmoid(
            self.regression_from_class_logit
        )
        component_regression = (
            uncoupled_component_regression
            + regression_from_class
            * (
                uncoupled_component_classification[..., 2]
                - uncoupled_component_classification[..., 0]
            )
        )
        component_classification = (
            uncoupled_component_classification
            + class_from_regression
            * uncoupled_component_regression.unsqueeze(-1)
            * self.polarity_basis.to(uncoupled_component_regression.dtype)
        )
        modality_regression, pair_regression = component_regression.split(3, dim=1)
        modality_classification, pair_classification = (
            component_classification[:, :3],
            component_classification[:, 3:],
        )
        unary_regression_contributions = unary_gates * modality_regression
        unary_classification_contributions = (
            unary_gates.unsqueeze(-1) * modality_classification
        )
        pair_regression_contributions = pair_gates * pair_regression
        pair_classification_contributions = pair_gates.unsqueeze(-1) * pair_classification
        regression_contributions = (
            unary_regression_contributions
            + pair_regression_contributions @ self.pair_allocation
        )
        classification_contributions = (
            unary_classification_contributions
            + torch.einsum(
                "bpc,pm->bmc", pair_classification_contributions, self.pair_allocation
            )
        )
        regression_raw = self.regression_bias + regression_contributions.sum(dim=1)
        regression = 3.0 * torch.tanh(regression_raw / 3.0)
        class_logits = self.class_bias + classification_contributions.sum(dim=1)

        temporal_weights = torch.stack([evidence[m]["weights"] for m in MODALITIES], dim=1)
        local_regression_evidence = torch.stack(
            [evidence[m]["local_regression"] for m in MODALITIES], dim=1
        )
        local_classification_evidence = torch.stack(
            [evidence[m]["local_classification"] for m in MODALITIES], dim=1
        )
        pair_temporal_weights = torch.stack(
            [pair_evidence[name]["weights"] for name in PAIR_NAMES], dim=1
        )
        pair_local_regression_evidence = torch.stack(
            [pair_evidence[name]["local_regression"] for name in PAIR_NAMES], dim=1
        )
        pair_local_classification_evidence = torch.stack(
            [pair_evidence[name]["local_classification"] for name in PAIR_NAMES], dim=1
        )
        # Distribute each contextual correction using the already learned
        # sparse temporal measure.  Since every temporal measure sums to one,
        # both the regression and classification explanations remain exactly
        # additive after introducing nonlinear global context.
        uncoupled_local_regression = (
            local_regression_evidence + regression_delta[:, :3, None]
        )
        uncoupled_local_classification = (
            local_classification_evidence
            + classification_delta[:, :3, None, :]
        )
        uncoupled_pair_local_regression = (
            pair_local_regression_evidence + regression_delta[:, 3:, None]
        )
        uncoupled_pair_local_classification = (
            pair_local_classification_evidence
            + classification_delta[:, 3:, None, :]
        )
        local_regression_evidence = (
            uncoupled_local_regression
            + regression_from_class
            * (
                uncoupled_local_classification[..., 2]
                - uncoupled_local_classification[..., 0]
            )
        )
        local_classification_evidence = (
            uncoupled_local_classification
            + class_from_regression
            * uncoupled_local_regression.unsqueeze(-1)
            * self.polarity_basis.to(uncoupled_local_regression.dtype)
        )
        pair_local_regression_evidence = (
            uncoupled_pair_local_regression
            + regression_from_class
            * (
                uncoupled_pair_local_classification[..., 2]
                - uncoupled_pair_local_classification[..., 0]
            )
        )
        pair_local_classification_evidence = (
            uncoupled_pair_local_classification
            + class_from_regression
            * uncoupled_pair_local_regression.unsqueeze(-1)
            * self.polarity_basis.to(uncoupled_pair_local_regression.dtype)
        )
        unary_local_regression_contributions = (
            unary_gates.unsqueeze(-1) * temporal_weights * local_regression_evidence
        )
        unary_local_classification_contributions = (
            unary_gates[:, :, None, None]
            * temporal_weights.unsqueeze(-1)
            * local_classification_evidence
        )
        pair_local_regression_contributions = (
            pair_gates.unsqueeze(-1)
            * pair_temporal_weights
            * pair_local_regression_evidence
        )
        pair_local_classification_contributions = (
            pair_gates[:, :, None, None]
            * pair_temporal_weights.unsqueeze(-1)
            * pair_local_classification_evidence
        )
        local_regression_contributions = (
            unary_local_regression_contributions
            + torch.einsum(
                "bpl,pm->bml", pair_local_regression_contributions, self.pair_allocation
            )
        )
        local_classification_contributions = (
            unary_local_classification_contributions
            + torch.einsum(
                "bplc,pm->bmlc",
                pair_local_classification_contributions,
                self.pair_allocation,
            )
        )
        conflict_scores = torch.stack(
            [conflict_scores_dict[name] for name in PAIR_NAMES], dim=1
        )
        pair_reliability = torch.stack(
            [pair_reliability_dict[name] for name in PAIR_NAMES], dim=1
        )
        return {
            "class_logits": class_logits,
            "class_probabilities": torch.softmax(class_logits, dim=-1),
            "regression": regression,
            "regression_raw": regression_raw,
            "regression_bias": self.regression_bias.expand_as(regression_raw),
            "class_bias": self.class_bias.unsqueeze(0).expand_as(class_logits),
            "component_gates": component_gates,
            "modality_gates": modality_gates,
            "modality_available": unary_available,
            "component_confidence": component_confidence.to(component_gates.dtype),
            "component_uncertainty": component_uncertainty.to(component_gates.dtype),
            "component_reliability": component_reliability.to(component_gates.dtype),
            "shared_private_orthogonality": shared_private_orthogonality.to(
                component_gates.dtype
            ),
            "temporal_weights": temporal_weights,
            "pair_temporal_weights": pair_temporal_weights,
            "conflict_scores": conflict_scores,
            "pair_reliability": pair_reliability,
            "modality_regression_evidence": modality_regression,
            "modality_classification_evidence": modality_classification,
            "pair_regression_evidence": pair_regression,
            "pair_classification_evidence": pair_classification,
            "unary_regression_contributions": unary_regression_contributions,
            "unary_classification_contributions": unary_classification_contributions,
            "pair_regression_contributions": pair_regression_contributions,
            "pair_classification_contributions": pair_classification_contributions,
            "regression_contributions": regression_contributions,
            "classification_contributions": classification_contributions,
            "unary_local_regression_contributions": unary_local_regression_contributions,
            "unary_local_classification_contributions": unary_local_classification_contributions,
            "pair_local_regression_contributions": pair_local_regression_contributions,
            "pair_local_classification_contributions": pair_local_classification_contributions,
            "local_regression_contributions": local_regression_contributions,
            "local_classification_contributions": local_classification_contributions,
        }


class ResidualTokenMLP(nn.Module):
    """Pre-normalized residual MLP used by the compact V2 encoders."""

    def __init__(self, d_model: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * multiplier, d_model),
            nn.Dropout(dropout),
        )
        self.residual_scale_logit = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        scale = torch.sigmoid(self.residual_scale_logit)
        x = x + scale * self.ffn(self.norm(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


class GatedTemporalEncoderV2(nn.Module):
    """Small residual token encoder with learned masked pooling."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        ff_multiplier: int,
        dropout: float,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionEncoding(d_model, max_sequence_length)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ResidualTokenMLP(d_model, ff_multiplier, dropout)
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.attention = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.GELU(),
            nn.Linear(max(16, d_model // 2), 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.projection(self.input_norm(x))
        # Preserve the projected modality features and add positional context.
        # The previous implementation replaced ``h`` with position encodings,
        # making every sample with the same mask indistinguishable.
        h = self.input_dropout(h + self.position(h))
        h = h * mask.unsqueeze(-1).to(h.dtype)
        for block in self.blocks:
            h = block(h, mask)
        h = self.output_norm(h) * mask.unsqueeze(-1).to(h.dtype)
        weights = masked_softmax(self.attention(h).squeeze(-1), mask, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        return h, pooled, weights


class HAFusionGatedV2(nn.Module):
    """Compact residual, cross-modal gated fusion for small aligned datasets.

    Unlike the original six-component evidence graph, V2 uses only enabled
    unary modalities plus a lightweight pair context.  The final prediction is
    still an exact sum of gated modality evidence, so the existing explanation
    and counterfactual-loss interfaces remain available.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dims = cfg["input_dims"]
        d_model = int(cfg["d_model"])
        dropout = float(cfg.get("dropout", 0.25))
        ff_multiplier = int(cfg.get("ff_multiplier", 2))
        n_layers = int(cfg.get("n_layers", 2))
        self.encoders = nn.ModuleDict(
            {
                modality: GatedTemporalEncoderV2(
                    input_dim=int(dims[modality]),
                    d_model=d_model,
                    n_layers=n_layers,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                    max_sequence_length=int(cfg.get("max_sequence_length", 512)),
                )
                for modality in MODALITIES
            }
        )
        enabled = tuple(cfg.get("enabled_modalities", MODALITIES))
        unknown = set(enabled) - set(MODALITIES)
        if unknown or not enabled:
            raise ValueError(f"Invalid enabled_modalities: {enabled}")
        self.enabled_modalities = enabled
        self.register_buffer(
            "enabled_mask",
            torch.tensor([name in enabled for name in MODALITIES], dtype=torch.bool),
            persistent=False,
        )
        pair_input_dim = 4 * d_model
        self.pair_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(pair_input_dim),
                    nn.Linear(pair_input_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in PAIR_NAMES
            ]
        )
        self.global_norm = nn.LayerNorm(d_model)
        self.context_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(2 * d_model),
                    nn.Linear(2 * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                    nn.Dropout(dropout),
                )
                for _ in MODALITIES
            ]
        )
        self.context_scale_logits = nn.Parameter(torch.full((len(MODALITIES),), -1.5))
        gate_input_dim = 4 * d_model
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(gate_input_dim),
                    nn.Linear(gate_input_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1),
                )
                for _ in MODALITIES
            ]
        )
        self.modality_prior = nn.Parameter(torch.zeros(len(MODALITIES)))
        self.gate_temperature = float(cfg.get("component_gate_temperature", 1.0))
        self.gate_floor = float(cfg.get("component_gate_floor", 0.04))
        self.modality_dropout = float(cfg.get("component_dropout", 0.10))
        if self.gate_temperature <= 0.0:
            raise ValueError("component_gate_temperature must be positive")
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("component_gate_floor must lie in [0,1)")
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("component_dropout must lie in [0,1)")
        self.evidence_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, max(32, d_model // 2)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(32, d_model // 2), 4),
                )
                for _ in MODALITIES
            ]
        )
        self.class_bias = nn.Parameter(torch.zeros(3))
        self.regression_bias = nn.Parameter(torch.zeros(()))
        self.class_from_regression_logit = nn.Parameter(
            torch.tensor(float(cfg.get("class_from_regression_logit", -1.0)))
        )
        self.register_buffer(
            "polarity_basis", torch.tensor([-1.0, 0.0, 1.0]), persistent=False
        )

    def _training_availability(self, available: Tensor) -> Tensor:
        if not self.training or self.modality_dropout <= 0.0:
            return available
        keep = torch.rand_like(available.float()) >= self.modality_dropout
        dropped = available & keep
        empty = ~dropped.any(dim=1)
        if empty.any():
            # Retain the first genuinely available component for each empty row.
            fallback = available.float().argmax(dim=1)
            dropped[empty, fallback[empty]] = True
        return dropped

    def forward(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        declared_masks = {name: batch[f"{name}_mask"].bool() for name in MODALITIES}
        masks = {
            name: declared_masks[name]
            & torch.isfinite(batch[name]).all(dim=-1)
            & (batch[name].detach().abs().amax(dim=-1) > 1e-12)
            for name in MODALITIES
        }
        safe_masks: Dict[str, Tensor] = {}
        tokens: list[Tensor] = []
        pooled: list[Tensor] = []
        temporal_weights: list[Tensor] = []
        for index, name in enumerate(MODALITIES):
            safe = masks[name].clone()
            empty = ~safe.any(dim=1)
            if empty.any():
                safe[empty, 0] = True
            safe_masks[name] = safe
            token, summary, weights = self.encoders[name](batch[name], safe)
            enabled = masks[name].any(dim=1) & self.enabled_mask[index]
            enabled_f = enabled[:, None].to(token.dtype)
            tokens.append(token * enabled_f[:, None])
            pooled.append(summary * enabled_f)
            temporal_weights.append(weights * enabled_f)

        pooled_tensor = torch.stack(pooled, dim=1)
        available = torch.stack(
            [
                masks[name].any(dim=1) & self.enabled_mask[index]
                for index, name in enumerate(MODALITIES)
            ],
            dim=1,
        )
        available_f = available.to(pooled_tensor.dtype)
        unary_context = (
            (pooled_tensor * available_f.unsqueeze(-1)).sum(dim=1)
            / available_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        pair_contexts = []
        pair_available = []
        for pair_index, (left, right) in enumerate(PAIR_MODALITY_INDICES):
            left_value = pooled_tensor[:, left]
            right_value = pooled_tensor[:, right]
            pair_input = torch.cat(
                [
                    left_value,
                    right_value,
                    left_value * right_value,
                    (left_value - right_value).abs(),
                ],
                dim=-1,
            )
            is_available = available[:, left] & available[:, right]
            pair_contexts.append(
                self.pair_projections[pair_index](pair_input)
                * is_available[:, None].to(pair_input.dtype)
            )
            pair_available.append(is_available)
        pair_context_tensor = torch.stack(pair_contexts, dim=1)
        pair_available_tensor = torch.stack(pair_available, dim=1)
        pair_available_f = pair_available_tensor.to(pair_context_tensor.dtype)
        pair_context = (
            (pair_context_tensor * pair_available_f.unsqueeze(-1)).sum(dim=1)
            / pair_available_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        global_context = self.global_norm(unary_context + pair_context)

        corrected_tokens = []
        corrected_pooled = []
        for index, name in enumerate(MODALITIES):
            correction = self.context_adapters[index](
                torch.cat([pooled_tensor[:, index], global_context], dim=-1)
            )
            scale = torch.sigmoid(self.context_scale_logits[index])
            corrected = (pooled_tensor[:, index] + scale * correction) * available_f[
                :, index, None
            ]
            corrected_pooled.append(corrected)
            corrected_tokens.append(
                (tokens[index] + scale * correction[:, None, :])
                * masks[name].unsqueeze(-1).to(tokens[index].dtype)
                * available_f[:, index, None, None]
            )
        corrected_pooled_tensor = torch.stack(corrected_pooled, dim=1)

        gate_logits = []
        for index in range(len(MODALITIES)):
            value = corrected_pooled_tensor[:, index]
            gate_input = torch.cat(
                [
                    value,
                    global_context,
                    (value - global_context).abs(),
                    value * global_context,
                ],
                dim=-1,
            )
            gate_logits.append(self.gates[index](gate_input).squeeze(-1))
        gate_logits_tensor = torch.stack(gate_logits, dim=1) + self.modality_prior
        effective_available = self._training_availability(available)
        modality_gates = masked_softmax(
            gate_logits_tensor / self.gate_temperature,
            effective_available,
            dim=1,
        )
        if self.gate_floor > 0.0:
            uniform = effective_available.to(modality_gates.dtype)
            uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
            modality_gates = (
                (1.0 - self.gate_floor) * modality_gates + self.gate_floor * uniform
            )

        local_regression = []
        local_classification = []
        coupling = 0.5 * torch.sigmoid(self.class_from_regression_logit)
        for index in range(len(MODALITIES)):
            evidence = self.evidence_heads[index](corrected_tokens[index])
            local_reg = 3.0 * torch.tanh(evidence[..., 0] / 3.0)
            local_cls = evidence[..., 1:] + (
                coupling
                * local_reg.unsqueeze(-1)
                * self.polarity_basis.to(local_reg.dtype)
            )
            local_regression.append(local_reg)
            local_classification.append(local_cls)
        local_regression_tensor = torch.stack(local_regression, dim=1)
        local_classification_tensor = torch.stack(local_classification, dim=1)
        temporal_weights_tensor = torch.stack(temporal_weights, dim=1)
        modality_regression = torch.sum(
            temporal_weights_tensor * local_regression_tensor, dim=2
        )
        modality_classification = torch.sum(
            temporal_weights_tensor.unsqueeze(-1) * local_classification_tensor,
            dim=2,
        )
        regression_contributions = modality_gates * modality_regression
        classification_contributions = (
            modality_gates.unsqueeze(-1) * modality_classification
        )
        regression_raw = self.regression_bias + regression_contributions.sum(dim=1)
        regression = 3.0 * torch.tanh(regression_raw / 3.0)
        class_logits = self.class_bias + classification_contributions.sum(dim=1)
        local_regression_contributions = (
            modality_gates[:, :, None]
            * temporal_weights_tensor
            * local_regression_tensor
        )
        local_classification_contributions = (
            modality_gates[:, :, None, None]
            * temporal_weights_tensor.unsqueeze(-1)
            * local_classification_tensor
        )

        batch_size, _, sequence_length = temporal_weights_tensor.shape
        zeros_pair = temporal_weights_tensor.new_zeros(batch_size, len(PAIR_NAMES))
        zeros_pair_local = temporal_weights_tensor.new_zeros(
            batch_size, len(PAIR_NAMES), sequence_length
        )
        zeros_pair_class = temporal_weights_tensor.new_zeros(
            batch_size, len(PAIR_NAMES), 3
        )
        zeros_pair_local_class = temporal_weights_tensor.new_zeros(
            batch_size, len(PAIR_NAMES), sequence_length, 3
        )
        return {
            "class_logits": class_logits,
            "class_probabilities": torch.softmax(class_logits, dim=-1),
            "regression": regression,
            "regression_raw": regression_raw,
            "regression_bias": self.regression_bias.expand_as(regression_raw),
            "class_bias": self.class_bias.unsqueeze(0).expand_as(class_logits),
            "component_gates": torch.cat([modality_gates, zeros_pair], dim=1),
            "modality_gates": modality_gates,
            "modality_available": available,
            "temporal_weights": temporal_weights_tensor,
            "pair_temporal_weights": zeros_pair_local,
            "conflict_scores": zeros_pair_local,
            "pair_reliability": zeros_pair_local,
            "modality_regression_evidence": modality_regression,
            "modality_classification_evidence": modality_classification,
            "pair_regression_evidence": zeros_pair,
            "pair_classification_evidence": zeros_pair_class,
            "unary_regression_contributions": regression_contributions,
            "unary_classification_contributions": classification_contributions,
            "pair_regression_contributions": zeros_pair,
            "pair_classification_contributions": zeros_pair_class,
            "regression_contributions": regression_contributions,
            "classification_contributions": classification_contributions,
            "unary_local_regression_contributions": local_regression_contributions,
            "unary_local_classification_contributions": local_classification_contributions,
            "pair_local_regression_contributions": zeros_pair_local,
            "pair_local_classification_contributions": zeros_pair_local_class,
            "local_regression_contributions": local_regression_contributions,
            "local_classification_contributions": local_classification_contributions,
        }


class LowRankTemporalEncoder(nn.Module):
    """Lightweight modality-specific temporal encoder for tensor fusion.

    The encoder deliberately avoids a deep Transformer on this small dataset.
    A train-only normalized projection is followed by a depthwise local branch,
    residual token MLPs, and learned masked pooling.  Each modality owns an
    independent instance so acoustic, visual, and textual temporal statistics
    are not forced through shared parameters.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        ff_multiplier: int,
        dropout: float,
        kernel_size: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("low-rank fusion kernel_size must be a positive odd integer")
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionEncoding(d_model, max_sequence_length)
        self.position_scale = nn.Parameter(torch.tensor(0.10))
        self.input_dropout = nn.Dropout(dropout)
        self.local_depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        self.local_pointwise = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.local_scale_logit = nn.Parameter(torch.tensor(-1.5))
        self.blocks = nn.ModuleList(
            [
                ResidualTokenMLP(d_model, ff_multiplier, dropout)
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.attention = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 2), 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        h = self.projection(self.input_norm(x))
        h = self.input_dropout(h + self.position_scale * self.position(h)) * mask_f
        local = self.local_pointwise(self.local_depthwise(h.transpose(1, 2))).transpose(1, 2)
        h = h + torch.sigmoid(self.local_scale_logit) * local * mask_f
        for block in self.blocks:
            h = block(h, mask)
        h = self.output_norm(h) * mask_f
        weights = masked_softmax(self.attention(h).squeeze(-1), mask, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        return h, pooled, weights


class LowRankTensorFusionNet(nn.Module):
    """Trimodal low-rank tensor fusion with a conservative residual path.

    This is a clean-room adaptation of the factorized tensor-product formula
    in Liu et al., ACL 2018.  Appending a constant one to each modality lets a
    single factorization represent unimodal, bimodal, and trimodal terms.  The
    learned tensor interaction is added to an attention-gated unimodal residual
    so optimization begins from a stable small-data model rather than relying
    on a high-order product from the first update.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        dims = cfg["input_dims"]
        d_model = int(cfg.get("d_model", 96))
        fusion_dim = int(cfg.get("fusion_dim", 128))
        rank = int(cfg.get("rank", 8))
        dropout = float(cfg.get("dropout", 0.25))
        if rank <= 0:
            raise ValueError("low-rank fusion rank must be positive")
        if fusion_dim <= 0:
            raise ValueError("low-rank fusion fusion_dim must be positive")
        self.rank = rank
        self.encoders = nn.ModuleDict(
            {
                modality: LowRankTemporalEncoder(
                    input_dim=int(dims[modality]),
                    d_model=d_model,
                    n_layers=int(cfg.get("n_layers", 1)),
                    ff_multiplier=int(cfg.get("ff_multiplier", 2)),
                    dropout=dropout,
                    kernel_size=int(cfg.get("conv_kernel_size", 3)),
                    max_sequence_length=int(cfg.get("max_sequence_length", 512)),
                )
                for modality in MODALITIES
            }
        )
        self.modality_dropout = float(cfg.get("modality_dropout", 0.10))
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("modality_dropout must lie in [0,1)")
        self.gate_temperature = float(cfg.get("gate_temperature", 1.0))
        if self.gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")
        self.modality_type = nn.Parameter(torch.zeros(len(MODALITIES), d_model))
        nn.init.normal_(self.modality_type, std=0.02)
        self.gate = nn.Sequential(
            nn.LayerNorm(4 * d_model),
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.gate_bias = nn.Parameter(torch.zeros(len(MODALITIES)))

        self.factors = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.empty(rank, d_model + 1, fusion_dim))
                for modality in MODALITIES
            }
        )
        for factor in self.factors.values():
            for rank_slice in factor:
                nn.init.xavier_uniform_(rank_slice)
        self.rank_logits = nn.Parameter(torch.zeros(rank))
        self.interaction_norm = nn.LayerNorm(fusion_dim)
        self.interaction_adapter = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.interaction_scale_logit = nn.Parameter(
            torch.tensor(float(cfg.get("interaction_scale_logit", -1.0)))
        )
        self.residual_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_block = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 2 * fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.Dropout(dropout),
        )
        self.fusion_scale_logit = nn.Parameter(torch.tensor(-1.0))
        head_hidden = max(32, fusion_dim // 2)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 3),
        )
        self.regression_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.neutral_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.unimodal_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, max(32, d_model // 2)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(32, d_model // 2), 4),
                )
                for _ in MODALITIES
            ]
        )
        self.class_bias = nn.Parameter(torch.zeros(3))
        self.regression_bias = nn.Parameter(torch.zeros(()))

    def _effective_availability(self, available: Tensor) -> Tensor:
        if not self.training or self.modality_dropout <= 0.0:
            return available
        keep = torch.rand_like(available.float()) >= self.modality_dropout
        effective = available & keep
        empty = ~effective.any(dim=1)
        if empty.any():
            fallback = available.float().argmax(dim=1)
            effective[empty, fallback[empty]] = True
        return effective

    def _low_rank_interaction(self, pooled: Tensor) -> Tensor:
        # The product is kept in FP32 even under AMP: multiplying three factor
        # responses in FP16 otherwise loses small but informative interactions.
        device_type = pooled.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            ones = torch.ones(
                pooled.size(0), len(MODALITIES), 1, device=pooled.device, dtype=torch.float32
            )
            augmented = torch.cat([ones, pooled.float()], dim=-1)
            projected = []
            for index, modality in enumerate(MODALITIES):
                projected.append(
                    torch.einsum(
                        "bd,rdf->brf", augmented[:, index], self.factors[modality].float()
                    )
                )
            product = projected[0] * projected[1] * projected[2]
            rank_weights = torch.softmax(self.rank_logits.float(), dim=0)
            interaction = torch.sum(rank_weights[None, :, None] * product, dim=1)
            interaction = self.interaction_norm(interaction)
        return interaction.to(pooled.dtype)

    def forward(self, batch: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        declared_masks = {name: batch[f"{name}_mask"].bool() for name in MODALITIES}
        masks = {
            name: declared_masks[name]
            & torch.isfinite(batch[name]).all(dim=-1)
            & (batch[name].detach().abs().amax(dim=-1) > 1e-12)
            for name in MODALITIES
        }
        tokens: list[Tensor] = []
        pooled: list[Tensor] = []
        temporal_weights: list[Tensor] = []
        for name in MODALITIES:
            safe_mask = masks[name].clone()
            empty = ~safe_mask.any(dim=1)
            if empty.any():
                safe_mask[empty, 0] = True
            token, summary, weights = self.encoders[name](batch[name], safe_mask)
            present = masks[name].any(dim=1)
            present_f = present.to(token.dtype)
            tokens.append(token * present_f[:, None, None])
            pooled.append(summary * present_f[:, None])
            temporal_weights.append(weights * present_f[:, None])
        pooled_tensor = torch.stack(pooled, dim=1)
        available = torch.stack([masks[name].any(dim=1) for name in MODALITIES], dim=1)
        effective = self._effective_availability(available)
        effective_f = effective.to(pooled_tensor.dtype)
        effective_pooled = pooled_tensor * effective_f.unsqueeze(-1)
        global_context = effective_pooled.sum(dim=1) / effective_f.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        typed = effective_pooled + self.modality_type.unsqueeze(0).to(pooled_tensor.dtype)
        global_expanded = global_context.unsqueeze(1).expand_as(typed)
        gate_features = torch.cat(
            [typed, global_expanded, (typed - global_expanded).abs(), typed * global_expanded],
            dim=-1,
        )
        gate_logits = self.gate(gate_features).squeeze(-1) + self.gate_bias
        modality_gates = masked_softmax(
            gate_logits / self.gate_temperature, effective, dim=1
        )
        residual = torch.sum(modality_gates.unsqueeze(-1) * effective_pooled, dim=1)
        residual = self.residual_projection(residual)
        interaction = self._low_rank_interaction(effective_pooled)
        fused = residual + torch.sigmoid(self.interaction_scale_logit) * self.interaction_adapter(
            interaction
        )
        fused = fused + torch.sigmoid(self.fusion_scale_logit) * self.fusion_block(fused)

        class_delta = self.classification_head(fused)
        regression_delta = self.regression_head(fused).squeeze(-1)
        neutral_logit = self.neutral_head(fused).squeeze(-1)
        class_logits = self.class_bias + class_delta
        regression_raw = self.regression_bias + regression_delta
        regression = 3.0 * torch.tanh(regression_raw / 3.0)

        temporal_weights_tensor = torch.stack(temporal_weights, dim=1)
        modality_outputs = torch.stack(
            [self.unimodal_heads[index](pooled_tensor[:, index]) for index in range(3)],
            dim=1,
        )
        modality_regression = 3.0 * torch.tanh(modality_outputs[..., 0] / 3.0)
        modality_classification = modality_outputs[..., 1:]
        # Allocate global tensor-fusion evidence back to the three modalities
        # with the learned residual gates.  This preserves the exact additive
        # explanation contract while keeping the high-order branch global.
        regression_contributions = modality_gates * regression_delta.unsqueeze(1)
        classification_contributions = modality_gates.unsqueeze(-1) * class_delta.unsqueeze(1)
        local_regression_contributions = (
            temporal_weights_tensor * regression_contributions.unsqueeze(-1)
        )
        local_classification_contributions = (
            temporal_weights_tensor.unsqueeze(-1)
            * classification_contributions.unsqueeze(2)
        )
        batch_size, _, sequence_length = temporal_weights_tensor.shape
        zeros_pair = temporal_weights_tensor.new_zeros(batch_size, len(PAIR_NAMES))
        zeros_pair_local = temporal_weights_tensor.new_zeros(
            batch_size, len(PAIR_NAMES), sequence_length
        )
        zeros_pair_class = temporal_weights_tensor.new_zeros(
            batch_size, len(PAIR_NAMES), 3
        )
        zeros_pair_local_class = temporal_weights_tensor.new_zeros(
            batch_size, len(PAIR_NAMES), sequence_length, 3
        )
        return {
            "class_logits": class_logits,
            "class_probabilities": torch.softmax(class_logits, dim=-1),
            "neutral_logit": neutral_logit,
            "regression": regression,
            "regression_raw": regression_raw,
            "regression_bias": self.regression_bias.expand_as(regression_raw),
            "class_bias": self.class_bias.unsqueeze(0).expand_as(class_logits),
            "component_gates": torch.cat([modality_gates, zeros_pair], dim=1),
            "modality_gates": modality_gates,
            "modality_available": available,
            "temporal_weights": temporal_weights_tensor,
            "pair_temporal_weights": zeros_pair_local,
            "conflict_scores": zeros_pair_local,
            "pair_reliability": zeros_pair_local,
            "modality_regression_evidence": modality_regression,
            "modality_classification_evidence": modality_classification,
            "pair_regression_evidence": zeros_pair,
            "pair_classification_evidence": zeros_pair_class,
            "unary_regression_contributions": regression_contributions,
            "unary_classification_contributions": classification_contributions,
            "pair_regression_contributions": zeros_pair,
            "pair_classification_contributions": zeros_pair_class,
            "regression_contributions": regression_contributions,
            "classification_contributions": classification_contributions,
            "unary_local_regression_contributions": local_regression_contributions,
            "unary_local_classification_contributions": local_classification_contributions,
            "pair_local_regression_contributions": zeros_pair_local,
            "pair_local_classification_contributions": zeros_pair_local_class,
            "local_regression_contributions": local_regression_contributions,
            "local_classification_contributions": local_classification_contributions,
        }


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    architecture = str(cfg.get("architecture", "hafusion")).strip().lower()
    if architecture in {"hafusion", "baseline", "c3_hafusion"}:
        return HAFusionNet(cfg)
    if architecture in {"gated_v2", "hafusion_gated_v2"}:
        return HAFusionGatedV2(cfg)
    if architecture in {"low_rank_tensor_fusion", "lmf", "hafusion_lmf"}:
        return LowRankTensorFusionNet(cfg)
    raise ValueError(f"Unknown model architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
