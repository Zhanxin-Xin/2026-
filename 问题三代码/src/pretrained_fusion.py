from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoModelForSequenceClassification


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


class ResidualMLP(nn.Module):
    def __init__(self, dimension: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.network = nn.Sequential(
            nn.Linear(dimension, dimension * multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * multiplier, dimension),
            nn.Dropout(dropout),
        )
        self.scale = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: Tensor) -> Tensor:
        return x + torch.sigmoid(self.scale) * self.network(self.norm(x))


class AttentiveTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        dropout: float,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dimension)
        self.projection = nn.Linear(input_dimension, hidden_dimension)
        self.local = nn.Sequential(
            nn.Conv1d(
                hidden_dimension,
                hidden_dimension,
                kernel_size=3,
                padding=1,
                groups=hidden_dimension,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dimension, hidden_dimension, kernel_size=1),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dimension,
            nhead=n_heads,
            dim_feedforward=hidden_dimension * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(hidden_dimension)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.Tanh(),
            nn.Linear(hidden_dimension // 2, 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mask = mask.bool()
        hidden = self.projection(self.input_norm(x))
        hidden = hidden + self.local(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.temporal(hidden, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden) * mask.unsqueeze(-1).to(hidden.dtype)
        scores = self.attention(hidden).squeeze(-1).float()
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).to(hidden.dtype)
        pooled = torch.sum(weights.unsqueeze(-1) * hidden, dim=1)
        return hidden, pooled, weights


class TemporalSpectralDynamicsAdapter(nn.Module):
    """Fuse time-domain motion and frequency-domain energy before fusion.

    The ordinary temporal encoder can learn these statistics implicitly, but
    small affect datasets rarely identify them reliably. This adapter exposes
    mean state, velocity, acceleration, and low/high spectral energy as a
    compact representation, then learns an interpretable text-conditioned
    gate into the existing modality state.
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        dropout: float,
        mix_logit: float = -2.0,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dimension)
        self.projection = nn.Linear(input_dimension, hidden_dimension)
        self.summary = nn.Sequential(
            nn.LayerNorm(5 * hidden_dimension),
            nn.Linear(5 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(hidden_dimension, multiplier=2, dropout=dropout),
        )
        self.route = nn.Sequential(
            nn.LayerNorm(3 * hidden_dimension),
            nn.Linear(3 * hidden_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.route[-1].weight)
        nn.init.zeros_(self.route[-1].bias)
        self.mix_logit = nn.Parameter(torch.tensor(float(mix_logit)))
        self.output_norm = nn.LayerNorm(hidden_dimension)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weight = mask.unsqueeze(-1).to(values.dtype)
        return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        text: Tensor,
        baseline: Tensor,
        sequence: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor]:
        mask = mask.bool()
        projected = F.gelu(self.projection(self.input_norm(sequence)))
        masked = projected * mask.unsqueeze(-1).to(projected.dtype)
        state = self._masked_mean(masked, mask)

        pair_mask = mask[:, 1:] & mask[:, :-1]
        velocity_sequence = (projected[:, 1:] - projected[:, :-1]).abs()
        velocity = self._masked_mean(velocity_sequence, pair_mask)
        if sequence.size(1) >= 3:
            acceleration_mask = pair_mask[:, 1:] & pair_mask[:, :-1]
            acceleration_sequence = (
                projected[:, 2:]
                - 2.0 * projected[:, 1:-1]
                + projected[:, :-2]
            ).abs()
            acceleration = self._masked_mean(
                acceleration_sequence, acceleration_mask
            )
        else:
            acceleration = torch.zeros_like(state)

        length = mask.sum(dim=1, keepdim=True).float().clamp_min(1.0)
        power = torch.fft.rfft(masked.float(), dim=1).abs().square()
        power = power / length.unsqueeze(-1)
        non_dc = power[:, 1:]
        if non_dc.size(1) > 0:
            split = max(1, non_dc.size(1) // 3)
            low_power = non_dc[:, :split].mean(dim=1)
            high_power = (
                non_dc[:, split:].mean(dim=1)
                if split < non_dc.size(1)
                else torch.zeros_like(low_power)
            )
        else:
            low_power = torch.zeros_like(state, dtype=torch.float32)
            high_power = torch.zeros_like(state, dtype=torch.float32)
        low_energy = torch.log1p(low_power).to(state.dtype)
        high_energy = torch.log1p(high_power).to(state.dtype)
        spectral = self.summary(
            torch.cat(
                [state, velocity, acceleration, low_energy, high_energy], dim=-1
            )
        )
        gate = torch.sigmoid(
            self.mix_logit
            + self.route(torch.cat([text, baseline, spectral], dim=-1)).squeeze(-1)
        )
        corrected = self.output_norm(
            baseline + gate.unsqueeze(-1) * (spectral - baseline)
        )
        return {
            "corrected": corrected,
            "spectral": spectral,
            "gate": gate,
            "velocity": velocity.float().mean(dim=-1),
            "acceleration": acceleration.float().mean(dim=-1),
            "low_energy": low_energy.float().mean(dim=-1),
            "high_energy": high_energy.float().mean(dim=-1),
        }


class MultimodalAdaptationGate(nn.Module):
    """Utterance-level MAG-style bounded visual shift for the text anchor."""

    def __init__(self, dimension: int, dropout: float, shift_scale: float) -> None:
        super().__init__()
        self.shift_scale = float(shift_scale)
        self.gate = nn.Sequential(
            nn.Linear(2 * dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
            nn.Sigmoid(),
        )
        self.shift = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
        )
        self.norm = nn.LayerNorm(dimension)

    def forward(self, text: Tensor, visual: Tensor) -> tuple[Tensor, Tensor]:
        proposed = self.gate(torch.cat([text, visual], dim=-1)) * self.shift(visual)
        text_norm = text.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        shift_norm = proposed.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.minimum(
            torch.ones_like(text_norm), self.shift_scale * text_norm / shift_norm
        ).to(proposed.dtype)
        bounded = proposed * scale
        return self.norm(text + bounded), bounded


class ModalityReliabilityGate(nn.Module):
    """Estimate whether a non-verbal modality should alter the text anchor."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(4 * dimension),
            nn.Linear(4 * dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )

    def forward(self, text: Tensor, modality: Tensor) -> Tensor:
        features = torch.cat(
            [text, modality, torch.abs(text - modality), text * modality], dim=-1
        )
        return torch.sigmoid(self.network(features))


class DirectionalContextRouter(nn.Module):
    """Choose whether previous or following dialogue context should be trusted."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.context_score = nn.Sequential(
            nn.LayerNorm(4 * dimension + 6),
            nn.Linear(4 * dimension + 6, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )
        self.reject_score = nn.Sequential(
            nn.LayerNorm(dimension + 4),
            nn.Linear(dimension + 4, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )
        self.direction_bias = nn.Parameter(torch.zeros(2))
        self.reject_bias = nn.Parameter(torch.tensor(1.0))
        # Begin as a conservative, sample-independent router. Gradients from the
        # task then learn when either direction deserves non-zero probability.
        nn.init.zeros_(self.context_score[-1].weight)
        nn.init.zeros_(self.context_score[-1].bias)
        nn.init.zeros_(self.reject_score[-1].weight)
        nn.init.zeros_(self.reject_score[-1].bias)

    @staticmethod
    def _confidence_diagnostics(probability: Tensor) -> Tensor:
        probability = probability.float().clamp_min(1e-8)
        top_two = torch.topk(probability, k=2, dim=-1).values
        uncertainty = 1.0 - top_two[:, 0]
        margin = top_two[:, 0] - top_two[:, 1]
        entropy = -(probability * probability.log()).sum(dim=-1) / math.log(
            probability.size(-1)
        )
        return torch.stack([uncertainty, margin, entropy], dim=-1)

    def _context_logit(
        self,
        text: Tensor,
        context: Tensor,
        confidence: Tensor,
        text_probability: Tensor,
        context_probability: Tensor,
        direction: int,
    ) -> Tensor:
        compatibility = F.cosine_similarity(
            text.float(), context.float(), dim=-1
        ).unsqueeze(-1)
        features = torch.cat(
            [
                text,
                context,
                torch.abs(text - context),
                text * context,
                confidence.to(text.dtype),
                compatibility.to(text.dtype),
                (text_probability.float() * context_probability.float())
                .sum(dim=-1, keepdim=True)
                .to(text.dtype),
                context_probability[:, 1:2].to(text.dtype),
            ],
            dim=-1,
        )
        return self.context_score(features).squeeze(-1) + self.direction_bias[direction]

    def forward(
        self,
        text: Tensor,
        previous: Tensor,
        following: Tensor,
        previous_available: Tensor,
        following_available: Tensor,
        text_probability: Tensor,
        previous_probability: Tensor,
        following_probability: Tensor,
    ) -> Tensor:
        confidence = self._confidence_diagnostics(text_probability)
        previous_logit = self._context_logit(
            text,
            previous,
            confidence,
            text_probability,
            previous_probability,
            direction=0,
        )
        following_logit = self._context_logit(
            text,
            following,
            confidence,
            text_probability,
            following_probability,
            direction=1,
        )
        agreement = F.cosine_similarity(
            previous.float(), following.float(), dim=-1
        ).unsqueeze(-1)
        reject_features = torch.cat(
            [text, confidence.to(text.dtype), agreement.to(text.dtype)], dim=-1
        )
        reject_logit = (
            self.reject_score(reject_features).squeeze(-1) + self.reject_bias
        )
        logits = torch.stack(
            [reject_logit, previous_logit, following_logit], dim=-1
        ).float()
        availability = torch.stack(
            [
                torch.ones_like(previous_available, dtype=torch.bool),
                previous_available.bool(),
                following_available.bool(),
            ],
            dim=-1,
        )
        return torch.softmax(logits.masked_fill(~availability, -1e4), dim=-1)


class SentimentTransitionExpert(nn.Module):
    """Correct a current posterior from signed, direction-aware context changes.

    The expert never forwards an absolute neighbour representation to the
    classifier.  Each edge is represented by current-minus-neighbour change,
    its magnitude, a multiplicative interaction, and differences between the
    utterance-level sentiment posteriors.  The final residual projection is
    zero-initialized so enabling this expert preserves the parent prediction at
    initialization.
    """

    def __init__(
        self,
        dimension: int,
        dropout: float,
        residual_scale_logit: float = -1.5,
    ) -> None:
        super().__init__()
        edge_feature_dimension = 3 * dimension + 9
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_feature_dimension),
            nn.Linear(edge_feature_dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(dimension, multiplier=2, dropout=dropout),
        )
        self.direction_embeddings = nn.Parameter(torch.zeros(2, dimension))
        nn.init.normal_(self.direction_embeddings, std=0.02)
        self.route_score = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )
        self.reject_score = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, dimension // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 4, 1),
        )
        self.direction_bias = nn.Parameter(torch.zeros(2))
        self.reject_bias = nn.Parameter(torch.tensor(1.0))
        self.class_residual = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 3),
        )
        nn.init.zeros_(self.class_residual[-1].weight)
        nn.init.zeros_(self.class_residual[-1].bias)
        self.residual_scale_logit = nn.Parameter(
            torch.tensor(float(residual_scale_logit))
        )

    @staticmethod
    def _confidence_diagnostics(probability: Tensor) -> Tensor:
        probability = probability.float().clamp_min(1e-8)
        top_two = torch.topk(probability, k=2, dim=-1).values
        uncertainty = 1.0 - top_two[:, 0]
        margin = top_two[:, 0] - top_two[:, 1]
        entropy = -(probability * probability.log()).sum(dim=-1) / math.log(
            probability.size(-1)
        )
        return torch.stack([uncertainty, margin, entropy], dim=-1)

    def _encode_edge(
        self,
        current: Tensor,
        neighbour: Tensor,
        current_probability: Tensor,
        neighbour_probability: Tensor,
        direction: int,
    ) -> Tensor:
        signed_change = current - neighbour
        features = torch.cat(
            [
                signed_change,
                torch.abs(signed_change),
                current * neighbour,
                (current_probability.float() - neighbour_probability.float()).to(
                    current.dtype
                ),
                (current_probability.float() * neighbour_probability.float()).to(
                    current.dtype
                ),
                self._confidence_diagnostics(current_probability).to(current.dtype),
            ],
            dim=-1,
        )
        return self.edge_encoder(features) + self.direction_embeddings[direction]

    def forward(
        self,
        current: Tensor,
        previous: Tensor,
        following: Tensor,
        previous_available: Tensor,
        following_available: Tensor,
        current_probability: Tensor,
        previous_probability: Tensor,
        following_probability: Tensor,
    ) -> dict[str, Tensor]:
        previous_edge = self._encode_edge(
            current,
            previous,
            current_probability,
            previous_probability,
            direction=0,
        )
        following_edge = self._encode_edge(
            current,
            following,
            current_probability,
            following_probability,
            direction=1,
        )
        confidence = self._confidence_diagnostics(current_probability).to(
            current.dtype
        )
        logits = torch.stack(
            [
                self.reject_score(confidence).squeeze(-1) + self.reject_bias,
                self.route_score(previous_edge).squeeze(-1)
                + self.direction_bias[0],
                self.route_score(following_edge).squeeze(-1)
                + self.direction_bias[1],
            ],
            dim=-1,
        ).float()
        availability = torch.stack(
            [
                torch.ones_like(previous_available, dtype=torch.bool),
                previous_available.bool(),
                following_available.bool(),
            ],
            dim=-1,
        )
        weights = torch.softmax(logits.masked_fill(~availability, -1e4), dim=-1)
        transition = (
            weights[:, 1:2].to(previous_edge.dtype) * previous_edge
            + weights[:, 2:3].to(following_edge.dtype) * following_edge
        )
        scale = torch.sigmoid(self.residual_scale_logit.float())
        residual = scale * self.class_residual(transition).float()
        return {
            "weights": weights,
            "transition": transition,
            "residual": residual,
            "scale": scale,
        }


class BoundedNeutralTransitionExpert(nn.Module):
    """Make a trust-region correction to only the Neutral-vs-Polar boundary.

    Unlike :class:`SentimentTransitionExpert`, this branch cannot change the
    Negative/Positive log-odds. Its scalar correction is hard-bounded by both
    the probability of accepting a context edge and the uncertainty of the
    already-trained parent posterior. Consequently an explicit reject route
    has a structural effect instead of relying on a downstream MLP to learn
    that rejection should imply a small residual.
    """

    def __init__(
        self,
        dimension: int,
        dropout: float,
        maximum_logit_shift: float = 1.0,
    ) -> None:
        super().__init__()
        edge_feature_dimension = 2 * dimension + 6
        hidden_dimension = max(dimension // 2, 32)
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_feature_dimension),
            nn.Linear(edge_feature_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.direction_embeddings = nn.Parameter(
            torch.zeros(2, hidden_dimension)
        )
        nn.init.normal_(self.direction_embeddings, std=0.02)
        self.route_score = nn.Sequential(
            nn.LayerNorm(hidden_dimension),
            nn.Linear(hidden_dimension, 1),
        )
        self.reject_score = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, 1),
        )
        self.direction_bias = nn.Parameter(torch.zeros(2))
        self.reject_bias = nn.Parameter(torch.tensor(1.0))
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(hidden_dimension + 6),
            nn.Linear(hidden_dimension + 6, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )
        nn.init.zeros_(self.boundary_head[-1].weight)
        nn.init.zeros_(self.boundary_head[-1].bias)
        self.maximum_logit_shift = float(maximum_logit_shift)
        if self.maximum_logit_shift <= 0.0:
            raise ValueError("maximum_logit_shift must be positive")

    @staticmethod
    def _confidence_diagnostics(probability: Tensor) -> Tensor:
        probability = probability.float().clamp_min(1e-8)
        top_two = torch.topk(probability, k=2, dim=-1).values
        uncertainty = 1.0 - top_two[:, 0]
        margin = top_two[:, 0] - top_two[:, 1]
        entropy = -(probability * probability.log()).sum(dim=-1) / math.log(
            probability.size(-1)
        )
        return torch.stack([uncertainty, margin, entropy], dim=-1)

    def _encode_edge(
        self,
        current: Tensor,
        neighbour: Tensor,
        current_probability: Tensor,
        neighbour_probability: Tensor,
        direction: int,
    ) -> Tensor:
        signed_change = current - neighbour
        features = torch.cat(
            [
                signed_change,
                torch.abs(signed_change),
                (current_probability.float() - neighbour_probability.float()).to(
                    current.dtype
                ),
                (current_probability.float() * neighbour_probability.float()).to(
                    current.dtype
                ),
            ],
            dim=-1,
        )
        return self.edge_encoder(features) + self.direction_embeddings[direction]

    def encode(
        self,
        current: Tensor,
        previous: Tensor,
        following: Tensor,
        previous_available: Tensor,
        following_available: Tensor,
        current_probability: Tensor,
        previous_probability: Tensor,
        following_probability: Tensor,
    ) -> dict[str, Tensor]:
        previous_edge = self._encode_edge(
            current,
            previous,
            current_probability,
            previous_probability,
            direction=0,
        )
        following_edge = self._encode_edge(
            current,
            following,
            current_probability,
            following_probability,
            direction=1,
        )
        confidence = self._confidence_diagnostics(current_probability).to(
            current.dtype
        )
        logits = torch.stack(
            [
                self.reject_score(confidence).squeeze(-1) + self.reject_bias,
                self.route_score(previous_edge).squeeze(-1)
                + self.direction_bias[0],
                self.route_score(following_edge).squeeze(-1)
                + self.direction_bias[1],
            ],
            dim=-1,
        ).float()
        availability = torch.stack(
            [
                torch.ones_like(previous_available, dtype=torch.bool),
                previous_available.bool(),
                following_available.bool(),
            ],
            dim=-1,
        )
        weights = torch.softmax(logits.masked_fill(~availability, -1e4), dim=-1)
        transition = (
            weights[:, 1:2].to(previous_edge.dtype) * previous_edge
            + weights[:, 2:3].to(following_edge.dtype) * following_edge
        )
        return {"weights": weights, "transition": transition}

    def correct(
        self,
        base_probability: Tensor,
        transition: Tensor,
        weights: Tensor,
    ) -> dict[str, Tensor]:
        base_probability = base_probability.float().clamp_min(1e-8)
        diagnostics = self._confidence_diagnostics(base_probability)
        raw_shift = self.boundary_head(
            torch.cat(
                [
                    transition,
                    base_probability.to(transition.dtype),
                    diagnostics.to(transition.dtype),
                ],
                dim=-1,
            )
        ).squeeze(-1).float()
        context_acceptance = (weights[:, 1] + weights[:, 2]).float()
        parent_uncertainty = (1.0 - base_probability.max(dim=-1).values).clamp(
            min=0.0, max=1.0
        )
        shift = (
            self.maximum_logit_shift
            * context_acceptance
            * parent_uncertainty
            * torch.tanh(raw_shift)
        )
        # Equal polar offsets preserve log p(Negative) - log p(Positive).
        correction = torch.stack(
            [-0.5 * shift, shift, -0.5 * shift], dim=-1
        )
        probability = torch.softmax(base_probability.log() + correction, dim=-1)
        return {
            "probabilities": probability,
            "residual": correction,
            "scale": context_acceptance * parent_uncertainty,
        }


class ConfidenceAwareFusionRouter(nn.Module):
    """Route a multimodal residual only when it complements the text anchor."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        input_dimension = 4 * dimension + 4
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )
        # Preserve the scalar-gate checkpoint exactly at initialization. This
        # lets a warm-started model learn only sample-specific deviations.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        text: Tensor,
        candidate: Tensor,
        audio_reliability: Tensor,
        visual_reliability: Tensor,
        text_probability: Tensor,
    ) -> Tensor:
        top_two = torch.topk(text_probability.float(), k=2, dim=-1).values
        uncertainty = 1.0 - top_two[:, 0]
        margin = top_two[:, 0] - top_two[:, 1]
        entropy = -(
            text_probability.float().clamp_min(1e-8)
            * text_probability.float().clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(text_probability.size(-1))
        diagnostics = torch.stack(
            [audio_reliability, visual_reliability, uncertainty, entropy - margin],
            dim=-1,
        ).to(text.dtype)
        features = torch.cat(
            [
                text,
                candidate,
                torch.abs(text - candidate),
                text * candidate,
                diagnostics,
            ],
            dim=-1,
        )
        return self.network(features).squeeze(-1)


class SharedPrivateFusion(nn.Module):
    """MISA-inspired decomposition followed by a small component transformer."""

    def __init__(self, dimension: int, dropout: float, n_heads: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
        )
        self.private_text = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU()
        )
        self.private_vision = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU()
        )
        self.type_embeddings = nn.Parameter(torch.zeros(5, dimension))
        nn.init.normal_(self.type_embeddings, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=n_heads,
            dim_feedforward=dimension * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=1)
        self.query = nn.Parameter(torch.zeros(dimension))
        nn.init.normal_(self.query, std=0.02)
        self.output = ResidualMLP(dimension, multiplier=2, dropout=dropout)

    def forward(self, text: Tensor, vision: Tensor) -> tuple[Tensor, Tensor]:
        shared_text = self.shared(text)
        shared_vision = self.shared(vision)
        private_text = self.private_text(text)
        private_vision = self.private_vision(vision)
        shared_consensus = 0.5 * (shared_text + shared_vision)
        components = torch.stack(
            [text, vision, shared_consensus, private_text, private_vision], dim=1
        )
        mixed = self.mixer(components + self.type_embeddings.unsqueeze(0))
        scores = torch.einsum("bkd,d->bk", mixed, self.query) / math.sqrt(mixed.size(-1))
        weights = torch.softmax(scores.float(), dim=1).to(mixed.dtype)
        fused = torch.sum(weights.unsqueeze(-1) * mixed, dim=1)
        return self.output(fused), weights


class TriModalSharedPrivateFusion(nn.Module):
    """MISA-style shared/private decomposition for text, audio, and vision."""

    def __init__(self, dimension: int, dropout: float, n_heads: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU()
        )
        self.private_text = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU()
        )
        self.private_audio = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU()
        )
        self.private_vision = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU()
        )
        self.type_embeddings = nn.Parameter(torch.zeros(7, dimension))
        nn.init.normal_(self.type_embeddings, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=n_heads,
            dim_feedforward=dimension * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=1)
        self.query = nn.Parameter(torch.zeros(dimension))
        nn.init.normal_(self.query, std=0.02)
        self.output = ResidualMLP(dimension, multiplier=2, dropout=dropout)

    def forward(
        self, text: Tensor, audio: Tensor, vision: Tensor
    ) -> tuple[Tensor, Tensor]:
        shared_consensus = (
            self.shared(text) + self.shared(audio) + self.shared(vision)
        ) / 3.0
        components = torch.stack(
            [
                text,
                audio,
                vision,
                shared_consensus,
                self.private_text(text),
                self.private_audio(audio),
                self.private_vision(vision),
            ],
            dim=1,
        )
        mixed = self.mixer(components + self.type_embeddings.unsqueeze(0))
        scores = torch.einsum("bkd,d->bk", mixed, self.query) / math.sqrt(
            mixed.size(-1)
        )
        weights = torch.softmax(scores.float(), dim=1).to(mixed.dtype)
        fused = torch.sum(weights.unsqueeze(-1) * mixed, dim=1)
        return self.output(fused), weights


class LowRankTriModalInteraction(nn.Module):
    """Bounded LMF-style multiplicative residual over pooled modalities."""

    def __init__(
        self,
        dimension: int,
        rank: int,
        dropout: float,
        maximum_mix: float = 0.35,
        initial_mix_logit: float = -2.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        if not 0.0 < maximum_mix <= 1.0:
            raise ValueError("maximum_mix must lie in (0, 1]")
        self.rank = int(rank)
        self.maximum_mix = float(maximum_mix)
        self.text_factor = nn.Parameter(
            torch.empty(rank, dimension + 1, dimension)
        )
        self.audio_factor = nn.Parameter(
            torch.empty(rank, dimension + 1, dimension)
        )
        self.vision_factor = nn.Parameter(
            torch.empty(rank, dimension + 1, dimension)
        )
        for factor in (self.text_factor, self.audio_factor, self.vision_factor):
            nn.init.xavier_uniform_(factor)
        self.output = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(dimension, multiplier=2, dropout=dropout),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * dimension + 2),
            nn.Linear(3 * dimension + 2, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(initial_mix_logit))

    def forward(
        self,
        shared_private: Tensor,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        audio_reliability: Tensor,
        visual_reliability: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ones = torch.ones(text.size(0), 1, dtype=text.dtype, device=text.device)
        augmented = [
            torch.cat([state, ones], dim=-1) for state in (text, audio, vision)
        ]
        text_factor = torch.einsum("bd,rdo->bro", augmented[0], self.text_factor)
        audio_factor = torch.einsum("bd,rdo->bro", augmented[1], self.audio_factor)
        vision_factor = torch.einsum("bd,rdo->bro", augmented[2], self.vision_factor)
        interaction = (text_factor * audio_factor * vision_factor).sum(dim=1)
        interaction = interaction / math.sqrt(float(self.rank))
        candidate = self.output(interaction) + text
        gate_features = torch.cat(
            [
                text,
                audio,
                vision,
                audio_reliability.unsqueeze(-1),
                visual_reliability.unsqueeze(-1),
            ],
            dim=-1,
        )
        mix = self.maximum_mix * torch.sigmoid(
            self.gate(gate_features).squeeze(-1).float()
        )
        fused = shared_private + mix.to(shared_private.dtype).unsqueeze(-1) * (
            candidate - shared_private
        )
        return fused, mix, candidate


class ModalityTokenReducer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        dimension: int,
        token_count: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dimension)
        self.projection = nn.Linear(input_dimension, dimension)
        self.queries = nn.Parameter(torch.empty(token_count, dimension))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(
            dimension, n_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        source = self.projection(self.input_norm(x))
        query = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        tokens, weights = self.attention(
            query=query,
            key=source,
            value=source,
            key_padding_mask=~mask.bool(),
            need_weights=True,
            average_attn_weights=True,
        )
        return self.output_norm(tokens + query), weights


class AdaptiveHyperModalityEncoder(nn.Module):
    """Language-guided hyper-modality tokens adapted from the ALMT principle."""

    def __init__(
        self,
        dimension: int,
        token_count: int,
        depth: int,
        n_heads: int,
        dropout: float,
        text_dimension: int = 768,
        audio_dimension: int = 74,
        vision_dimension: int = 35,
    ) -> None:
        super().__init__()
        reducer_args = (dimension, token_count, n_heads, dropout)
        self.text_reducer = ModalityTokenReducer(
            text_dimension, *reducer_args
        )
        self.audio_reducer = ModalityTokenReducer(
            audio_dimension, *reducer_args
        )
        self.vision_reducer = ModalityTokenReducer(
            vision_dimension, *reducer_args
        )
        self.hyper_tokens = nn.Parameter(torch.empty(token_count, dimension))
        nn.init.normal_(self.hyper_tokens, std=0.02)
        self.text_layers = nn.ModuleList()
        self.audio_cross = nn.ModuleList()
        self.vision_cross = nn.ModuleList()
        self.query_norms = nn.ModuleList()
        self.hyper_norms = nn.ModuleList()
        self.hyper_gates = nn.ParameterList()
        for _ in range(depth):
            self.text_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=dimension,
                    nhead=n_heads,
                    dim_feedforward=2 * dimension,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            self.audio_cross.append(
                nn.MultiheadAttention(
                    dimension, n_heads, dropout=dropout, batch_first=True
                )
            )
            self.vision_cross.append(
                nn.MultiheadAttention(
                    dimension, n_heads, dropout=dropout, batch_first=True
                )
            )
            self.query_norms.append(nn.LayerNorm(dimension))
            self.hyper_norms.append(nn.LayerNorm(dimension))
            self.hyper_gates.append(nn.Parameter(torch.tensor(-1.0)))
        self.fusion_query = nn.Parameter(torch.empty(1, dimension))
        nn.init.normal_(self.fusion_query, std=0.02)
        self.fusion_attention = nn.MultiheadAttention(
            dimension, n_heads, dropout=dropout, batch_first=True
        )
        self.output = ResidualMLP(dimension, multiplier=2, dropout=dropout)

    def forward(
        self,
        text: Tensor,
        text_mask: Tensor,
        audio: Tensor,
        audio_mask: Tensor,
        vision: Tensor,
        vision_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        text_tokens, text_weights = self.text_reducer(text, text_mask)
        audio_tokens, _ = self.audio_reducer(audio, audio_mask)
        vision_tokens, _ = self.vision_reducer(vision, vision_mask)
        hyper = self.hyper_tokens.unsqueeze(0).expand(text.size(0), -1, -1)
        for text_layer, audio_cross, vision_cross, query_norm, hyper_norm, gate in zip(
            self.text_layers,
            self.audio_cross,
            self.vision_cross,
            self.query_norms,
            self.hyper_norms,
            self.hyper_gates,
        ):
            text_tokens = text_layer(text_tokens)
            query = query_norm(text_tokens)
            audio_context, _ = audio_cross(
                query=query, key=audio_tokens, value=audio_tokens, need_weights=False
            )
            vision_context, _ = vision_cross(
                query=query, key=vision_tokens, value=vision_tokens, need_weights=False
            )
            hyper = hyper_norm(
                hyper + torch.sigmoid(gate) * (audio_context + vision_context)
            )
        evidence = torch.cat([text_tokens, hyper], dim=1)
        query = self.fusion_query.unsqueeze(0).expand(text.size(0), -1, -1)
        fused, _ = self.fusion_attention(
            query=query, key=evidence, value=evidence, need_weights=False
        )
        return self.output(fused[:, 0]), hyper, text_weights


class ClassPrototypeRouter(nn.Module):
    """Label-query cross-attention with learnable class prototypes.

    Each class owns a query that extracts its evidence from all available token
    streams. The class-conditioned vector is compared with its own prototype,
    so Neutral is not forced to share one linear surface with both polar ends.
    """

    def __init__(self, dimension: int, dropout: float, n_heads: int) -> None:
        super().__init__()
        self.n_heads = int(n_heads)
        self.class_queries = nn.Parameter(torch.empty(3, dimension))
        self.class_prototypes = nn.Parameter(torch.empty(3, dimension))
        nn.init.normal_(self.class_queries, std=0.02)
        nn.init.normal_(self.class_prototypes, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.evidence = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.embedding_projection = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )

    def forward(
        self,
        fused: Tensor,
        evidence_tokens: Tensor,
        evidence_mask: Tensor,
        evidence_weight: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = fused.size(0)
        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = None
        key_padding_mask = ~evidence_mask.bool()
        if evidence_weight is not None:
            log_weight = evidence_weight.float().clamp_min(1e-8).log()
            log_weight = log_weight.masked_fill(~evidence_mask.bool(), -1e4)
            attention_mask = (
                log_weight.unsqueeze(1)
                .expand(-1, queries.size(1), -1)
                .repeat_interleave(self.n_heads, dim=0)
            )
            key_padding_mask = None
        attended, attention = self.cross_attention(
            query=queries,
            key=evidence_tokens,
            value=evidence_tokens,
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        class_states = self.output_norm(attended + queries + fused.unsqueeze(1))
        normalized_states = F.normalize(class_states.float(), dim=-1)
        normalized_prototypes = F.normalize(self.class_prototypes.float(), dim=-1)
        scale = self.logit_scale.float().exp().clamp(max=100.0)
        prototype_similarity = torch.einsum(
            "bcd,cd->bc", normalized_states, normalized_prototypes
        )
        logits = scale * prototype_similarity + self.evidence(class_states).squeeze(-1)
        embedding = F.normalize(self.embedding_projection(fused).float(), dim=-1)
        return logits, embedding, attention


class ZeroInflatedOrdinalExpert(nn.Module):
    """Factor Neutral-vs-Polar and Negative-vs-Positive without replacing CE."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 3),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        neutral_logit, polarity_logit, magnitude_logit = self.network(x).unbind(-1)
        p_neutral = torch.sigmoid(neutral_logit.float())
        p_positive = torch.sigmoid(polarity_logit.float())
        probabilities = torch.stack(
            [
                (1.0 - p_neutral) * (1.0 - p_positive),
                p_neutral,
                (1.0 - p_neutral) * p_positive,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        magnitude = 3.0 * torch.sigmoid(magnitude_logit.float())
        regression = (1.0 - p_neutral) * torch.tanh(polarity_logit.float()) * magnitude
        return {
            "neutral_logit": neutral_logit,
            "polarity_logit": polarity_logit,
            "magnitude_logit": magnitude_logit,
            "probabilities": probabilities,
            "regression": regression,
        }


class PolarDistanceNeutralExpert(nn.Module):
    """Decouple text polarity from a multimodal Neutral boundary.

    The current-utterance text anchor is the only input allowed to determine
    Negative-versus-Positive odds.  Context and non-verbal evidence can only
    change the Neutral mass, so a noisy modality cannot reverse the polar
    ordering.  Neutral is modelled as a learned interval around the text
    polarity boundary, with a bounded evidence residual near that boundary.
    """

    def __init__(
        self,
        dimension: int,
        dropout: float,
        maximum_boundary_mix: float = 0.50,
        maximum_evidence_shift: float = 1.50,
        boundary_mix_logit: float = 0.0,
        distance_scale_raw: float = -1.5,
        maximum_polarity_shift: float = 0.0,
        polarity_prior_maximum_shift: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < maximum_boundary_mix <= 1.0:
            raise ValueError("maximum_boundary_mix must lie in (0, 1]")
        if maximum_evidence_shift <= 0.0:
            raise ValueError("maximum_evidence_shift must be positive")
        if maximum_polarity_shift < 0.0:
            raise ValueError("maximum_polarity_shift cannot be negative")
        if polarity_prior_maximum_shift < 0.0:
            raise ValueError("polarity_prior_maximum_shift cannot be negative")
        self.maximum_boundary_mix = float(maximum_boundary_mix)
        self.maximum_evidence_shift = float(maximum_evidence_shift)
        self.maximum_polarity_shift = float(maximum_polarity_shift)
        self.polarity_prior_maximum_shift = float(
            polarity_prior_maximum_shift
        )
        self.polarity = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )
        self.neutral_anchor = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )
        evidence_dimension = 4 * dimension + 3
        self.neutral_evidence = nn.Sequential(
            nn.LayerNorm(evidence_dimension),
            nn.Linear(evidence_dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )
        if self.maximum_polarity_shift > 0.0:
            # A soft current-utterance anchor: contextual/non-verbal evidence
            # may repair polarity only through a bounded, uncertainty-aware
            # trust route.  The original hard invariant remains the default.
            self.polarity_correction = nn.Sequential(
                nn.LayerNorm(evidence_dimension),
                nn.Linear(evidence_dimension, dimension),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dimension, 1),
            )
            self.polarity_trust = nn.Sequential(
                nn.LayerNorm(evidence_dimension),
                nn.Linear(evidence_dimension, dimension // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dimension // 2, 1),
            )
            nn.init.constant_(self.polarity_trust[-1].bias, -1.0)
        self.magnitude = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )
        self.boundary_mix_logit = nn.Parameter(
            torch.tensor(float(boundary_mix_logit))
        )
        self.distance_scale_raw = nn.Parameter(
            torch.tensor(float(distance_scale_raw))
        )

    def forward(
        self,
        text_anchor: Tensor,
        boundary_state: Tensor,
        base_probability: Tensor,
        context_reliability: Tensor,
        audio_reliability: Tensor,
        visual_reliability: Tensor,
        polarity_prior_logit: Tensor | None = None,
    ) -> dict[str, Tensor]:
        polarity_model_logit = self.polarity(text_anchor).squeeze(-1).float()
        polarity_prior_residual = polarity_model_logit.new_zeros(
            polarity_model_logit.shape
        )
        polarity_prior_value = polarity_model_logit.new_zeros(
            polarity_model_logit.shape
        )
        if polarity_prior_logit is not None:
            polarity_prior_value = polarity_prior_logit.float()
            polarity_prior_residual = self.polarity_prior_maximum_shift * torch.tanh(
                polarity_model_logit
            )
            polarity_anchor_logit = (
                polarity_prior_value + polarity_prior_residual
            )
        else:
            polarity_anchor_logit = polarity_model_logit
        anchor_polar_probability = torch.sigmoid(polarity_anchor_logit)
        polar_uncertainty = 4.0 * anchor_polar_probability * (
            1.0 - anchor_polar_probability
        )
        reliability = torch.stack(
            [context_reliability, audio_reliability, visual_reliability], dim=-1
        ).float()
        evidence = torch.cat(
            [
                text_anchor,
                boundary_state,
                torch.abs(text_anchor - boundary_state),
                text_anchor * boundary_state,
                reliability.to(text_anchor.dtype),
            ],
            dim=-1,
        )
        polarity_residual = polarity_anchor_logit.new_zeros(
            polarity_anchor_logit.shape
        )
        polarity_trust = polarity_anchor_logit.new_zeros(
            polarity_anchor_logit.shape
        )
        if self.maximum_polarity_shift > 0.0:
            polarity_trust = torch.sigmoid(
                self.polarity_trust(evidence).squeeze(-1).float()
            )
            polarity_residual = (
                self.maximum_polarity_shift
                * polarity_trust
                * torch.tanh(
                    self.polarity_correction(evidence).squeeze(-1).float()
                )
                * (0.25 + 0.75 * polar_uncertainty)
            )
        polarity_logit = polarity_anchor_logit + polarity_residual
        polar_probability = torch.sigmoid(polarity_logit)
        # Evidence is strongest around the polar decision boundary and remains
        # bounded everywhere.  This is a trust region, not a free class-logit
        # residual.
        evidence_shift = self.maximum_evidence_shift * torch.tanh(
            self.neutral_evidence(evidence).squeeze(-1).float()
        ) * (0.5 + 0.5 * polar_uncertainty)
        distance_scale = F.softplus(self.distance_scale_raw.float())
        boundary_logit = (
            self.neutral_anchor(text_anchor).squeeze(-1).float()
            - distance_scale * polarity_logit.abs()
            + evidence_shift
        )
        boundary_probability = torch.sigmoid(boundary_logit)
        boundary_mix = self.maximum_boundary_mix * torch.sigmoid(
            self.boundary_mix_logit.float()
        )
        neutral_probability = (
            (1.0 - boundary_mix) * base_probability[:, 1].float()
            + boundary_mix * boundary_probability
        ).clamp(1e-6, 1.0 - 1e-6)
        probabilities = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - polar_probability),
                neutral_probability,
                (1.0 - neutral_probability) * polar_probability,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        magnitude = 3.0 * torch.sigmoid(
            self.magnitude(boundary_state).squeeze(-1).float()
        )
        regression = (
            (1.0 - neutral_probability)
            * torch.tanh(polarity_logit)
            * magnitude
        )
        return {
            "neutral_logit": torch.logit(neutral_probability),
            "polarity_logit": polarity_logit,
            "polarity_anchor_logit": polarity_anchor_logit,
            "polarity_prior_logit": polarity_prior_value,
            "polarity_prior_residual": polarity_prior_residual,
            "polarity_residual": polarity_residual,
            "polarity_trust": polarity_trust,
            "probabilities": probabilities,
            "regression": regression,
            "boundary_probability": boundary_probability,
            "boundary_mix": boundary_mix,
            "evidence_shift": evidence_shift,
            "distance_scale": distance_scale,
            "polar_uncertainty": polar_uncertainty,
        }


class OrderedIntervalExpert(nn.Module):
    """Rank-consistent Negative < Neutral < Positive interval expert."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )
        self.center = nn.Parameter(torch.zeros(()))
        self.width_raw = nn.Parameter(torch.tensor(0.0))
        self.temperature_raw = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        score = self.score(x).squeeze(-1).float()
        half_width = 0.10 + F.softplus(self.width_raw.float())
        temperature = 0.10 + F.softplus(self.temperature_raw.float())
        lower = self.center.float() - half_width
        upper = self.center.float() + half_width
        threshold_logits = torch.stack(
            [(score - lower) / temperature, (score - upper) / temperature],
            dim=-1,
        )
        above_lower = torch.sigmoid(threshold_logits[:, 0])
        above_upper = torch.sigmoid(threshold_logits[:, 1])
        probabilities = torch.stack(
            [1.0 - above_lower, above_lower - above_upper, above_upper],
            dim=-1,
        ).clamp_min(1e-8)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return {
            "threshold_logits": threshold_logits,
            "probabilities": probabilities,
            "score": score,
            "half_width": half_width,
        }


class LatentAffectSubcenterHead(nn.Module):
    """A bounded multi-prototype class geometry for heterogeneous affect modes."""

    def __init__(
        self,
        dimension: int,
        subcenters: int = 3,
        maximum_mix: float = 0.35,
        initial_mix_logit: float = -3.0,
        initial_scale: float = 10.0,
    ) -> None:
        super().__init__()
        if subcenters < 2:
            raise ValueError("subcenters must be at least two")
        self.subcenters = int(subcenters)
        self.maximum_mix = float(maximum_mix)
        self.centers = nn.Parameter(torch.empty(3, self.subcenters, dimension))
        nn.init.normal_(self.centers, std=0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))
        self.mix_logit = nn.Parameter(torch.tensor(float(initial_mix_logit)))

    def forward(self, embedding: Tensor) -> dict[str, Tensor]:
        normalized_embedding = F.normalize(embedding.float(), dim=-1)
        normalized_centers = F.normalize(self.centers.float(), dim=-1)
        cosine = torch.einsum(
            "bd,ckd->bck", normalized_embedding, normalized_centers
        )
        scale = self.log_scale.float().exp().clamp(1.0, 50.0)
        logits = torch.logsumexp(scale * cosine, dim=-1) - math.log(
            self.subcenters
        )
        center_similarity = torch.einsum(
            "ckd,cjd->ckj", normalized_centers, normalized_centers
        )
        off_diagonal = ~torch.eye(
            self.subcenters, dtype=torch.bool, device=embedding.device
        ).unsqueeze(0)
        diversity = center_similarity.masked_select(off_diagonal).square().mean()
        return {
            "logits": logits,
            "mix": self.maximum_mix * torch.sigmoid(self.mix_logit.float()),
            "diversity": diversity,
            "scale": scale,
        }


class ConflictIgnoranceNeutralHead(nn.Module):
    """Bound Neutral mass using explicit cross-modal conflict and ignorance.

    Each modality emits a binary polarity opinion and a non-negative evidence
    strength. Reliable but opposing opinions create conflict; weak total
    evidence creates ignorance. The final correction is uncertainty-gated and
    can only change Neutral-vs-Polar odds. Negative-vs-Positive odds therefore
    remain exactly equal to the trained parent model. A zero-initialized final
    layer makes the module an exact identity transformation at initialization.
    """

    def __init__(
        self,
        dimension: int,
        dropout: float,
        maximum_neutral_logit_shift: float = 0.75,
        evidence_prior: float = 2.0,
    ) -> None:
        super().__init__()
        hidden = max(dimension // 4, 32)
        self.opinion_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dimension),
                    nn.Linear(dimension, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 2),
                )
                for _ in range(3)
            ]
        )
        # parent posterior (3), polarity opinions (3), reliable strengths (3),
        # conflict/ignorance/balance/uncertainty/parent-neutral (5), and two
        # non-verbal reliabilities (2).
        diagnostic_dimension = 16
        self.boundary = nn.Sequential(
            nn.LayerNorm(diagnostic_dimension),
            nn.Linear(diagnostic_dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.boundary[-1].weight)
        nn.init.zeros_(self.boundary[-1].bias)
        self.maximum_neutral_logit_shift = float(maximum_neutral_logit_shift)
        self.evidence_prior = float(evidence_prior)
        if self.maximum_neutral_logit_shift <= 0.0:
            raise ValueError("maximum_neutral_logit_shift must be positive")
        if self.evidence_prior <= 0.0:
            raise ValueError("evidence_prior must be positive")

    def forward(
        self,
        parent_probability: Tensor,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        audio_reliability: Tensor,
        visual_reliability: Tensor,
    ) -> dict[str, Tensor]:
        parent_probability = parent_probability.float().clamp_min(1e-8)
        reliability = torch.stack(
            [
                torch.ones_like(audio_reliability, dtype=torch.float32),
                audio_reliability.float().clamp(0.0, 1.0),
                visual_reliability.float().clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        opinions = torch.stack(
            [
                head(state).float()
                for head, state in zip(
                    self.opinion_heads, (text, audio, vision), strict=True
                )
            ],
            dim=1,
        )
        polarity_probability = torch.sigmoid(opinions[..., 0])
        evidence_strength = F.softplus(opinions[..., 1]) * reliability
        positive_evidence = (evidence_strength * polarity_probability).sum(dim=1)
        negative_evidence = (
            evidence_strength * (1.0 - polarity_probability)
        ).sum(dim=1)
        total_evidence = positive_evidence + negative_evidence
        ignorance = self.evidence_prior / (self.evidence_prior + total_evidence)
        conflict = (
            2.0
            * torch.minimum(positive_evidence, negative_evidence)
            / total_evidence.clamp_min(1e-6)
        )
        balance = (
            (positive_evidence - negative_evidence)
            / total_evidence.clamp_min(1e-6)
        )
        top_two = torch.topk(parent_probability, k=2, dim=-1).values
        posterior_uncertainty = (
            1.0 - (top_two[:, 0] - top_two[:, 1])
        ).clamp(0.0, 1.0)
        diagnostics = torch.cat(
            [
                parent_probability,
                polarity_probability,
                evidence_strength,
                conflict.unsqueeze(-1),
                ignorance.unsqueeze(-1),
                balance.unsqueeze(-1),
                posterior_uncertainty.unsqueeze(-1),
                parent_probability[:, 1:2],
                reliability[:, 1:],
            ],
            dim=-1,
        )
        raw_shift = torch.tanh(self.boundary(diagnostics).squeeze(-1).float())
        neutral_shift = (
            self.maximum_neutral_logit_shift
            * posterior_uncertainty
            * raw_shift
        )
        parent_neutral = parent_probability[:, 1].clamp(1e-6, 1.0 - 1e-6)
        neutral_logit = torch.logit(parent_neutral) + neutral_shift
        neutral_probability = torch.sigmoid(neutral_logit)
        polar_parent = parent_probability[:, [0, 2]]
        positive_within_polar = (
            polar_parent[:, 1] / polar_parent.sum(dim=-1).clamp_min(1e-8)
        )
        probability = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_within_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_within_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probability = probability / probability.sum(dim=-1, keepdim=True)
        return {
            "probabilities": probability,
            "neutral_logit": torch.logit(
                neutral_probability.clamp(1e-6, 1.0 - 1e-6)
            ),
            "polarity_logit": torch.log(
                positive_within_polar.clamp_min(1e-8)
            )
            - torch.log((1.0 - positive_within_polar).clamp_min(1e-8)),
            "regression": 3.0 * (probability[:, 2] - probability[:, 0]),
            "neutral_shift": neutral_shift,
            "conflict": conflict,
            "ignorance": ignorance,
            "evidence_strength": evidence_strength,
        }


class AlignedTemporalIncongruityHead(nn.Module):
    """Read word-aligned tri-modal mismatch before utterance pooling.

    The pretrained text branch is a strong semantic anchor, but the legacy
    aligned features retain local acoustic and visual events that disappear
    when every stream is pooled independently. This head projects the aligned
    text/audio/vision frames into a compact shared space, represents signed
    multiplicative agreement and absolute disagreement for all modality pairs,
    and sparsely pools only valid frames. Its two bounded axes alter
    Neutral-vs-Polar and (more conservatively) Positive-vs-Negative odds.
    """

    def __init__(
        self,
        text_dimension: int,
        audio_dimension: int,
        vision_dimension: int,
        hidden_dimension: int = 64,
        dropout: float = 0.15,
        maximum_neutral_logit_shift: float = 0.75,
        maximum_polarity_logit_shift: float = 0.25,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dimension)
        if hidden < 16:
            raise ValueError("hidden_dimension must be at least 16")
        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(size), nn.Linear(size, hidden), nn.GELU())
                for size in (text_dimension, audio_dimension, vision_dimension)
            ]
        )
        self.interaction = nn.Sequential(
            nn.LayerNorm(6 * hidden),
            nn.Linear(6 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
        )
        self.temporal_score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.axes = nn.Sequential(
            nn.LayerNorm(hidden + 5),
            nn.Linear(hidden + 5, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.axes[-1].weight)
        nn.init.zeros_(self.axes[-1].bias)
        self.maximum_neutral_logit_shift = float(maximum_neutral_logit_shift)
        self.maximum_polarity_logit_shift = float(maximum_polarity_logit_shift)

    def forward(
        self,
        parent_probability: Tensor,
        legacy_text: Tensor,
        audio: Tensor,
        vision: Tensor,
        text_mask: Tensor,
        audio_mask: Tensor,
        vision_mask: Tensor,
        audio_reliability: Tensor,
        visual_reliability: Tensor,
    ) -> dict[str, Tensor]:
        parent_probability = parent_probability.float().clamp_min(1e-8)
        text_state, audio_state, vision_state = [
            projection(value)
            for projection, value in zip(
                self.projections, (legacy_text, audio, vision), strict=True
            )
        ]
        valid = text_mask.bool() & audio_mask.bool() & vision_mask.bool()
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if empty.any():
            safe_valid[empty, 0] = True
        interactions = self.interaction(
            torch.cat(
                [
                    text_state * audio_state,
                    text_state * vision_state,
                    audio_state * vision_state,
                    torch.abs(text_state - audio_state),
                    torch.abs(text_state - vision_state),
                    torch.abs(audio_state - vision_state),
                ],
                dim=-1,
            )
        )
        scores = self.temporal_score(interactions).squeeze(-1).float()
        weights = torch.softmax(scores.masked_fill(~safe_valid, -1e4), dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(weights.unsqueeze(-1).to(interactions.dtype) * interactions, dim=1)
        top_two = torch.topk(parent_probability, k=2, dim=-1).values
        uncertainty = (1.0 - (top_two[:, 0] - top_two[:, 1])).clamp(0.0, 1.0)
        axes = torch.tanh(
            self.axes(
                torch.cat(
                    [
                        pooled,
                        parent_probability.to(pooled.dtype),
                        audio_reliability.unsqueeze(-1).to(pooled.dtype),
                        visual_reliability.unsqueeze(-1).to(pooled.dtype),
                    ],
                    dim=-1,
                )
            ).float()
        )
        neutral_shift = self.maximum_neutral_logit_shift * uncertainty * axes[:, 0]
        polarity_shift = self.maximum_polarity_logit_shift * uncertainty * axes[:, 1]
        parent_neutral = parent_probability[:, 1].clamp(1e-6, 1.0 - 1e-6)
        neutral_probability = torch.sigmoid(torch.logit(parent_neutral) + neutral_shift)
        polar_parent = parent_probability[:, [0, 2]]
        positive_within_polar = (
            polar_parent[:, 1] / polar_parent.sum(dim=-1).clamp_min(1e-8)
        ).clamp(1e-6, 1.0 - 1e-6)
        positive_within_polar = torch.sigmoid(
            torch.logit(positive_within_polar) + polarity_shift
        )
        probability = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_within_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_within_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probability = probability / probability.sum(dim=-1, keepdim=True)
        entropy = -(
            weights * weights.clamp_min(1e-8).log()
        ).sum(dim=-1)
        return {
            "probabilities": probability,
            "neutral_logit": torch.logit(
                neutral_probability.clamp(1e-6, 1.0 - 1e-6)
            ),
            "polarity_logit": torch.logit(positive_within_polar),
            "regression": 3.0 * (probability[:, 2] - probability[:, 0]),
            "neutral_shift": neutral_shift,
            "polarity_shift": polarity_shift,
            "attention_entropy": entropy,
        }


class ContrastiveFeatureDecompositionHead(nn.Module):
    """ConFEDE/HyCon-inspired shared-private tri-modal consensus head.

    A weight-shared projector extracts class-relevant coordinates from every
    modality while three private projectors retain complementary residue. The
    private vectors are orthogonalized sample-wise against their shared vectors
    before classification. Shared embeddings are exposed for cross-modal
    supervised contrastive learning, and each modality has its own auxiliary
    classifier/distillation path. A zero-initialized bounded logit residual
    makes the complete branch an exact parent identity at initialization.
    """

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int = 128,
        dropout: float = 0.15,
        maximum_logit_shift: float = 0.50,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dimension)
        if hidden < 16:
            raise ValueError("hidden_dimension must be at least 16")
        self.shared = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden, bias=False),
            nn.GELU(),
        )
        self.private = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dimension),
                    nn.Linear(dimension, hidden, bias=False),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.modality_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(2 * hidden),
                    nn.Linear(2 * hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 3),
                )
                for _ in range(3)
            ]
        )
        self.consensus = nn.Sequential(
            nn.LayerNorm(6 * hidden),
            nn.Linear(6 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(2 * hidden, multiplier=2, dropout=dropout),
            nn.Linear(2 * hidden, 3),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(11),
            nn.Linear(11, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 3),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.maximum_logit_shift = float(maximum_logit_shift)
        if self.maximum_logit_shift <= 0.0:
            raise ValueError("maximum_logit_shift must be positive")

    def forward(
        self,
        parent_probability: Tensor,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        audio_reliability: Tensor,
        visual_reliability: Tensor,
    ) -> dict[str, Tensor]:
        parent_probability = parent_probability.float().clamp_min(1e-8)
        states = (text, audio, vision)
        shared = torch.stack([self.shared(state) for state in states], dim=1)
        private = torch.stack(
            [head(state) for head, state in zip(self.private, states, strict=True)],
            dim=1,
        )
        projection = (private * shared).sum(dim=-1, keepdim=True)
        projection = projection / shared.square().sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        private = private - projection * shared
        modality_features = torch.cat([shared, private], dim=-1)
        modality_logits = torch.stack(
            [
                head(modality_features[:, index])
                for index, head in enumerate(self.modality_classifiers)
            ],
            dim=1,
        ).float()
        consensus_logits = self.consensus(
            torch.cat([shared.flatten(1), private.flatten(1)], dim=-1)
        ).float()
        modality_probability = torch.softmax(modality_logits, dim=-1)
        reliability = torch.stack(
            [
                torch.ones_like(audio_reliability, dtype=torch.float32),
                audio_reliability.float().clamp(0.0, 1.0),
                visual_reliability.float().clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        reliability = reliability / reliability.sum(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        modality_consensus = torch.sum(
            reliability.unsqueeze(-1) * modality_probability, dim=1
        )
        consensus_probability = torch.softmax(consensus_logits, dim=-1)
        agreement = (modality_consensus * parent_probability).sum(dim=-1, keepdim=True)
        entropy = -(
            modality_consensus * modality_consensus.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / math.log(3.0)
        residual_features = torch.cat(
            [
                parent_probability,
                consensus_probability,
                modality_consensus,
                agreement,
                entropy,
            ],
            dim=-1,
        )
        residual = self.maximum_logit_shift * torch.tanh(
            self.residual(residual_features).float()
        )
        probability = torch.softmax(parent_probability.log() + residual, dim=-1)
        positive_within_polar = (
            probability[:, 2]
            / (probability[:, 0] + probability[:, 2]).clamp_min(1e-8)
        )
        orthogonality = F.cosine_similarity(
            shared.float(), private.float(), dim=-1, eps=1e-6
        ).abs()
        return {
            "probabilities": probability,
            "neutral_logit": torch.logit(
                probability[:, 1].clamp(1e-6, 1.0 - 1e-6)
            ),
            "polarity_logit": torch.logit(
                positive_within_polar.clamp(1e-6, 1.0 - 1e-6)
            ),
            "regression": 3.0 * (probability[:, 2] - probability[:, 0]),
            "parent_probability": parent_probability,
            "shared_embeddings": F.normalize(shared.float(), dim=-1),
            "modality_logits": modality_logits,
            "modality_reliability": reliability,
            "consensus_logits": consensus_logits,
            "residual": residual,
            "orthogonality": orthogonality,
        }


class TextConditionalInnovationHead(nn.Module):
    """TFR-Net-inspired text-conditional non-verbal innovation expert.

    The text anchor predicts the acoustic and visual states expected from its
    literal semantics.  Only the unpredictable non-verbal residue is offered
    to the decision branch.  This separates useful incongruity from ordinary
    cross-modal scale differences.  Reconstruction targets are fixed
    affine-free normalizations of the frozen parent states, while a
    zero-initialized bounded two-axis correction preserves the parent posterior
    exactly at initialization.
    """

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int = 128,
        dropout: float = 0.15,
        maximum_neutral_logit_shift: float = 0.75,
        maximum_polarity_logit_shift: float = 0.15,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dimension)
        if hidden < 16:
            raise ValueError("hidden_dimension must be at least 16")
        self.audio_target_norm = nn.LayerNorm(
            dimension, elementwise_affine=False
        )
        self.vision_target_norm = nn.LayerNorm(
            dimension, elementwise_affine=False
        )

        def predictor() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualMLP(hidden, multiplier=2, dropout=dropout),
                nn.Linear(hidden, dimension),
            )

        def projector() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, hidden, bias=False),
                nn.GELU(),
            )

        self.audio_predictor = predictor()
        self.vision_predictor = predictor()
        self.text_projector = projector()
        self.audio_innovation_projector = projector()
        self.vision_innovation_projector = projector()
        self.joint = nn.Sequential(
            nn.LayerNorm(5 * hidden),
            nn.Linear(5 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(2 * hidden, multiplier=2, dropout=dropout),
        )
        self.innovation_classifier = nn.Linear(2 * hidden, 3)
        self.innovation_neutral = nn.Linear(2 * hidden, 1)
        # parent posterior (3), two reliabilities, two normalized errors
        self.axes = nn.Sequential(
            nn.LayerNorm(2 * hidden + 7),
            nn.Linear(2 * hidden + 7, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.axes[-1].weight)
        nn.init.zeros_(self.axes[-1].bias)
        self.maximum_neutral_logit_shift = float(
            maximum_neutral_logit_shift
        )
        self.maximum_polarity_logit_shift = float(
            maximum_polarity_logit_shift
        )
        if self.maximum_neutral_logit_shift <= 0.0:
            raise ValueError("maximum_neutral_logit_shift must be positive")
        if self.maximum_polarity_logit_shift < 0.0:
            raise ValueError("maximum_polarity_logit_shift cannot be negative")

    def forward(
        self,
        parent_probability: Tensor,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        audio_reliability: Tensor,
        visual_reliability: Tensor,
    ) -> dict[str, Tensor]:
        parent_probability = parent_probability.float().clamp_min(1e-8)
        audio_target = self.audio_target_norm(audio.float())
        vision_target = self.vision_target_norm(vision.float())
        predicted_audio = self.audio_predictor(text).float()
        predicted_vision = self.vision_predictor(text).float()
        audio_innovation = audio_target - predicted_audio
        vision_innovation = vision_target - predicted_vision
        text_state = self.text_projector(text)
        audio_state = self.audio_innovation_projector(audio_innovation)
        vision_state = self.vision_innovation_projector(vision_innovation)
        joint = self.joint(
            torch.cat(
                [
                    text_state,
                    audio_state,
                    vision_state,
                    audio_state * vision_state,
                    torch.abs(audio_state - vision_state),
                ],
                dim=-1,
            )
        )
        audio_error = audio_innovation.square().mean(dim=-1).sqrt()
        vision_error = vision_innovation.square().mean(dim=-1).sqrt()
        normalized_errors = torch.stack(
            [audio_error, vision_error], dim=-1
        )
        normalized_errors = normalized_errors / (
            1.0 + normalized_errors
        )
        axes = torch.tanh(
            self.axes(
                torch.cat(
                    [
                        joint,
                        parent_probability,
                        audio_reliability.float().unsqueeze(-1),
                        visual_reliability.float().unsqueeze(-1),
                        normalized_errors,
                    ],
                    dim=-1,
                )
            ).float()
        )
        neutral_shift = self.maximum_neutral_logit_shift * axes[:, 0]
        polarity_shift = self.maximum_polarity_logit_shift * axes[:, 1]
        parent_neutral = parent_probability[:, 1].clamp(1e-6, 1.0 - 1e-6)
        neutral_probability = torch.sigmoid(
            torch.logit(parent_neutral) + neutral_shift
        )
        parent_polar = parent_probability[:, [0, 2]]
        positive_within_polar = (
            parent_polar[:, 1]
            / parent_polar.sum(dim=-1).clamp_min(1e-8)
        ).clamp(1e-6, 1.0 - 1e-6)
        positive_within_polar = torch.sigmoid(
            torch.logit(positive_within_polar) + polarity_shift
        )
        probability = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_within_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_within_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probability = probability / probability.sum(dim=-1, keepdim=True)
        return {
            "probabilities": probability,
            "neutral_logit": torch.logit(
                neutral_probability.clamp(1e-6, 1.0 - 1e-6)
            ),
            "polarity_logit": torch.logit(
                positive_within_polar.clamp(1e-6, 1.0 - 1e-6)
            ),
            "regression": 3.0 * (probability[:, 2] - probability[:, 0]),
            "parent_probability": parent_probability,
            "innovation_logits": self.innovation_classifier(joint).float(),
            "innovation_neutral_logit": self.innovation_neutral(joint)
            .squeeze(-1)
            .float(),
            "predicted_audio": predicted_audio,
            "predicted_vision": predicted_vision,
            "audio_target": audio_target.detach(),
            "vision_target": vision_target.detach(),
            "audio_error": audio_error,
            "vision_error": vision_error,
            "neutral_shift": neutral_shift,
            "polarity_shift": polarity_shift,
        }


class AgreementAwareEmotionEvidence(nn.Module):
    """Fuse a frozen emotion space without allowing it to dominate the task model.

    The representation path and the two semantic decision axes start as exact
    no-ops.  Training may learn a bounded change to Neutral-vs-Polar and a
    smaller bounded change to Positive-vs-Negative.  This preserves a strong
    task-domain anchor while exposing independently pretrained emotion evidence.
    """

    def __init__(
        self,
        primary_dimension: int,
        emotion_dimension: int,
        dropout: float,
        maximum_representation_shift: float = 0.35,
        maximum_neutral_logit_shift: float = 0.75,
        maximum_polarity_logit_shift: float = 0.25,
    ) -> None:
        super().__init__()
        self.maximum_representation_shift = float(maximum_representation_shift)
        self.maximum_neutral_logit_shift = float(maximum_neutral_logit_shift)
        self.maximum_polarity_logit_shift = float(maximum_polarity_logit_shift)
        self.emotion_projection = nn.Sequential(
            nn.LayerNorm(2 * emotion_dimension),
            nn.Linear(2 * emotion_dimension, primary_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(primary_dimension, multiplier=2, dropout=dropout),
        )
        representation_dimension = 4 * primary_dimension + 4
        self.representation_gate = nn.Sequential(
            nn.LayerNorm(representation_dimension),
            nn.Linear(representation_dimension, primary_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(primary_dimension, 1),
        )
        self.representation_delta = nn.Sequential(
            nn.LayerNorm(representation_dimension),
            nn.Linear(representation_dimension, primary_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(primary_dimension, primary_dimension),
        )
        nn.init.zeros_(self.representation_delta[-1].weight)
        nn.init.zeros_(self.representation_delta[-1].bias)
        self.representation_norm = nn.LayerNorm(primary_dimension)

        decision_dimension = 4 * primary_dimension + 11
        self.decision_reliability = nn.Sequential(
            nn.LayerNorm(decision_dimension),
            nn.Linear(decision_dimension, primary_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(primary_dimension, 1),
        )
        self.decision_delta = nn.Sequential(
            nn.LayerNorm(decision_dimension),
            nn.Linear(decision_dimension, primary_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(primary_dimension, 2),
        )
        nn.init.zeros_(self.decision_delta[-1].weight)
        nn.init.zeros_(self.decision_delta[-1].bias)
        self.auxiliary = nn.Sequential(
            nn.LayerNorm(primary_dimension),
            nn.Dropout(dropout),
            nn.Linear(primary_dimension, 3),
        )

    @staticmethod
    def entropy(probability: Tensor) -> Tensor:
        value = probability.float().clamp_min(1e-8)
        return -(value * value.log()).sum(dim=-1, keepdim=True) / math.log(3.0)

    @staticmethod
    def joint_features(primary: Tensor, emotion: Tensor) -> list[Tensor]:
        return [primary, emotion, torch.abs(primary - emotion), primary * emotion]

    def adapt(
        self,
        primary: Tensor,
        emotion_encoded: Tensor,
        emotion_mask: Tensor,
        emotion_probability: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        emotion = self.emotion_projection(
            torch.cat(
                [
                    emotion_encoded[:, 0],
                    masked_mean(emotion_encoded, emotion_mask.bool()),
                ],
                dim=-1,
            )
        )
        features = torch.cat(
            self.joint_features(primary, emotion)
            + [emotion_probability.float(), self.entropy(emotion_probability)],
            dim=-1,
        )
        reliability = torch.sigmoid(self.representation_gate(features)).squeeze(-1)
        delta = self.maximum_representation_shift * torch.tanh(
            self.representation_delta(features)
        )
        adapted = self.representation_norm(
            primary + reliability.unsqueeze(-1) * delta
        )
        return adapted, emotion, reliability, self.auxiliary(emotion)

    def correct(
        self,
        probability: Tensor,
        primary: Tensor,
        emotion: Tensor,
        emotion_probability: Tensor,
    ) -> dict[str, Tensor]:
        probability = probability.float().clamp(1e-7, 1.0 - 1e-7)
        emotion_probability = emotion_probability.float().clamp(1e-7, 1.0 - 1e-7)
        features = torch.cat(
            self.joint_features(primary, emotion)
            + [
                probability,
                emotion_probability,
                torch.abs(probability - emotion_probability),
                self.entropy(probability),
                self.entropy(emotion_probability),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.decision_reliability(features)).squeeze(-1)
        raw_delta = torch.tanh(self.decision_delta(features)) * reliability.unsqueeze(-1)
        neutral_delta = self.maximum_neutral_logit_shift * raw_delta[:, 0]
        polarity_delta = self.maximum_polarity_logit_shift * raw_delta[:, 1]

        neutral_logit = torch.logit(probability[:, 1]) + neutral_delta
        neutral_probability = torch.sigmoid(neutral_logit)
        base_polar_total = (probability[:, 0] + probability[:, 2]).clamp_min(1e-7)
        positive_within_polar = (probability[:, 2] / base_polar_total).clamp(
            1e-7, 1.0 - 1e-7
        )
        polarity_logit = torch.logit(positive_within_polar) + polarity_delta
        positive_within_polar = torch.sigmoid(polarity_logit)
        corrected = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_within_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_within_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        return {
            "probabilities": corrected / corrected.sum(dim=-1, keepdim=True),
            "reliability": reliability,
            "neutral_delta": neutral_delta,
            "polarity_delta": polarity_delta,
            "agreement": (probability * emotion_probability).sum(dim=-1),
        }


class NeutralOrthogonalEnergyBottleneck(nn.Module):
    """Interpretable Neutral evidence aggregation with an identity-safe residual.

    Each modality is decomposed into learned Neutral and Polar subspaces.  A
    modality-specific bounded log-odds residual is combined as a reliability
    weighted product of experts.  Cross-modal disagreement attenuates that
    residual instead of becoming positive Neutral evidence by itself.  The
    final layers of the residual experts are zero initialized, so inserting the
    module into an already trained parent preserves every parent probability.
    """

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        dropout: float,
        maximum_neutral_logit_shift: float = 0.75,
        disagreement_temperature: float = 0.35,
        variant: str = "residual_mlp",
        consensus_mode: str = "soft",
    ) -> None:
        super().__init__()
        if hidden_dimension < 2:
            raise ValueError("hidden_dimension must be at least two")
        if maximum_neutral_logit_shift <= 0.0:
            raise ValueError("maximum_neutral_logit_shift must be positive")
        if disagreement_temperature <= 0.0:
            raise ValueError("disagreement_temperature must be positive")
        self.maximum_neutral_logit_shift = float(maximum_neutral_logit_shift)
        self.disagreement_temperature = float(disagreement_temperature)
        self.hidden_dimension = int(hidden_dimension)
        self.variant = str(variant).strip().lower()
        if self.variant not in {"residual_mlp", "signed_contrast"}:
            raise ValueError(
                "neutral energy variant must be residual_mlp or signed_contrast"
            )
        self.consensus_mode = str(consensus_mode).strip().lower()
        if self.consensus_mode not in {"soft", "unanimity_veto"}:
            raise ValueError(
                "neutral energy consensus mode must be soft or unanimity_veto"
            )
        if self.variant != "signed_contrast" and self.consensus_mode != "soft":
            raise ValueError(
                "unanimity_veto is only defined for signed_contrast evidence"
            )

        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(
                        dimension,
                        elementwise_affine=self.variant != "signed_contrast",
                    ),
                    nn.Linear(
                        dimension,
                        2 * hidden_dimension,
                        bias=self.variant != "signed_contrast",
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(3)
            ]
        )
        if self.variant == "residual_mlp":
            diagnostic_dimension = 2 * hidden_dimension + 4
            self.neutral_residuals = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(diagnostic_dimension),
                        nn.Linear(diagnostic_dimension, hidden_dimension),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dimension, 1),
                    )
                    for _ in range(3)
                ]
            )
            self.evidence_strengths = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(diagnostic_dimension),
                        nn.Linear(diagnostic_dimension, hidden_dimension),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dimension, 1),
                    )
                    for _ in range(3)
                ]
            )
            for residual in self.neutral_residuals:
                nn.init.zeros_(residual[-1].weight)
                nn.init.zeros_(residual[-1].bias)
        else:
            # Bias-free evidence axes cannot learn a fold-specific constant
            # offset.  Their difference gives the signed Neutral-vs-Polar
            # evidence used by EXP189.
            self.neutral_axes = nn.ModuleList(
                [nn.Linear(hidden_dimension, 1, bias=False) for _ in range(3)]
            )
            self.polar_axes = nn.ModuleList(
                [nn.Linear(hidden_dimension, 1, bias=False) for _ in range(3)]
            )
            for axis in (*self.neutral_axes, *self.polar_axes):
                nn.init.zeros_(axis.weight)

    def forward(
        self,
        base_probability: Tensor,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        external_reliability: Tensor | None = None,
        modality_available: Tensor | None = None,
    ) -> dict[str, Tensor]:
        base_probability = base_probability.float().clamp_min(1e-8)
        base_probability = base_probability / base_probability.sum(
            dim=-1, keepdim=True
        )
        states = (text, audio, vision)
        batch_size = base_probability.size(0)
        if external_reliability is None:
            external_reliability = base_probability.new_ones(batch_size, 3)
        else:
            external_reliability = external_reliability.float().clamp(0.0, 1.0)
        if modality_available is None:
            modality_available = base_probability.new_ones(batch_size, 3)
        else:
            modality_available = modality_available.to(
                device=base_probability.device, dtype=base_probability.dtype
            ).clamp(0.0, 1.0)
        if external_reliability.shape != (batch_size, 3):
            raise ValueError("external_reliability must have shape [batch, 3]")
        if modality_available.shape != (batch_size, 3):
            raise ValueError("modality_available must have shape [batch, 3]")
        if bool((modality_available.sum(dim=-1) <= 0).any()):
            raise ValueError("at least one modality must be available per sample")

        parent_neutral_logit = torch.logit(
            base_probability[:, 1].clamp(1e-7, 1.0 - 1e-7)
        )
        parent_uncertainty = 1.0 - base_probability.max(dim=-1).values
        parent_entropy = -(
            base_probability * base_probability.log()
        ).sum(dim=-1) / math.log(3.0)

        modality_logits = []
        reliability = []
        neutral_energy = []
        polar_energy = []
        orthogonality = []
        residuals = []
        raw_contrasts = []
        for index, state in enumerate(states):
            projected = self.projectors[index](state)
            neutral_state, polar_state = projected.chunk(2, dim=-1)
            neutral_norm = neutral_state.float().pow(2).mean(dim=-1)
            polar_norm = polar_state.float().pow(2).mean(dim=-1)
            cosine = F.cosine_similarity(
                neutral_state.float(), polar_state.float(), dim=-1, eps=1e-6
            )
            orthogonal_error = cosine.pow(2)
            if self.variant == "residual_mlp":
                features = torch.cat(
                    [
                        neutral_state.float(),
                        polar_state.float(),
                        neutral_norm.unsqueeze(-1),
                        polar_norm.unsqueeze(-1),
                        parent_uncertainty.unsqueeze(-1),
                        parent_entropy.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                raw_contrast = self.neutral_residuals[index](features).squeeze(
                    -1
                ).float()
                residual = self.maximum_neutral_logit_shift * torch.tanh(
                    raw_contrast
                )
                learned_reliability = torch.sigmoid(
                    self.evidence_strengths[index](features).squeeze(-1).float()
                )
                modality_logit = parent_neutral_logit + residual
                neutral_measure = neutral_norm
                polar_measure = polar_norm
            else:
                neutral_score = self.neutral_axes[index](
                    neutral_state.float()
                ).squeeze(-1)
                polar_score = self.polar_axes[index](
                    polar_state.float()
                ).squeeze(-1)
                raw_contrast = neutral_score - polar_score
                residual = self.maximum_neutral_logit_shift * torch.tanh(
                    raw_contrast
                )
                # Evidence magnitude raises confidence symmetrically; it never
                # chooses the Neutral direction.
                learned_reliability = 0.5 + 0.5 * torch.tanh(
                    raw_contrast.abs() / self.disagreement_temperature
                )
                modality_logit = raw_contrast
                neutral_measure = F.softplus(neutral_score)
                polar_measure = F.softplus(polar_score)
            # The upstream fusion gates remain useful evidence-quality priors,
            # but never completely silence a modality during early training.
            learned_reliability = learned_reliability * (
                0.25 + 0.75 * external_reliability[:, index]
            )
            available = modality_available[:, index]
            residual = residual * available
            raw_contrast = raw_contrast * available
            modality_logit = modality_logit * available
            learned_reliability = learned_reliability * available

            residuals.append(residual)
            raw_contrasts.append(raw_contrast)
            modality_logits.append(modality_logit)
            reliability.append(learned_reliability)
            neutral_energy.append(neutral_measure)
            polar_energy.append(polar_measure)
            orthogonality.append(orthogonal_error)

        residual_tensor = torch.stack(residuals, dim=-1)
        raw_contrast_tensor = torch.stack(raw_contrasts, dim=-1)
        modality_logit_tensor = torch.stack(modality_logits, dim=-1)
        reliability_tensor = torch.stack(reliability, dim=-1)
        raw_weight = reliability_tensor.clamp_min(1e-6) * modality_available
        modality_weight = raw_weight / raw_weight.sum(dim=-1, keepdim=True).clamp_min(
            1e-6
        )

        poe_residual = (modality_weight * residual_tensor).sum(dim=-1)
        disagreement = (
            modality_weight * (residual_tensor - poe_residual.unsqueeze(-1)).pow(2)
        ).sum(dim=-1)
        agreement = torch.exp(
            -disagreement / (self.disagreement_temperature**2)
        ).clamp(0.0, 1.0)
        if self.variant == "signed_contrast":
            signed_vote = torch.tanh(
                raw_contrast_tensor / self.disagreement_temperature
            )
            if self.consensus_mode == "unanimity_veto":
                available = modality_available.bool()
                unanimous_positive = torch.logical_or(
                    raw_contrast_tensor > 0.0, ~available
                ).all(dim=-1)
                unanimous_negative = torch.logical_or(
                    raw_contrast_tensor < 0.0, ~available
                ).all(dim=-1)
                available_strength = torch.where(
                    available,
                    signed_vote.abs(),
                    torch.ones_like(signed_vote),
                )
                weakest_evidence = available_strength.amin(dim=-1)
                unanimous_signed = torch.where(
                    unanimous_positive,
                    weakest_evidence,
                    torch.where(
                        unanimous_negative,
                        -weakest_evidence,
                        torch.zeros_like(weakest_evidence),
                    ),
                )
                coherence = unanimous_signed.abs()
                unanimous = unanimous_positive | unanimous_negative
            else:
                coherence = torch.abs(
                    (modality_weight * signed_vote).sum(dim=-1)
                )
                unanimous = torch.ones_like(coherence, dtype=torch.bool)
            boundary_gate = (
                4.0 * base_probability[:, 1] * (1.0 - base_probability[:, 1])
            ).clamp(0.0, 1.0)
        else:
            coherence = torch.ones_like(agreement)
            boundary_gate = torch.ones_like(agreement)
            unanimous = torch.ones_like(coherence, dtype=torch.bool)
        attenuation = agreement * coherence * boundary_gate
        modality_contribution = (
            attenuation.unsqueeze(-1) * modality_weight * residual_tensor
        )
        neutral_shift = modality_contribution.sum(dim=-1)
        neutral_logit = parent_neutral_logit + neutral_shift
        neutral_probability = torch.sigmoid(neutral_logit)

        polar_total = (base_probability[:, 0] + base_probability[:, 2]).clamp_min(
            1e-8
        )
        positive_within_polar = (
            base_probability[:, 2] / polar_total
        ).clamp(1e-7, 1.0 - 1e-7)
        probability = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_within_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_within_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probability = probability / probability.sum(dim=-1, keepdim=True)

        return {
            "probabilities": probability,
            "parent_probability": base_probability.detach(),
            "neutral_logit": neutral_logit,
            "polarity_logit": torch.logit(positive_within_polar),
            "modality_neutral_logits": modality_logit_tensor,
            "modality_neutral_probabilities": torch.sigmoid(
                modality_logit_tensor
            ),
            "modality_reliability": reliability_tensor,
            "modality_weights": modality_weight,
            "modality_contributions": modality_contribution,
            "neutral_energy": torch.stack(neutral_energy, dim=-1),
            "polar_energy": torch.stack(polar_energy, dim=-1),
            "orthogonality": torch.stack(orthogonality, dim=-1),
            "poe_residual": poe_residual,
            "disagreement": disagreement,
            "agreement": agreement,
            "coherence": coherence,
            "unanimous": unanimous.float(),
            "boundary_gate": boundary_gate,
            "neutral_shift": neutral_shift,
            "uncertainty": 1.0 - reliability_tensor,
        }


class BackgroundDeviationAdapter(nn.Module):
    """Turn leave-one-out video context into a bounded representation residual.

    The current utterance is never part of its own reference.  A zero-initialized
    residual makes the adapter an exact identity when introduced into a trained
    parent, while its gate and signed deviation remain directly inspectable.
    """

    def __init__(
        self,
        dimension: int,
        dropout: float,
        maximum_shift: float = 0.35,
        uncertainty_gated: bool = False,
        l2_trust_region: bool = False,
    ) -> None:
        super().__init__()
        self.uncertainty_gated = bool(uncertainty_gated)
        self.l2_trust_region = bool(l2_trust_region)
        feature_dimension = 5 * dimension + 2 + int(self.uncertainty_gated)
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_dimension),
            nn.Linear(feature_dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(dimension, multiplier=2, dropout=dropout),
        )
        self.gate = nn.Linear(dimension, 1)
        self.delta = nn.Linear(dimension, dimension)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.maximum_shift = float(maximum_shift)
        if self.maximum_shift <= 0.0:
            raise ValueError("video background maximum shift must be positive")

    def forward(
        self,
        text: Tensor,
        current: Tensor,
        centered: Tensor,
        reliability: Tensor,
        available: Tensor,
        log_group_size: Tensor,
        parent_uncertainty: Tensor | None = None,
    ) -> dict[str, Tensor]:
        difference = current - centered
        scalar_features = [
            reliability.float().unsqueeze(-1),
            log_group_size.float().unsqueeze(-1),
        ]
        if self.uncertainty_gated:
            if parent_uncertainty is None:
                raise ValueError("parent uncertainty is required by the background gate")
            scalar_features.append(parent_uncertainty.float().unsqueeze(-1))
        hidden = self.encoder(
            torch.cat(
                [
                    text,
                    current,
                    centered,
                    difference,
                    difference.abs(),
                    *scalar_features,
                ],
                dim=-1,
            )
        )
        gate = torch.sigmoid(self.gate(hidden).squeeze(-1).float())
        gate = gate * available.float()
        if self.uncertainty_gated:
            assert parent_uncertainty is not None
            gate = gate * parent_uncertainty.float().clamp(0.0, 1.0)
        delta_direction = torch.tanh(self.delta(hidden).float())
        if self.l2_trust_region:
            direction_norm = delta_direction.norm(dim=-1, keepdim=True)
            delta_direction = delta_direction / direction_norm.clamp_min(1.0)
        delta = self.maximum_shift * delta_direction
        corrected = current + gate.unsqueeze(-1).to(current.dtype) * delta.to(
            current.dtype
        )
        return {
            "corrected": corrected,
            "gate": gate,
            "delta_norm": delta.norm(dim=-1),
            "background_similarity": F.cosine_similarity(
                current.float(), centered.float(), dim=-1, eps=1e-6
            ),
        }


class LeaveOneOutVideoBackgroundDeconfounder(nn.Module):
    """Separate persistent video background from utterance affect per modality."""

    def __init__(
        self,
        dimension: int,
        dropout: float,
        maximum_shift: float = 0.35,
        uncertainty_gated: bool = False,
        l2_trust_region: bool = False,
    ) -> None:
        super().__init__()
        self.audio = BackgroundDeviationAdapter(
            dimension,
            dropout,
            maximum_shift=maximum_shift,
            uncertainty_gated=uncertainty_gated,
            l2_trust_region=l2_trust_region,
        )
        self.vision = BackgroundDeviationAdapter(
            dimension,
            dropout,
            maximum_shift=maximum_shift,
            uncertainty_gated=uncertainty_gated,
            l2_trust_region=l2_trust_region,
        )


class VideoRelativeNeutralHead(nn.Module):
    """Use cross-modal video-relative deviations only on Neutral-vs-Polar odds."""

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int = 128,
        dropout: float = 0.15,
        maximum_neutral_logit_shift: float = 0.75,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dimension)

        def projector() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(dimension),
                nn.Linear(dimension, hidden, bias=False),
                nn.GELU(),
            )

        self.text_projector = projector()
        self.audio_projector = projector()
        self.vision_projector = projector()
        # Five hidden blocks plus parent probability (3), modality reliability
        # (2), background similarity (2), anchor uncertainty, group size, and
        # signed cross-modal agreement (3 + 2 + 2 + 1 + 1 + 1 = 10).
        self.joint = nn.Sequential(
            nn.LayerNorm(5 * hidden + 10),
            nn.Linear(5 * hidden + 10, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(hidden, multiplier=2, dropout=dropout),
        )
        self.trust = nn.Linear(hidden, 1)
        self.neutral_axis = nn.Linear(hidden, 1)
        nn.init.zeros_(self.neutral_axis.weight)
        nn.init.zeros_(self.neutral_axis.bias)
        self.maximum_neutral_logit_shift = float(maximum_neutral_logit_shift)
        if self.maximum_neutral_logit_shift <= 0.0:
            raise ValueError("video-relative Neutral shift must be positive")

    def forward(
        self,
        parent_probability: Tensor,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        audio_centered: Tensor,
        vision_centered: Tensor,
        audio_reliability: Tensor,
        vision_reliability: Tensor,
        available: Tensor,
        log_group_size: Tensor,
        parent_uncertainty: Tensor,
    ) -> dict[str, Tensor]:
        parent_probability = parent_probability.float().clamp_min(1e-8)
        audio_deviation = audio - audio_centered
        vision_deviation = vision - vision_centered
        text_state = self.text_projector(text)
        audio_state = self.audio_projector(audio_deviation)
        vision_state = self.vision_projector(vision_deviation)
        crossmodal_agreement = F.cosine_similarity(
            audio_state.float(), vision_state.float(), dim=-1, eps=1e-6
        )
        audio_similarity = F.cosine_similarity(
            audio.float(), audio_centered.float(), dim=-1, eps=1e-6
        )
        vision_similarity = F.cosine_similarity(
            vision.float(), vision_centered.float(), dim=-1, eps=1e-6
        )
        hidden = self.joint(
            torch.cat(
                [
                    text_state,
                    audio_state,
                    vision_state,
                    audio_state * vision_state,
                    torch.abs(audio_state - vision_state),
                    parent_probability,
                    audio_reliability.float().unsqueeze(-1),
                    vision_reliability.float().unsqueeze(-1),
                    audio_similarity.unsqueeze(-1),
                    vision_similarity.unsqueeze(-1),
                    parent_uncertainty.float().unsqueeze(-1),
                    log_group_size.float().unsqueeze(-1),
                    crossmodal_agreement.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        agreement_gate = 0.5 * (crossmodal_agreement + 1.0)
        trust = torch.sigmoid(self.trust(hidden).squeeze(-1).float())
        trust = (
            trust
            * available.float()
            * parent_uncertainty.float().clamp(0.0, 1.0)
            * agreement_gate.clamp(0.0, 1.0)
        )
        raw_shift = self.maximum_neutral_logit_shift * torch.tanh(
            self.neutral_axis(hidden).squeeze(-1).float()
        )
        neutral_shift = trust * raw_shift
        parent_neutral = parent_probability[:, 1].clamp(1e-6, 1.0 - 1e-6)
        neutral_probability = torch.sigmoid(
            torch.logit(parent_neutral) + neutral_shift
        )
        parent_polar = parent_probability[:, [0, 2]]
        positive_within_polar = (
            parent_polar[:, 1] / parent_polar.sum(dim=-1).clamp_min(1e-8)
        )
        probability = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_within_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_within_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probability = probability / probability.sum(dim=-1, keepdim=True)
        return {
            "probabilities": probability,
            "neutral_shift": neutral_shift,
            "trust": trust,
            "crossmodal_agreement": crossmodal_agreement,
            "audio_similarity": audio_similarity,
            "vision_similarity": vision_similarity,
        }


class PairwiseSignedVideoRelativeNeutralHead(nn.Module):
    """Learn a signed Neutral direction from train-only within-video pairs.

    Current and leave-one-out reference features pass through the *same*
    modality map before subtraction.  Consequently each modality difference
    is antisymmetric under swapping current/reference inputs.  A text query
    reads the direction of that difference, while a non-negative router can
    only choose how much to trust audio or vision; it cannot manufacture a
    class bias.  The final update changes Neutral-vs-Polar odds only and thus
    preserves the parent's Negative/Positive conditional odds exactly.
    """

    def __init__(
        self,
        dimension: int,
        audio_dimension: int,
        vision_dimension: int,
        semantic_dimension: int = 768,
        hidden_dimension: int = 96,
        dropout: float = 0.15,
        maximum_neutral_logit_shift: float = 0.75,
        initial_correction_scale_logit: float = 0.0,
        initial_pair_temperature: float = 2.0,
        use_semantic_reference: bool = False,
        use_absolute_neutral_anchor: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dimension)
        if hidden <= 0:
            raise ValueError("pairwise video hidden dimension must be positive")
        if maximum_neutral_logit_shift <= 0.0:
            raise ValueError("pairwise video Neutral shift must be positive")
        if initial_pair_temperature <= 0.0:
            raise ValueError("pairwise video temperature must be positive")

        self.text_map = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden),
            nn.Tanh(),
        )

        def modality_map(input_dimension: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dimension),
                nn.Linear(input_dimension, hidden, bias=False),
                nn.Tanh(),
            )

        self.audio_map = modality_map(audio_dimension)
        self.vision_map = modality_map(vision_dimension)
        self.audio_query = nn.Linear(hidden, hidden, bias=False)
        self.vision_query = nn.Linear(hidden, hidden, bias=False)
        self.use_semantic_reference = bool(use_semantic_reference)
        self.use_absolute_neutral_anchor = bool(use_absolute_neutral_anchor)
        if self.use_semantic_reference:
            self.semantic_map = modality_map(semantic_dimension)
            self.semantic_query = nn.Linear(hidden, hidden, bias=False)
        route_dimension = 2 * hidden + 6

        def route() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(route_dimension),
                nn.Linear(route_dimension, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        self.audio_route = route()
        self.vision_route = route()
        if self.use_semantic_reference:
            self.semantic_route = route()
        self.pair_temperature_parameter = nn.Parameter(
            torch.tensor(
                math.log(math.expm1(float(initial_pair_temperature)))
            )
        )
        self.correction_scale_logit = nn.Parameter(
            torch.tensor(float(initial_correction_scale_logit))
        )
        if self.use_absolute_neutral_anchor:
            anchor_dimension = (
                (4 if self.use_semantic_reference else 3) * hidden + 7
            )
            self.neutral_anchor = nn.Sequential(
                nn.LayerNorm(anchor_dimension),
                nn.Linear(anchor_dimension, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.neutral_anchor[-1].weight)
            nn.init.zeros_(self.neutral_anchor[-1].bias)
            self.anchor_scale_logit = nn.Parameter(torch.zeros(()))
        self.maximum_neutral_logit_shift = float(maximum_neutral_logit_shift)

    @staticmethod
    def _signed_evidence(query: Tensor, difference: Tensor) -> Tensor:
        query = F.normalize(query.float(), dim=-1, eps=1e-6)
        difference = F.normalize(difference.float(), dim=-1, eps=1e-6)
        return (query * difference).sum(dim=-1)

    def forward(
        self,
        parent_probability: Tensor,
        text: Tensor,
        audio_current: Tensor,
        audio_reference: Tensor,
        vision_current: Tensor,
        vision_reference: Tensor,
        audio_reliability: Tensor,
        vision_reliability: Tensor,
        available: Tensor,
        log_group_size: Tensor,
        parent_uncertainty: Tensor,
        semantic_current: Tensor | None = None,
        semantic_reference: Tensor | None = None,
    ) -> dict[str, Tensor]:
        parent_probability = parent_probability.float().clamp_min(1e-8)
        text_state = self.text_map(text)
        audio_difference = (
            self.audio_map(audio_current) - self.audio_map(audio_reference)
        )
        vision_difference = (
            self.vision_map(vision_current) - self.vision_map(vision_reference)
        )
        available_float = available.float().unsqueeze(-1)
        audio_difference = audio_difference * available_float
        vision_difference = vision_difference * available_float
        audio_evidence = self._signed_evidence(
            self.audio_query(text_state), audio_difference
        )
        vision_evidence = self._signed_evidence(
            self.vision_query(text_state), vision_difference
        )
        metadata = torch.cat(
            [
                parent_probability,
                audio_reliability.float().unsqueeze(-1),
                vision_reliability.float().unsqueeze(-1),
                log_group_size.float().unsqueeze(-1),
            ],
            dim=-1,
        ).to(text_state.dtype)
        audio_route_logit = self.audio_route(
            torch.cat(
                [text_state, torch.abs(audio_difference), metadata], dim=-1
            )
        ).squeeze(-1)
        vision_route_logit = self.vision_route(
            torch.cat(
                [text_state, torch.abs(vision_difference), metadata], dim=-1
            )
        ).squeeze(-1)
        route_logits = [audio_route_logit, vision_route_logit]
        evidence = [audio_evidence, vision_evidence]
        semantic_difference = None
        semantic_evidence = torch.zeros_like(audio_evidence)
        if self.use_semantic_reference:
            if semantic_current is None or semantic_reference is None:
                raise ValueError(
                    "semantic video reference tensors are required by this head"
                )
            semantic_difference = (
                self.semantic_map(semantic_current)
                - self.semantic_map(semantic_reference)
            ) * available_float
            semantic_evidence = self._signed_evidence(
                self.semantic_query(text_state), semantic_difference
            )
            semantic_route_logit = self.semantic_route(
                torch.cat(
                    [text_state, torch.abs(semantic_difference), metadata], dim=-1
                )
            ).squeeze(-1)
            route_logits.append(semantic_route_logit)
            evidence.append(semantic_evidence)
        routes = torch.softmax(torch.stack(route_logits, dim=-1).float(), dim=-1)
        evidence_tensor = torch.stack(evidence, dim=-1)
        pair_products = []
        for left in range(evidence_tensor.size(1)):
            for right in range(left + 1, evidence_tensor.size(1)):
                pair_products.append(
                    evidence_tensor[:, left] * evidence_tensor[:, right]
                )
        crossmodal_agreement = torch.stack(pair_products, dim=-1).mean(dim=-1)
        # A disagreement reduces trust but never reverses the signed evidence.
        agreement_gate = 0.5 + 0.5 * torch.sigmoid(
            4.0 * crossmodal_agreement
        )
        routed_evidence = (routes * evidence_tensor).sum(dim=-1)
        temperature = F.softplus(self.pair_temperature_parameter.float())
        relative_logit = temperature * routed_evidence
        reliability_values = [
            audio_reliability.float(),
            vision_reliability.float(),
        ]
        if self.use_semantic_reference:
            reliability_values.append(torch.ones_like(audio_reliability).float())
        routed_reliability = (
            routes * torch.stack(reliability_values, dim=-1)
        ).sum(dim=-1)
        trust = (
            available.float()
            * parent_uncertainty.float().clamp(0.0, 1.0)
            * agreement_gate
            * (0.5 + 0.5 * routed_reliability.clamp(0.0, 1.0))
        )
        correction_scale = torch.sigmoid(self.correction_scale_logit.float())
        relative_shift = (
            self.maximum_neutral_logit_shift
            * correction_scale
            * trust
            * torch.tanh(relative_logit)
        )
        anchor_logit = torch.zeros_like(relative_logit)
        anchor_shift = torch.zeros_like(relative_logit)
        if self.use_absolute_neutral_anchor:
            anchor_blocks = [
                text_state,
                torch.abs(audio_difference),
                torch.abs(vision_difference),
            ]
            if semantic_difference is not None:
                anchor_blocks.append(torch.abs(semantic_difference))
            anchor_logit = self.neutral_anchor(
                torch.cat(
                    anchor_blocks
                    + [metadata, relative_logit.unsqueeze(-1).to(text_state.dtype)],
                    dim=-1,
                )
            ).squeeze(-1).float()
            anchor_shift = (
                self.maximum_neutral_logit_shift
                * torch.sigmoid(self.anchor_scale_logit.float())
                * parent_uncertainty.float().clamp(0.0, 1.0)
                * torch.tanh(anchor_logit)
            )
        neutral_shift = relative_shift + anchor_shift
        parent_neutral = parent_probability[:, 1].clamp(1e-6, 1.0 - 1e-6)
        neutral_logit = torch.logit(parent_neutral) + neutral_shift
        neutral_probability = torch.sigmoid(neutral_logit)
        parent_polar = parent_probability[:, [0, 2]]
        positive_within_polar = (
            parent_polar[:, 1] / parent_polar.sum(dim=-1).clamp_min(1e-8)
        )
        probability = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_within_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_within_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probability = probability / probability.sum(dim=-1, keepdim=True)
        return {
            "probabilities": probability,
            "neutral_shift": neutral_shift,
            "trust": trust,
            "relative_logit": relative_logit,
            "audio_evidence": audio_evidence,
            "vision_evidence": vision_evidence,
            "crossmodal_agreement": crossmodal_agreement,
            "audio_route": routes[:, 0],
            "vision_route": routes[:, 1],
            "correction_scale": correction_scale,
            "semantic_evidence": semantic_evidence,
            "semantic_route": (
                routes[:, 2]
                if self.use_semantic_reference
                else torch.zeros_like(audio_evidence)
            ),
            "anchor_logit": anchor_logit,
            "anchor_shift": anchor_shift,
            "neutral_logit": neutral_logit,
        }


class CrossVideoBilateralRelationHead(nn.Module):
    """Calibrate Neutral evidence against balanced cross-video exemplars.

    The two residual heads answer different questions: Neutral versus Negative
    and Neutral versus Positive.  Every relation is computed against an equal
    number of outer-fold training exemplars from each class. Audio and vision
    contribute only agreement-based trust; the final scalar update is applied
    to Neutral-vs-Polar odds and therefore cannot alter Negative/Positive odds.
    """

    def __init__(
        self,
        dimension: int,
        semantic_dimension: int,
        audio_dimension: int,
        vision_dimension: int,
        hidden_dimension: int = 96,
        dropout: float = 0.15,
        maximum_neutral_logit_shift: float = 1.25,
        retrieval_temperature: float = 0.20,
        odd_evidence_only: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dimension)
        if hidden <= 0:
            raise ValueError("cross-video relation hidden dimension must be positive")
        if maximum_neutral_logit_shift <= 0.0:
            raise ValueError("cross-video maximum Neutral shift must be positive")
        if retrieval_temperature <= 0.0:
            raise ValueError("cross-video retrieval temperature must be positive")

        def relation_map(input_dimension: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dimension),
                nn.Linear(input_dimension, hidden, bias=False),
                nn.Tanh(),
            )

        self.semantic_map = relation_map(semantic_dimension)
        self.audio_map = relation_map(audio_dimension)
        self.vision_map = relation_map(vision_dimension)
        self.text_query = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden),
            nn.Tanh(),
        )
        self.odd_evidence_only = bool(odd_evidence_only)
        # Four relation channels each provide Neutral affinity, side affinity,
        # and their signed margin; the remaining fields expose reliability,
        # the frozen parent's side log-odds, uncertainty, and agreement.
        relation_dimension = 17

        def boundary() -> nn.Sequential:
            network = nn.Sequential(
                nn.LayerNorm(relation_dimension),
                nn.Linear(relation_dimension, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
            return network

        if self.odd_evidence_only:
            # Exact identity at initialization. The scalar can learn evidence
            # orientation, but no additive bias can expand Neutral globally.
            self.relation_scale_parameter = nn.Parameter(torch.zeros(()))
        else:
            self.negative_neutral_boundary = boundary()
            self.positive_neutral_boundary = boundary()
        self.maximum_neutral_logit_shift = float(maximum_neutral_logit_shift)
        self.retrieval_temperature = float(retrieval_temperature)

    @staticmethod
    def _cosine(current: Tensor, references: Tensor) -> Tensor:
        current = F.normalize(current.float(), dim=-1, eps=1e-6)
        references = F.normalize(references.float(), dim=-1, eps=1e-6)
        return torch.einsum("bd,bckd->bck", current, references)

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
        masked = logits.float().masked_fill(~mask.bool(), -1e4)
        weight = torch.softmax(masked, dim=-1) * mask.float()
        return weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _side_feature(
        affinities: list[Tensor],
        neutral_index: int,
        polar_index: int,
        audio_reliability: Tensor,
        vision_reliability: Tensor,
        parent_side_log_odds: Tensor,
        parent_uncertainty: Tensor,
        agreement: Tensor,
    ) -> Tensor:
        blocks = []
        for affinity in affinities:
            neutral = affinity[:, neutral_index]
            polar = affinity[:, polar_index]
            blocks.extend([neutral, polar, neutral - polar])
        blocks.extend(
            [
                audio_reliability.float(),
                vision_reliability.float(),
                parent_side_log_odds.float(),
                parent_uncertainty.float(),
                agreement.float(),
            ]
        )
        return torch.stack(blocks, dim=-1)

    def forward(
        self,
        parent_probability: Tensor,
        text: Tensor,
        semantic_current: Tensor,
        audio_current: Tensor,
        vision_current: Tensor,
        semantic_references: Tensor,
        audio_references: Tensor,
        vision_references: Tensor,
        retrieval_similarity: Tensor,
        reference_mask: Tensor,
        audio_reliability: Tensor,
        vision_reliability: Tensor,
        parent_uncertainty: Tensor,
    ) -> dict[str, Tensor]:
        parent_probability = parent_probability.float().clamp_min(1e-8)
        semantic_current_state = self.semantic_map(semantic_current)
        audio_current_state = self.audio_map(audio_current)
        vision_current_state = self.vision_map(vision_current)
        semantic_reference_state = self.semantic_map(semantic_references)
        audio_reference_state = self.audio_map(audio_references)
        vision_reference_state = self.vision_map(vision_references)

        semantic_similarity = self._cosine(
            semantic_current_state, semantic_reference_state
        )
        audio_similarity = self._cosine(audio_current_state, audio_reference_state)
        vision_similarity = self._cosine(
            vision_current_state, vision_reference_state
        )
        query = F.normalize(self.text_query(text).float(), dim=-1, eps=1e-6)
        query_reference = F.normalize(
            semantic_reference_state.float(), dim=-1, eps=1e-6
        )
        contextual_similarity = torch.einsum(
            "bd,bckd->bck", query, query_reference
        )
        attention = self._masked_softmax(
            (
                retrieval_similarity.float()
                + semantic_similarity
                + contextual_similarity
            )
            / self.retrieval_temperature,
            reference_mask,
        )

        def aggregate(value: Tensor) -> Tensor:
            return (attention * value.float()).sum(dim=-1)

        semantic_affinity = aggregate(
            0.5 * (semantic_similarity + contextual_similarity)
            if self.odd_evidence_only
            else semantic_similarity
        )
        audio_affinity = aggregate(audio_similarity)
        vision_affinity = aggregate(vision_similarity)
        retrieval_affinity = aggregate(retrieval_similarity)
        affinities = [
            semantic_affinity,
            audio_affinity,
            vision_affinity,
            retrieval_affinity,
        ]

        semantic_margin = torch.stack(
            [
                semantic_affinity[:, 1] - semantic_affinity[:, 0],
                semantic_affinity[:, 1] - semantic_affinity[:, 2],
            ],
            dim=-1,
        )
        audio_margin = torch.stack(
            [
                audio_affinity[:, 1] - audio_affinity[:, 0],
                audio_affinity[:, 1] - audio_affinity[:, 2],
            ],
            dim=-1,
        )
        vision_margin = torch.stack(
            [
                vision_affinity[:, 1] - vision_affinity[:, 0],
                vision_affinity[:, 1] - vision_affinity[:, 2],
            ],
            dim=-1,
        )
        reliability_sum = (
            audio_reliability.float() + vision_reliability.float()
        ).clamp_min(1e-6)
        agreement = (
            audio_reliability.float().unsqueeze(-1)
            * torch.exp(-torch.abs(audio_margin - semantic_margin))
            + vision_reliability.float().unsqueeze(-1)
            * torch.exp(-torch.abs(vision_margin - semantic_margin))
        ) / reliability_sum.unsqueeze(-1)

        parent_left = torch.log(
            parent_probability[:, 1] / parent_probability[:, 0]
        )
        parent_right = torch.log(
            parent_probability[:, 1] / parent_probability[:, 2]
        )
        left_feature = self._side_feature(
            affinities,
            1,
            0,
            audio_reliability,
            vision_reliability,
            parent_left,
            parent_uncertainty,
            agreement[:, 0],
        )
        right_feature = self._side_feature(
            affinities,
            1,
            2,
            audio_reliability,
            vision_reliability,
            parent_right,
            parent_uncertainty,
            agreement[:, 1],
        )
        relation_scale = semantic_margin.new_zeros(())
        if self.odd_evidence_only:
            relation_scale = 12.0 * torch.tanh(
                self.relation_scale_parameter.float()
            )
            left_residual = self.maximum_neutral_logit_shift * torch.tanh(
                relation_scale * semantic_margin[:, 0]
            )
            right_residual = self.maximum_neutral_logit_shift * torch.tanh(
                relation_scale * semantic_margin[:, 1]
            )
        else:
            left_residual = self.maximum_neutral_logit_shift * torch.tanh(
                self.negative_neutral_boundary(left_feature).squeeze(-1).float()
            )
            right_residual = self.maximum_neutral_logit_shift * torch.tanh(
                self.positive_neutral_boundary(right_feature).squeeze(-1).float()
            )
        left_logit = parent_left + left_residual
        right_logit = parent_right + right_residual
        complete = reference_mask.bool().any(dim=-1).all(dim=-1).float()
        crossmodal_agreement = agreement.mean(dim=-1)
        trust = (
            complete
            * parent_uncertainty.float().clamp(0.0, 1.0)
            * (0.5 + 0.5 * crossmodal_agreement.clamp(0.0, 1.0))
        )
        if self.odd_evidence_only:
            polar_total = (
                parent_probability[:, 0] + parent_probability[:, 2]
            ).clamp_min(1e-8)
            negative_route = parent_probability[:, 0] / polar_total
            positive_route = parent_probability[:, 2] / polar_total
            neutral_shift = trust * (
                negative_route * left_residual
                + positive_route * right_residual
            )
        else:
            neutral_shift = trust * 0.5 * (left_residual + right_residual)
        parent_neutral = parent_probability[:, 1].clamp(1e-6, 1.0 - 1e-6)
        neutral_logit = torch.logit(parent_neutral) + neutral_shift
        neutral_probability = torch.sigmoid(neutral_logit)
        parent_polar = parent_probability[:, [0, 2]]
        positive_within_polar = (
            parent_polar[:, 1] / parent_polar.sum(dim=-1).clamp_min(1e-8)
        )
        probabilities = torch.stack(
            [
                (1.0 - neutral_probability) * (1.0 - positive_within_polar),
                neutral_probability,
                (1.0 - neutral_probability) * positive_within_polar,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return {
            "parent_probability": parent_probability.detach(),
            "probabilities": probabilities,
            "neutral_shift": neutral_shift,
            "trust": trust,
            "left_logit": left_logit,
            "right_logit": right_logit,
            "left_residual": left_residual,
            "right_residual": right_residual,
            "semantic_margin": semantic_margin,
            "audio_margin": audio_margin,
            "vision_margin": vision_margin,
            "crossmodal_agreement": crossmodal_agreement,
            "negative_affinity": semantic_affinity[:, 0],
            "neutral_affinity": semantic_affinity[:, 1],
            "positive_affinity": semantic_affinity[:, 2],
            "neutral_logit": neutral_logit,
            "relation_scale": relation_scale,
        }


class PretrainedTextFusionNet(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        self.use_pretrained_classifier = bool(
            cfg.get("use_pretrained_classifier", False)
        )
        model_name = str(cfg["pretrained_model"])
        local_files_only = bool(cfg.get("local_files_only", False))
        revision = cfg.get("revision")
        if self.use_pretrained_classifier:
            self.text_encoder = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                revision=revision,
                local_files_only=local_files_only,
            )
            labels = [
                str(self.text_encoder.config.id2label[index]).strip().lower()
                for index in range(int(self.text_encoder.config.num_labels))
            ]
            expected = ["negative", "neutral", "positive"]
            configured_groups = cfg.get("external_class_groups")
            if configured_groups is not None:
                if not isinstance(configured_groups, Mapping):
                    raise ValueError("external_class_groups must be a mapping")
                normalized_groups = {
                    str(name).strip().lower(): [
                        str(label).strip().lower() for label in values
                    ]
                    for name, values in configured_groups.items()
                }
                if set(normalized_groups) != set(expected):
                    raise ValueError(
                        "external_class_groups must define Negative/Neutral/Positive"
                    )
                flattened = [
                    label for name in expected for label in normalized_groups[name]
                ]
                if sorted(flattened) != sorted(labels):
                    raise ValueError(
                        "external_class_groups must partition every pretrained label once"
                    )
                self.external_class_groups = [
                    [labels.index(label) for label in normalized_groups[name]]
                    for name in expected
                ]
                self.external_class_order = []
            else:
                if set(labels) != set(expected):
                    raise ValueError(
                        "The pretrained classifier must expose Negative/Neutral/Positive "
                        "labels or external_class_groups"
                    )
                self.external_class_order = [labels.index(name) for name in expected]
                self.external_class_groups = []
        else:
            self.text_encoder = AutoModel.from_pretrained(
                model_name,
                revision=revision,
                local_files_only=local_files_only,
            )
            self.external_class_order = [0, 1, 2]
            self.external_class_groups = []
        text_dimension = int(self.text_encoder.config.hidden_size)
        if bool(cfg.get("gradient_checkpointing", True)):
            # PyTorch 2.7 can revisit DeBERTa's relative-position graph twice
            # with the legacy re-entrant checkpoint implementation.  The
            # non-reentrant variant is the recommended modern path and avoids
            # the second-backward failure while preserving activation savings.
            self.text_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.parameter_efficient_encoder = bool(cfg.get("use_lora", False))
        if self.parameter_efficient_encoder:
            default_targets = (
                ["query", "value"]
                if self.text_encoder.config.model_type == "roberta"
                else ["query_proj", "value_proj"]
            )
            lora_config = LoraConfig(
                task_type=(
                    TaskType.SEQ_CLS
                    if self.use_pretrained_classifier
                    else TaskType.FEATURE_EXTRACTION
                ),
                r=int(cfg.get("lora_rank", 8)),
                lora_alpha=int(cfg.get("lora_alpha", 16)),
                lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                target_modules=list(cfg.get("lora_target_modules", default_targets)),
                bias="none",
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
            self.encoder_adapter_parameter_names = {
                name
                for name, parameter in self.text_encoder.named_parameters()
                if parameter.requires_grad
            }
        else:
            self.encoder_adapter_parameter_names = set()
        self.use_layerwise_pooling = bool(cfg.get("use_layerwise_pooling", False))
        self.layer_mix_count = int(cfg.get("layer_mix_count", 4))
        if self.layer_mix_count < 1:
            raise ValueError("layer_mix_count must be positive")
        if self.use_layerwise_pooling:
            available_layers = int(getattr(self.text_encoder.config, "num_hidden_layers", 0))
            if available_layers and self.layer_mix_count > available_layers:
                raise ValueError(
                    "layer_mix_count cannot exceed the pretrained encoder depth"
                )
            # ELMo-style scalar mixing preserves useful lower-level lexical
            # evidence that can be attenuated in the final task-agnostic layer.
            self.layer_mix_logits = nn.Parameter(torch.zeros(self.layer_mix_count))
            self.layerwise_residual = bool(cfg.get("layerwise_residual", False))
            if self.layerwise_residual:
                self.layer_mix_scale_logit = nn.Parameter(
                    torch.tensor(float(cfg.get("layer_mix_scale_logit", -2.0)))
                )
            else:
                self.layer_mix_scale = nn.Parameter(torch.ones(()))
        fusion_dimension = int(cfg.get("fusion_dimension", 256))
        dropout = float(cfg.get("dropout", 0.15))
        self.use_emotion_evidence = bool(cfg.get("use_emotion_evidence", False))
        if self.use_emotion_evidence:
            self.emotion_encoder = AutoModelForSequenceClassification.from_pretrained(
                str(cfg["emotion_pretrained_model"]),
                revision=cfg.get("emotion_revision"),
                local_files_only=local_files_only,
            )
            for parameter in self.emotion_encoder.parameters():
                parameter.requires_grad_(False)
            emotion_labels = [
                str(self.emotion_encoder.config.id2label[index]).strip().lower()
                for index in range(int(self.emotion_encoder.config.num_labels))
            ]
            emotion_groups = cfg.get("emotion_class_groups")
            if not isinstance(emotion_groups, Mapping):
                raise ValueError(
                    "emotion_class_groups must map Negative/Neutral/Positive labels"
                )
            expected_groups = ["negative", "neutral", "positive"]
            normalized_groups = {
                str(name).strip().lower(): [
                    str(label).strip().lower() for label in values
                ]
                for name, values in emotion_groups.items()
            }
            if set(normalized_groups) != set(expected_groups):
                raise ValueError(
                    "emotion_class_groups must define Negative/Neutral/Positive"
                )
            flattened = [
                label
                for name in expected_groups
                for label in normalized_groups[name]
            ]
            if sorted(flattened) != sorted(emotion_labels):
                raise ValueError(
                    "emotion_class_groups must partition every emotion label once"
                )
            self.emotion_class_groups = [
                [emotion_labels.index(label) for label in normalized_groups[name]]
                for name in expected_groups
            ]
            self.emotion_evidence = AgreementAwareEmotionEvidence(
                primary_dimension=fusion_dimension,
                emotion_dimension=int(self.emotion_encoder.config.hidden_size),
                dropout=dropout,
                maximum_representation_shift=float(
                    cfg.get("emotion_maximum_representation_shift", 0.35)
                ),
                maximum_neutral_logit_shift=float(
                    cfg.get("emotion_maximum_neutral_logit_shift", 0.75)
                ),
                maximum_polarity_logit_shift=float(
                    cfg.get("emotion_maximum_polarity_logit_shift", 0.25)
                ),
            )
        else:
            self.emotion_class_groups = []
        self.use_context = bool(cfg.get("use_context", False))
        self.use_directional_context = bool(
            cfg.get("use_directional_context", False)
        )
        self.use_context_transition = bool(
            cfg.get("use_context_transition", False)
        )
        self.use_bounded_neutral_transition = bool(
            cfg.get("use_bounded_neutral_transition", False)
        )
        if sum(
            int(value)
            for value in (
                self.use_context,
                self.use_directional_context,
                self.use_context_transition,
                self.use_bounded_neutral_transition,
            )
        ) > 1:
            raise ValueError(
                "context, directional-context, and transition modes are "
                "mutually exclusive"
            )
        self.use_legacy_text = bool(cfg.get("use_legacy_text", False))
        self.use_almt_hyper = bool(cfg.get("use_almt_hyper", False))
        self.use_audio = bool(cfg.get("use_audio", False))
        self.use_vision = bool(cfg.get("use_vision", False))
        self.use_spectral_dynamics = bool(cfg.get("use_spectral_dynamics", False))
        if self.use_spectral_dynamics and not (self.use_audio and self.use_vision):
            raise ValueError("spectral dynamics requires both audio and vision")
        self.use_video_background_deconfounder = bool(
            cfg.get("use_video_background_deconfounder", False)
        )
        self.use_video_relative_neutral_head = bool(
            cfg.get("use_video_relative_neutral_head", False)
        )
        self.use_pairwise_signed_video_neutral_head = bool(
            cfg.get("use_pairwise_signed_video_neutral_head", False)
        )
        self.use_cross_video_bilateral_relation = bool(
            cfg.get("use_cross_video_bilateral_relation", False)
        )
        self.pairwise_video_use_semantic_reference = bool(
            cfg.get("pairwise_video_use_semantic_reference", False)
        )
        self.pairwise_video_use_absolute_neutral_anchor = bool(
            cfg.get("pairwise_video_use_absolute_neutral_anchor", False)
        )
        if self.use_video_background_deconfounder and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "video background deconfounding requires audio and vision"
            )
        if self.use_video_relative_neutral_head and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "video-relative Neutral head requires audio and vision"
            )
        if self.use_pairwise_signed_video_neutral_head and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "pairwise signed video Neutral head requires audio and vision"
            )
        if self.use_cross_video_bilateral_relation and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "cross-video bilateral relation requires audio and vision"
            )
        if sum(
            int(value)
            for value in (
                self.use_video_relative_neutral_head,
                self.use_pairwise_signed_video_neutral_head,
                self.use_cross_video_bilateral_relation,
            )
        ) > 1:
            raise ValueError(
                "video-relative Neutral heads are mutually exclusive"
            )
        self.use_low_rank_interaction = bool(
            cfg.get("use_low_rank_interaction", False)
        )
        if self.use_low_rank_interaction and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "use_low_rank_interaction requires both audio and vision"
            )
        self.use_hurdle = bool(cfg.get("use_hurdle", False))
        self.use_polar_distance_neutral = bool(
            cfg.get("use_polar_distance_neutral", False)
        )
        self.use_external_polarity_anchor = bool(
            cfg.get("use_external_polarity_anchor", False)
        )
        if self.use_external_polarity_anchor and not (
            self.use_pretrained_classifier and self.use_polar_distance_neutral
        ):
            raise ValueError(
                "use_external_polarity_anchor requires both "
                "use_pretrained_classifier and use_polar_distance_neutral"
            )
        if self.use_hurdle and self.use_polar_distance_neutral:
            raise ValueError(
                "use_hurdle and use_polar_distance_neutral are mutually exclusive"
            )
        self.use_prototype_router = bool(cfg.get("use_prototype_router", False))
        self.use_subcenter_head = bool(cfg.get("use_subcenter_head", False))
        self.use_conflict_ignorance_neutral = bool(
            cfg.get("use_conflict_ignorance_neutral", False)
        )
        if self.use_conflict_ignorance_neutral and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "use_conflict_ignorance_neutral requires audio and vision"
            )
        self.use_aligned_temporal_incongruity = bool(
            cfg.get("use_aligned_temporal_incongruity", False)
        )
        if self.use_aligned_temporal_incongruity and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "use_aligned_temporal_incongruity requires audio and vision"
            )
        if (
            self.use_aligned_temporal_incongruity
            and self.use_conflict_ignorance_neutral
        ):
            raise ValueError(
                "aligned temporal incongruity and conflict/ignorance heads "
                "must be evaluated separately"
            )
        self.use_contrastive_feature_decomposition = bool(
            cfg.get("use_contrastive_feature_decomposition", False)
        )
        if self.use_contrastive_feature_decomposition and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "use_contrastive_feature_decomposition requires audio and vision"
            )
        if self.use_contrastive_feature_decomposition and (
            self.use_conflict_ignorance_neutral
            or self.use_aligned_temporal_incongruity
        ):
            raise ValueError(
                "contrastive feature decomposition, aligned incongruity, and "
                "conflict/ignorance heads must be evaluated separately"
            )
        self.decomposition_dimension = int(
            cfg.get("decomposition_dimension", 128)
        )
        self.use_text_conditional_innovation = bool(
            cfg.get("use_text_conditional_innovation", False)
        )
        if self.use_text_conditional_innovation and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "use_text_conditional_innovation requires audio and vision"
            )
        if self.use_text_conditional_innovation and (
            self.use_conflict_ignorance_neutral
            or self.use_aligned_temporal_incongruity
            or self.use_contrastive_feature_decomposition
        ):
            raise ValueError(
                "text-conditional innovation and other structural correction "
                "heads must be evaluated separately"
            )
        self.use_neutral_energy_bottleneck = bool(
            cfg.get("use_neutral_energy_bottleneck", False)
        )
        if self.use_neutral_energy_bottleneck and not (
            self.use_audio and self.use_vision
        ):
            raise ValueError(
                "use_neutral_energy_bottleneck requires audio and vision"
            )
        if self.use_neutral_energy_bottleneck and (
            self.use_hurdle
            or self.use_polar_distance_neutral
            or self.use_conflict_ignorance_neutral
            or self.use_aligned_temporal_incongruity
            or self.use_contrastive_feature_decomposition
            or self.use_text_conditional_innovation
        ):
            raise ValueError(
                "the Neutral energy bottleneck must be evaluated without other "
                "decision-correction heads"
            )
        self.use_context_weighted_prototype = bool(
            cfg.get("use_context_weighted_prototype", False)
        )
        if self.use_context_weighted_prototype and not self.use_directional_context:
            raise ValueError(
                "use_context_weighted_prototype requires use_directional_context"
            )
        self.use_ordinal_expert = bool(cfg.get("use_ordinal_expert", False))
        self.hurdle_mix = float(cfg.get("hurdle_mix", 0.25))
        if not 0.0 <= self.hurdle_mix <= 1.0:
            raise ValueError("hurdle_mix must lie in [0,1]")

        self.text_pool = nn.Sequential(
            nn.LayerNorm(2 * text_dimension),
            nn.Linear(2 * text_dimension, fusion_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualMLP(fusion_dimension, multiplier=2, dropout=dropout),
        )
        self.text_token_projection = nn.Sequential(
            nn.LayerNorm(text_dimension),
            nn.Linear(text_dimension, fusion_dimension),
            nn.GELU(),
        )
        if self.use_context:
            self.context_type_embeddings = nn.Parameter(
                torch.zeros(2, fusion_dimension)
            )
            nn.init.normal_(self.context_type_embeddings, std=0.02)
            self.context_projection = nn.Sequential(
                nn.LayerNorm(text_dimension),
                nn.Linear(text_dimension, fusion_dimension),
                nn.GELU(),
            )
            self.context_adaptation = MultimodalAdaptationGate(
                dimension=fusion_dimension,
                dropout=dropout,
                shift_scale=float(cfg.get("context_shift_scale", 0.35)),
            )
            self.context_reliability = ModalityReliabilityGate(
                fusion_dimension, dropout
            )
            self.context_norm = nn.LayerNorm(fusion_dimension)
        if self.use_directional_context:
            self.direction_embeddings = nn.Parameter(
                torch.zeros(2, fusion_dimension)
            )
            nn.init.normal_(self.direction_embeddings, std=0.02)
            self.directional_context_pool = nn.Sequential(
                nn.LayerNorm(2 * text_dimension),
                nn.Linear(2 * text_dimension, fusion_dimension),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualMLP(fusion_dimension, multiplier=2, dropout=dropout),
            )
            self.previous_context_adaptation = MultimodalAdaptationGate(
                dimension=fusion_dimension,
                dropout=dropout,
                shift_scale=float(cfg.get("context_shift_scale", 0.35)),
            )
            self.following_context_adaptation = MultimodalAdaptationGate(
                dimension=fusion_dimension,
                dropout=dropout,
                shift_scale=float(cfg.get("context_shift_scale", 0.35)),
            )
            self.directional_context_router = DirectionalContextRouter(
                fusion_dimension, dropout
            )
            self.directional_context_norm = nn.LayerNorm(fusion_dimension)
        if self.use_context_transition:
            self.sentiment_transition_expert = SentimentTransitionExpert(
                dimension=fusion_dimension,
                dropout=dropout,
                residual_scale_logit=float(
                    cfg.get("context_transition_scale_logit", -1.5)
                ),
            )
        if self.use_bounded_neutral_transition:
            self.bounded_neutral_transition_expert = (
                BoundedNeutralTransitionExpert(
                    dimension=fusion_dimension,
                    dropout=dropout,
                    maximum_logit_shift=float(
                        cfg.get("bounded_neutral_maximum_logit_shift", 1.0)
                    ),
                )
            )
        if self.use_legacy_text:
            self.legacy_text_encoder = AttentiveTemporalEncoder(
                input_dimension=int(cfg.get("legacy_text_dimension", 768)),
                hidden_dimension=fusion_dimension,
                dropout=dropout,
                n_heads=int(cfg.get("fusion_heads", 4)),
            )
            self.legacy_text_adaptation = MultimodalAdaptationGate(
                dimension=fusion_dimension,
                dropout=dropout,
                shift_scale=float(cfg.get("legacy_text_shift_scale", 0.35)),
            )
            self.legacy_text_reliability = ModalityReliabilityGate(
                fusion_dimension, dropout
            )
            self.legacy_text_norm = nn.LayerNorm(fusion_dimension)
        if self.use_almt_hyper:
            self.adaptive_hyper = AdaptiveHyperModalityEncoder(
                dimension=fusion_dimension,
                token_count=int(cfg.get("hyper_token_count", 4)),
                depth=int(cfg.get("hyper_depth", 2)),
                n_heads=int(cfg.get("fusion_heads", 4)),
                dropout=dropout,
                text_dimension=int(cfg.get("legacy_text_dimension", 768)),
                audio_dimension=int(cfg.get("audio_dimension", 74)),
                vision_dimension=int(cfg.get("vision_dimension", 35)),
            )
            self.hyper_adaptation = MultimodalAdaptationGate(
                dimension=fusion_dimension,
                dropout=dropout,
                shift_scale=float(cfg.get("hyper_shift_scale", 0.50)),
            )
            self.hyper_reliability = ModalityReliabilityGate(
                fusion_dimension, dropout
            )
            self.hyper_norm = nn.LayerNorm(fusion_dimension)
        if self.use_audio:
            self.audio_encoder = AttentiveTemporalEncoder(
                input_dimension=int(cfg.get("audio_dimension", 74)),
                hidden_dimension=fusion_dimension,
                dropout=dropout,
                n_heads=int(cfg.get("fusion_heads", 4)),
            )
            self.audio_adaptation = MultimodalAdaptationGate(
                dimension=fusion_dimension,
                dropout=dropout,
                shift_scale=float(cfg.get("adaptation_shift_scale", 0.5)),
            )
            self.audio_reliability = ModalityReliabilityGate(
                fusion_dimension, dropout
            )
            if self.use_spectral_dynamics:
                self.audio_spectral_dynamics = TemporalSpectralDynamicsAdapter(
                    input_dimension=int(cfg.get("audio_dimension", 74)),
                    hidden_dimension=fusion_dimension,
                    dropout=dropout,
                    mix_logit=float(cfg.get("spectral_mix_logit", -2.0)),
                )
        if self.use_vision:
            self.vision_encoder = AttentiveTemporalEncoder(
                input_dimension=int(cfg.get("vision_dimension", 35)),
                hidden_dimension=fusion_dimension,
                dropout=dropout,
                n_heads=int(cfg.get("fusion_heads", 4)),
            )
            self.adaptation = MultimodalAdaptationGate(
                dimension=fusion_dimension,
                dropout=dropout,
                shift_scale=float(cfg.get("adaptation_shift_scale", 0.5)),
            )
            self.visual_reliability = ModalityReliabilityGate(
                fusion_dimension, dropout
            )
            if self.use_spectral_dynamics:
                self.vision_spectral_dynamics = TemporalSpectralDynamicsAdapter(
                    input_dimension=int(cfg.get("vision_dimension", 35)),
                    hidden_dimension=fusion_dimension,
                    dropout=dropout,
                    mix_logit=float(cfg.get("spectral_mix_logit", -2.0)),
                )
        if self.use_video_background_deconfounder:
            self.video_background_deconfounder = (
                LeaveOneOutVideoBackgroundDeconfounder(
                    dimension=fusion_dimension,
                    dropout=dropout,
                    maximum_shift=float(
                        cfg.get("video_background_maximum_shift", 0.35)
                    ),
                    uncertainty_gated=bool(
                        cfg.get("video_background_uncertainty_gate", False)
                    ),
                    l2_trust_region=bool(
                        cfg.get("video_background_l2_trust_region", False)
                    ),
                )
            )
        if self.use_video_relative_neutral_head:
            self.video_relative_neutral_head = VideoRelativeNeutralHead(
                dimension=fusion_dimension,
                hidden_dimension=int(
                    cfg.get("video_relative_neutral_hidden_dimension", 128)
                ),
                dropout=dropout,
                maximum_neutral_logit_shift=float(
                    cfg.get("video_relative_neutral_maximum_logit_shift", 0.75)
                ),
            )
        if self.use_pairwise_signed_video_neutral_head:
            self.pairwise_signed_video_neutral_head = (
                PairwiseSignedVideoRelativeNeutralHead(
                    dimension=fusion_dimension,
                    audio_dimension=int(cfg.get("audio_dimension", 74)),
                    vision_dimension=int(cfg.get("vision_dimension", 35)),
                    semantic_dimension=int(
                        cfg.get("legacy_text_dimension", 768)
                    ),
                    hidden_dimension=int(
                        cfg.get("pairwise_video_hidden_dimension", 96)
                    ),
                    dropout=dropout,
                    maximum_neutral_logit_shift=float(
                        cfg.get("pairwise_video_maximum_logit_shift", 0.75)
                    ),
                    initial_correction_scale_logit=float(
                        cfg.get("pairwise_video_correction_scale_logit", 0.0)
                    ),
                    initial_pair_temperature=float(
                        cfg.get("pairwise_video_initial_temperature", 2.0)
                    ),
                    use_semantic_reference=(
                        self.pairwise_video_use_semantic_reference
                    ),
                    use_absolute_neutral_anchor=(
                        self.pairwise_video_use_absolute_neutral_anchor
                    ),
                )
            )
        if self.use_cross_video_bilateral_relation:
            self.cross_video_bilateral_relation = CrossVideoBilateralRelationHead(
                dimension=fusion_dimension,
                semantic_dimension=int(cfg.get("legacy_text_dimension", 768)),
                audio_dimension=int(cfg.get("audio_dimension", 74)),
                vision_dimension=int(cfg.get("vision_dimension", 35)),
                hidden_dimension=int(
                    cfg.get("cross_video_relation_hidden_dimension", 96)
                ),
                dropout=dropout,
                maximum_neutral_logit_shift=float(
                    cfg.get("cross_video_maximum_neutral_logit_shift", 1.25)
                ),
                retrieval_temperature=float(
                    cfg.get("cross_video_retrieval_temperature", 0.20)
                ),
                odd_evidence_only=bool(
                    cfg.get("cross_video_odd_evidence_only", False)
                ),
            )
        if self.use_audio and self.use_vision:
            self.tri_fusion = TriModalSharedPrivateFusion(
                dimension=fusion_dimension,
                dropout=dropout,
                n_heads=int(cfg.get("fusion_heads", 4)),
            )
            if self.use_low_rank_interaction:
                self.low_rank_interaction = LowRankTriModalInteraction(
                    dimension=fusion_dimension,
                    rank=int(cfg.get("low_rank_interaction_rank", 4)),
                    dropout=dropout,
                    maximum_mix=float(
                        cfg.get("low_rank_interaction_maximum_mix", 0.35)
                    ),
                    initial_mix_logit=float(
                        cfg.get("low_rank_interaction_mix_logit", -2.0)
                    ),
                )
        elif self.use_audio or self.use_vision:
            self.fusion = SharedPrivateFusion(
                dimension=fusion_dimension,
                dropout=dropout,
                n_heads=int(cfg.get("fusion_heads", 4)),
            )
        if self.use_audio or self.use_vision:
            self.multimodal_norm = nn.LayerNorm(fusion_dimension)
            self.fusion_scale_logit = nn.Parameter(
                torch.tensor(float(cfg.get("fusion_scale_logit", -2.0)))
            )
            self.use_dynamic_fusion_router = bool(
                cfg.get("use_dynamic_fusion_router", False)
            )
            if self.use_dynamic_fusion_router:
                self.dynamic_fusion_router = ConfidenceAwareFusionRouter(
                    dimension=fusion_dimension,
                    dropout=dropout,
                )
        else:
            self.use_dynamic_fusion_router = False
        if self.use_prototype_router:
            self.prototype_router = ClassPrototypeRouter(
                dimension=fusion_dimension,
                dropout=dropout,
                n_heads=int(cfg.get("fusion_heads", 4)),
            )
            self.prototype_mix_logit = nn.Parameter(
                torch.tensor(float(cfg.get("prototype_mix_logit", -1.5)))
            )
        if self.use_subcenter_head:
            self.subcenter_head = LatentAffectSubcenterHead(
                dimension=fusion_dimension,
                subcenters=int(cfg.get("subcenter_count", 3)),
                maximum_mix=float(cfg.get("subcenter_maximum_mix", 0.35)),
                initial_mix_logit=float(cfg.get("subcenter_mix_logit", -3.0)),
                initial_scale=float(cfg.get("subcenter_initial_scale", 10.0)),
            )
        if self.use_conflict_ignorance_neutral:
            self.conflict_ignorance_neutral = ConflictIgnoranceNeutralHead(
                dimension=fusion_dimension,
                dropout=dropout,
                maximum_neutral_logit_shift=float(
                    cfg.get("conflict_maximum_neutral_logit_shift", 0.75)
                ),
                evidence_prior=float(cfg.get("conflict_evidence_prior", 2.0)),
            )
        if self.use_aligned_temporal_incongruity:
            self.aligned_temporal_incongruity = AlignedTemporalIncongruityHead(
                text_dimension=int(cfg.get("legacy_text_dimension", 768)),
                audio_dimension=int(cfg.get("audio_dimension", 74)),
                vision_dimension=int(cfg.get("vision_dimension", 35)),
                hidden_dimension=int(cfg.get("aligned_interaction_dimension", 64)),
                dropout=dropout,
                maximum_neutral_logit_shift=float(
                    cfg.get("aligned_maximum_neutral_logit_shift", 0.75)
                ),
                maximum_polarity_logit_shift=float(
                    cfg.get("aligned_maximum_polarity_logit_shift", 0.25)
                ),
            )
        if self.use_contrastive_feature_decomposition:
            self.contrastive_feature_decomposition = (
                ContrastiveFeatureDecompositionHead(
                    dimension=fusion_dimension,
                    hidden_dimension=self.decomposition_dimension,
                    dropout=dropout,
                    maximum_logit_shift=float(
                        cfg.get("decomposition_maximum_logit_shift", 0.50)
                    ),
                )
            )
        if self.use_text_conditional_innovation:
            self.text_conditional_innovation = TextConditionalInnovationHead(
                dimension=fusion_dimension,
                hidden_dimension=int(cfg.get("innovation_hidden_dimension", 128)),
                dropout=dropout,
                maximum_neutral_logit_shift=float(
                    cfg.get("innovation_maximum_neutral_logit_shift", 0.75)
                ),
                maximum_polarity_logit_shift=float(
                    cfg.get("innovation_maximum_polarity_logit_shift", 0.15)
                ),
            )
        if self.use_neutral_energy_bottleneck:
            self.neutral_energy_bottleneck = NeutralOrthogonalEnergyBottleneck(
                dimension=fusion_dimension,
                hidden_dimension=int(
                    cfg.get("neutral_energy_hidden_dimension", 64)
                ),
                dropout=dropout,
                maximum_neutral_logit_shift=float(
                    cfg.get("neutral_energy_maximum_logit_shift", 0.75)
                ),
                disagreement_temperature=float(
                    cfg.get("neutral_energy_disagreement_temperature", 0.35)
                ),
                variant=str(
                    cfg.get("neutral_energy_variant", "residual_mlp")
                ),
                consensus_mode=str(
                    cfg.get("neutral_energy_consensus_mode", "soft")
                ),
            )
        if self.use_pretrained_classifier:
            self.external_maximum_mix = float(
                cfg.get("external_maximum_mix", 1.0)
            )
            if not 0.0 <= self.external_maximum_mix <= 1.0:
                raise ValueError("external_maximum_mix must lie in [0, 1]")
            self.external_mix_logit = nn.Parameter(
                torch.tensor(float(cfg.get("external_mix_logit", -1.0)))
            )
        if self.use_ordinal_expert:
            self.ordinal_expert = OrderedIntervalExpert(fusion_dimension, dropout)
            self.ordinal_mix_logit = nn.Parameter(
                torch.tensor(float(cfg.get("ordinal_mix_logit", -1.5)))
            )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dimension),
            nn.Linear(fusion_dimension, fusion_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dimension, 3),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(fusion_dimension),
            nn.Linear(fusion_dimension, fusion_dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dimension // 2, 1),
        )
        self.text_auxiliary = nn.Linear(fusion_dimension, 3)
        self.hurdle = ZeroInflatedOrdinalExpert(fusion_dimension, dropout)
        if self.use_polar_distance_neutral:
            self.polar_distance_neutral = PolarDistanceNeutralExpert(
                dimension=fusion_dimension,
                dropout=dropout,
                maximum_boundary_mix=float(
                    cfg.get("neutral_boundary_maximum_mix", 0.50)
                ),
                maximum_evidence_shift=float(
                    cfg.get("neutral_boundary_maximum_shift", 1.50)
                ),
                boundary_mix_logit=float(
                    cfg.get("neutral_boundary_mix_logit", 0.0)
                ),
                distance_scale_raw=float(
                    cfg.get("neutral_distance_scale_raw", -1.5)
                ),
                maximum_polarity_shift=float(
                    cfg.get("neutral_polarity_maximum_shift", 0.0)
                ),
                polarity_prior_maximum_shift=float(
                    cfg.get("external_polarity_anchor_maximum_shift", 1.0)
                ),
            )

    def set_text_encoder_trainable(self, enabled: bool) -> None:
        if self.parameter_efficient_encoder:
            for name, parameter in self.text_encoder.named_parameters():
                parameter.requires_grad_(
                    enabled and name in self.encoder_adapter_parameter_names
                )
        else:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad_(enabled)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        text_output = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=(
                self.use_pretrained_classifier or self.use_layerwise_pooling
            ),
        )
        if self.use_pretrained_classifier:
            final_encoded = text_output.hidden_states[-1]
            raw_external_logits = text_output.logits.float()
            if self.external_class_groups:
                raw_external_probability = torch.softmax(raw_external_logits, dim=-1)
                external_probability = torch.stack(
                    [
                        raw_external_probability[:, indices].sum(dim=1)
                        for indices in self.external_class_groups
                    ],
                    dim=1,
                ).clamp_min(1e-8)
                external_probability = external_probability / external_probability.sum(
                    dim=1, keepdim=True
                )
                external_logits = external_probability.log()
            else:
                external_logits = raw_external_logits[:, self.external_class_order]
        else:
            final_encoded = text_output.last_hidden_state
            external_logits = final_encoded.new_zeros(final_encoded.size(0), 3).float()
        layer_mix_weights = final_encoded.new_zeros(1).float()
        layer_mix_scale = final_encoded.new_zeros(()).float()
        if self.use_layerwise_pooling:
            hidden_states = text_output.hidden_states
            if hidden_states is None:
                raise RuntimeError("Layerwise pooling requires encoder hidden states")
            selected = torch.stack(hidden_states[-self.layer_mix_count :], dim=0)
            layer_mix_weights = torch.softmax(self.layer_mix_logits.float(), dim=0)
            mixed_encoded = torch.einsum(
                "l,lbsd->bsd", layer_mix_weights.to(selected.dtype), selected
            )
            if self.layerwise_residual:
                layer_mix_scale = torch.sigmoid(self.layer_mix_scale_logit.float())
                encoded = final_encoded + layer_mix_scale.to(final_encoded.dtype) * (
                    mixed_encoded - final_encoded
                )
            else:
                layer_mix_scale = self.layer_mix_scale.float()
                encoded = self.layer_mix_scale * mixed_encoded
        else:
            encoded = final_encoded
        attention_mask = batch["attention_mask"].bool()
        token_type_ids = batch.get(
            "token_type_ids", torch.zeros_like(batch["attention_mask"])
        ).long()
        current_mask = attention_mask
        if self.use_context:
            current_mask = attention_mask & token_type_ids.eq(0)
        text = self.text_pool(
            torch.cat([encoded[:, 0], masked_mean(encoded, current_mask)], dim=-1)
        )
        emotion_state = torch.zeros_like(text)
        emotion_probability = text.new_full((text.size(0), 3), 1.0 / 3.0).float()
        emotion_representation_reliability = text.new_zeros(text.size(0)).float()
        emotion_auxiliary_logits = text.new_zeros(text.size(0), 3).float()
        if self.use_emotion_evidence:
            if "emotion_input_ids" not in batch:
                raise KeyError(
                    "emotion_input_ids are required when use_emotion_evidence is enabled"
                )
            # The external emotion tower is a fixed semantic coordinate system.
            # Keeping it in eval/no-grad mode prevents fold-specific drift and
            # makes only the bounded agreement module task-trainable.
            self.emotion_encoder.eval()
            with torch.no_grad():
                emotion_output = self.emotion_encoder(
                    input_ids=batch["emotion_input_ids"],
                    attention_mask=batch["emotion_attention_mask"],
                    output_hidden_states=True,
                )
                raw_emotion_probability = torch.softmax(
                    emotion_output.logits.float(), dim=-1
                )
                emotion_probability = torch.stack(
                    [
                        raw_emotion_probability[:, indices].sum(dim=1)
                        for indices in self.emotion_class_groups
                    ],
                    dim=1,
                ).clamp_min(1e-8)
                emotion_probability = emotion_probability / emotion_probability.sum(
                    dim=1, keepdim=True
                )
                emotion_encoded = emotion_output.hidden_states[-1]
            (
                text,
                emotion_state,
                emotion_representation_reliability,
                emotion_auxiliary_logits,
            ) = self.emotion_evidence.adapt(
                text,
                emotion_encoded,
                batch["emotion_attention_mask"],
                emotion_probability,
            )
        # EXP019 jointly encodes the current utterance and causal context.  The
        # structured boundary head needs an uncontaminated polarity anchor, so
        # it receives a second, current-only view when context is enabled.
        text_anchor = text
        if self.use_polar_distance_neutral and self.use_context:
            if "current_input_ids" not in batch:
                raise KeyError(
                    "current_input_ids are required for the polar-distance "
                    "Neutral expert with joint context"
                )
            anchor_output = self.text_encoder(
                input_ids=batch["current_input_ids"],
                attention_mask=batch["current_attention_mask"],
                output_hidden_states=self.use_pretrained_classifier,
            )
            anchor_encoded = (
                anchor_output.hidden_states[-1]
                if self.use_pretrained_classifier
                else anchor_output.last_hidden_state
            )
            anchor_mask = batch["current_attention_mask"].bool()
            text_anchor = self.text_pool(
                torch.cat(
                    [
                        anchor_encoded[:, 0],
                        masked_mean(anchor_encoded, anchor_mask),
                    ],
                    dim=-1,
                )
            )
        text_tokens = self.text_token_projection(encoded)
        context_shift = torch.zeros_like(text)
        context_reliability = text.new_zeros(text.size(0))
        context_reject_weight = text.new_ones(text.size(0))
        previous_context_weight = text.new_zeros(text.size(0))
        following_context_weight = text.new_zeros(text.size(0))
        previous_context_tokens = None
        following_context_tokens = None
        previous_context_mask = None
        following_context_mask = None
        context_transition_residual = text.new_zeros(text.size(0), 3).float()
        context_transition_scale = text.new_zeros(()).float()
        context_transition_norm = text.new_zeros(text.size(0)).float()
        bounded_transition_state = None
        if self.use_context:
            context_mask = attention_mask & token_type_ids.eq(1)
            type_index = token_type_ids.clamp(min=0, max=1)
            text_tokens = text_tokens + self.context_type_embeddings[type_index]
            context = self.context_projection(masked_mean(encoded, context_mask))
            _, context_shift = self.context_adaptation(text, context)
            context_available = batch.get(
                "context_available", context_mask.any(dim=1)
            ).to(text.dtype)
            context_reliability = (
                self.context_reliability(text, context).squeeze(-1)
                * context_available
            )
            text = self.context_norm(
                text + context_reliability.unsqueeze(-1) * context_shift
            )
            context_reject_weight = 1.0 - context_reliability
        if self.use_directional_context:
            previous_attention_mask = batch["previous_attention_mask"].bool()
            following_attention_mask = batch["following_attention_mask"].bool()
            previous_available = batch["previous_context_available"].bool()
            following_available = batch["following_context_available"].bool()
            previous_output = self.text_encoder(
                input_ids=batch["previous_input_ids"],
                attention_mask=batch["previous_attention_mask"],
                output_hidden_states=self.use_pretrained_classifier,
            )
            following_output = self.text_encoder(
                input_ids=batch["following_input_ids"],
                attention_mask=batch["following_attention_mask"],
                output_hidden_states=self.use_pretrained_classifier,
            )
            if self.use_pretrained_classifier:
                previous_encoded = previous_output.hidden_states[-1]
                following_encoded = following_output.hidden_states[-1]
            else:
                previous_encoded = previous_output.last_hidden_state
                following_encoded = following_output.last_hidden_state
            previous_context = self.directional_context_pool(
                torch.cat(
                    [
                        previous_encoded[:, 0],
                        masked_mean(previous_encoded, previous_attention_mask),
                    ],
                    dim=-1,
                )
            ) + self.direction_embeddings[0]
            following_context = self.directional_context_pool(
                torch.cat(
                    [
                        following_encoded[:, 0],
                        masked_mean(following_encoded, following_attention_mask),
                    ],
                    dim=-1,
                )
            ) + self.direction_embeddings[1]
            _, previous_shift = self.previous_context_adaptation(
                text, previous_context
            )
            _, following_shift = self.following_context_adaptation(
                text, following_context
            )
            current_probability = torch.softmax(
                self.text_auxiliary(text).float(), dim=-1
            )
            previous_probability = torch.softmax(
                self.text_auxiliary(previous_context).float(), dim=-1
            )
            following_probability = torch.softmax(
                self.text_auxiliary(following_context).float(), dim=-1
            )
            context_weights = self.directional_context_router(
                text,
                previous_context,
                following_context,
                previous_available,
                following_available,
                current_probability,
                previous_probability,
                following_probability,
            )
            context_reject_weight = context_weights[:, 0]
            previous_context_weight = context_weights[:, 1]
            following_context_weight = context_weights[:, 2]
            context_reliability = (
                previous_context_weight + following_context_weight
            )
            context_shift = (
                previous_context_weight.unsqueeze(-1) * previous_shift
                + following_context_weight.unsqueeze(-1) * following_shift
            )
            text = self.directional_context_norm(text + context_shift)
            previous_context_tokens = (
                self.text_token_projection(previous_encoded)
                + self.direction_embeddings[0]
            )
            following_context_tokens = (
                self.text_token_projection(following_encoded)
                + self.direction_embeddings[1]
            )
            previous_context_mask = (
                previous_attention_mask & previous_available.unsqueeze(-1)
            )
            following_context_mask = (
                following_attention_mask & following_available.unsqueeze(-1)
            )
        if self.use_context_transition or self.use_bounded_neutral_transition:
            previous_attention_mask = batch["previous_attention_mask"].bool()
            following_attention_mask = batch["following_attention_mask"].bool()
            previous_available = batch["previous_context_available"].bool()
            following_available = batch["following_context_available"].bool()
            previous_output = self.text_encoder(
                input_ids=batch["previous_input_ids"],
                attention_mask=batch["previous_attention_mask"],
                output_hidden_states=self.use_pretrained_classifier,
            )
            following_output = self.text_encoder(
                input_ids=batch["following_input_ids"],
                attention_mask=batch["following_attention_mask"],
                output_hidden_states=self.use_pretrained_classifier,
            )
            if self.use_pretrained_classifier:
                previous_encoded = previous_output.hidden_states[-1]
                following_encoded = following_output.hidden_states[-1]
            else:
                previous_encoded = previous_output.last_hidden_state
                following_encoded = following_output.last_hidden_state
            previous_context = self.text_pool(
                torch.cat(
                    [
                        previous_encoded[:, 0],
                        masked_mean(previous_encoded, previous_attention_mask),
                    ],
                    dim=-1,
                )
            )
            following_context = self.text_pool(
                torch.cat(
                    [
                        following_encoded[:, 0],
                        masked_mean(following_encoded, following_attention_mask),
                    ],
                    dim=-1,
                )
            )
            current_probability = torch.softmax(
                self.text_auxiliary(text).float(), dim=-1
            )
            previous_probability = torch.softmax(
                self.text_auxiliary(previous_context).float(), dim=-1
            )
            following_probability = torch.softmax(
                self.text_auxiliary(following_context).float(), dim=-1
            )
            if self.use_context_transition:
                transition_output = self.sentiment_transition_expert(
                    text,
                    previous_context,
                    following_context,
                    previous_available,
                    following_available,
                    current_probability,
                    previous_probability,
                    following_probability,
                )
            else:
                transition_output = self.bounded_neutral_transition_expert.encode(
                    text,
                    previous_context,
                    following_context,
                    previous_available,
                    following_available,
                    current_probability,
                    previous_probability,
                    following_probability,
                )
            context_weights = transition_output["weights"]
            context_reject_weight = context_weights[:, 0]
            previous_context_weight = context_weights[:, 1]
            following_context_weight = context_weights[:, 2]
            context_reliability = (
                previous_context_weight + following_context_weight
            )
            if self.use_context_transition:
                context_transition_residual = transition_output["residual"]
                context_transition_scale = transition_output["scale"]
                context_transition_norm = context_transition_residual.norm(dim=-1)
            else:
                bounded_transition_state = transition_output
        hyper_text_weights = text.new_zeros(text.size(0), 1, 1)
        hyper_shift = torch.zeros_like(text)
        hyper_reliability = text.new_zeros(text.size(0))
        hyper_tokens = None
        if self.use_almt_hyper:
            hyper_vector, hyper_tokens, hyper_text_weights = self.adaptive_hyper(
                batch["legacy_text"],
                batch["legacy_text_mask"],
                batch["audio"],
                batch["audio_mask"],
                batch["vision"],
                batch["vision_mask"],
            )
            _, hyper_shift = self.hyper_adaptation(text, hyper_vector)
            hyper_reliability = self.hyper_reliability(
                text, hyper_vector
            ).squeeze(-1)
            text = self.hyper_norm(
                text + hyper_reliability.unsqueeze(-1) * hyper_shift
            )
        legacy_text_weights = text.new_zeros(text.size(0), 1)
        legacy_text_shift = torch.zeros_like(text)
        legacy_text_reliability = text.new_zeros(text.size(0))
        legacy_text_hidden = None
        if self.use_legacy_text:
            (
                legacy_text_hidden,
                legacy_text,
                legacy_text_weights,
            ) = self.legacy_text_encoder(
                batch["legacy_text"], batch["legacy_text_mask"]
            )
            _, legacy_text_shift = self.legacy_text_adaptation(text, legacy_text)
            legacy_text_reliability = self.legacy_text_reliability(
                text, legacy_text
            ).squeeze(-1)
            text = self.legacy_text_norm(
                text
                + legacy_text_reliability.unsqueeze(-1) * legacy_text_shift
            )
        text_auxiliary_logits = self.text_auxiliary(text)
        text_auxiliary_probability = torch.softmax(
            text_auxiliary_logits.float(), dim=-1
        )
        text_auxiliary_top_two = torch.topk(
            text_auxiliary_probability, k=2, dim=-1
        ).values
        video_background_anchor_uncertainty = (
            1.0 - (text_auxiliary_top_two[:, 0] - text_auxiliary_top_two[:, 1])
        ).clamp(0.0, 1.0)
        audio_weights = text.new_zeros(text.size(0), 1)
        visual_weights = text.new_zeros(text.size(0), 1)
        component_weights = text.new_ones(text.size(0), 1)
        audio_shift = torch.zeros_like(text)
        adaptation_shift = torch.zeros_like(text)
        audio_reliability = text.new_zeros(text.size(0))
        visual_reliability = text.new_zeros(text.size(0))
        audio_background_gate = text.new_zeros(text.size(0))
        vision_background_gate = text.new_zeros(text.size(0))
        audio_background_delta_norm = text.new_zeros(text.size(0))
        vision_background_delta_norm = text.new_zeros(text.size(0))
        audio_background_similarity = text.new_zeros(text.size(0))
        vision_background_similarity = text.new_zeros(text.size(0))
        fusion_scale = text.new_zeros(())
        low_rank_interaction_mix = text.new_zeros(text.size(0))
        low_rank_interaction_norm = text.new_zeros(text.size(0))
        spectral_audio_gate = text.new_zeros(text.size(0))
        spectral_vision_gate = text.new_zeros(text.size(0))
        spectral_audio_velocity = text.new_zeros(text.size(0))
        spectral_vision_velocity = text.new_zeros(text.size(0))
        spectral_audio_acceleration = text.new_zeros(text.size(0))
        spectral_vision_acceleration = text.new_zeros(text.size(0))
        spectral_audio_low_energy = text.new_zeros(text.size(0))
        spectral_vision_low_energy = text.new_zeros(text.size(0))
        spectral_audio_high_energy = text.new_zeros(text.size(0))
        spectral_vision_high_energy = text.new_zeros(text.size(0))
        evidence_tokens = [text_tokens]
        evidence_masks = [attention_mask]
        evidence_weights = [torch.ones_like(attention_mask, dtype=text.dtype)]
        if self.use_emotion_evidence:
            evidence_tokens.append(emotion_state.unsqueeze(1))
            evidence_masks.append(
                torch.ones(
                    (text.size(0), 1), dtype=torch.bool, device=text.device
                )
            )
            evidence_weights.append(
                emotion_representation_reliability.unsqueeze(-1).to(text.dtype)
            )
        if self.use_directional_context:
            assert previous_context_tokens is not None
            assert following_context_tokens is not None
            assert previous_context_mask is not None
            assert following_context_mask is not None
            evidence_tokens.extend(
                [previous_context_tokens, following_context_tokens]
            )
            evidence_masks.extend(
                [previous_context_mask, following_context_mask]
            )
            evidence_weights.extend(
                [
                    previous_context_weight.unsqueeze(-1).expand_as(
                        previous_context_mask
                    ),
                    following_context_weight.unsqueeze(-1).expand_as(
                        following_context_mask
                    ),
                ]
            )
        if self.use_almt_hyper:
            assert hyper_tokens is not None
            evidence_tokens.append(hyper_tokens)
            evidence_masks.append(
                torch.ones(
                    hyper_tokens.shape[:2],
                    dtype=torch.bool,
                    device=hyper_tokens.device,
                )
            )
            evidence_weights.append(
                torch.ones(
                    hyper_tokens.shape[:2],
                    dtype=text.dtype,
                    device=hyper_tokens.device,
                )
            )
        if self.use_legacy_text:
            assert legacy_text_hidden is not None
            evidence_tokens.append(legacy_text_hidden)
            evidence_masks.append(batch["legacy_text_mask"].bool())
            evidence_weights.append(
                torch.ones_like(batch["legacy_text_mask"], dtype=text.dtype)
            )
        fused = text
        audio = None
        vision = None
        audio_centered = None
        vision_centered = None
        if self.use_audio:
            audio_hidden, audio, audio_weights = self.audio_encoder(
                batch["audio"], batch["audio_mask"]
            )
            if self.use_spectral_dynamics:
                audio_spectral = self.audio_spectral_dynamics(
                    text, audio, batch["audio"], batch["audio_mask"]
                )
                audio = audio_spectral["corrected"]
                spectral_audio_gate = audio_spectral["gate"]
                spectral_audio_velocity = audio_spectral["velocity"]
                spectral_audio_acceleration = audio_spectral["acceleration"]
                spectral_audio_low_energy = audio_spectral["low_energy"]
                spectral_audio_high_energy = audio_spectral["high_energy"]
                evidence_tokens.append(audio_spectral["spectral"].unsqueeze(1))
                evidence_masks.append(
                    torch.ones(
                        text.size(0), 1, dtype=torch.bool, device=text.device
                    )
                )
                evidence_weights.append(text.new_ones(text.size(0), 1))
            if (
                self.use_video_background_deconfounder
                or self.use_video_relative_neutral_head
            ):
                available = batch["video_reference_available"].bool()
                audio_centered_input = (
                    batch["audio"]
                    - batch["video_audio_reference"].unsqueeze(1)
                ) * batch["audio_mask"].unsqueeze(-1).to(batch["audio"].dtype)
                _, audio_centered, _ = self.audio_encoder(
                    audio_centered_input, batch["audio_mask"]
                )
            if self.use_video_background_deconfounder:
                assert audio_centered is not None
                preliminary_reliability = self.audio_reliability(
                    text, audio
                ).squeeze(-1)
                background = self.video_background_deconfounder.audio(
                    text,
                    audio,
                    audio_centered,
                    preliminary_reliability,
                    available,
                    batch["video_group_log_size"],
                    video_background_anchor_uncertainty,
                )
                audio = background["corrected"]
                audio_background_gate = background["gate"]
                audio_background_delta_norm = background["delta_norm"]
                audio_background_similarity = background[
                    "background_similarity"
                ]
            _, audio_shift = self.audio_adaptation(text, audio)
            audio_reliability = self.audio_reliability(text, audio).squeeze(-1)
            evidence_tokens.append(audio_hidden)
            evidence_masks.append(batch["audio_mask"].bool())
            evidence_weights.append(
                torch.ones_like(batch["audio_mask"], dtype=text.dtype)
            )
        if self.use_vision:
            vision_hidden, vision, visual_weights = self.vision_encoder(
                batch["vision"], batch["vision_mask"]
            )
            if self.use_spectral_dynamics:
                vision_spectral = self.vision_spectral_dynamics(
                    text, vision, batch["vision"], batch["vision_mask"]
                )
                vision = vision_spectral["corrected"]
                spectral_vision_gate = vision_spectral["gate"]
                spectral_vision_velocity = vision_spectral["velocity"]
                spectral_vision_acceleration = vision_spectral["acceleration"]
                spectral_vision_low_energy = vision_spectral["low_energy"]
                spectral_vision_high_energy = vision_spectral["high_energy"]
                evidence_tokens.append(vision_spectral["spectral"].unsqueeze(1))
                evidence_masks.append(
                    torch.ones(
                        text.size(0), 1, dtype=torch.bool, device=text.device
                    )
                )
                evidence_weights.append(text.new_ones(text.size(0), 1))
            if (
                self.use_video_background_deconfounder
                or self.use_video_relative_neutral_head
            ):
                available = batch["video_reference_available"].bool()
                vision_centered_input = (
                    batch["vision"]
                    - batch["video_vision_reference"].unsqueeze(1)
                ) * batch["vision_mask"].unsqueeze(-1).to(batch["vision"].dtype)
                _, vision_centered, _ = self.vision_encoder(
                    vision_centered_input, batch["vision_mask"]
                )
            if self.use_video_background_deconfounder:
                assert vision_centered is not None
                preliminary_reliability = self.visual_reliability(
                    text, vision
                ).squeeze(-1)
                background = self.video_background_deconfounder.vision(
                    text,
                    vision,
                    vision_centered,
                    preliminary_reliability,
                    available,
                    batch["video_group_log_size"],
                    video_background_anchor_uncertainty,
                )
                vision = background["corrected"]
                vision_background_gate = background["gate"]
                vision_background_delta_norm = background["delta_norm"]
                vision_background_similarity = background[
                    "background_similarity"
                ]
            _, adaptation_shift = self.adaptation(text, vision)
            visual_reliability = self.visual_reliability(text, vision).squeeze(-1)
            evidence_tokens.append(vision_hidden)
            evidence_masks.append(batch["vision_mask"].bool())
            evidence_weights.append(
                torch.ones_like(batch["vision_mask"], dtype=text.dtype)
            )
        if self.use_audio or self.use_vision:
            adapted_text = text
            if self.use_audio:
                adapted_text = (
                    adapted_text + audio_reliability.unsqueeze(-1) * audio_shift
                )
            if self.use_vision:
                adapted_text = (
                    adapted_text
                    + visual_reliability.unsqueeze(-1) * adaptation_shift
                )
            adapted_text = self.multimodal_norm(adapted_text)
            if self.use_audio and self.use_vision:
                assert audio is not None and vision is not None
                fusion_candidate, component_weights = self.tri_fusion(
                    adapted_text, audio, vision
                )
                if self.use_low_rank_interaction:
                    (
                        fusion_candidate,
                        low_rank_interaction_mix,
                        low_rank_candidate,
                    ) = self.low_rank_interaction(
                        fusion_candidate,
                        adapted_text,
                        audio,
                        vision,
                        audio_reliability,
                        visual_reliability,
                    )
                    low_rank_interaction_norm = (
                        low_rank_candidate.float() - adapted_text.float()
                    ).norm(dim=-1)
            else:
                modality = audio if self.use_audio else vision
                assert modality is not None
                fusion_candidate, component_weights = self.fusion(
                    adapted_text, modality
                )
            if self.use_dynamic_fusion_router:
                text_probability = torch.softmax(text_auxiliary_logits.float(), dim=-1)
                route_delta = self.dynamic_fusion_router(
                    adapted_text,
                    fusion_candidate,
                    audio_reliability,
                    visual_reliability,
                    text_probability,
                )
                fusion_scale = torch.sigmoid(self.fusion_scale_logit + route_delta)
                fused = adapted_text + fusion_scale.unsqueeze(-1) * (
                    fusion_candidate - adapted_text
                )
            else:
                fusion_scale = torch.sigmoid(self.fusion_scale_logit)
                fused = adapted_text + fusion_scale * (fusion_candidate - adapted_text)

        direct_logits = self.classifier(fused)
        direct_probability = torch.softmax(direct_logits.float(), dim=-1)
        prototype_logits = direct_logits.float()
        prototype_attention = text.new_zeros(text.size(0), 3, 1)
        contrastive_embedding = F.normalize(fused.float(), dim=-1)
        prototype_mix = text.new_zeros(())
        base_probability = direct_probability
        if self.use_prototype_router:
            prototype_logits, contrastive_embedding, prototype_attention = (
                self.prototype_router(
                    fused,
                    torch.cat(evidence_tokens, dim=1),
                    torch.cat(evidence_masks, dim=1),
                    (
                        torch.cat(evidence_weights, dim=1)
                        if self.use_context_weighted_prototype
                        else None
                    ),
                )
            )
            prototype_probability = torch.softmax(
                prototype_logits.float(), dim=-1
            )
            prototype_mix = torch.sigmoid(self.prototype_mix_logit)
            base_probability = (
                (1.0 - prototype_mix) * direct_probability
                + prototype_mix * prototype_probability
            ).clamp_min(1e-8)
        video_relative_neutral_shift = text.new_zeros(text.size(0)).float()
        video_relative_neutral_trust = text.new_zeros(text.size(0)).float()
        video_relative_crossmodal_agreement = text.new_zeros(text.size(0)).float()
        video_relative_audio_similarity = text.new_zeros(text.size(0)).float()
        video_relative_vision_similarity = text.new_zeros(text.size(0)).float()
        if self.use_video_relative_neutral_head:
            assert audio is not None and vision is not None
            assert audio_centered is not None and vision_centered is not None
            video_relative = self.video_relative_neutral_head(
                base_probability,
                text,
                audio,
                vision,
                audio_centered,
                vision_centered,
                audio_reliability,
                visual_reliability,
                batch["video_reference_available"].bool(),
                batch["video_group_log_size"],
                video_background_anchor_uncertainty,
            )
            base_probability = video_relative["probabilities"]
            video_relative_neutral_shift = video_relative["neutral_shift"]
            video_relative_neutral_trust = video_relative["trust"]
            video_relative_crossmodal_agreement = video_relative[
                "crossmodal_agreement"
            ]
            video_relative_audio_similarity = video_relative["audio_similarity"]
            video_relative_vision_similarity = video_relative["vision_similarity"]
        pairwise_video_relative_shift = text.new_zeros(text.size(0)).float()
        pairwise_video_relative_trust = text.new_zeros(text.size(0)).float()
        pairwise_video_relative_logit = text.new_zeros(text.size(0)).float()
        pairwise_video_audio_evidence = text.new_zeros(text.size(0)).float()
        pairwise_video_vision_evidence = text.new_zeros(text.size(0)).float()
        pairwise_video_crossmodal_agreement = text.new_zeros(text.size(0)).float()
        pairwise_video_audio_route = text.new_zeros(text.size(0)).float()
        pairwise_video_vision_route = text.new_zeros(text.size(0)).float()
        pairwise_video_correction_scale = text.new_zeros(()).float()
        pairwise_video_semantic_evidence = text.new_zeros(text.size(0)).float()
        pairwise_video_semantic_route = text.new_zeros(text.size(0)).float()
        pairwise_video_anchor_logit = text.new_zeros(text.size(0)).float()
        pairwise_video_anchor_shift = text.new_zeros(text.size(0)).float()
        pairwise_video_neutral_logit = torch.logit(
            base_probability[:, 1].float().clamp(1e-6, 1.0 - 1e-6)
        )
        if self.use_pairwise_signed_video_neutral_head:
            assert audio is not None and vision is not None
            audio_current = masked_mean(
                batch["audio"], batch["audio_mask"].bool()
            )
            vision_current = masked_mean(
                batch["vision"], batch["vision_mask"].bool()
            )
            semantic_current = None
            semantic_reference = None
            if self.pairwise_video_use_semantic_reference:
                semantic_current = masked_mean(
                    batch["legacy_text"], batch["legacy_text_mask"].bool()
                )
                semantic_reference = batch["video_text_reference"]
            pairwise_video = self.pairwise_signed_video_neutral_head(
                base_probability,
                text,
                audio_current,
                batch["video_audio_reference"],
                vision_current,
                batch["video_vision_reference"],
                audio_reliability,
                visual_reliability,
                batch["video_reference_available"].bool(),
                batch["video_group_log_size"],
                video_background_anchor_uncertainty,
                semantic_current,
                semantic_reference,
            )
            base_probability = pairwise_video["probabilities"]
            pairwise_video_relative_shift = pairwise_video["neutral_shift"]
            pairwise_video_relative_trust = pairwise_video["trust"]
            pairwise_video_relative_logit = pairwise_video["relative_logit"]
            pairwise_video_audio_evidence = pairwise_video["audio_evidence"]
            pairwise_video_vision_evidence = pairwise_video["vision_evidence"]
            pairwise_video_crossmodal_agreement = pairwise_video[
                "crossmodal_agreement"
            ]
            pairwise_video_audio_route = pairwise_video["audio_route"]
            pairwise_video_vision_route = pairwise_video["vision_route"]
            pairwise_video_correction_scale = pairwise_video[
                "correction_scale"
            ]
            pairwise_video_semantic_evidence = pairwise_video[
                "semantic_evidence"
            ]
            pairwise_video_semantic_route = pairwise_video["semantic_route"]
            pairwise_video_anchor_logit = pairwise_video["anchor_logit"]
            pairwise_video_anchor_shift = pairwise_video["anchor_shift"]
            pairwise_video_neutral_logit = pairwise_video["neutral_logit"]
        cross_video_relation = None
        if self.use_cross_video_bilateral_relation:
            assert audio is not None and vision is not None
            semantic_current = masked_mean(
                batch["legacy_text"], batch["legacy_text_mask"].bool()
            )
            audio_current = masked_mean(
                batch["audio"], batch["audio_mask"].bool()
            )
            vision_current = masked_mean(
                batch["vision"], batch["vision_mask"].bool()
            )
            cross_video_relation = self.cross_video_bilateral_relation(
                base_probability,
                text,
                semantic_current,
                audio_current,
                vision_current,
                batch["cross_video_semantic_references"],
                batch["cross_video_audio_references"],
                batch["cross_video_vision_references"],
                batch["cross_video_retrieval_similarity"],
                batch["cross_video_reference_mask"].bool(),
                audio_reliability,
                visual_reliability,
                video_background_anchor_uncertainty,
            )
            base_probability = cross_video_relation["probabilities"]
        subcenter_logits = direct_logits.float()
        subcenter_mix = text.new_zeros(()).float()
        subcenter_diversity = text.new_zeros(()).float()
        subcenter_scale = text.new_zeros(()).float()
        if self.use_subcenter_head:
            subcenter = self.subcenter_head(fused)
            subcenter_logits = subcenter["logits"]
            subcenter_mix = subcenter["mix"]
            subcenter_diversity = subcenter["diversity"]
            subcenter_scale = subcenter["scale"]
            subcenter_probability = torch.softmax(subcenter_logits, dim=-1)
            base_probability = (
                (1.0 - subcenter_mix) * base_probability
                + subcenter_mix * subcenter_probability
            ).clamp_min(1e-8)
        if self.use_context_transition:
            base_probability = torch.softmax(
                base_probability.float().clamp_min(1e-8).log()
                + context_transition_residual,
                dim=-1,
            )
        if self.use_bounded_neutral_transition:
            if bounded_transition_state is None:
                raise RuntimeError("bounded transition state was not constructed")
            bounded_correction = self.bounded_neutral_transition_expert.correct(
                base_probability,
                bounded_transition_state["transition"],
                bounded_transition_state["weights"],
            )
            base_probability = bounded_correction["probabilities"]
            context_transition_residual = bounded_correction["residual"]
            context_transition_scale = bounded_correction["scale"]
            context_transition_norm = context_transition_residual.norm(dim=-1)
        ordinal_threshold_logits = direct_logits.new_zeros(text.size(0), 2).float()
        ordinal_probability = direct_probability
        ordinal_score = text.new_zeros(text.size(0)).float()
        ordinal_half_width = text.new_zeros(())
        ordinal_mix = text.new_zeros(())
        if self.use_ordinal_expert:
            ordinal = self.ordinal_expert(fused)
            ordinal_threshold_logits = ordinal["threshold_logits"]
            ordinal_probability = ordinal["probabilities"]
            ordinal_score = ordinal["score"]
            ordinal_half_width = ordinal["half_width"]
            ordinal_mix = torch.sigmoid(self.ordinal_mix_logit)
            base_probability = (
                (1.0 - ordinal_mix) * base_probability
                + ordinal_mix * ordinal_probability
            ).clamp_min(1e-8)
        external_mix = text.new_zeros(())
        external_probability = torch.softmax(external_logits, dim=-1)
        if self.use_pretrained_classifier:
            external_mix = self.external_maximum_mix * torch.sigmoid(
                self.external_mix_logit
            )
            base_probability = (
                (1.0 - external_mix) * base_probability
                + external_mix * external_probability
            ).clamp_min(1e-8)
        emotion_decision_reliability = text.new_zeros(text.size(0)).float()
        emotion_neutral_residual = text.new_zeros(text.size(0)).float()
        emotion_polarity_residual = text.new_zeros(text.size(0)).float()
        emotion_agreement = text.new_zeros(text.size(0)).float()
        if self.use_emotion_evidence:
            emotion_correction = self.emotion_evidence.correct(
                base_probability,
                fused,
                emotion_state,
                emotion_probability,
            )
            base_probability = emotion_correction["probabilities"]
            emotion_decision_reliability = emotion_correction["reliability"]
            emotion_neutral_residual = emotion_correction["neutral_delta"]
            emotion_polarity_residual = emotion_correction["polarity_delta"]
            emotion_agreement = emotion_correction["agreement"]
        conflict_ignorance = None
        if self.use_conflict_ignorance_neutral:
            assert audio is not None and vision is not None
            conflict_ignorance = self.conflict_ignorance_neutral(
                base_probability,
                text,
                audio,
                vision,
                audio_reliability,
                visual_reliability,
            )
            base_probability = conflict_ignorance["probabilities"]
        aligned_incongruity = None
        if self.use_aligned_temporal_incongruity:
            aligned_incongruity = self.aligned_temporal_incongruity(
                base_probability,
                batch["legacy_text"],
                batch["audio"],
                batch["vision"],
                batch["legacy_text_mask"],
                batch["audio_mask"],
                batch["vision_mask"],
                audio_reliability,
                visual_reliability,
            )
            base_probability = aligned_incongruity["probabilities"]
        feature_decomposition = None
        if self.use_contrastive_feature_decomposition:
            assert audio is not None and vision is not None
            feature_decomposition = self.contrastive_feature_decomposition(
                base_probability,
                text,
                audio,
                vision,
                audio_reliability,
                visual_reliability,
            )
            base_probability = feature_decomposition["probabilities"]
        conditional_innovation = None
        if self.use_text_conditional_innovation:
            assert audio is not None and vision is not None
            conditional_innovation = self.text_conditional_innovation(
                base_probability,
                text,
                audio,
                vision,
                audio_reliability,
                visual_reliability,
            )
            base_probability = conditional_innovation["probabilities"]
        neutral_energy_bottleneck = None
        if self.use_neutral_energy_bottleneck:
            assert audio is not None and vision is not None
            modality_available = torch.stack(
                [
                    torch.ones_like(audio_reliability),
                    batch["audio_mask"].bool().any(dim=1).to(text.dtype),
                    batch["vision_mask"].bool().any(dim=1).to(text.dtype),
                ],
                dim=-1,
            )
            external_reliability = torch.stack(
                [
                    torch.ones_like(audio_reliability),
                    audio_reliability,
                    visual_reliability,
                ],
                dim=-1,
            )
            neutral_energy_bottleneck = self.neutral_energy_bottleneck(
                base_probability,
                text,
                audio,
                vision,
                external_reliability,
                modality_available,
            )
            base_probability = neutral_energy_bottleneck["probabilities"]
        direct_regression_raw = self.regressor(fused).squeeze(-1).float()
        direct_regression = 3.0 * torch.tanh(direct_regression_raw / 3.0)
        if neutral_energy_bottleneck is not None:
            neutral_energy_bottleneck["regression"] = direct_regression
        hurdle = self.hurdle(fused)
        polar_distance = None
        if self.use_polar_distance_neutral:
            polar_distance = self.polar_distance_neutral(
                text_anchor,
                fused,
                base_probability,
                context_reliability,
                audio_reliability,
                visual_reliability,
                (
                    external_logits[:, 2] - external_logits[:, 0]
                    if self.use_external_polarity_anchor
                    else None
                ),
            )
            probabilities = polar_distance["probabilities"]
            class_logits = probabilities.log()
            # Preserve the existing regression path while the structured
            # regression output remains an explicitly supervised auxiliary.
            regression = direct_regression
            structured = polar_distance
        elif self.use_hurdle:
            probabilities = (
                (1.0 - self.hurdle_mix) * base_probability
                + self.hurdle_mix * hurdle["probabilities"]
            ).clamp_min(1e-8)
            class_logits = probabilities.log()
            regression = (
                (1.0 - self.hurdle_mix) * direct_regression
                + self.hurdle_mix * hurdle["regression"]
            )
            structured = hurdle
        else:
            probabilities = base_probability
            class_logits = (
                probabilities.log()
                if (
                    self.use_prototype_router
                    or self.use_subcenter_head
                    or self.use_emotion_evidence
                    or self.use_conflict_ignorance_neutral
                    or self.use_aligned_temporal_incongruity
                    or self.use_contrastive_feature_decomposition
                    or self.use_text_conditional_innovation
                    or self.use_neutral_energy_bottleneck
                    or self.use_cross_video_bilateral_relation
                )
                else direct_logits.float()
            )
            regression = direct_regression
            structured = (
                neutral_energy_bottleneck
                if neutral_energy_bottleneck is not None
                else (
                    conditional_innovation
                    if conditional_innovation is not None
                    else (
                        feature_decomposition
                        if feature_decomposition is not None
                        else (
                            aligned_incongruity
                            if aligned_incongruity is not None
                            else (
                                conflict_ignorance
                                if conflict_ignorance is not None
                                else hurdle
                            )
                        )
                    )
                )
            )
        return {
            "class_logits": class_logits,
            "class_probabilities": probabilities,
            "regression": regression,
            "direct_regression": direct_regression,
            "text_auxiliary_logits": text_auxiliary_logits,
            "prototype_logits": prototype_logits,
            "contrastive_embedding": contrastive_embedding,
            # Exposed for train-only retrieval datastores. This tensor does not
            # participate in the forward decision and preserves old checkpoints.
            "retrieval_embedding": F.normalize(fused.float(), dim=-1),
            "prototype_attention": prototype_attention,
            "prototype_mix": prototype_mix,
            "subcenter_logits": subcenter_logits,
            "subcenter_mix": subcenter_mix,
            "subcenter_diversity": subcenter_diversity,
            "subcenter_scale": subcenter_scale,
            "ordinal_threshold_logits": ordinal_threshold_logits,
            "ordinal_probabilities": ordinal_probability,
            "ordinal_score": ordinal_score,
            "ordinal_half_width": ordinal_half_width,
            "ordinal_mix": ordinal_mix,
            "external_logits": external_logits,
            "external_mix": external_mix,
            "emotion_auxiliary_logits": emotion_auxiliary_logits,
            "emotion_probability": emotion_probability,
            "emotion_representation_reliability": emotion_representation_reliability,
            "emotion_decision_reliability": emotion_decision_reliability,
            "emotion_neutral_residual": emotion_neutral_residual,
            "emotion_polarity_residual": emotion_polarity_residual,
            "emotion_agreement": emotion_agreement,
            "conflict_neutral_shift": (
                conflict_ignorance["neutral_shift"]
                if conflict_ignorance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "conflict_mass": (
                conflict_ignorance["conflict"]
                if conflict_ignorance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "ignorance_mass": (
                conflict_ignorance["ignorance"]
                if conflict_ignorance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "aligned_neutral_shift": (
                aligned_incongruity["neutral_shift"]
                if aligned_incongruity is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "aligned_polarity_shift": (
                aligned_incongruity["polarity_shift"]
                if aligned_incongruity is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "aligned_attention_entropy": (
                aligned_incongruity["attention_entropy"]
                if aligned_incongruity is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "decomposition_parent_probability": (
                feature_decomposition["parent_probability"]
                if feature_decomposition is not None
                else base_probability.detach()
            ),
            "decomposition_shared_embeddings": (
                feature_decomposition["shared_embeddings"]
                if feature_decomposition is not None
                else text.new_zeros(
                    text.size(0), 3, self.decomposition_dimension
                ).float()
            ),
            "decomposition_modality_logits": (
                feature_decomposition["modality_logits"]
                if feature_decomposition is not None
                else text.new_zeros(text.size(0), 3, 3).float()
            ),
            "decomposition_modality_reliability": (
                feature_decomposition["modality_reliability"]
                if feature_decomposition is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "decomposition_consensus_logits": (
                feature_decomposition["consensus_logits"]
                if feature_decomposition is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "decomposition_residual_norm": (
                feature_decomposition["residual"].float().norm(dim=-1)
                if feature_decomposition is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "decomposition_orthogonality": (
                feature_decomposition["orthogonality"]
                if feature_decomposition is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "innovation_logits": (
                conditional_innovation["innovation_logits"]
                if conditional_innovation is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "innovation_parent_probability": (
                conditional_innovation["parent_probability"]
                if conditional_innovation is not None
                else base_probability.detach()
            ),
            "innovation_neutral_logit": (
                conditional_innovation["innovation_neutral_logit"]
                if conditional_innovation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "innovation_predicted_audio": (
                conditional_innovation["predicted_audio"]
                if conditional_innovation is not None
                else torch.zeros_like(text).float()
            ),
            "innovation_predicted_vision": (
                conditional_innovation["predicted_vision"]
                if conditional_innovation is not None
                else torch.zeros_like(text).float()
            ),
            "innovation_audio_target": (
                conditional_innovation["audio_target"]
                if conditional_innovation is not None
                else torch.zeros_like(text).float()
            ),
            "innovation_vision_target": (
                conditional_innovation["vision_target"]
                if conditional_innovation is not None
                else torch.zeros_like(text).float()
            ),
            "innovation_audio_error": (
                conditional_innovation["audio_error"]
                if conditional_innovation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "innovation_vision_error": (
                conditional_innovation["vision_error"]
                if conditional_innovation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "innovation_neutral_shift": (
                conditional_innovation["neutral_shift"]
                if conditional_innovation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "innovation_polarity_shift": (
                conditional_innovation["polarity_shift"]
                if conditional_innovation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "neutral_energy_parent_probability": (
                neutral_energy_bottleneck["parent_probability"]
                if neutral_energy_bottleneck is not None
                else base_probability.detach()
            ),
            "neutral_energy_modality_logits": (
                neutral_energy_bottleneck["modality_neutral_logits"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "neutral_energy_modality_probabilities": (
                neutral_energy_bottleneck["modality_neutral_probabilities"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "neutral_energy_reliability": (
                neutral_energy_bottleneck["modality_reliability"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "neutral_energy_weights": (
                neutral_energy_bottleneck["modality_weights"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "neutral_energy_contributions": (
                neutral_energy_bottleneck["modality_contributions"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "neutral_energy_values": (
                neutral_energy_bottleneck["neutral_energy"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "polar_energy_values": (
                neutral_energy_bottleneck["polar_energy"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "neutral_energy_orthogonality": (
                neutral_energy_bottleneck["orthogonality"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0), 3).float()
            ),
            "neutral_energy_poe_residual": (
                neutral_energy_bottleneck["poe_residual"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "neutral_energy_disagreement": (
                neutral_energy_bottleneck["disagreement"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "neutral_energy_agreement": (
                neutral_energy_bottleneck["agreement"]
                if neutral_energy_bottleneck is not None
                else text.new_ones(text.size(0)).float()
            ),
            "neutral_energy_coherence": (
                neutral_energy_bottleneck["coherence"]
                if neutral_energy_bottleneck is not None
                else text.new_ones(text.size(0)).float()
            ),
            "neutral_energy_unanimous": (
                neutral_energy_bottleneck["unanimous"]
                if neutral_energy_bottleneck is not None
                else text.new_ones(text.size(0)).float()
            ),
            "neutral_energy_boundary_gate": (
                neutral_energy_bottleneck["boundary_gate"]
                if neutral_energy_bottleneck is not None
                else text.new_ones(text.size(0)).float()
            ),
            "neutral_energy_shift": (
                neutral_energy_bottleneck["neutral_shift"]
                if neutral_energy_bottleneck is not None
                else text.new_zeros(text.size(0)).float()
            ),
            **{
                f"neutral_energy_{modality}_{quantity}": (
                    neutral_energy_bottleneck[source][:, index]
                    if neutral_energy_bottleneck is not None
                    else text.new_zeros(text.size(0)).float()
                )
                for modality, index in (
                    ("text", 0),
                    ("audio", 1),
                    ("vision", 2),
                )
                for quantity, source in (
                    ("probability", "modality_neutral_probabilities"),
                    ("reliability", "modality_reliability"),
                    ("weight", "modality_weights"),
                    ("contribution", "modality_contributions"),
                    ("neutral_energy", "neutral_energy"),
                    ("polar_energy", "polar_energy"),
                )
            },
            "neutral_logit": structured["neutral_logit"],
            "polarity_logit": structured["polarity_logit"],
            "hurdle_regression": structured["regression"],
            "neutral_boundary_probability": (
                polar_distance["boundary_probability"]
                if polar_distance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "neutral_boundary_mix": (
                polar_distance["boundary_mix"]
                if polar_distance is not None
                else text.new_zeros(()).float()
            ),
            "neutral_evidence_shift": (
                polar_distance["evidence_shift"]
                if polar_distance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "neutral_distance_scale": (
                polar_distance["distance_scale"]
                if polar_distance is not None
                else text.new_zeros(()).float()
            ),
            "polar_uncertainty": (
                polar_distance["polar_uncertainty"]
                if polar_distance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "polarity_anchor_logit": (
                polar_distance["polarity_anchor_logit"]
                if polar_distance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "polarity_prior_logit": (
                polar_distance["polarity_prior_logit"]
                if polar_distance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "polarity_prior_residual": (
                polar_distance["polarity_prior_residual"]
                if polar_distance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "polarity_residual": (
                polar_distance["polarity_residual"]
                if polar_distance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "polarity_trust": (
                polar_distance["polarity_trust"]
                if polar_distance is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "hyper_text_weights": hyper_text_weights,
            "legacy_text_weights": legacy_text_weights,
            "audio_weights": audio_weights,
            "visual_weights": visual_weights,
            "component_weights": component_weights,
            "context_reliability": context_reliability,
            "context_reject_weight": context_reject_weight,
            "previous_context_weight": previous_context_weight,
            "following_context_weight": following_context_weight,
            "context_transition_scale": context_transition_scale,
            "context_transition_norm": context_transition_norm,
            "hyper_reliability": hyper_reliability,
            "legacy_text_reliability": legacy_text_reliability,
            "audio_reliability": audio_reliability,
            "visual_reliability": visual_reliability,
            "audio_background_gate": audio_background_gate,
            "vision_background_gate": vision_background_gate,
            "audio_background_delta_norm": audio_background_delta_norm,
            "vision_background_delta_norm": vision_background_delta_norm,
            "audio_background_similarity": audio_background_similarity,
            "vision_background_similarity": vision_background_similarity,
            "video_background_anchor_uncertainty": (
                video_background_anchor_uncertainty
            ),
            "video_relative_neutral_shift": video_relative_neutral_shift,
            "video_relative_neutral_trust": video_relative_neutral_trust,
            "video_relative_crossmodal_agreement": (
                video_relative_crossmodal_agreement
            ),
            "video_relative_audio_similarity": video_relative_audio_similarity,
            "video_relative_vision_similarity": video_relative_vision_similarity,
            "pairwise_video_relative_shift": pairwise_video_relative_shift,
            "pairwise_video_relative_trust": pairwise_video_relative_trust,
            "pairwise_video_relative_logit": pairwise_video_relative_logit,
            "pairwise_video_audio_evidence": pairwise_video_audio_evidence,
            "pairwise_video_vision_evidence": pairwise_video_vision_evidence,
            "pairwise_video_crossmodal_agreement": (
                pairwise_video_crossmodal_agreement
            ),
            "pairwise_video_audio_route": pairwise_video_audio_route,
            "pairwise_video_vision_route": pairwise_video_vision_route,
            "pairwise_video_correction_scale": pairwise_video_correction_scale,
            "pairwise_video_semantic_evidence": (
                pairwise_video_semantic_evidence
            ),
            "pairwise_video_semantic_route": pairwise_video_semantic_route,
            "pairwise_video_anchor_logit": pairwise_video_anchor_logit,
            "pairwise_video_anchor_shift": pairwise_video_anchor_shift,
            "pairwise_video_neutral_logit": pairwise_video_neutral_logit,
            "cross_video_neutral_shift": (
                cross_video_relation["neutral_shift"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_parent_probability": (
                cross_video_relation["parent_probability"]
                if cross_video_relation is not None
                else base_probability.detach()
            ),
            "cross_video_relation_trust": (
                cross_video_relation["trust"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_left_logit": (
                cross_video_relation["left_logit"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_right_logit": (
                cross_video_relation["right_logit"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_left_residual": (
                cross_video_relation["left_residual"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_right_residual": (
                cross_video_relation["right_residual"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_semantic_left_margin": (
                cross_video_relation["semantic_margin"][:, 0]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_semantic_right_margin": (
                cross_video_relation["semantic_margin"][:, 1]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_audio_left_margin": (
                cross_video_relation["audio_margin"][:, 0]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_audio_right_margin": (
                cross_video_relation["audio_margin"][:, 1]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_vision_left_margin": (
                cross_video_relation["vision_margin"][:, 0]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_vision_right_margin": (
                cross_video_relation["vision_margin"][:, 1]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_crossmodal_agreement": (
                cross_video_relation["crossmodal_agreement"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_negative_affinity": (
                cross_video_relation["negative_affinity"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_neutral_affinity": (
                cross_video_relation["neutral_affinity"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_positive_affinity": (
                cross_video_relation["positive_affinity"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_neutral_logit": (
                cross_video_relation["neutral_logit"]
                if cross_video_relation is not None
                else text.new_zeros(text.size(0)).float()
            ),
            "cross_video_relation_scale": (
                cross_video_relation["relation_scale"]
                if cross_video_relation is not None
                else text.new_zeros(()).float()
            ),
            "context_shift_norm": context_shift.float().norm(dim=-1),
            "hyper_shift_norm": hyper_shift.float().norm(dim=-1),
            "legacy_text_shift_norm": legacy_text_shift.float().norm(dim=-1),
            "audio_shift_norm": audio_shift.float().norm(dim=-1),
            "adaptation_shift_norm": adaptation_shift.float().norm(dim=-1),
            "fusion_scale": fusion_scale,
            "low_rank_interaction_mix": low_rank_interaction_mix,
            "low_rank_interaction_norm": low_rank_interaction_norm,
            "spectral_audio_gate": spectral_audio_gate,
            "spectral_vision_gate": spectral_vision_gate,
            "spectral_audio_velocity": spectral_audio_velocity,
            "spectral_vision_velocity": spectral_vision_velocity,
            "spectral_audio_acceleration": spectral_audio_acceleration,
            "spectral_vision_acceleration": spectral_vision_acceleration,
            "spectral_audio_low_energy": spectral_audio_low_energy,
            "spectral_vision_low_energy": spectral_vision_low_energy,
            "spectral_audio_high_energy": spectral_audio_high_energy,
            "spectral_vision_high_energy": spectral_vision_high_energy,
            "layer_mix_weights": layer_mix_weights,
            "layer_mix_scale": layer_mix_scale,
        }
