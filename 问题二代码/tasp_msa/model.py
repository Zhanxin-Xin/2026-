from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

MODALITIES = ("text", "audio", "vision")
INPUT_DIMS = {"text": 768, "audio": 74, "vision": 35}


def make_encoder(hidden: int, heads: int, layers: int, ffn: int, dropout: float) -> nn.Module:
    layer = nn.TransformerEncoderLayer(
        hidden, heads, ffn, dropout, activation="gelu", batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(hidden))


class SafeAttentionPool(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, 1))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.bool()
        safe = mask.clone()
        empty = ~safe.any(1)
        if empty.any():
            safe[empty, 0] = True
        logits = self.score(value).squeeze(-1).masked_fill(~safe, -1e4)
        weights = torch.softmax(logits, -1) * safe.to(value.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = (value * weights.unsqueeze(-1)).sum(1)
        if empty.any():
            pooled = pooled.masked_fill(empty[:, None], 0)
            weights = weights.masked_fill(empty[:, None], 0)
        return pooled, weights


class TASPMsa(nn.Module):
    """Compact model that completes only task-relevant shared semantics."""

    def __init__(
        self,
        hidden_dim: int = 96,
        shared_dim: int = 64,
        private_dim: int = 32,
        fusion_dim: int = 96,
        max_length: int = 50,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
        fusion_dropout: float = 0.2,
        use_proxy: bool = True,
        use_shared_specific: bool = True,
        use_reliability: bool = True,
        use_hierarchical: bool = True,
        anchor_mode: str | None = None,
    ) -> None:
        super().__init__()
        if anchor_mode not in (None, "text", "audio", "vision", "symmetric"):
            raise ValueError(f"unsupported anchor_mode={anchor_mode!r}")
        self.hidden_dim = hidden_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.max_length = max_length
        self.use_proxy = use_proxy
        self.use_shared_specific = use_shared_specific
        self.use_reliability = use_reliability
        self.use_hierarchical = use_hierarchical
        self.anchor_mode = anchor_mode
        self.config = dict(
            hidden_dim=hidden_dim, shared_dim=shared_dim, private_dim=private_dim,
            fusion_dim=fusion_dim, max_length=max_length, nhead=nhead,
            num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, fusion_dropout=fusion_dropout, use_proxy=use_proxy,
            use_shared_specific=use_shared_specific, use_reliability=use_reliability,
            use_hierarchical=use_hierarchical, anchor_mode=anchor_mode,
        )
        self.projection = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(INPUT_DIMS[modality], hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(dropout),
            ) for modality in MODALITIES
        })
        self.position = nn.Embedding(max_length, hidden_dim)
        self.modality = nn.Embedding(3, hidden_dim)
        self.missing_state = nn.Embedding(2, hidden_dim)
        self.mask_token = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            for modality in MODALITIES
        })
        self.encoder = nn.ModuleDict({
            modality: make_encoder(hidden_dim, nhead, num_layers, dim_feedforward, dropout)
            for modality in MODALITIES
        })
        self.shared_head = nn.ModuleDict({
            modality: nn.Sequential(nn.Linear(hidden_dim, shared_dim), nn.LayerNorm(shared_dim), nn.GELU())
            for modality in MODALITIES
        })
        self.private_head = nn.ModuleDict({
            modality: nn.Sequential(nn.Linear(hidden_dim, private_dim), nn.LayerNorm(private_dim), nn.GELU())
            for modality in MODALITIES
        })
        self.shared_pool = nn.ModuleDict({m: SafeAttentionPool(shared_dim) for m in MODALITIES})
        self.private_pool = nn.ModuleDict({m: SafeAttentionPool(private_dim) for m in MODALITIES})

        # Each proxy sees all available shared summaries and explicit missing ratios.
        proxy_input = shared_dim * 3 + 3
        self.proxy = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(proxy_input, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(128, shared_dim * 2),
            ) for modality in MODALITIES
        })
        self.private_to_shared = nn.ModuleDict({
            modality: nn.Linear(private_dim, shared_dim, bias=False) for modality in MODALITIES
        })
        def residual_network() -> nn.Module:
            return nn.Sequential(
                nn.Linear(shared_dim * 2, shared_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(shared_dim, shared_dim),
            )

        def reliability_network() -> nn.Module:
            return nn.Sequential(
                nn.Linear(shared_dim + 3, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, 1)
            )

        if anchor_mode is None:
            # Preserve the legacy parameter layout for checkpoint compatibility.
            self.audio_residual = residual_network()
            self.vision_residual = residual_network()
            self.reliability = nn.ModuleDict({
                modality: reliability_network() for modality in ("audio", "vision")
            })
        else:
            # Anchor-selection ablation: every candidate owns the same modules so all
            # variants have exactly the same stored parameter count.
            self.anchor_residual = nn.ModuleDict({m: residual_network() for m in MODALITIES})
            self.reliability = nn.ModuleDict({m: reliability_network() for m in MODALITIES})
        self.fusion = nn.Sequential(
            nn.Linear(shared_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(),
            nn.Dropout(fusion_dropout), nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim), nn.GELU(),
        )
        self.neutral_head = nn.Sequential(
            nn.Dropout(fusion_dropout), nn.Linear(fusion_dim, 48), nn.GELU(), nn.Linear(48, 1)
        )
        self.sign_head = nn.Sequential(
            nn.Dropout(fusion_dropout), nn.Linear(fusion_dim, 48), nn.GELU(), nn.Linear(48, 1)
        )
        self.regression_head = nn.Sequential(
            nn.Dropout(fusion_dropout), nn.Linear(fusion_dim, 48), nn.GELU(), nn.Linear(48, 1), nn.Tanh()
        )
        self.private_modality_classifier = nn.Linear(private_dim, 3)
        self.private_orthogonal_projection = nn.Linear(private_dim, shared_dim, bias=False)
        self.flat_classification_head = nn.Sequential(
            nn.Dropout(fusion_dropout), nn.Linear(fusion_dim, 48), nn.GELU(), nn.Linear(48, 3)
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        valid_mask_text: torch.Tensor,
        valid_mask_audio: torch.Tensor,
        valid_mask_vision: torch.Tensor,
        missing_mask_text: torch.Tensor | None = None,
        missing_mask_audio: torch.Tensor | None = None,
        missing_mask_vision: torch.Tensor | None = None,
        sample_proxy: bool = False,
    ) -> dict[str, Any]:
        values = {"text": text, "audio": audio, "vision": vision}
        valid = {
            "text": valid_mask_text.bool(), "audio": valid_mask_audio.bool(),
            "vision": valid_mask_vision.bool(),
        }
        supplied = {
            "text": missing_mask_text, "audio": missing_mask_audio,
            "vision": missing_mask_vision,
        }
        observed = {
            m: valid[m] & (supplied[m].bool() if supplied[m] is not None else torch.ones_like(valid[m]))
            for m in MODALITIES
        }
        length = text.shape[1]
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds max_length={self.max_length}")
        position = self.position(torch.arange(length, device=text.device))[None]
        encoded: dict[str, torch.Tensor] = {}
        shared_sequence: dict[str, torch.Tensor] = {}
        private_sequence: dict[str, torch.Tensor] = {}
        shared_observed: dict[str, torch.Tensor] = {}
        private_observed: dict[str, torch.Tensor] = {}
        attention: dict[str, torch.Tensor] = {}
        ratios: dict[str, torch.Tensor] = {}
        for index, modality in enumerate(MODALITIES):
            base = self.projection[modality](values[modality]) + position + self.modality.weight[index][None, None]
            observed_token = base + self.missing_state.weight[1][None, None]
            missing_token = (
                self.mask_token[modality].expand_as(base) + position
                + self.modality.weight[index][None, None] + self.missing_state.weight[0][None, None]
            )
            token = torch.where(observed[modality].unsqueeze(-1), observed_token, missing_token)
            encoded[modality] = self.encoder[modality](token, src_key_padding_mask=~valid[modality])
            shared_sequence[modality] = self.shared_head[modality](encoded[modality])
            private_sequence[modality] = self.private_head[modality](encoded[modality])
            shared_observed[modality], attention[modality] = self.shared_pool[modality](
                shared_sequence[modality], observed[modality]
            )
            private_observed[modality], _ = self.private_pool[modality](
                private_sequence[modality], observed[modality]
            )
            if not self.use_shared_specific:
                private_observed[modality] = torch.zeros_like(private_observed[modality])
            ratios[modality] = (
                (valid[modality] & ~observed[modality]).sum(1).float()
                / valid[modality].sum(1).clamp_min(1).float()
            )

        ratio_tensor = torch.stack([ratios[m] for m in MODALITIES], -1)
        proxy_input = torch.cat([shared_observed[m] for m in MODALITIES] + [ratio_tensor], -1)
        proxy_mean: dict[str, torch.Tensor] = {}
        proxy_logvar: dict[str, torch.Tensor] = {}
        completed_shared: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            proxy_mean[modality], proxy_logvar[modality] = self.proxy[modality](proxy_input).chunk(2, -1)
            proxy_logvar[modality] = proxy_logvar[modality].clamp(-6, 3)
            proxy_value = proxy_mean[modality]
            if sample_proxy and self.training:
                proxy_value = proxy_value + torch.randn_like(proxy_value) * torch.exp(
                    0.5 * proxy_logvar[modality]
                )
            ratio = ratios[modality][:, None]
            completed_shared[modality] = (
                (1 - ratio) * shared_observed[modality] + ratio * proxy_value
                if self.use_proxy else shared_observed[modality]
            )

        auxiliary_residuals: dict[str, torch.Tensor] = {}
        reliability: dict[str, torch.Tensor] = {}
        fusion_weights: torch.Tensor | None = None
        if self.anchor_mode is None:
            anchor = completed_shared["text"]
            candidates = (("audio", self.audio_residual), ("vision", self.vision_residual))
            for modality, network in candidates:
                availability = 1 - ratios[modality]
                private = self.private_to_shared[modality](private_observed[modality]) * availability[:, None]
                auxiliary_residuals[modality] = network(
                    torch.cat([completed_shared[modality], private], -1)
                )
                uncertainty = torch.exp(proxy_logvar[modality]).mean(-1)
                if not self.use_proxy:
                    uncertainty = torch.zeros_like(uncertainty)
                agreement = F.cosine_similarity(completed_shared[modality], anchor, dim=-1)
                gate_input = torch.cat([
                    completed_shared[modality], ratios[modality][:, None],
                    uncertainty[:, None], agreement[:, None],
                ], -1)
                reliability[modality] = (
                    torch.sigmoid(self.reliability[modality](gate_input))
                    if self.use_reliability else torch.full_like(uncertainty[:, None], 0.5)
                )
            sentiment = (
                anchor + reliability["audio"] * auxiliary_residuals["audio"]
                + reliability["vision"] * auxiliary_residuals["vision"]
            )
        else:
            anchor = completed_shared.get(self.anchor_mode)
            raw_gate_logits: list[torch.Tensor] = []
            for modality in MODALITIES:
                availability = 1 - ratios[modality]
                private = self.private_to_shared[modality](private_observed[modality]) * availability[:, None]
                auxiliary_residuals[modality] = self.anchor_residual[modality](
                    torch.cat([completed_shared[modality], private], -1)
                )
                uncertainty = torch.exp(proxy_logvar[modality]).mean(-1)
                if not self.use_proxy:
                    uncertainty = torch.zeros_like(uncertainty)
                if self.anchor_mode == "symmetric":
                    peers = [completed_shared[m] for m in MODALITIES if m != modality]
                    reference = torch.stack(peers, 0).mean(0)
                else:
                    reference = anchor
                agreement = F.cosine_similarity(completed_shared[modality], reference, dim=-1)
                gate_input = torch.cat([
                    completed_shared[modality], ratios[modality][:, None],
                    uncertainty[:, None], agreement[:, None],
                ], -1)
                raw_gate_logits.append(self.reliability[modality](gate_input))
            gate_logits = torch.cat(raw_gate_logits, -1)
            if self.anchor_mode == "symmetric":
                fusion_weights = (
                    torch.softmax(gate_logits, -1) if self.use_reliability
                    else torch.full_like(gate_logits, 1 / 3)
                )
                reliability = {
                    modality: fusion_weights[:, index:index + 1]
                    for index, modality in enumerate(MODALITIES)
                }
                sentiment = sum(
                    reliability[m] * auxiliary_residuals[m] for m in MODALITIES
                )
            else:
                reliability = {
                    modality: (
                        torch.sigmoid(gate_logits[:, index:index + 1])
                        if self.use_reliability else torch.full_like(gate_logits[:, index:index + 1], 0.5)
                    ) for index, modality in enumerate(MODALITIES)
                }
                sentiment = anchor + sum(
                    reliability[m] * auxiliary_residuals[m]
                    for m in MODALITIES if m != self.anchor_mode
                )
        fused = self.fusion(sentiment)

        neutral_logit = self.neutral_head(fused).squeeze(-1)
        sign_logit = self.sign_head(fused).squeeze(-1)
        p_neutral = torch.sigmoid(neutral_logit)
        p_positive_given_non_neutral = torch.sigmoid(sign_logit)
        probabilities = torch.stack([
            (1 - p_neutral) * (1 - p_positive_given_non_neutral),
            p_neutral,
            (1 - p_neutral) * p_positive_given_non_neutral,
        ], -1).clamp_min(1e-7)
        classification_logits = (
            torch.log(probabilities) if self.use_hierarchical
            else self.flat_classification_head(fused)
        )
        if not self.use_hierarchical:
            probabilities = torch.softmax(classification_logits, -1)
        private_logits = torch.cat([
            self.private_modality_classifier(private_observed[m]) for m in MODALITIES
        ], 0)
        return {
            "classification_logits": classification_logits,
            "class_probabilities": probabilities,
            "neutral_logit": neutral_logit,
            "sign_logit": sign_logit,
            "regression": self.regression_head(fused) * 3,
            "fused_features": fused,
            "shared_observed": shared_observed,
            "private_observed": private_observed,
            "completed_shared": completed_shared,
            "proxy_mean": proxy_mean,
            "proxy_logvar": proxy_logvar,
            "missing_ratios": ratios,
            "reliability": reliability,
            "fusion_weights": fusion_weights,
            "anchor_mode": self.anchor_mode or "text",
            "auxiliary_residuals": auxiliary_residuals,
            "temporal_attention": attention,
            "private_modality_logits": private_logits,
            "encoded_features": encoded,
        }
