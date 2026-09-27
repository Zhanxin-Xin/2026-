from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def pearson_loss(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    # Correlation and logarithmic regularizers are deliberately evaluated in
    # FP32.  In FP16, 1e-8 underflows to zero; a constant-label mini-batch then
    # evaluates 0/0 and poisons the complete multi-task loss.
    prediction = prediction.float()
    target = target.float()
    x = prediction - prediction.mean()
    y = target - target.mean()
    numerator = torch.sum(x * y)
    x_energy = torch.sum(x.square())
    y_energy = torch.sum(y.square())
    denominator = torch.sqrt((x_energy * y_energy).clamp_min(eps))
    correlation = (numerator / denominator).clamp(-1.0, 1.0)
    informative = (x_energy > eps) & (y_energy > eps)
    correlation = torch.where(informative, correlation, torch.zeros_like(correlation))
    return 1.0 - correlation


def predicted_modality_importance(outputs: Mapping[str, Tensor], eps: float = 1e-8) -> Tensor:
    reg = outputs["regression_contributions"].float().abs()
    reg_sum = reg.sum(dim=1, keepdim=True)
    fallback = outputs["modality_gates"].float()
    reg = torch.where(reg_sum > eps, reg / reg_sum.clamp_min(eps), fallback)
    probs = outputs["class_probabilities"]
    top2 = torch.topk(probs, k=2, dim=1).indices
    contributions = outputs["classification_contributions"].float()
    pred_idx = top2[:, 0, None, None].expand(-1, contributions.size(1), 1)
    runner_idx = top2[:, 1, None, None].expand(-1, contributions.size(1), 1)
    pred_contrib = contributions.gather(2, pred_idx).squeeze(-1)
    runner_contrib = contributions.gather(2, runner_idx).squeeze(-1)
    cls = (pred_contrib - runner_contrib).abs()
    cls_sum = cls.sum(dim=1, keepdim=True)
    cls = torch.where(cls_sum > eps, cls / cls_sum.clamp_min(eps), fallback)
    importance = 0.5 * (reg + cls)
    importance_sum = importance.sum(dim=1, keepdim=True)
    return torch.where(
        importance_sum > eps,
        importance / importance_sum.clamp_min(eps),
        fallback,
    )


def ablation_modality_importance(
    full_outputs: Mapping[str, Tensor],
    ablated_outputs: Sequence[Mapping[str, Tensor]],
    eps: float = 1e-8,
) -> Tensor:
    predicted_class = full_outputs["class_probabilities"].argmax(dim=1)
    batch_idx = torch.arange(predicted_class.size(0), device=predicted_class.device)
    full_confidence = full_outputs["class_probabilities"].float()[batch_idx, predicted_class]
    effects = []
    for ablated in ablated_outputs:
        cls_delta = (
            full_confidence
            - ablated["class_probabilities"].float()[batch_idx, predicted_class]
        ).abs()
        reg_delta = (
            full_outputs["regression"].float() - ablated["regression"].float()
        ).abs() / 3.0
        effects.append(0.5 * (cls_delta + reg_delta))
    effect = torch.stack(effects, dim=1)
    effect_sum = effect.sum(dim=1, keepdim=True)
    fallback = full_outputs["modality_gates"].float()
    return torch.where(effect_sum > eps, effect / effect_sum.clamp_min(eps), fallback)


class MultitaskEvidenceLoss(nn.Module):
    def __init__(
        self,
        cfg: Mapping[str, float],
        class_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
        distillation_temperature: float = 2.0,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.weights = dict(cfg)
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = label_smoothing
        self.distillation_temperature = distillation_temperature
        self.focal_gamma = float(focal_gamma)
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")

    def forward(
        self,
        outputs: Mapping[str, Tensor],
        class_target: Tensor,
        regression_target: Tensor,
        ablated_outputs: Optional[Sequence[Mapping[str, Tensor]]] = None,
        faithfulness_target: Optional[Tensor] = None,
        teacher_outputs: Optional[Mapping[str, Tensor]] = None,
        regularizer_scale: float = 1.0,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        logits = outputs["class_logits"].float()
        regression = outputs["regression"].float()
        regression_target = regression_target.float()
        class_weights = (
            None if self.class_weights is None else self.class_weights.float()
        )
        if self.focal_gamma > 0.0:
            per_sample_ce = F.cross_entropy(
                logits,
                class_target,
                weight=class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            true_probability = torch.softmax(logits, dim=-1).gather(
                1, class_target.unsqueeze(1)
            ).squeeze(1)
            cls = (
                (1.0 - true_probability).clamp_min(0.0).pow(self.focal_gamma)
                * per_sample_ce
            ).mean()
        else:
            # Preserve the original weighted-mean reduction exactly when focal
            # loss is disabled so legacy checkpoints/runs remain reproducible.
            cls = F.cross_entropy(
                logits,
                class_target,
                weight=class_weights,
                label_smoothing=self.label_smoothing,
            )
        reg = F.smooth_l1_loss(regression, regression_target, beta=0.5)
        corr = pearson_loss(regression, regression_target)
        expected_direction = (
            outputs["class_probabilities"].float()[:, 2]
            - outputs["class_probabilities"].float()[:, 0]
        )
        consistency = F.smooth_l1_loss(expected_direction, regression / 3.0)

        attention_parts = [outputs["temporal_weights"]]
        if "pair_temporal_weights" in outputs:
            attention_parts.append(outputs["pair_temporal_weights"])
        attention = torch.cat(attention_parts, dim=1).float().clamp_min(0.0)
        # Keep the probability itself outside the logarithm clamp: an exact
        # sparse zero contributes exactly zero, while log() and its gradient
        # always receive a representable FP32 value.
        safe_attention = attention.clamp_min(1e-8)
        entropy = -(attention * safe_attention.log()).sum(dim=-1).mean()
        total_variation = (attention[:, :, 1:] - attention[:, :, :-1]).abs().mean()
        mean_gate = outputs["modality_gates"].float().mean(dim=0).clamp_min(1e-8)
        gate_balance = torch.sum(mean_gate * torch.log(mean_gate * 3.0))

        faithfulness = regression.new_zeros(())
        if faithfulness_target is not None or ablated_outputs is not None:
            target_importance = (
                faithfulness_target
                if faithfulness_target is not None
                else ablation_modality_importance(outputs, ablated_outputs).detach()
            )
            predicted_importance = predicted_modality_importance(outputs)
            faithfulness = F.mse_loss(predicted_importance, target_importance.float())

        distillation = regression.new_zeros(())
        if teacher_outputs is not None:
            temperature = self.distillation_temperature
            teacher_probability = torch.softmax(
                teacher_outputs["class_logits"].detach().float() / temperature, dim=-1
            )
            classification_distillation = F.kl_div(
                torch.log_softmax(logits / temperature, dim=-1),
                teacher_probability,
                reduction="batchmean",
            ) * (temperature**2)
            regression_distillation = F.smooth_l1_loss(
                regression, teacher_outputs["regression"].detach().float(), beta=0.25
            )
            distillation = 0.5 * (
                classification_distillation + regression_distillation
            )

        interaction_l1 = regression.new_zeros(())
        if "pair_regression_contributions" in outputs:
            interaction_l1 = outputs["pair_regression_contributions"].float().abs().mean()

        unimodal_auxiliary = regression.new_zeros(())
        if (
            "modality_classification_evidence" in outputs
            and "modality_regression_evidence" in outputs
        ):
            modality_logits = outputs["modality_classification_evidence"].float()
            modality_regression = outputs["modality_regression_evidence"].float()
            available = outputs.get("modality_available")
            if available is None:
                available = torch.ones(
                    modality_logits.shape[:2],
                    dtype=torch.bool,
                    device=modality_logits.device,
                )
            else:
                available = available.bool()
            repeated_targets = class_target[:, None].expand(-1, modality_logits.size(1))
            auxiliary_cls = F.cross_entropy(
                modality_logits[available],
                repeated_targets[available],
                weight=None if self.class_weights is None else self.class_weights.float(),
                label_smoothing=self.label_smoothing,
            )
            auxiliary_reg = F.smooth_l1_loss(
                modality_regression[available],
                regression_target[:, None].expand_as(modality_regression)[available],
                beta=0.5,
            )
            # Deep supervision prevents an early gate collapse from starving a
            # complete modality stream of predictive gradients.
            unimodal_auxiliary = 0.5 * (auxiliary_cls + auxiliary_reg)

        neutral_auxiliary = regression.new_zeros(())
        if "neutral_logit" in outputs:
            neutral_target = (class_target == 1).to(logits.dtype)
            neutral_auxiliary = F.binary_cross_entropy_with_logits(
                outputs["neutral_logit"].float(), neutral_target
            )

        components = {
            "classification": cls,
            "regression": reg,
            "pearson": corr,
            "consistency": consistency,
            "attention_entropy": entropy,
            "attention_total_variation": total_variation,
            "gate_balance": gate_balance,
            "faithfulness": faithfulness,
            "distillation": distillation,
            "interaction_l1": interaction_l1,
            "unimodal_auxiliary": unimodal_auxiliary,
            "neutral_auxiliary": neutral_auxiliary,
        }
        regularized = {
            "attention_entropy",
            "attention_total_variation",
            "gate_balance",
            "faithfulness",
            "interaction_l1",
        }
        total = sum(
            self.weights.get(name, 0.0)
            * value
            * (regularizer_scale if name in regularized else 1.0)
            for name, value in components.items()
        )
        return total, components
