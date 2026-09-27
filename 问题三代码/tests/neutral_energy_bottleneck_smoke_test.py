from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import (
    NeutralOrthogonalEnergyBottleneck,
    PretrainedTextFusionNet,
)
from src.train_pretrained_fusion import compute_loss, resolve_loss_config


def _head(device: torch.device) -> NeutralOrthogonalEnergyBottleneck:
    return NeutralOrthogonalEnergyBottleneck(
        dimension=32,
        hidden_dimension=12,
        dropout=0.0,
        maximum_neutral_logit_shift=0.75,
        disagreement_temperature=0.35,
    ).to(device)


def _inputs(device: torch.device, batch: int = 9) -> tuple[torch.Tensor, ...]:
    parent = torch.softmax(torch.randn(batch, 3, device=device), dim=-1)
    states = tuple(torch.randn(batch, 32, device=device) for _ in range(3))
    reliability = torch.ones(batch, 3, device=device)
    availability = torch.ones(batch, 3, device=device)
    return parent, *states, reliability, availability


def _has_gradient(parameter: torch.Tensor) -> bool:
    return parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)


def test_identity_normalization_and_polar_odds(device: torch.device) -> None:
    head = _head(device)
    parent, text, audio, vision, reliability, availability = _inputs(device)
    output = head(
        parent, text, audio, vision, reliability, availability
    )
    if not torch.allclose(output["probabilities"], parent, atol=2e-6, rtol=2e-6):
        raise AssertionError("zero initialization must preserve the parent posterior")
    if not torch.allclose(
        output["probabilities"].sum(dim=-1),
        torch.ones(parent.size(0), device=device),
        atol=1e-6,
    ):
        raise AssertionError("corrected probabilities must be normalized")
    parent_odds = parent[:, 2] / parent[:, 0]
    corrected_odds = output["probabilities"][:, 2] / output["probabilities"][:, 0]
    if not torch.allclose(parent_odds, corrected_odds, atol=2e-6, rtol=2e-6):
        raise AssertionError("Positive/Negative odds must remain unchanged")


def test_conflict_attenuates_and_deletion_is_explicit(device: torch.device) -> None:
    head = _head(device)
    parent, text, audio, vision, reliability, availability = _inputs(device)
    with torch.no_grad():
        for residual in head.neutral_residuals:
            residual[-1].bias.fill_(1.0)
    aligned = head(parent, text, audio, vision, reliability, availability)
    with torch.no_grad():
        head.neutral_residuals[2][-1].bias.fill_(-1.0)
    conflict = head(parent, text, audio, vision, reliability, availability)
    if not bool((conflict["neutral_shift"] >= -1e-7).all()):
        raise AssertionError("symmetric conflict must not invent negative evidence")
    if not bool(
        (conflict["neutral_shift"] <= aligned["neutral_shift"] + 1e-7).all()
    ):
        raise AssertionError("conflict must attenuate, not amplify, Neutral evidence")
    if not bool((conflict["disagreement"] > aligned["disagreement"]).all()):
        raise AssertionError("the conflict diagnostic must respond to disagreement")

    availability = availability.clone()
    availability[:, 2] = 0.0
    deleted = head(parent, text, audio, vision, reliability, availability)
    if not torch.equal(
        deleted["modality_weights"][:, 2],
        torch.zeros_like(deleted["modality_weights"][:, 2]),
    ):
        raise AssertionError("a deleted modality must receive zero PoE weight")
    if not torch.equal(
        deleted["modality_contributions"][:, 2],
        torch.zeros_like(deleted["modality_contributions"][:, 2]),
    ):
        raise AssertionError("a deleted modality must have zero decision contribution")


def test_all_modalities_receive_gradients(device: torch.device) -> None:
    head = _head(device)
    parent, text, audio, vision, reliability, availability = _inputs(device)
    output = head(parent, text, audio, vision, reliability, availability)
    target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device)
    neutral_target = target.eq(1).float().unsqueeze(-1).expand(-1, 3)
    evidence_target = 1.0 - torch.abs(
        output["modality_neutral_probabilities"].detach() - neutral_target
    )
    energy_margin = output["neutral_energy"] - output["polar_energy"]
    sign = target.eq(1).float().mul(2.0).sub(1.0).unsqueeze(-1)
    loss = F.nll_loss(output["probabilities"].clamp_min(1e-8).log(), target)
    loss = loss + 0.20 * F.binary_cross_entropy_with_logits(
        output["modality_neutral_logits"], neutral_target
    )
    loss = loss + 0.05 * F.binary_cross_entropy(
        output["modality_reliability"].clamp(1e-6, 1.0 - 1e-6),
        evidence_target,
    )
    loss = loss + 0.02 * output["orthogonality"].mean()
    loss = loss + 0.05 * F.softplus(-sign * energy_margin).mean()
    loss.backward()
    for index in range(3):
        if not _has_gradient(head.neutral_residuals[index][-1].weight):
            raise AssertionError(f"modality {index} residual received no gradient")
        if not _has_gradient(head.evidence_strengths[index][-1].weight):
            raise AssertionError(f"modality {index} reliability received no gradient")
        if not _has_gradient(head.projectors[index][1].weight):
            raise AssertionError(f"modality {index} energy projector received no gradient")


def test_signed_contrast_is_bias_free_and_conflict_safe(
    device: torch.device,
) -> None:
    head = NeutralOrthogonalEnergyBottleneck(
        dimension=4,
        hidden_dimension=2,
        dropout=0.0,
        maximum_neutral_logit_shift=0.75,
        disagreement_temperature=0.35,
        variant="signed_contrast",
    ).to(device)
    if any(axis.bias is not None for axis in (*head.neutral_axes, *head.polar_axes)):
        raise AssertionError("signed evidence axes must not expose a bias")
    parent = torch.tensor([[0.35, 0.30, 0.35]], device=device).repeat(4, 1)
    states = [torch.randn(4, 4, device=device) for _ in range(3)]
    initial = head(parent, *states)
    if not torch.allclose(initial["probabilities"], parent, atol=2e-6, rtol=2e-6):
        raise AssertionError("signed contrast must also initialize as identity")

    with torch.no_grad():
        for projector in head.projectors:
            projector[1].weight.copy_(torch.eye(4, device=device))
        for neutral_axis, polar_axis in zip(head.neutral_axes, head.polar_axes):
            neutral_axis.weight.copy_(torch.tensor([[1.0, 0.0]], device=device))
            polar_axis.weight.copy_(torch.tensor([[1.0, 0.0]], device=device))
    aligned_state = torch.tensor([[2.0, 0.0, -2.0, 0.0]], device=device).repeat(
        4, 1
    )
    opposite_state = torch.tensor(
        [[-2.0, 0.0, 2.0, 0.0]], device=device
    ).repeat(4, 1)
    aligned = head(parent, aligned_state, aligned_state, aligned_state)
    conflict = head(parent, aligned_state, aligned_state, opposite_state)
    if not bool((aligned["neutral_shift"] > 0.0).all()):
        raise AssertionError("aligned Neutral evidence must yield a positive shift")
    if not bool(
        (conflict["neutral_shift"] <= aligned["neutral_shift"] + 1e-7).all()
    ):
        raise AssertionError("signed conflict must attenuate the Neutral shift")
    if not bool((conflict["coherence"] < aligned["coherence"]).all()):
        raise AssertionError("the signed coherence gate must expose conflict")

    veto = NeutralOrthogonalEnergyBottleneck(
        dimension=4,
        hidden_dimension=2,
        dropout=0.0,
        maximum_neutral_logit_shift=0.75,
        disagreement_temperature=0.35,
        variant="signed_contrast",
        consensus_mode="unanimity_veto",
    ).to(device)
    veto.load_state_dict(head.state_dict())
    veto_aligned = veto(parent, aligned_state, aligned_state, aligned_state)
    veto_conflict = veto(parent, aligned_state, aligned_state, opposite_state)
    if not bool((veto_aligned["unanimous"] == 1.0).all()):
        raise AssertionError("aligned evidence must pass the unanimity gate")
    if not torch.equal(
        veto_conflict["neutral_shift"],
        torch.zeros_like(veto_conflict["neutral_shift"]),
    ):
        raise AssertionError("one dissenting modality must force exact abstention")
    if not bool((veto_conflict["unanimous"] == 0.0).all()):
        raise AssertionError("the unanimity diagnostic must expose the veto")

    train_head = NeutralOrthogonalEnergyBottleneck(
        dimension=32,
        hidden_dimension=12,
        dropout=0.0,
        variant="signed_contrast",
    ).to(device)
    parent, text, audio, vision, reliability, availability = _inputs(device)
    output = train_head(
        parent, text, audio, vision, reliability, availability
    )
    target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device)
    neutral_target = target.eq(1).float().unsqueeze(-1).expand(-1, 3)
    sign = target.eq(1).float().mul(2.0).sub(1.0).unsqueeze(-1)
    margin = output["neutral_energy"] - output["polar_energy"]
    loss = F.binary_cross_entropy_with_logits(
        output["modality_neutral_logits"], neutral_target
    )
    loss = loss + 0.10 * F.softplus(-sign * margin).mean()
    loss = loss + 0.02 * output["orthogonality"].mean()
    loss.backward()
    for index in range(3):
        if not _has_gradient(train_head.neutral_axes[index].weight):
            raise AssertionError(f"signed Neutral axis {index} received no gradient")
        if not _has_gradient(train_head.polar_axes[index].weight):
            raise AssertionError(f"signed Polar axis {index} received no gradient")
        if not _has_gradient(train_head.projectors[index][1].weight):
            raise AssertionError(f"signed projector {index} received no gradient")


def test_full_model_and_loss(device: torch.device) -> None:
    name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    encoded = tokenizer(
        ["This is clearly useful.", "The outcome is neutral.", "This is poor."],
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    full_batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "legacy_text": torch.randn(3, 6, 768, device=device),
        "legacy_text_mask": torch.ones(3, 6, dtype=torch.bool, device=device),
        "audio": torch.randn(3, 6, 74, device=device),
        "audio_mask": torch.ones(3, 6, dtype=torch.bool, device=device),
        "vision": torch.randn(3, 6, 35, device=device),
        "vision_mask": torch.ones(3, 6, dtype=torch.bool, device=device),
        "class_label": torch.tensor([2, 1, 0], device=device),
        "regression_label": torch.tensor([1.0, 0.0, -1.0], device=device),
    }
    model = PretrainedTextFusionNet(
        {
            "pretrained_model": name,
            "local_files_only": True,
            "gradient_checkpointing": False,
            "fusion_dimension": 64,
            "fusion_heads": 4,
            "dropout": 0.0,
            "use_audio": True,
            "use_vision": True,
            "use_dynamic_fusion_router": True,
            "use_prototype_router": True,
            "use_neutral_energy_bottleneck": True,
            "neutral_energy_hidden_dimension": 16,
            "audio_dimension": 74,
            "vision_dimension": 35,
        }
    ).to(device)
    model.eval()
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model(full_batch)
    if not torch.allclose(
        output["class_probabilities"],
        output["neutral_energy_parent_probability"],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError("the end-to-end bottleneck must initialize as identity")
    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        loss_cfg = resolve_loss_config(
            {
                "classification": 1.0,
                "neutral_auxiliary": 0.08,
                "polarity_auxiliary": 0.05,
                "neutral_energy_prior_balanced": True,
                "neutral_energy_classification": 0.20,
                "neutral_evidence_calibration": 0.05,
                "neutral_energy_consistency": 0.03,
                "neutral_energy_orthogonality": 0.02,
                "neutral_energy_sparsity": 0.01,
                "neutral_energy_separation": 0.05,
            },
            full_batch["class_label"].detach().cpu().numpy(),
        )
        if loss_cfg["_neutral_energy_positive_weight"] != 2.0:
            raise AssertionError("fold-local Neutral prior ratio was resolved incorrectly")
        loss, components = compute_loss(
            output,
            full_batch,
            torch.ones(3, device=device),
            loss_cfg,
        )
    if not torch.isfinite(loss):
        raise AssertionError("full bottleneck loss must remain finite")
    if not all(torch.isfinite(value) for value in components.values()):
        raise AssertionError("every loss component must remain finite")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260924)
    test_identity_normalization_and_polar_odds(device)
    test_conflict_attenuates_and_deletion_is_explicit(device)
    test_all_modalities_receive_gradients(device)
    test_signed_contrast_is_bias_free_and_conflict_safe(device)
    test_full_model_and_loss(device)
    print("Neutral orthogonal energy bottleneck smoke test passed")


if __name__ == "__main__":
    main()
