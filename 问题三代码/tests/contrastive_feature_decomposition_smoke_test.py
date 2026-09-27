from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import (
    ContrastiveFeatureDecompositionHead,
    PretrainedTextFusionNet,
)
from src.train_pretrained_fusion import compute_loss, supervised_contrastive_loss


def _has_nonzero_gradient(parameter: torch.Tensor) -> bool:
    return parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this architecture smoke test")
    device = torch.device("cuda")
    torch.manual_seed(20260924)
    batch, dimension, hidden = 9, 32, 16
    head = ContrastiveFeatureDecompositionHead(
        dimension=dimension,
        hidden_dimension=hidden,
        dropout=0.0,
        maximum_logit_shift=0.50,
    ).to(device)
    parent = torch.softmax(torch.randn(batch, 3, device=device), dim=-1)
    states = [torch.randn(batch, dimension, device=device) for _ in range(3)]
    audio_reliability = torch.rand(batch, device=device)
    visual_reliability = torch.rand(batch, device=device)
    output = head(
        parent,
        *states,
        audio_reliability,
        visual_reliability,
    )

    if not torch.allclose(output["probabilities"], parent, atol=2e-6, rtol=2e-6):
        raise AssertionError("zero initialization must preserve parent probabilities")
    if not torch.allclose(
        output["probabilities"].sum(dim=-1),
        torch.ones(batch, device=device),
        atol=1e-6,
    ):
        raise AssertionError("probabilities must be normalized")
    if not torch.isfinite(output["shared_embeddings"]).all():
        raise AssertionError("shared contrastive embeddings must remain finite")
    if float(output["orthogonality"].max()) > 2e-5:
        raise AssertionError("shared/private decomposition is not orthogonal")

    target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device)
    main_loss = F.nll_loss(output["probabilities"].clamp_min(1e-8).log(), target)
    consensus_loss = F.cross_entropy(output["consensus_logits"], target)
    modality_target = target.unsqueeze(1).expand(-1, 3).reshape(-1)
    modality_loss = F.cross_entropy(
        output["modality_logits"].reshape(-1, 3), modality_target
    )
    parent_target = output["parent_probability"].detach().unsqueeze(1).expand(
        -1, 3, -1
    )
    distillation = F.kl_div(
        F.log_softmax(output["modality_logits"], dim=-1),
        parent_target,
        reduction="none",
    ).sum(dim=-1)
    distillation = (
        distillation * output["modality_reliability"]
    ).sum(dim=-1).mean()
    contrastive_target = target.unsqueeze(1).expand(-1, 3).reshape(-1)
    contrastive = supervised_contrastive_loss(
        output["shared_embeddings"].reshape(-1, hidden),
        contrastive_target,
    )
    loss = (
        main_loss
        + 0.20 * consensus_loss
        + 0.10 * modality_loss
        + 0.05 * distillation
        + 0.05 * contrastive
    )
    loss.backward()

    if not _has_nonzero_gradient(head.residual[-1].weight):
        raise AssertionError("bounded residual must receive task gradients")
    if not _has_nonzero_gradient(head.shared[1].weight):
        raise AssertionError("shared projector must receive auxiliary gradients")
    if not _has_nonzero_gradient(head.private[0][1].weight):
        raise AssertionError("private projector must receive auxiliary gradients")
    if not _has_nonzero_gradient(head.modality_classifiers[0][-1].weight):
        raise AssertionError("modality classifiers must receive auxiliary gradients")
    if not _has_nonzero_gradient(head.consensus[-1].weight):
        raise AssertionError("consensus classifier must receive auxiliary gradients")

    # Exercise the actual end-to-end model and production loss wiring with a
    # locally cached lightweight backbone.  The zero-initialized decomposition
    # residual must still expose the unmodified parent posterior.
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
            "use_contrastive_feature_decomposition": True,
            "decomposition_dimension": 16,
            "decomposition_maximum_logit_shift": 0.50,
            "audio_dimension": 74,
            "vision_dimension": 35,
        }
    ).to(device)
    model.eval()
    full_output = model(full_batch)
    if not torch.allclose(
        full_output["class_probabilities"],
        full_output["decomposition_parent_probability"],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError("full model decomposition must initialize as identity")
    loss_cfg = {
        "classification": 1.0,
        "neutral_auxiliary": 0.05,
        "polarity_auxiliary": 0.05,
        "decomposition_classification": 0.20,
        "unimodal_classification": 0.10,
        "unimodal_distillation": 0.05,
        "crossmodal_contrastive": 0.05,
        "label_smoothing": 0.03,
    }
    full_loss, components = compute_loss(
        full_output,
        full_batch,
        torch.ones(3, device=device),
        loss_cfg,
    )
    if not torch.isfinite(full_loss):
        raise AssertionError("full decomposition loss must remain finite")
    if not all(torch.isfinite(value) for value in components.values()):
        raise AssertionError("all full-model loss components must remain finite")
    print("contrastive feature decomposition CUDA smoke test passed")


if __name__ == "__main__":
    main()
