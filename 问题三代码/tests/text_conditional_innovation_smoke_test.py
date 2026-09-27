from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PretrainedTextFusionNet, TextConditionalInnovationHead
from src.train_pretrained_fusion import compute_loss


def _has_gradient(parameter: torch.Tensor) -> bool:
    return parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this architecture smoke test")
    device = torch.device("cuda")
    torch.manual_seed(20260925)
    batch, dimension = 9, 32
    head = TextConditionalInnovationHead(
        dimension=dimension,
        hidden_dimension=16,
        dropout=0.0,
    ).to(device)
    parent = torch.softmax(torch.randn(batch, 3, device=device), dim=-1)
    text = torch.randn(batch, dimension, device=device)
    audio = torch.randn(batch, dimension, device=device)
    vision = torch.randn(batch, dimension, device=device)
    output = head(
        parent,
        text,
        audio,
        vision,
        torch.rand(batch, device=device),
        torch.rand(batch, device=device),
    )
    if not torch.allclose(output["probabilities"], parent, atol=2e-6, rtol=2e-6):
        raise AssertionError("zero initialization must preserve parent probabilities")
    if not torch.allclose(
        output["probabilities"].sum(dim=-1),
        torch.ones(batch, device=device),
        atol=1e-6,
    ):
        raise AssertionError("probabilities must be normalized")
    if not torch.isfinite(output["audio_error"]).all():
        raise AssertionError("audio innovation errors must remain finite")
    if not torch.isfinite(output["vision_error"]).all():
        raise AssertionError("vision innovation errors must remain finite")

    target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], device=device)
    neutral_target = target.eq(1).float()
    loss = F.nll_loss(output["probabilities"].clamp_min(1e-8).log(), target)
    loss = loss + 0.20 * F.cross_entropy(output["innovation_logits"], target)
    loss = loss + 0.15 * F.binary_cross_entropy_with_logits(
        output["innovation_neutral_logit"], neutral_target
    )
    loss = loss + 0.10 * F.smooth_l1_loss(
        output["predicted_audio"], output["audio_target"]
    )
    loss = loss + 0.10 * F.smooth_l1_loss(
        output["predicted_vision"], output["vision_target"]
    )
    loss.backward()
    if not _has_gradient(head.axes[-1].weight):
        raise AssertionError("bounded decision axes must receive task gradients")
    if not _has_gradient(head.audio_predictor[-1].weight):
        raise AssertionError("audio predictor must receive reconstruction gradients")
    if not _has_gradient(head.vision_predictor[-1].weight):
        raise AssertionError("vision predictor must receive reconstruction gradients")
    if not _has_gradient(head.audio_innovation_projector[1].weight):
        raise AssertionError("innovation projector must receive auxiliary gradients")
    if not _has_gradient(head.innovation_classifier.weight):
        raise AssertionError("innovation classifier must receive gradients")

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
            "use_text_conditional_innovation": True,
            "innovation_hidden_dimension": 16,
            "audio_dimension": 74,
            "vision_dimension": 35,
        }
    ).to(device)
    model.eval()
    full_output = model(full_batch)
    if not torch.allclose(
        full_output["class_probabilities"],
        full_output["innovation_parent_probability"],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError("full innovation model must initialize as identity")
    full_loss, components = compute_loss(
        full_output,
        full_batch,
        torch.ones(3, device=device),
        {
            "classification": 1.0,
            "neutral_auxiliary": 0.05,
            "polarity_auxiliary": 0.02,
            "innovation_classification": 0.20,
            "innovation_neutral": 0.15,
            "audio_reconstruction": 0.10,
            "vision_reconstruction": 0.10,
            "label_smoothing": 0.03,
        },
    )
    if not torch.isfinite(full_loss):
        raise AssertionError("full innovation loss must remain finite")
    if not all(torch.isfinite(value) for value in components.values()):
        raise AssertionError("all full-model loss components must remain finite")
    print("text-conditional innovation CUDA smoke test passed")


if __name__ == "__main__":
    main()
