from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PolarDistanceNeutralExpert, PretrainedTextFusionNet


def test_polar_odds_are_invariant_to_multimodal_boundary_evidence() -> None:
    torch.manual_seed(17)
    dimension = 16
    model = PolarDistanceNeutralExpert(
        dimension=dimension,
        dropout=0.0,
        maximum_boundary_mix=0.5,
        maximum_evidence_shift=1.5,
    )
    anchor = torch.randn(4, dimension)
    boundary = torch.randn(4, dimension)
    base = torch.softmax(torch.randn(4, 3), dim=-1)
    reliability = torch.rand(4)
    output_a = model(anchor, boundary, base, reliability, reliability, reliability)
    output_b = model(
        anchor,
        boundary + 2.0,
        base.roll(1, dims=0),
        1.0 - reliability,
        1.0 - reliability,
        1.0 - reliability,
    )
    odds_a = output_a["probabilities"][:, 2] / output_a["probabilities"][:, 0]
    odds_b = output_b["probabilities"][:, 2] / output_b["probabilities"][:, 0]
    assert torch.allclose(odds_a, odds_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        output_a["probabilities"].sum(dim=1), torch.ones(4), atol=1e-6
    )
    assert not torch.allclose(
        output_a["probabilities"][:, 1], output_b["probabilities"][:, 1]
    )
    loss = -output_a["probabilities"].clamp_min(1e-8).log().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_contextual_trimodal_integration() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    texts = ["The result is acceptable.", "I absolutely loved this."]
    contexts = ["The earlier remark was unclear.", "They sounded enthusiastic."]
    paired = tokenizer(
        texts,
        text_pair=contexts,
        padding="max_length",
        truncation="longest_first",
        max_length=24,
        return_tensors="pt",
    )
    current = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    batch = {
        "input_ids": paired["input_ids"].to(device),
        "attention_mask": paired["attention_mask"].to(device),
        "token_type_ids": paired["token_type_ids"].to(device),
        "context_available": torch.ones(2, dtype=torch.bool, device=device),
        "current_input_ids": current["input_ids"].to(device),
        "current_attention_mask": current["attention_mask"].to(device),
        "legacy_text": torch.randn(2, 8, 768, device=device),
        "legacy_text_mask": torch.ones(2, 8, dtype=torch.bool, device=device),
        "audio": torch.randn(2, 8, 74, device=device),
        "audio_mask": torch.ones(2, 8, dtype=torch.bool, device=device),
        "vision": torch.randn(2, 8, 35, device=device),
        "vision_mask": torch.ones(2, 8, dtype=torch.bool, device=device),
    }
    model = PretrainedTextFusionNet(
        {
            "pretrained_model": name,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "fusion_dimension": 64,
            "fusion_heads": 4,
            "dropout": 0.1,
            "use_context": True,
            "use_audio": True,
            "use_vision": True,
            "use_prototype_router": True,
            "use_polar_distance_neutral": True,
            "audio_dimension": 74,
            "vision_dimension": 35,
        }
    ).to(device)
    output = model(batch)
    assert output["class_probabilities"].shape == (2, 3)
    assert torch.allclose(
        output["class_probabilities"].sum(dim=1),
        torch.ones(2, device=device),
        atol=1e-5,
    )
    assert torch.isfinite(output["neutral_logit"]).all()
    assert torch.isfinite(output["polarity_logit"]).all()
    loss = torch.nn.functional.cross_entropy(
        output["class_logits"], torch.tensor([1, 2], device=device)
    )
    loss.backward()
    assert model.polar_distance_neutral.polarity[-1].weight.grad is not None


if __name__ == "__main__":
    test_polar_odds_are_invariant_to_multimodal_boundary_evidence()
    test_contextual_trimodal_integration()
    print("Polar-distance Neutral expert smoke test passed.")
