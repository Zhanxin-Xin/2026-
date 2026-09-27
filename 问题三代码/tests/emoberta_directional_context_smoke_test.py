from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PretrainedTextFusionNet


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the EmoBERTa directional smoke test")

    device = torch.device("cuda")
    name = "tae898/emoberta-base"
    revision = "64377bdd2a1d7bc5ecdac9a4fbd219002663df1e"
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        local_files_only=True,
    )

    def encode(texts: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=24,
            return_tensors="pt",
        )

    current = encode(["I feel fine", "This is terrible"])
    previous = encode(["Earlier I was uncertain", ""])
    following = encode(["", "Later it became calmer"])
    generator = torch.Generator().manual_seed(20260924)
    batch = {
        "input_ids": current["input_ids"].to(device),
        "attention_mask": current["attention_mask"].to(device),
        "token_type_ids": torch.zeros_like(current["input_ids"]).to(device),
        "context_available": torch.tensor([True, True], device=device),
        "previous_input_ids": previous["input_ids"].to(device),
        "previous_attention_mask": previous["attention_mask"].to(device),
        "previous_context_available": torch.tensor([True, False], device=device),
        "following_input_ids": following["input_ids"].to(device),
        "following_attention_mask": following["attention_mask"].to(device),
        "following_context_available": torch.tensor([False, True], device=device),
        "audio": torch.randn(2, 3, 74, generator=generator).to(device),
        "audio_mask": torch.ones(2, 3, dtype=torch.bool, device=device),
        "vision": torch.randn(2, 3, 35, generator=generator).to(device),
        "vision_mask": torch.ones(2, 3, dtype=torch.bool, device=device),
    }
    model = PretrainedTextFusionNet(
        {
            "pretrained_model": name,
            "revision": revision,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "use_pretrained_classifier": True,
            "external_class_groups": {
                "negative": ["anger", "sadness", "disgust", "fear"],
                "neutral": ["neutral", "surprise"],
                "positive": ["joy"],
            },
            "fusion_dimension": 64,
            "fusion_heads": 4,
            "dropout": 0.1,
            "use_directional_context": True,
            "use_context_weighted_prototype": True,
            "use_audio": True,
            "use_vision": True,
            "use_prototype_router": True,
            "audio_dimension": 74,
            "vision_dimension": 35,
            "context_shift_scale": 0.35,
        }
    ).to(device)
    output = model(batch)
    route = torch.stack(
        [
            output["context_reject_weight"],
            output["previous_context_weight"],
            output["following_context_weight"],
        ],
        dim=-1,
    )
    assert output["class_probabilities"].shape == (2, 3)
    assert output["external_logits"].shape == (2, 3)
    assert torch.isfinite(output["class_probabilities"]).all()
    assert torch.allclose(route.sum(dim=-1), torch.ones(2, device=device), atol=1e-6)
    assert output["following_context_weight"][0].item() == 0.0
    assert output["previous_context_weight"][1].item() == 0.0
    loss = F.nll_loss(
        output["class_probabilities"].clamp_min(1e-8).log(),
        torch.tensor([1, 0], device=device),
    )
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.directional_context_router.parameters()
    )
    print(
        {
            "loss": float(loss.detach()),
            "route": route.detach().cpu().tolist(),
            "external_mix": float(output["external_mix"].detach()),
        }
    )
    print("EmoBERTa directional context smoke test passed.")


if __name__ == "__main__":
    main()
