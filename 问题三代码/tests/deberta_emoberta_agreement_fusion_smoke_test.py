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
        raise RuntimeError("CUDA is required for the dual-evidence smoke test")
    device = torch.device("cuda")
    primary_name = "microsoft/deberta-v3-base"
    emotion_name = "tae898/emoberta-base"
    emotion_revision = "64377bdd2a1d7bc5ecdac9a4fbd219002663df1e"
    primary_tokenizer = AutoTokenizer.from_pretrained(
        primary_name, local_files_only=True
    )
    emotion_tokenizer = AutoTokenizer.from_pretrained(
        emotion_name, revision=emotion_revision, local_files_only=True
    )
    texts = ["The result was ordinary.", "I am delighted with it."]
    primary = primary_tokenizer(
        texts, padding="max_length", truncation=True, max_length=24,
        return_tensors="pt",
    )
    emotion = emotion_tokenizer(
        texts, padding="max_length", truncation=True, max_length=24,
        return_tensors="pt",
    )
    generator = torch.Generator().manual_seed(20260925)
    batch = {
        "input_ids": primary["input_ids"].to(device),
        "attention_mask": primary["attention_mask"].to(device),
        "token_type_ids": primary.get(
            "token_type_ids", torch.zeros_like(primary["input_ids"])
        ).to(device),
        "context_available": torch.zeros(2, dtype=torch.bool, device=device),
        "emotion_input_ids": emotion["input_ids"].to(device),
        "emotion_attention_mask": emotion["attention_mask"].to(device),
        "emotion_token_type_ids": emotion.get(
            "token_type_ids", torch.zeros_like(emotion["input_ids"])
        ).to(device),
        "audio": torch.randn(2, 3, 74, generator=generator).to(device),
        "audio_mask": torch.ones(2, 3, dtype=torch.bool, device=device),
        "vision": torch.randn(2, 3, 35, generator=generator).to(device),
        "vision_mask": torch.ones(2, 3, dtype=torch.bool, device=device),
    }
    model = PretrainedTextFusionNet(
        {
            "pretrained_model": primary_name,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "use_emotion_evidence": True,
            "emotion_pretrained_model": emotion_name,
            "emotion_revision": emotion_revision,
            "emotion_class_groups": {
                "negative": ["anger", "sadness", "disgust", "fear"],
                "neutral": ["neutral", "surprise"],
                "positive": ["joy"],
            },
            "fusion_dimension": 64,
            "fusion_heads": 4,
            "dropout": 0.1,
            "use_dynamic_fusion_router": True,
            "use_audio": True,
            "use_vision": True,
            "use_prototype_router": True,
            "audio_dimension": 74,
            "vision_dimension": 35,
        }
    ).to(device)
    output = model(batch)
    probability = output["class_probabilities"]
    assert probability.shape == (2, 3)
    assert torch.isfinite(probability).all()
    assert torch.allclose(
        probability.sum(dim=-1), torch.ones(2, device=device), atol=1e-6
    )
    assert output["emotion_probability"].shape == (2, 3)
    assert not any(parameter.requires_grad for parameter in model.emotion_encoder.parameters())
    loss = F.nll_loss(
        probability.clamp_min(1e-8).log(), torch.tensor([1, 2], device=device)
    )
    loss.backward()
    assert model.emotion_evidence.decision_delta[-1].weight.grad is not None
    assert model.emotion_evidence.representation_delta[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.emotion_encoder.parameters())
    print(
        {
            "loss": float(loss.detach()),
            "representation_reliability": output[
                "emotion_representation_reliability"
            ].detach().cpu().tolist(),
            "decision_reliability": output[
                "emotion_decision_reliability"
            ].detach().cpu().tolist(),
        }
    )
    print("DeBERTa + EmoBERTa agreement fusion smoke test passed.")


if __name__ == "__main__":
    main()
