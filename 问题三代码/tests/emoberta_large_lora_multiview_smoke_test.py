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
        raise RuntimeError("CUDA is required for the EmoBERTa-large LoRA test")
    device = torch.device("cuda")
    name = "tae898/emoberta-large"
    revision = "8934b68e8b0d9fc3cd961cc7e7605533c7081e59"
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, local_files_only=True
    )
    encoded = tokenizer(
        ["The result was ordinary.", "I am delighted with it."],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "token_type_ids": torch.zeros_like(encoded["input_ids"]).to(device),
        "context_available": torch.zeros(2, dtype=torch.bool, device=device),
        "audio": torch.randn(2, 3, 74, device=device),
        "audio_mask": torch.ones(2, 3, dtype=torch.bool, device=device),
        "vision": torch.randn(2, 3, 35, device=device),
        "vision_mask": torch.ones(2, 3, dtype=torch.bool, device=device),
    }
    model = PretrainedTextFusionNet(
        {
            "pretrained_model": name,
            "revision": revision,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "use_pretrained_classifier": True,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "external_class_groups": {
                "negative": ["anger", "sadness", "disgust", "fear"],
                "neutral": ["neutral", "surprise"],
                "positive": ["joy"],
            },
            "use_layerwise_pooling": True,
            "layer_mix_count": 4,
            "layerwise_residual": True,
            "fusion_dimension": 64,
            "fusion_heads": 4,
            "dropout": 0.1,
            "use_audio": True,
            "use_vision": True,
            "use_prototype_router": True,
            "use_low_rank_interaction": True,
            "low_rank_interaction_rank": 4,
            "audio_dimension": 74,
            "vision_dimension": 35,
        }
    ).to(device)
    output = model(batch)
    assert output["class_probabilities"].shape == (2, 3)
    assert torch.isfinite(output["class_probabilities"]).all()
    assert output["layer_mix_weights"].shape == (4,)
    assert torch.allclose(
        output["layer_mix_weights"].sum(), torch.ones((), device=device), atol=1e-6
    )
    loss = F.nll_loss(
        output["class_probabilities"].log(), torch.tensor([1, 2], device=device)
    )
    loss.backward()
    lora_gradients = [
        parameter.grad
        for name, parameter in model.text_encoder.named_parameters()
        if "lora_" in name and parameter.grad is not None
    ]
    assert lora_gradients
    assert all(torch.isfinite(gradient).all() for gradient in lora_gradients)
    assert model.layer_mix_logits.grad is not None
    assert torch.isfinite(model.layer_mix_logits.grad).all()
    print(
        {
            "loss": float(loss.detach()),
            "layer_mix": output["layer_mix_weights"].detach().cpu().tolist(),
            "lora_gradient_tensors": len(lora_gradients),
        }
    )
    print("EmoBERTa-large LoRA multiview smoke test passed.")


if __name__ == "__main__":
    main()
