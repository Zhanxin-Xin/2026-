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
        raise RuntimeError("CUDA is required for this smoke test")
    device = torch.device("cuda")
    name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    encoded = tokenizer(
        ["This was excellent.", "It was an ordinary day."],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "legacy_text": torch.randn(2, 10, 768, device=device),
        "legacy_text_mask": torch.ones(2, 10, dtype=torch.bool, device=device),
        "audio": torch.randn(2, 10, 74, device=device),
        "audio_mask": torch.ones(2, 10, dtype=torch.bool, device=device),
        "vision": torch.randn(2, 10, 35, device=device),
        "vision_mask": torch.ones(2, 10, dtype=torch.bool, device=device),
    }
    model = PretrainedTextFusionNet(
        {
            "pretrained_model": name,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "use_pretrained_classifier": True,
            "fusion_dimension": 128,
            "fusion_heads": 4,
            "dropout": 0.1,
            "use_legacy_text": True,
            "use_audio": True,
            "use_vision": True,
            "use_prototype_router": True,
        }
    ).to(device)
    output = model(batch)
    assert output["class_logits"].shape == (2, 3)
    assert output["external_logits"].shape == (2, 3)
    assert 0.0 < float(output["external_mix"]) < 1.0
    loss = F.cross_entropy(
        output["class_logits"], torch.tensor([2, 1], device=device)
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    print(
        {
            "device": str(device),
            "loss": float(loss.detach()),
            "external_mix": float(output["external_mix"].detach()),
        }
    )
    print("Pretrained classifier expert smoke test passed.")


if __name__ == "__main__":
    main()
