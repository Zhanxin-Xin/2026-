from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.train_subjectivity_neutral_expert import (
    SubjectivityResidualNeutralExpert,
    load_trainable_state,
    trainable_state_dict,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    device = torch.device("cuda")
    cfg = {
        "pretrained_model": "AIWizards/mdeberta-v3-base-subjectivity-english",
        "revision": "4f3923d5f7d7e5391b159cd5394fba5606bbc2de",
        "local_files_only": True,
        "gradient_checkpointing": True,
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "lora_target_modules": ["query_proj", "value_proj"],
        "residual_hidden": 16,
        "dropout": 0.1,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/mdeberta-v3-base",
        revision="a0484667b22365f84929a935b5e50a51f71f159d",
        local_files_only=True,
    )
    encoded = tokenizer(
        ["It was an ordinary day.", "This was absolutely wonderful!"],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    batch = {name: value.to(device) for name, value in encoded.items()}
    batch["class_label"] = torch.tensor([1, 2], device=device)
    model = SubjectivityResidualNeutralExpert(cfg).to(device)
    outputs = model(batch)
    assert outputs["neutral_logit"].shape == (2,)
    assert outputs["sentiment_logits"].shape == (2, 3)
    loss = F.binary_cross_entropy_with_logits(
        outputs["neutral_logit"], torch.tensor([1.0, 0.0], device=device)
    ) + 0.1 * F.cross_entropy(outputs["sentiment_logits"], batch["class_label"])
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    state = trainable_state_dict(model)
    load_trainable_state(model, state)
    print(
        {
            "device": str(device),
            "loss": float(loss.detach()),
            "trainable_state_tensors": len(state),
            "temperature": float(outputs["temperature"].detach()),
        }
    )
    print("Subjectivity Neutral expert smoke test passed.")


if __name__ == "__main__":
    main()

