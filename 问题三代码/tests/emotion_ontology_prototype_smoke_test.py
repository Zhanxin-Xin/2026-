from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.train_subjectivity_neutral_expert import SubjectivityResidualNeutralExpert


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test")
    device = torch.device("cuda")
    revision = "d75048347613a25d77de8cf6412eaae9fa7b26be"
    name = "SamLowe/roberta-base-go_emotions"
    cfg = {
        "pretrained_model": name,
        "revision": revision,
        "local_files_only": True,
        "prior_mode": "neutral_label",
        "gradient_checkpointing": True,
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "lora_target_modules": ["query", "value"],
        "residual_hidden": 16,
        "use_prototype_head": True,
        "prototype_dimension": 16,
        "prototype_temperature": 0.2,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, local_files_only=True
    )
    encoded = tokenizer(
        ["It was an ordinary day.", "This was absolutely wonderful!"],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in encoded.items()}
    model = SubjectivityResidualNeutralExpert(cfg).to(device)
    outputs = model(batch)
    assert outputs["neutral_logit"].shape == (2,)
    assert outputs["prototype_margin"].shape == (2,)
    assert float(outputs["prototype_scale"].detach()) > 0.0
    loss = F.binary_cross_entropy_with_logits(
        outputs["neutral_logit"], torch.tensor([1.0, 0.0], device=device)
    )
    loss.backward()
    assert model.prototypes.grad is not None
    print(
        {
            "device": str(device),
            "loss": float(loss.detach()),
            "prototype_scale": float(outputs["prototype_scale"].detach()),
        }
    )
    print("Emotion ontology prototype smoke test passed.")


if __name__ == "__main__":
    main()

