from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PretrainedTextFusionNet
from src.train_pretrained_fusion import build_directional_contexts


def main() -> None:
    arrays = SimpleNamespace(
        ids=np.asarray(["videoA$_$0", "videoA$_$1", "videoA$_$2", "videoB$_$0"]),
        raw_text=np.asarray(["first", "middle", "last", "isolated"]),
        size=4,
    )
    previous, following = build_directional_contexts(arrays, window=1)
    assert previous == ["", "first", "middle", ""]
    assert following == ["middle", "last", "", ""]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the directional context smoke test")
    device = torch.device("cuda")
    name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    current = tokenizer(
        ["current zero", "current one", "current isolated"],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    previous_encoded = tokenizer(
        ["", "previous one", ""],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    following_encoded = tokenizer(
        ["following zero", "", ""],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    batch = {
        "input_ids": current["input_ids"].to(device),
        "attention_mask": current["attention_mask"].to(device),
        "token_type_ids": current["token_type_ids"].to(device),
        "context_available": torch.tensor([True, True, False], device=device),
        "previous_input_ids": previous_encoded["input_ids"].to(device),
        "previous_attention_mask": previous_encoded["attention_mask"].to(device),
        "previous_context_available": torch.tensor(
            [False, True, False], device=device
        ),
        "following_input_ids": following_encoded["input_ids"].to(device),
        "following_attention_mask": following_encoded["attention_mask"].to(device),
        "following_context_available": torch.tensor(
            [True, False, False], device=device
        ),
    }
    model = PretrainedTextFusionNet(
        {
            "pretrained_model": name,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "fusion_dimension": 64,
            "fusion_heads": 4,
            "dropout": 0.1,
            "use_directional_context": True,
            "use_prototype_router": True,
            "use_context_weighted_prototype": True,
            "context_shift_scale": 0.35,
        }
    ).to(device)
    output = model(batch)
    weights = torch.stack(
        [
            output["context_reject_weight"],
            output["previous_context_weight"],
            output["following_context_weight"],
        ],
        dim=-1,
    )
    assert output["class_logits"].shape == (3, 3)
    assert torch.isfinite(output["class_logits"]).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, device=device), atol=1e-6)
    assert output["previous_context_weight"][0].item() == 0.0
    assert output["following_context_weight"][1].item() == 0.0
    assert output["context_reject_weight"][2].item() == 1.0
    assert torch.isfinite(output["prototype_attention"]).all()
    loss = F.cross_entropy(
        output["class_logits"], torch.tensor([0, 1, 2], device=device)
    )
    loss.backward()
    router_gradients = [
        parameter.grad
        for parameter in model.directional_context_router.parameters()
        if parameter.grad is not None
    ]
    assert router_gradients
    assert all(torch.isfinite(gradient).all() for gradient in router_gradients)
    print(
        {
            "device": str(device),
            "loss": float(loss.detach()),
            "weights": weights.detach().cpu().tolist(),
        }
    )
    print("Directional context router smoke test passed.")


if __name__ == "__main__":
    main()
