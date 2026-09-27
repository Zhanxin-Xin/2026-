from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PretrainedTextFusionNet
from src.train_pretrained_fusion import (
    hierarchical_best_view_consistency,
    hierarchical_rdrop_consistency,
)
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/exp205_hierarchical_rdrop_trimodal.yaml",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hierarchical R-Drop smoke test")
    device = torch.device("cuda")
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["pretrained_model"], local_files_only=True
    )
    encoded = tokenizer(
        [
            "This is a clearly positive outcome.",
            "Nothing about the result stands out.",
            "The outcome is disappointing.",
            "The evidence remains mixed and uncertain.",
        ],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    batch_size = encoded["input_ids"].size(0)
    batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "token_type_ids": encoded["token_type_ids"].to(device),
        "context_available": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "legacy_text": torch.randn(batch_size, 12, 768, device=device),
        "legacy_text_mask": torch.ones(batch_size, 12, dtype=torch.bool, device=device),
        "audio": torch.randn(batch_size, 12, 74, device=device),
        "audio_mask": torch.ones(batch_size, 12, dtype=torch.bool, device=device),
        "vision": torch.randn(batch_size, 12, 35, device=device),
        "vision_mask": torch.ones(batch_size, 12, dtype=torch.bool, device=device),
        "video_reference_available": torch.zeros(
            batch_size, dtype=torch.bool, device=device
        ),
        "video_group_log_size": torch.zeros(batch_size, device=device),
        "video_audio_reference": torch.zeros(batch_size, 74, device=device),
        "video_vision_reference": torch.zeros(batch_size, 35, device=device),
        "class_label": torch.tensor([2, 1, 0, 1], device=device),
        "regression_label": torch.tensor([1.0, 0.0, -1.0, 0.0], device=device),
    }
    model = PretrainedTextFusionNet(model_cfg).to(device).train()
    first = model(batch)
    second = model(batch)
    if float(cfg["loss"].get("hierarchical_rdrop_consistency", 0.0)) > 0.0:
        consistency, components = hierarchical_rdrop_consistency(
            first, second, cfg["loss"]
        )
        coefficient = float(cfg["loss"]["hierarchical_rdrop_consistency"])
        neutral_key = "hierarchical_rdrop_neutral"
        polarity_key = "hierarchical_rdrop_polarity"
    else:
        consistency, components = hierarchical_best_view_consistency(
            first, second, batch, cfg["loss"]
        )
        coefficient = float(cfg["loss"]["hierarchical_best_view_consistency"])
        neutral_key = "hierarchical_best_view_neutral"
        polarity_key = "hierarchical_best_view_polarity"
    target = batch["class_label"]
    supervised = 0.5 * (
        F.cross_entropy(first["class_logits"], target)
        + F.cross_entropy(second["class_logits"], target)
    )
    loss = supervised + coefficient * consistency
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert consistency.item() > 0.0
    print(
        {
            "device": str(device),
            "batch_size": batch_size,
            "loss": float(loss.detach()),
            "consistency": float(consistency.detach()),
            "neutral_consistency": float(components[neutral_key].detach()),
            "polarity_consistency": float(components[polarity_key].detach()),
            "allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }
    )
    print("Hierarchical R-Drop CUDA smoke test passed.")


if __name__ == "__main__":
    main()
