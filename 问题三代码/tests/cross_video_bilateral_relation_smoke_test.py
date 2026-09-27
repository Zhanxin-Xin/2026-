from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PretrainedTextFusionNet
from src.train_pretrained_fusion import compute_loss, resolve_loss_config


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this architecture smoke test")
    device = torch.device("cuda")
    torch.manual_seed(20260926)
    name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    encoded = tokenizer(
        ["This is poor.", "This is neutral.", "This is clearly useful."],
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    batch_size, references = 3, 2
    batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "token_type_ids": encoded["token_type_ids"].to(device),
        "context_available": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "legacy_text": torch.randn(batch_size, 6, 768, device=device),
        "legacy_text_mask": torch.ones(batch_size, 6, dtype=torch.bool, device=device),
        "audio": torch.randn(batch_size, 6, 74, device=device),
        "audio_mask": torch.ones(batch_size, 6, dtype=torch.bool, device=device),
        "vision": torch.randn(batch_size, 6, 35, device=device),
        "vision_mask": torch.ones(batch_size, 6, dtype=torch.bool, device=device),
        "class_label": torch.tensor([0, 1, 2], device=device),
        "regression_label": torch.tensor([-1.0, 0.0, 1.0], device=device),
        "cross_video_semantic_references": torch.randn(
            batch_size, 3, references, 768, device=device
        ),
        "cross_video_audio_references": torch.randn(
            batch_size, 3, references, 74, device=device
        ),
        "cross_video_vision_references": torch.randn(
            batch_size, 3, references, 35, device=device
        ),
        "cross_video_retrieval_similarity": torch.rand(
            batch_size, 3, references, device=device
        ),
        "cross_video_reference_mask": torch.ones(
            batch_size, 3, references, dtype=torch.bool, device=device
        ),
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
            "use_prototype_router": True,
            "use_cross_video_bilateral_relation": True,
            "cross_video_relation_hidden_dimension": 24,
            "legacy_text_dimension": 768,
            "audio_dimension": 74,
            "vision_dimension": 35,
        }
    ).to(device)
    model.eval()
    output = model(batch)
    if not torch.allclose(
        output["cross_video_left_residual"],
        torch.zeros(batch_size, device=device),
        atol=1e-7,
    ):
        raise AssertionError("cross-video relation must initialize as parent identity")
    if not torch.isfinite(output["class_probabilities"]).all():
        raise AssertionError("cross-video probabilities must remain finite")
    parent_odds = output["cross_video_parent_probability"][:, 2] / output[
        "cross_video_parent_probability"
    ][:, 0]
    child_odds = output["class_probabilities"][:, 2] / output[
        "class_probabilities"
    ][:, 0]
    if not torch.allclose(parent_odds, child_odds, atol=1e-5, rtol=1e-5):
        raise AssertionError("cross-video correction must preserve polar odds")
    loss_cfg = resolve_loss_config(
        {
            "classification": 1.0,
            "cross_video_bilateral_relation": 0.5,
            "label_smoothing": 0.03,
        },
        batch["class_label"].cpu().numpy(),
    )
    loss, components = compute_loss(
        output, batch, torch.ones(3, device=device), loss_cfg
    )
    if not torch.isfinite(loss):
        raise AssertionError("cross-video full-model loss must remain finite")
    if not torch.isfinite(components["cross_video_bilateral_relation"]):
        raise AssertionError("bilateral auxiliary loss must remain finite")
    loss.backward()
    boundary_gradient = model.cross_video_bilateral_relation.negative_neutral_boundary[
        -1
    ].weight.grad
    if boundary_gradient is None or not bool(boundary_gradient.abs().sum() > 0):
        raise AssertionError("bilateral relation boundary must receive gradients")
    print("cross-video bilateral relation CUDA smoke test passed")


if __name__ == "__main__":
    main()
