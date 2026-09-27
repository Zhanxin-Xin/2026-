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
        raise RuntimeError("CUDA is required for the pretrained fusion smoke test")
    device = torch.device("cuda")
    name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    encoded = tokenizer(
        ["This was surprisingly good.", "The result was neither good nor bad."],
        text_pair=["Earlier context was uncertain.", "A previous neutral remark."],
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    batch = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "token_type_ids": encoded["token_type_ids"].to(device),
        "context_available": torch.ones(2, dtype=torch.bool, device=device),
        "legacy_text": torch.randn(2, 12, 768, device=device),
        "legacy_text_mask": torch.ones(2, 12, dtype=torch.bool, device=device),
        "audio": torch.randn(2, 12, 74, device=device),
        "audio_mask": torch.ones(2, 12, dtype=torch.bool, device=device),
        "vision": torch.randn(2, 12, 35, device=device),
        "vision_mask": torch.ones(2, 12, dtype=torch.bool, device=device),
        "video_reference_available": torch.tensor(
            [True, False], dtype=torch.bool, device=device
        ),
        "video_group_log_size": torch.log1p(
            torch.tensor([2.0, 1.0], device=device)
        ),
        "video_audio_reference": torch.randn(2, 74, device=device),
        "video_vision_reference": torch.randn(2, 35, device=device),
    }
    model = PretrainedTextFusionNet(
        {
            "pretrained_model": name,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "use_layerwise_pooling": True,
            "layer_mix_count": 4,
            "use_lora": True,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["query_proj", "value_proj"],
            "fusion_dimension": 128,
            "fusion_heads": 4,
            "dropout": 0.1,
            "use_context": True,
            "use_legacy_text": True,
            "use_almt_hyper": True,
            "hyper_token_count": 4,
            "hyper_depth": 2,
            "use_audio": True,
            "use_vision": True,
            "use_spectral_dynamics": True,
            "use_video_background_deconfounder": True,
            "video_background_uncertainty_gate": True,
            "video_background_l2_trust_region": True,
            "video_background_maximum_shift": 0.35,
            "use_video_relative_neutral_head": True,
            "video_relative_neutral_hidden_dimension": 64,
            "video_relative_neutral_maximum_logit_shift": 0.75,
            "use_low_rank_interaction": True,
            "low_rank_interaction_rank": 2,
            "use_dynamic_fusion_router": True,
            "use_prototype_router": True,
            "use_ordinal_expert": True,
            "use_hurdle": True,
            "hurdle_mix": 0.2,
            "legacy_text_dimension": 768,
            "audio_dimension": 74,
            "vision_dimension": 35,
            "adaptation_shift_scale": 0.5,
        }
    ).to(device)
    output = model(batch)
    assert output["class_logits"].shape == (2, 3)
    assert output["regression"].shape == (2,)
    assert output["legacy_text_weights"].shape == (2, 12)
    assert output["hyper_text_weights"].shape == (2, 4, 12)
    assert output["audio_weights"].shape == (2, 12)
    assert output["visual_weights"].shape == (2, 12)
    assert output["spectral_audio_gate"].shape == (2,)
    assert output["spectral_vision_gate"].shape == (2,)
    assert torch.all((output["spectral_audio_gate"] > 0.0))
    assert torch.all((output["spectral_audio_gate"] < 1.0))
    assert torch.isfinite(output["spectral_audio_high_energy"]).all()
    assert torch.isfinite(output["spectral_vision_high_energy"]).all()
    assert output["prototype_logits"].shape == (2, 3)
    assert output["ordinal_threshold_logits"].shape == (2, 2)
    assert output["ordinal_probabilities"].shape == (2, 3)
    assert output["prototype_attention"].shape == (2, 3, 66)
    assert output["layer_mix_weights"].shape == (4,)
    assert output["fusion_scale"].shape == (2,)
    assert output["low_rank_interaction_mix"].shape == (2,)
    assert output["audio_background_gate"].shape == (2,)
    assert output["vision_background_gate"].shape == (2,)
    assert output["audio_background_delta_norm"].shape == (2,)
    assert output["vision_background_delta_norm"].shape == (2,)
    assert output["audio_background_gate"][1].item() == 0.0
    assert output["vision_background_gate"][1].item() == 0.0
    assert torch.all(
        output["audio_background_gate"]
        <= output["video_background_anchor_uncertainty"] + 1e-6
    )
    assert torch.all(
        output["vision_background_gate"]
        <= output["video_background_anchor_uncertainty"] + 1e-6
    )
    assert torch.count_nonzero(output["audio_background_delta_norm"]) == 0
    assert torch.count_nonzero(output["vision_background_delta_norm"]) == 0
    assert torch.all(output["audio_background_delta_norm"] <= 0.35 + 1e-6)
    assert torch.all(output["vision_background_delta_norm"] <= 0.35 + 1e-6)
    assert output["video_relative_neutral_shift"].shape == (2,)
    assert output["video_relative_neutral_trust"].shape == (2,)
    assert output["video_relative_crossmodal_agreement"].shape == (2,)
    assert torch.count_nonzero(output["video_relative_neutral_shift"]) == 0
    assert torch.all(output["low_rank_interaction_mix"] <= 0.35)
    assert torch.isfinite(output["class_logits"]).all()
    assert torch.isfinite(output["regression"]).all()
    assert torch.allclose(
        output["class_probabilities"].sum(dim=1),
        torch.ones(2, device=device),
        atol=1e-5,
    )
    loss = F.cross_entropy(
        output["class_logits"], torch.tensor([2, 1], device=device)
    ) + F.smooth_l1_loss(
        output["regression"], torch.tensor([1.0, 0.0], device=device)
    )
    loss.backward()
    trainable_gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert trainable_gradients
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)
    assert any("lora_" in name for name, _ in model.text_encoder.named_parameters())
    print(
        {
            "device": str(device),
            "loss": float(loss.detach()),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    print("Pretrained fusion smoke test passed.")


if __name__ == "__main__":
    main()
