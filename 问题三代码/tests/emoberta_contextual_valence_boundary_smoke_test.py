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
        raise RuntimeError("CUDA is required for the valence-boundary smoke test")
    torch.manual_seed(20260924)
    device = torch.device("cuda")
    name = "tae898/emoberta-base"
    revision = "64377bdd2a1d7bc5ecdac9a4fbd219002663df1e"
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, local_files_only=True
    )

    def encode(texts: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=24,
            return_tensors="pt",
        )

    current = encode(["The outcome is ordinary.", "I hated the outcome."])
    previous = encode(["Nothing special happened.", "They expected good news."])
    following = encode(["The discussion moved on.", "Everyone became quiet."])
    batch = {
        "input_ids": current["input_ids"].to(device),
        "attention_mask": current["attention_mask"].to(device),
        "token_type_ids": torch.zeros_like(current["input_ids"]).to(device),
        "context_available": torch.ones(2, dtype=torch.bool, device=device),
        "previous_input_ids": previous["input_ids"].to(device),
        "previous_attention_mask": previous["attention_mask"].to(device),
        "previous_context_available": torch.ones(2, dtype=torch.bool, device=device),
        "following_input_ids": following["input_ids"].to(device),
        "following_attention_mask": following["attention_mask"].to(device),
        "following_context_available": torch.ones(2, dtype=torch.bool, device=device),
        "audio": torch.randn(2, 4, 74, device=device),
        "audio_mask": torch.ones(2, 4, dtype=torch.bool, device=device),
        "vision": torch.randn(2, 4, 35, device=device),
        "vision_mask": torch.ones(2, 4, dtype=torch.bool, device=device),
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
            "use_polar_distance_neutral": True,
            "use_external_polarity_anchor": True,
            "external_polarity_anchor_maximum_shift": 1.0,
            "use_low_rank_interaction": True,
            "low_rank_interaction_rank": 4,
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
    # The full model enforces that every contextual/non-verbal correction can
    # move Neutral mass but cannot alter the current-utterance polar odds.
    positive_given_polar = probability[:, 2] / (
        probability[:, 0] + probability[:, 2]
    ).clamp_min(1e-8)
    assert torch.allclose(
        positive_given_polar,
        torch.sigmoid(output["polarity_logit"]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isfinite(output["neutral_boundary_probability"]).all()
    assert torch.isfinite(output["low_rank_interaction_mix"]).all()
    expected_prior = output["external_logits"][:, 2] - output["external_logits"][:, 0]
    assert torch.allclose(output["polarity_prior_logit"], expected_prior, atol=1e-6)
    assert output["polarity_prior_residual"].abs().max().item() <= 1.0 + 1e-6
    assert torch.allclose(
        output["polarity_anchor_logit"],
        output["polarity_prior_logit"] + output["polarity_prior_residual"],
        atol=1e-6,
    )
    loss = F.nll_loss(
        probability.clamp_min(1e-8).log(),
        torch.tensor([1, 0], device=device),
    )
    loss.backward()
    assert model.polar_distance_neutral.polarity[-1].weight.grad is not None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.low_rank_interaction.parameters()
    )
    print(
        {
            "loss": float(loss.detach()),
            "neutral_boundary": output["neutral_boundary_probability"]
            .detach()
            .cpu()
            .tolist(),
            "positive_given_polar": positive_given_polar.detach().cpu().tolist(),
        }
    )
    print("EmoBERTa contextual valence-boundary smoke test passed.")


if __name__ == "__main__":
    main()
