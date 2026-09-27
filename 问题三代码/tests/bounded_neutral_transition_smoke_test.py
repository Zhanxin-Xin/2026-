from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.pretrained_fusion import PretrainedTextFusionNet


def encode(tokenizer: object, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=24,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded transition smoke test")
    device = torch.device("cuda")
    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    current = encode(tokenizer, ["I am calm", "this is fine", "nothing changed"], device)
    previous = encode(tokenizer, ["I was furious", "", ""], device)
    following = encode(tokenizer, ["", "this became awful", ""], device)
    batch = {
        "input_ids": current["input_ids"],
        "attention_mask": current["attention_mask"],
        "token_type_ids": current.get("token_type_ids", torch.zeros_like(current["input_ids"])),
        "context_available": torch.tensor([True, True, False], device=device),
        "previous_input_ids": previous["input_ids"],
        "previous_attention_mask": previous["attention_mask"],
        "previous_context_available": torch.tensor([True, False, False], device=device),
        "following_input_ids": following["input_ids"],
        "following_attention_mask": following["attention_mask"],
        "following_context_available": torch.tensor([False, True, False], device=device),
    }
    parent_cfg = {
        "pretrained_model": model_name,
        "local_files_only": True,
        "gradient_checkpointing": True,
        "fusion_dimension": 64,
        "fusion_heads": 4,
        "dropout": 0.1,
        "use_prototype_router": True,
    }
    torch.manual_seed(7)
    parent = PretrainedTextFusionNet(parent_cfg).to(device).eval()
    bounded_cfg = dict(parent_cfg)
    bounded_cfg["use_bounded_neutral_transition"] = True
    torch.manual_seed(11)
    bounded = PretrainedTextFusionNet(bounded_cfg).to(device).eval()
    compatible = {
        key: value
        for key, value in parent.state_dict().items()
        if key in bounded.state_dict() and value.shape == bounded.state_dict()[key].shape
    }
    bounded.load_state_dict(compatible, strict=False)

    with torch.no_grad():
        parent_output = parent(batch)
        initial_output = bounded(batch)
    assert torch.allclose(
        parent_output["class_probabilities"],
        initial_output["class_probabilities"],
        atol=1e-6,
        rtol=1e-6,
    )
    weights = torch.stack(
        [
            initial_output["context_reject_weight"],
            initial_output["previous_context_weight"],
            initial_output["following_context_weight"],
        ],
        dim=-1,
    )
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, device=device))
    assert initial_output["following_context_weight"][0].item() == 0.0
    assert initial_output["previous_context_weight"][1].item() == 0.0
    assert initial_output["context_reject_weight"][2].item() == 1.0
    assert initial_output["prototype_attention"].shape[-1] == 24
    assert torch.count_nonzero(initial_output["context_transition_norm"]) == 0

    # Match isolated training: the frozen parent (including its dropout) stays
    # in eval mode while only the new residual branch is stochastic/trainable.
    bounded.eval()
    bounded.bounded_neutral_transition_expert.train()
    optimizer = torch.optim.SGD(
        bounded.bounded_neutral_transition_expert.parameters(), lr=0.1
    )
    target = torch.tensor([1, 0, 2], device=device)
    for _ in range(2):
        output = bounded(batch)
        loss = F.cross_entropy(output["class_logits"], target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    output = bounded(batch)
    loss = F.cross_entropy(output["class_logits"], target)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in bounded.bounded_neutral_transition_expert.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(
        parameter.grad is not None
        for parameter in bounded.bounded_neutral_transition_expert.route_score.parameters()
    )
    probabilities = output["class_probabilities"].float().clamp_min(1e-8)
    parent_probabilities = parent_output["class_probabilities"].float().clamp_min(1e-8)
    polar_log_odds = probabilities[:, 0].log() - probabilities[:, 2].log()
    parent_polar_log_odds = parent_probabilities[:, 0].log() - parent_probabilities[:, 2].log()
    assert torch.allclose(polar_log_odds, parent_polar_log_odds, atol=2e-5, rtol=2e-5)
    assert output["context_transition_norm"][2].item() == 0.0
    assert torch.isfinite(output["class_logits"]).all()
    print(
        {
            "device": str(device),
            "parent_max_delta": float(
                (parent_output["class_probabilities"] - initial_output["class_probabilities"])
                .abs()
                .max()
            ),
            "weights": weights.cpu().tolist(),
            "isolated_residual_norm": float(output["context_transition_norm"][2]),
            "maximum_polar_log_odds_delta": float(
                (polar_log_odds - parent_polar_log_odds).abs().max()
            ),
        }
    )
    print("Bounded Neutral-transition smoke test passed.")


if __name__ == "__main__":
    main()
