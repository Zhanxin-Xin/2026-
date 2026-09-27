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
        raise RuntimeError("CUDA is required for the sentiment-transition smoke test")
    device = torch.device("cuda")
    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    current = encode(
        tokenizer,
        ["I am calm", "this is fine", "nothing changed"],
        device,
    )
    previous = encode(tokenizer, ["I was furious", "", ""], device)
    following = encode(tokenizer, ["", "this became awful", ""], device)
    batch = {
        "input_ids": current["input_ids"],
        "attention_mask": current["attention_mask"],
        "token_type_ids": current.get(
            "token_type_ids", torch.zeros_like(current["input_ids"])
        ),
        "context_available": torch.tensor([True, True, False], device=device),
        "previous_input_ids": previous["input_ids"],
        "previous_attention_mask": previous["attention_mask"],
        "previous_context_available": torch.tensor(
            [True, False, False], device=device
        ),
        "following_input_ids": following["input_ids"],
        "following_attention_mask": following["attention_mask"],
        "following_context_available": torch.tensor(
            [False, True, False], device=device
        ),
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
    transition_cfg = dict(parent_cfg)
    transition_cfg["use_context_transition"] = True
    torch.manual_seed(11)
    transition = PretrainedTextFusionNet(transition_cfg).to(device).eval()
    compatible = {
        key: value
        for key, value in parent.state_dict().items()
        if key in transition.state_dict()
        and value.shape == transition.state_dict()[key].shape
    }
    transition.load_state_dict(compatible, strict=False)

    with torch.no_grad():
        parent_output = parent(batch)
        transition_output = transition(batch)
    assert torch.allclose(
        parent_output["class_probabilities"],
        transition_output["class_probabilities"],
        atol=1e-6,
        rtol=1e-6,
    )
    weights = torch.stack(
        [
            transition_output["context_reject_weight"],
            transition_output["previous_context_weight"],
            transition_output["following_context_weight"],
        ],
        dim=-1,
    )
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, device=device))
    assert transition_output["following_context_weight"][0].item() == 0.0
    assert transition_output["previous_context_weight"][1].item() == 0.0
    assert transition_output["context_reject_weight"][2].item() == 1.0
    # No absolute neighbour tokens may expand the prototype evidence sequence.
    assert transition_output["prototype_attention"].shape[-1] == 24
    assert torch.count_nonzero(transition_output["context_transition_norm"]) == 0

    transition.train()
    optimizer = torch.optim.SGD(
        transition.sentiment_transition_expert.parameters(), lr=0.1
    )
    target = torch.tensor([1, 0, 2], device=device)
    first = transition(batch)
    first_loss = F.cross_entropy(first["class_logits"], target)
    first_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = transition(batch)
    second_loss = F.cross_entropy(second["class_logits"], target)
    second_loss.backward()
    gradients = [
        parameter.grad
        for parameter in transition.sentiment_transition_expert.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(
        parameter.grad is not None
        for parameter in transition.sentiment_transition_expert.route_score.parameters()
    )
    assert torch.isfinite(second["class_logits"]).all()
    print(
        {
            "device": str(device),
            "parent_max_delta": float(
                (
                    parent_output["class_probabilities"]
                    - transition_output["class_probabilities"]
                )
                .abs()
                .max()
            ),
            "weights": weights.cpu().tolist(),
            "loss": float(second_loss.detach()),
        }
    )
    print("Sentiment-transition residual smoke test passed.")


if __name__ == "__main__":
    main()
