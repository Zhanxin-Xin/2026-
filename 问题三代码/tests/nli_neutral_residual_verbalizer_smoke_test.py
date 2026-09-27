from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.train_nli_semantic_expert import NLILabelSemanticExpert


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the residual-verbalizer smoke test")
    device = torch.device("cuda")
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    revision = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
    hypotheses = [
        "The speaker expresses negative sentiment.",
        "The speaker expresses neither positive nor negative sentiment.",
        "The speaker makes a factual statement without expressing an opinion.",
        "The speaker expresses positive sentiment.",
    ]
    texts = ["This is wonderful.", "It is simply a table."]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=True
    )
    encoded = tokenizer(
        [text for text in texts for _ in hypotheses],
        hypotheses * len(texts),
        padding="max_length",
        truncation="longest_first",
        max_length=48,
        return_tensors="pt",
    )
    batch = {
        name: value.reshape(2, 4, 48).to(device)
        for name, value in encoded.items()
    }
    model = NLILabelSemanticExpert(
        {
            "pretrained_model": model_name,
            "revision": revision,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "semantic_mode": "neutral_residual_verbalizer",
            "neutral_prompt_gate_type": "competitive",
            "neutral_prompt_mix_logit": -2.0,
            "neutral_competition_scale": 1.0,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["query_proj", "value_proj"],
        }
    ).to(device)
    output = model(batch)
    assert output["class_logits"].shape == (2, 3)
    assert output["regression"].shape == (2,)
    assert output["neutral_prompt_gate"].shape == (2,)
    assert torch.all(output["neutral_prompt_gate"] > 0.04)
    assert torch.all(output["neutral_prompt_gate"] < 0.28)
    loss = F.cross_entropy(
        output["class_logits"], torch.tensor([2, 1], device=device)
    )
    loss.backward()
    assert model.neutral_competition_scale.grad is not None
    assert torch.isfinite(model.neutral_competition_scale.grad).all()
    print(
        {
            "loss": float(loss.detach()),
            "neutral_prompt_gate": output["neutral_prompt_gate"].detach().cpu().tolist(),
        }
    )
    print("NLI neutral residual-verbalizer smoke test passed.")


if __name__ == "__main__":
    main()
