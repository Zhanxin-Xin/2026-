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
        raise RuntimeError("CUDA is required for the NLI dual-view smoke test")
    device = torch.device("cuda")
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    revision = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
    hypotheses = [
        "The speaker expresses negative sentiment.",
        "The speaker expresses neither positive nor negative sentiment.",
        "The speaker expresses positive sentiment.",
    ]
    current = ["This is wonderful.", "It is simply a table."]
    contextual = [
        "Previous utterances: I expected a disaster. Current utterance: This is wonderful.",
        "Previous utterances: We need the facts. Current utterance: It is simply a table.",
    ]
    premises = [
        text
        for views in zip(current, contextual)
        for text in views
        for _ in hypotheses
    ]
    paired_hypotheses = hypotheses * (len(current) * 2)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=True
    )
    encoded = tokenizer(
        premises,
        paired_hypotheses,
        padding="max_length",
        truncation="longest_first",
        max_length=48,
        return_tensors="pt",
    )
    batch = {
        name: value.reshape(2, 2, 3, 48).to(device)
        for name, value in encoded.items()
    }
    batch["context_available"] = torch.tensor(
        [True, True], dtype=torch.bool, device=device
    )
    model = NLILabelSemanticExpert(
        {
            "pretrained_model": model_name,
            "revision": revision,
            "local_files_only": True,
            "gradient_checkpointing": True,
            "hypotheses": hypotheses,
            "context_dual_view": True,
            "context_mix_logit": -2.0,
            "context_gate_hidden": 16,
            "use_relation_adapter": True,
            "relation_adapter_hidden": 24,
            "use_neutral_hurdle": True,
            "neutral_hurdle_residual": True,
            "neutral_hurdle_hidden": 24,
            "neutral_prior_logit": -1.25,
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_target_modules": ["query_proj", "value_proj"],
        }
    ).to(device)
    output = model(batch)
    assert output["class_logits"].shape == (2, 3)
    assert output["regression"].shape == (2,)
    assert output["context_gate"].shape == (2, 3)
    assert output["neutral_hurdle_logit"].shape == (2,)
    assert torch.isfinite(output["class_logits"]).all()
    assert torch.isfinite(output["context_gate"]).all()
    loss = F.cross_entropy(
        output["class_logits"], torch.tensor([2, 1], device=device)
    )
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert model.relation_adapter[-1].weight.grad is not None
    print(
        {
            "loss": float(loss.detach()),
            "context_gate": output["context_gate"].detach().cpu().tolist(),
        }
    )
    print("NLI context dual-view smoke test passed.")


if __name__ == "__main__":
    main()
