from __future__ import annotations

from pathlib import Path

import torch

from utils.checkpoint import load_checkpoint
from tasp_msa.model import MODALITIES, TASPMsa
from tests.test_model import inputs, sample_batch


def test_legacy_checkpoint_compatibility():
    checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints/best_model.pth"
    if not checkpoint_path.exists():
        return
    checkpoint = load_checkpoint(checkpoint_path)
    model = TASPMsa(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)


def test_all_anchor_modes_forward_backward_and_parameter_match():
    batch = sample_batch()
    counts = []
    for mode in ("text", "audio", "vision", "symmetric"):
        model = TASPMsa(anchor_mode=mode)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        output = model(**inputs(batch, "double_view"))
        assert output["classification_logits"].shape == (3, 3)
        assert output["regression"].shape == (3, 1)
        assert output["anchor_mode"] == mode
        assert set(output["reliability"]) == set(MODALITIES)
        assert all(torch.isfinite(value).all() for value in output["reliability"].values())
        if mode == "symmetric":
            assert output["fusion_weights"].shape == (3, 3)
            assert torch.allclose(
                output["fusion_weights"].sum(-1), torch.ones(3), atol=1e-6
            )
        loss = output["classification_logits"].mean() + output["regression"].mean()
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
    assert len(set(counts)) == 1
