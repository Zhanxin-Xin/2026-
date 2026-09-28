from __future__ import annotations

import torch

from tasp_msa.model import MODALITIES, TASPMsa
from train import all_losses


def sample_batch(batch_size: int = 3):
    lengths = torch.tensor([50, 30, 15])[:batch_size]
    batch = {
        "text": torch.randn(batch_size, 50, 768),
        "audio": torch.randn(batch_size, 50, 74),
        "vision": torch.randn(batch_size, 50, 35),
        "classification_label": torch.tensor([0, 1, 2])[:batch_size],
        "regression_label": torch.tensor([-1.2, 0.0, 1.4])[:batch_size],
    }
    for modality in MODALITIES:
        batch[f"valid_mask_{modality}"] = torch.arange(50)[None] < lengths[:, None]
    for view, modalities in (("single_view", ("audio",)), ("double_view", ("audio", "vision"))):
        content = {}
        for modality in MODALITIES:
            content[f"masked_{modality}"] = batch[modality].clone()
            content[f"missing_mask_{modality}"] = torch.ones(batch_size, 50, dtype=torch.bool)
            if modality in modalities:
                content[f"masked_{modality}"][:, 5:10] = 0
                content[f"missing_mask_{modality}"][:, 5:10] = False
        batch[view] = content
    return batch


def inputs(batch, view="full"):
    source = batch if view == "full" else batch[view]
    result = {}
    for modality in MODALITIES:
        result[modality] = batch[modality] if view == "full" else source[f"masked_{modality}"]
        result[f"valid_mask_{modality}"] = batch[f"valid_mask_{modality}"]
        result[f"missing_mask_{modality}"] = (
            torch.ones_like(batch[f"valid_mask_{modality}"])
            if view == "full" else source[f"missing_mask_{modality}"]
        )
    return result


def test_forward_shapes_probabilities_and_range():
    batch = sample_batch()
    model = TASPMsa()
    output = model(**inputs(batch, "double_view"))
    assert output["classification_logits"].shape == (3, 3)
    assert output["regression"].shape == (3, 1)
    assert torch.allclose(output["class_probabilities"].sum(-1), torch.ones(3), atol=1e-6)
    assert output["regression"].min() >= -3 and output["regression"].max() <= 3
    assert output["fused_features"].shape == (3, 96)
    for modality in MODALITIES:
        assert output["proxy_mean"][modality].shape == (3, 64)
        assert torch.isfinite(output["proxy_logvar"][modality]).all()


def test_no_hidden_target_leakage():
    batch = sample_batch()
    model = TASPMsa(dropout=0.0).eval()
    model_inputs = inputs(batch, "double_view")
    first = model(**model_inputs)["proxy_mean"]["audio"]
    # Alter ground truth only; masked model input remains the same object/tensor.
    batch["audio"][:, 5:10] += 1000
    second = model(**model_inputs)["proxy_mean"]["audio"]
    assert torch.allclose(first, second)


def test_all_losses_backward_finite():
    batch = sample_batch()
    model = TASPMsa()
    full = model(**inputs(batch, "full"))
    single = model(**inputs(batch, "single_view"))
    double = model(**inputs(batch, "double_view"))
    cfg = {
        "lambda_regression": 0.7, "lambda_proxy": 0.1,
        "lambda_consistency": 0.08, "lambda_representation": 0.05,
        "lambda_private_modality": 0.02, "lambda_orthogonality": 0.01,
        "lambda_shared_alignment": 0.01, "label_smoothing": 0.02,
    }
    losses = all_losses(model, full, single, double, batch, cfg)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_text_anchor_residual_gates():
    output = TASPMsa()(**inputs(sample_batch(), "double_view"))
    for modality in ("audio", "vision"):
        gate = output["reliability"][modality]
        assert gate.shape == (3, 1)
        assert bool(((gate >= 0) & (gate <= 1)).all())
