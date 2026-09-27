from __future__ import annotations

import torch

from src.pretrained_fusion import TemporalSpectralDynamicsAdapter


def test_spectral_adapter_is_mask_invariant_and_explainable() -> None:
    torch.manual_seed(7)
    adapter = TemporalSpectralDynamicsAdapter(6, 12, dropout=0.0).eval()
    text = torch.randn(2, 12)
    baseline = torch.randn(2, 12)
    prefix = torch.randn(2, 5, 6)
    first = torch.cat([prefix, torch.zeros(2, 3, 6)], dim=1)
    second = torch.cat([prefix, torch.randn(2, 3, 6) * 100.0], dim=1)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]] * 2, dtype=torch.bool)
    output_a = adapter(text, baseline, first, mask)
    output_b = adapter(text, baseline, second, mask)
    for name in output_a:
        assert torch.allclose(output_a[name], output_b[name], atol=1e-6)
        assert torch.isfinite(output_a[name]).all()
    assert output_a["corrected"].shape == (2, 12)
    assert torch.all((output_a["gate"] > 0.0) & (output_a["gate"] < 1.0))


def test_temporal_velocity_detects_dynamic_signal_and_backpropagates() -> None:
    torch.manual_seed(11)
    adapter = TemporalSpectralDynamicsAdapter(4, 8, dropout=0.0)
    text = torch.randn(2, 8)
    baseline = torch.randn(2, 8)
    feature_pattern = torch.tensor([0.0, 1.0, 2.0, 4.0]).view(1, 1, 4)
    constant = feature_pattern.expand(1, 8, 4)
    alternating_sign = torch.tensor([-1.0, 1.0] * 4).view(1, 8, 1)
    alternating = alternating_sign * feature_pattern
    sequence = torch.cat([constant, alternating], dim=0).requires_grad_()
    mask = torch.ones(2, 8, dtype=torch.bool)
    output = adapter(text, baseline, sequence, mask)
    assert output["velocity"][1] > output["velocity"][0]
    loss = output["corrected"].square().mean() + output["high_energy"].mean()
    loss.backward()
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()
    assert sequence.grad.abs().sum().item() > 0.0
