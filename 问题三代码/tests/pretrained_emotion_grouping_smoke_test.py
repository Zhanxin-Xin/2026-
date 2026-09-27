from __future__ import annotations

import torch


def grouped_probability(logits: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    raw = torch.softmax(logits, dim=-1)
    result = torch.stack([raw[:, indices].sum(dim=1) for indices in groups], dim=1)
    return result / result.sum(dim=1, keepdim=True)


def main() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, 2.0, -1.0, -1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0, 2.0, 1.0, 0.5, 0.0],
        ]
    )
    # EmoBERTa order: neutral, joy, surprise, anger, sadness, disgust, fear.
    probability = grouped_probability(logits, [[3, 4, 5, 6], [0, 2], [1]])
    assert probability.shape == (2, 3)
    assert np_allclose(probability.sum(dim=1), torch.ones(2))
    assert probability[0].argmax().item() == 1
    assert probability[1].argmax().item() == 0
    print("pretrained emotion grouping smoke test passed")


def np_allclose(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.allclose(left, right, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    main()
