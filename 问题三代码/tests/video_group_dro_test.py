from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from src.group_robust import VideoGroupDRO, build_video_group_metadata, video_group_name
from src.pretrained_fusion import (
    CrossVideoBilateralRelationHead,
    PairwiseSignedVideoRelativeNeutralHead,
)
from src.train_pretrained_fusion import (
    EncodedTextDataset,
    build_cross_video_relation_references,
)


def test_video_group_metadata_uses_grouped_oof_prefix() -> None:
    identifiers = np.asarray(
        ["video-b$_$0", "video-a$_$0", "video-b$_$1", "singleton"],
        dtype=object,
    )
    metadata = build_video_group_metadata(identifiers)

    assert video_group_name("video-b$_$17") == "video-b"
    assert video_group_name("singleton") == "singleton"
    assert metadata.names == ("singleton", "video-a", "video-b")
    assert metadata.group_count == 3
    assert metadata.index.tolist() == [2, 1, 2, 0]
    assert metadata.size.tolist() == [2.0, 1.0, 2.0, 1.0]


def test_equal_video_risk_and_nonzero_gradient() -> None:
    controller = VideoGroupDRO(
        group_count=2,
        sample_count=4,
        eta=0.5,
        device=torch.device("cpu"),
    )
    # Group 0 has three utterances, group 1 only one. Equal-video risk is
    # (mean([1, 2, 3]) + mean([6])) / 2 = 4, not the sample mean 3.
    losses = torch.tensor([1.0, 2.0, 3.0, 6.0], requires_grad=True)
    value = controller.loss(
        losses,
        torch.tensor([0, 0, 0, 1]),
        torch.tensor([3.0, 3.0, 3.0, 1.0]),
    )
    assert value.item() == pytest.approx(4.0)
    value.backward()
    assert torch.all(losses.grad != 0)
    assert losses.grad[3].item() == pytest.approx(3.0 * losses.grad[0].item())


def test_high_risk_video_receives_more_adversarial_weight() -> None:
    controller = VideoGroupDRO(
        group_count=2,
        sample_count=4,
        eta=1.0,
        device=torch.device("cpu"),
    )
    controller.loss(
        torch.tensor([0.2, 0.4, 1.0, 3.0]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([2.0, 2.0, 2.0, 2.0]),
    )
    diagnostics = controller.finish_epoch()

    assert controller.weights[1] > controller.weights[0]
    assert diagnostics["video_group_mean_risk"] == pytest.approx(1.15)
    assert diagnostics["video_group_worst_decile_risk"] == pytest.approx(2.0)
    assert 1.0 < diagnostics["video_group_effective_count"] < 2.0


def test_finish_epoch_rejects_unobserved_training_video() -> None:
    controller = VideoGroupDRO(
        group_count=3,
        sample_count=3,
        eta=0.25,
        device=torch.device("cpu"),
    )
    controller.loss(
        torch.tensor([0.5, 0.7]),
        torch.tensor([0, 1]),
        torch.tensor([1.0, 1.0]),
    )
    with pytest.raises(RuntimeError, match="did not observe 1"):
        controller.finish_epoch()


def test_dataset_reference_excludes_current_utterance_and_singletons() -> None:
    audio = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    vision = audio[..., :1].copy()
    arrays = SimpleNamespace(
        size=3,
        ids=np.asarray(["shared$_$0", "shared$_$1", "solo$_$0"], dtype=object),
        features={
            "text": np.zeros((3, 2, 1), dtype=np.float32),
            "audio": audio,
            "vision": vision,
        },
        masks={
            "text": np.ones((3, 2), dtype=np.float32),
            "audio": np.ones((3, 2), dtype=np.float32),
            "vision": np.ones((3, 2), dtype=np.float32),
        },
        class_labels=np.asarray([0, 1, 2]),
        regression_labels=np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
    )
    tokenized = {
        "input_ids": torch.zeros(3, 2, dtype=torch.long),
        "attention_mask": torch.ones(3, 2, dtype=torch.long),
        "token_type_ids": torch.zeros(3, 2, dtype=torch.long),
        "context_available": torch.zeros(3, dtype=torch.bool),
    }
    dataset = EncodedTextDataset(arrays, tokenized)

    assert torch.allclose(
        dataset[0]["video_audio_reference"], torch.tensor([6.0, 7.0])
    )
    assert torch.allclose(
        dataset[1]["video_audio_reference"], torch.tensor([2.0, 3.0])
    )
    assert torch.count_nonzero(dataset[2]["video_audio_reference"]) == 0
    assert torch.allclose(
        dataset[0]["video_text_reference"], torch.tensor([0.0])
    )
    assert bool(dataset[0]["video_reference_available"])
    assert not bool(dataset[2]["video_reference_available"])

    supervised = EncodedTextDataset(
        arrays, tokenized, include_video_pair_supervision=True
    )
    assert supervised[0]["video_pair_signed_difference"].item() == pytest.approx(-1.0)
    assert supervised[0]["video_pair_direction"].item() == pytest.approx(0.0)
    assert supervised[0]["video_pair_weight"].item() == pytest.approx(1.0)
    assert supervised[1]["video_pair_signed_difference"].item() == pytest.approx(1.0)
    assert supervised[1]["video_pair_direction"].item() == pytest.approx(1.0)
    assert supervised[1]["video_pair_weight"].item() == pytest.approx(1.0)
    assert supervised[2]["video_pair_weight"].item() == pytest.approx(0.0)
    assert "video_pair_direction" not in dataset[0]


def test_pairwise_signed_head_is_antisymmetric_and_polar_odds_invariant() -> None:
    torch.manual_seed(17)
    head = PairwiseSignedVideoRelativeNeutralHead(
        dimension=8,
        audio_dimension=2,
        vision_dimension=1,
        hidden_dimension=6,
        dropout=0.0,
    ).eval()
    parent = torch.tensor([[0.40, 0.20, 0.40], [0.20, 0.30, 0.50]])
    text = torch.randn(2, 8)
    audio_current = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    audio_reference = torch.tensor([[0.5, -1.0], [0.0, 0.0]])
    vision_current = torch.tensor([[2.0], [1.0]])
    vision_reference = torch.tensor([[-2.0], [0.0]])
    common = {
        "parent_probability": parent,
        "text": text,
        "audio_reliability": torch.tensor([0.7, 0.8]),
        "vision_reliability": torch.tensor([0.8, 0.7]),
        "available": torch.tensor([True, False]),
        "log_group_size": torch.log1p(torch.tensor([2.0, 1.0])),
        "parent_uncertainty": torch.tensor([0.8, 0.8]),
    }
    forward = head(
        audio_current=audio_current,
        audio_reference=audio_reference,
        vision_current=vision_current,
        vision_reference=vision_reference,
        **common,
    )
    swapped = head(
        audio_current=audio_reference,
        audio_reference=audio_current,
        vision_current=vision_reference,
        vision_reference=vision_current,
        **common,
    )

    assert torch.allclose(
        forward["relative_logit"], -swapped["relative_logit"], atol=1e-6
    )
    assert torch.allclose(
        forward["neutral_shift"], -swapped["neutral_shift"], atol=1e-6
    )
    assert torch.allclose(forward["probabilities"][1], parent[1], atol=1e-6)
    parent_polar_odds = parent[:, 2] / parent[:, 0]
    child_polar_odds = (
        forward["probabilities"][:, 2] / forward["probabilities"][:, 0]
    )
    assert torch.allclose(child_polar_odds, parent_polar_odds, atol=1e-6)


def test_pairwise_semantic_anchor_receives_pair_and_class_gradients() -> None:
    torch.manual_seed(23)
    head = PairwiseSignedVideoRelativeNeutralHead(
        dimension=8,
        audio_dimension=2,
        vision_dimension=1,
        semantic_dimension=5,
        hidden_dimension=6,
        dropout=0.0,
        use_semantic_reference=True,
        use_absolute_neutral_anchor=True,
    )
    output = head(
        parent_probability=torch.tensor(
            [[0.40, 0.20, 0.40], [0.20, 0.30, 0.50]]
        ),
        text=torch.randn(2, 8),
        audio_current=torch.randn(2, 2),
        audio_reference=torch.randn(2, 2),
        vision_current=torch.randn(2, 1),
        vision_reference=torch.randn(2, 1),
        audio_reliability=torch.tensor([0.7, 0.8]),
        vision_reliability=torch.tensor([0.8, 0.7]),
        available=torch.tensor([True, True]),
        log_group_size=torch.log1p(torch.tensor([2.0, 2.0])),
        parent_uncertainty=torch.tensor([0.8, 0.7]),
        semantic_current=torch.randn(2, 5),
        semantic_reference=torch.randn(2, 5),
    )
    loss = F.binary_cross_entropy_with_logits(
        output["relative_logit"], torch.tensor([1.0, 0.0])
    ) + F.binary_cross_entropy_with_logits(
        output["anchor_logit"], torch.tensor([1.0, 0.0])
    )
    loss.backward()

    assert output["semantic_evidence"].shape == (2,)
    assert output["semantic_route"].shape == (2,)
    assert torch.allclose(
        output["audio_route"]
        + output["vision_route"]
        + output["semantic_route"],
        torch.ones(2),
        atol=1e-6,
    )
    assert head.semantic_map[1].weight.grad is not None
    assert head.neutral_anchor[-1].weight.grad is not None


def test_cross_video_reference_bank_is_class_balanced_and_group_isolated() -> None:
    ids = np.asarray(
        [
            "shared$_$0",
            "negative-other$_$0",
            "neutral-a$_$0",
            "neutral-b$_$0",
            "positive-a$_$0",
            "positive-b$_$0",
        ],
        dtype=object,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    text = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3) + 1.0
    arrays = SimpleNamespace(
        size=6,
        ids=ids,
        features={
            "text": text,
            "audio": text[..., :2].copy(),
            "vision": text[..., :1].copy(),
        },
        masks={name: np.ones((6, 2), dtype=np.float32) for name in ("text", "audio", "vision")},
        class_labels=labels,
        regression_labels=np.linspace(-1.0, 1.0, 6, dtype=np.float32),
    )
    query = SimpleNamespace(
        size=1,
        ids=np.asarray(["shared$_$99"], dtype=object),
        features={name: value[:1].copy() for name, value in arrays.features.items()},
        masks={name: value[:1].copy() for name, value in arrays.masks.items()},
        # Deliberately absent: retrieval must not inspect query labels.
        class_labels=None,
        regression_labels=None,
    )

    references = build_cross_video_relation_references(arrays, query, 1)
    selected = references["bank_indices"][0, :, 0]

    assert references["mask"].all()
    assert labels[selected].tolist() == [0, 1, 2]
    selected_groups = [str(ids[index]).rsplit("$_$", 1)[0] for index in selected]
    assert "shared" not in selected_groups


def test_cross_video_bilateral_head_preserves_polar_odds_and_has_gradients() -> None:
    torch.manual_seed(29)
    head = CrossVideoBilateralRelationHead(
        dimension=8,
        semantic_dimension=5,
        audio_dimension=2,
        vision_dimension=1,
        hidden_dimension=6,
        dropout=0.0,
    )
    parent = torch.tensor([[0.40, 0.20, 0.40], [0.20, 0.30, 0.50]])
    inputs = {
        "parent_probability": parent,
        "text": torch.randn(2, 8),
        "semantic_current": torch.randn(2, 5),
        "audio_current": torch.randn(2, 2),
        "vision_current": torch.randn(2, 1),
        "semantic_references": torch.randn(2, 3, 2, 5),
        "audio_references": torch.randn(2, 3, 2, 2),
        "vision_references": torch.randn(2, 3, 2, 1),
        "retrieval_similarity": torch.rand(2, 3, 2),
        "reference_mask": torch.ones(2, 3, 2, dtype=torch.bool),
        "audio_reliability": torch.tensor([0.7, 0.8]),
        "vision_reliability": torch.tensor([0.8, 0.7]),
        "parent_uncertainty": torch.tensor([0.9, 0.8]),
    }
    initial = head(**inputs)
    assert torch.allclose(initial["probabilities"], parent, atol=1e-6)

    with torch.no_grad():
        head.negative_neutral_boundary[-1].weight.normal_(std=0.1)
        head.positive_neutral_boundary[-1].weight.normal_(std=0.1)
    output = head(**inputs)
    parent_odds = parent[:, 2] / parent[:, 0]
    child_odds = output["probabilities"][:, 2] / output["probabilities"][:, 0]
    assert torch.allclose(child_odds, parent_odds, atol=1e-6)
    loss = F.binary_cross_entropy_with_logits(
        output["left_logit"], torch.tensor([1.0, 0.0])
    ) + F.binary_cross_entropy_with_logits(
        output["right_logit"], torch.tensor([1.0, 0.0])
    )
    loss.backward()

    assert head.semantic_map[1].weight.grad is not None
    assert head.audio_map[1].weight.grad is not None
    assert head.vision_map[1].weight.grad is not None
    assert head.negative_neutral_boundary[-1].weight.grad is not None


def test_cross_video_odd_relation_has_no_additive_bias_and_learns_scale() -> None:
    torch.manual_seed(31)
    head = CrossVideoBilateralRelationHead(
        dimension=8,
        semantic_dimension=5,
        audio_dimension=2,
        vision_dimension=1,
        hidden_dimension=6,
        dropout=0.0,
        odd_evidence_only=True,
    )
    parent = torch.tensor([[0.40, 0.20, 0.40], [0.20, 0.30, 0.50]])
    output = head(
        parent_probability=parent,
        text=torch.randn(2, 8),
        semantic_current=torch.randn(2, 5),
        audio_current=torch.randn(2, 2),
        vision_current=torch.randn(2, 1),
        semantic_references=torch.randn(2, 3, 2, 5),
        audio_references=torch.randn(2, 3, 2, 2),
        vision_references=torch.randn(2, 3, 2, 1),
        retrieval_similarity=torch.rand(2, 3, 2),
        reference_mask=torch.ones(2, 3, 2, dtype=torch.bool),
        audio_reliability=torch.tensor([0.7, 0.8]),
        vision_reliability=torch.tensor([0.8, 0.7]),
        parent_uncertainty=torch.tensor([0.9, 0.8]),
    )

    assert not hasattr(head, "negative_neutral_boundary")
    assert torch.allclose(output["probabilities"], parent, atol=1e-6)
    loss = F.binary_cross_entropy_with_logits(
        output["left_logit"], torch.tensor([1.0, 0.0])
    ) + F.binary_cross_entropy_with_logits(
        output["right_logit"], torch.tensor([1.0, 0.0])
    )
    loss.backward()
    assert head.relation_scale_parameter.grad is not None
    assert head.relation_scale_parameter.grad.abs().item() > 0.0
