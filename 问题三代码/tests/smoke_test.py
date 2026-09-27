from __future__ import annotations

import sys
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data import (
    FeatureNormalizer,
    MultimodalDataset,
    attach_labels_from_excel,
    compute_class_weights,
    load_feature_source,
    parse_split,
)
from src.explain import build_explanation_rows, build_video_mapping, find_video, local_importance
from src.losses import MultitaskEvidenceLoss, pearson_loss
from src.model import HAFusionNet, masked_sparsemax
from src.q3_reporting import export_labeled_prediction_views


def synthetic_data(n: int = 9, length: int = 10):
    rng = np.random.default_rng(7)
    valid_lengths = np.asarray([length - i % 4 for i in range(n)])
    mask = np.arange(length)[None, :] < valid_lengths[:, None]
    text = rng.normal(size=(n, length, 12)).astype(np.float32) * mask[..., None]
    audio = rng.normal(size=(n, length, 6)).astype(np.float32) * mask[..., None]
    vision = rng.normal(size=(n, length, 5)).astype(np.float32) * mask[..., None]
    text_bert = np.zeros((n, 3, length), dtype=np.int64)
    text_bert[:, 1, :] = mask
    return {
        "train": {
            "id": np.asarray([f"video$_${i}" for i in range(n)], dtype=object),
            "raw_text": np.asarray(["a synthetic multimodal sample" for _ in range(n)]),
            "text": text,
            "audio": audio,
            "vision": vision,
            "text_bert": text_bert,
            "classification_labels": np.asarray([-1, 0, 1] * 3),
            "regression_labels": np.linspace(-2.5, 2.5, n, dtype=np.float32),
        }
    }


def test_end_to_end():
    arrays = parse_split(synthetic_data(), "train", "text_shared", require_labels=True)
    arrays = FeatureNormalizer().fit(arrays).transform(arrays)
    dataset = MultimodalDataset(arrays)
    batch = torch.utils.data.default_collate([dataset[i] for i in range(6)])
    cfg = {
        "input_dims": {"text": 12, "audio": 6, "vision": 5},
        "d_model": 24,
        "n_heads": 4,
        "n_layers": 1,
        "ff_multiplier": 2,
        "dropout": 0.1,
        "conv_kernel_sizes": [3, 5],
        "sparse_attention": True,
        "evidence_temperature": 0.8,
        "use_cross_context": True,
        "use_conflict_context": True,
        "use_pairwise_evidence": True,
        "fixed_modality_gate": False,
    }
    model = HAFusionNet(cfg)
    outputs = model(batch)
    assert outputs["class_logits"].shape == (6, 3)
    assert outputs["regression"].shape == (6,)
    assert torch.all(outputs["regression"].abs() <= 3.0 + 1e-6)
    assert torch.allclose(outputs["class_probabilities"].sum(1), torch.ones(6), atol=1e-5)
    assert torch.allclose(outputs["temporal_weights"].sum(2), torch.ones(6, 3), atol=1e-5)
    assert torch.allclose(
        outputs["local_regression_contributions"].sum(2),
        outputs["regression_contributions"],
        atol=1e-5,
    )
    assert torch.allclose(
        outputs["local_classification_contributions"].sum(2),
        outputs["classification_contributions"],
        atol=1e-5,
    )
    assert outputs["conflict_scores"].shape == (6, 3, 10)
    assert outputs["pair_regression_contributions"].shape == (6, 3)
    assert torch.allclose(
        outputs["pair_local_regression_contributions"].sum(2),
        outputs["pair_regression_contributions"],
        atol=1e-5,
    )
    assert torch.allclose(
        outputs["regression_raw"] - outputs["regression_bias"],
        outputs["regression_contributions"].sum(1),
        atol=1e-5,
    )
    assert torch.allclose(
        outputs["class_logits"] - outputs["class_bias"],
        outputs["classification_contributions"].sum(1),
        atol=1e-5,
    )
    assert torch.isfinite(local_importance(outputs)).all()

    loss_cfg = {
        "classification": 1.0,
        "regression": 1.0,
        "pearson": 0.2,
        "consistency": 0.1,
        "attention_entropy": 0.01,
        "attention_total_variation": 0.01,
        "gate_balance": 0.001,
        "faithfulness": 0.0,
    }
    criterion = MultitaskEvidenceLoss(
        loss_cfg, compute_class_weights(arrays.class_labels), label_smoothing=0.02
    )
    loss, _ = criterion(outputs, batch["class_label"], batch["regression_label"])
    assert torch.isfinite(loss)
    loss.backward()

    masks = torch.stack(
        [batch["text_mask"], batch["audio_mask"], batch["vision_mask"]], dim=1
    ).numpy()
    summary, evidence = build_explanation_rows(
        ids=batch["id"],
        raw_texts=batch["raw_text"],
        masks=masks,
        outputs=outputs,
        explanation_cfg={"cumulative_mass": 0.7, "max_segments_per_modality": 3},
    )
    assert len(summary) == 6
    assert not evidence.empty
    assert np.allclose(
        summary[["text_importance", "audio_importance", "vision_importance"]].sum(axis=1),
        1.0,
        atol=1e-5,
    )
    summary["true_label"] = ["Negative", "Neutral", "Positive"] * 2
    summary["true_intensity"] = batch["regression_label"].numpy()
    with tempfile.TemporaryDirectory() as temp_dir:
        exported = export_labeled_prediction_views(summary, temp_dir, "valid")
        assert "absolute_regression_error" in exported
        assert (Path(temp_dir) / "valid_classification_predictions.csv").is_file()
        assert (Path(temp_dir) / "valid_regression_predictions.csv").is_file()

    compact_summary, compact_local = build_explanation_rows(
        ids=batch["id"],
        raw_texts=batch["raw_text"],
        masks=masks,
        outputs=outputs,
        explanation_cfg={"cumulative_mass": 0.7, "max_segments_per_modality": 3},
        include_local_table=False,
    )
    assert len(compact_summary) == 6
    assert compact_local.empty


def test_numerically_stable_losses():
    # Exact sparse zeros and a constant-label mini-batch previously produced
    # NaN under FP16 because 1e-8/1e-12 underflowed to zero.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    batch_size, length = 4, 5
    logits = torch.randn(batch_size, 3, device=device, dtype=dtype, requires_grad=True)
    regression = torch.zeros(batch_size, device=device, dtype=dtype, requires_grad=True)
    sparse = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 0.0]] * 3],
        device=device,
        dtype=dtype,
    ).expand(batch_size, -1, -1).clone().requires_grad_()
    outputs = {
        "class_logits": logits,
        "class_probabilities": torch.softmax(logits, dim=-1),
        "regression": regression,
        "temporal_weights": sparse,
        "pair_temporal_weights": sparse,
        "modality_gates": torch.full((batch_size, 3), 1.0 / 3.0, device=device, dtype=dtype),
        "regression_contributions": torch.zeros(batch_size, 3, device=device, dtype=dtype),
        "classification_contributions": torch.zeros(batch_size, 3, 3, device=device, dtype=dtype),
        "pair_regression_contributions": torch.zeros(batch_size, 3, device=device, dtype=dtype),
        "modality_classification_evidence": torch.zeros(batch_size, 3, 3, device=device, dtype=dtype),
        "modality_regression_evidence": torch.zeros(batch_size, 3, device=device, dtype=dtype),
    }
    targets = torch.ones(batch_size, device=device)
    labels = torch.ones(batch_size, dtype=torch.long, device=device)
    cfg = {
        "classification": 1.0,
        "regression": 1.0,
        "pearson": 0.2,
        "consistency": 0.1,
        "attention_entropy": 0.01,
        "attention_total_variation": 0.01,
        "gate_balance": 0.01,
        "faithfulness": 0.0,
        "distillation": 0.0,
        "interaction_l1": 0.01,
        "unimodal_auxiliary": 0.1,
    }
    criterion = MultitaskEvidenceLoss(cfg).to(device)
    loss, components = criterion(outputs, labels, targets)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert torch.isfinite(pearson_loss(regression, targets))
    loss.backward()
    assert sparse.grad is not None and torch.isfinite(sparse.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert regression.grad is not None and torch.isfinite(regression.grad).all()


def test_sparsemax_mask():
    logits = torch.tensor([[2.0, 1.0, -2.0, 10.0]])
    mask = torch.tensor([[True, True, True, False]])
    output = masked_sparsemax(logits, mask)
    assert torch.allclose(output.sum(1), torch.ones(1))
    assert output[0, 3] == 0


def test_attachment4_directory_loader():
    source = synthetic_data(n=2, length=10)["train"]
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir)
        (directory / "videos").mkdir()
        for index in range(2):
            sample = {
                "text": source["text"][index],
                "audio": source["audio"][index],
                "vision": source["vision"][index],
                "text_bert": source["text_bert"][index],
                "raw_text": source["raw_text"][index],
            }
            with (directory / f"{index + 1:02d}.pkl").open("wb") as f:
                pickle.dump(sample, f)
        merged = load_feature_source(directory, split_name="test")
        arrays = parse_split(merged, "test", "text_shared", require_labels=False)
        assert arrays.size == 2
        assert arrays.ids.tolist() == ["01", "02"]
        assert arrays.features["text"].shape == (2, 10, 12)


def test_excel_label_attachment():
    split = synthetic_data(n=3, length=10)["train"]
    split.pop("classification_labels")
    split.pop("regression_labels")
    with tempfile.TemporaryDirectory() as temp_dir:
        excel_path = Path(temp_dir) / "label.xlsx"
        pd.DataFrame(
            {
                "video_id": ["video"] * 3,
                "clip_id": [0, 1, 2],
                "label": [-1.5, 0.0, 2.0],
            }
        ).to_excel(excel_path, index=False)
        data = attach_labels_from_excel({"train": split}, excel_path)
        arrays = parse_split(data, "train", "text_shared", require_labels=True)
        assert arrays.class_labels.tolist() == [0, 1, 2]
        assert np.allclose(arrays.regression_labels, [-1.5, 0.0, 2.0])


def test_video_lookup():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "videos"
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "1.MOV").write_bytes(b"not-a-real-video")
        (root / "video_a").mkdir()
        (root / "video_a" / "clip_b.avi").write_bytes(b"not-a-real-video")
        assert find_video(root, "01").name == "1.MOV"
        assert find_video(root, "video_a$_$clip_b").name == "clip_b.avi"
        mapping = build_video_mapping(["01", "video_a$_$clip_b", "missing"], root)
        assert mapping["video_found"].tolist() == [True, True, False]


if __name__ == "__main__":
    test_sparsemax_mask()
    test_attachment4_directory_loader()
    test_excel_label_attachment()
    test_video_lookup()
    test_numerically_stable_losses()
    test_end_to_end()
    print("All C3-HAFusion smoke tests passed.")
