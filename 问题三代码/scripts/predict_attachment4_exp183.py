"""Run the frozen heterogeneous EXP183 deployment on unlabeled Attachment 4.

Seven heterogeneous members are loaded one at a time so the complete pipeline
fits an 8 GiB GPU.  Besides the frozen prediction, the script repeats inference
under three explicit interventions (blank/zero text, zero audio, zero vision)
and converts the final-deployment response into normalized modality importance.
No Attachment-4 labels or metrics are used.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    CLASS_NAMES,
    MODALITIES,
    FeatureNormalizer,
    MultimodalDataset,
    SplitArrays,
    load_feature_source,
    load_pickle,
    parse_split,
)
from src.explain import build_video_mapping  # noqa: E402
from src.model import build_model  # noqa: E402
from src.pretrained_fusion import PretrainedTextFusionNet  # noqa: E402
from src.train_nli_semantic_expert import (  # noqa: E402
    LabelPairDataset,
    NLILabelSemanticExpert,
    encode_label_pairs,
    move_batch as move_nli_batch,
)
from src.train_neutral_authenticity_arbitrator import (  # noqa: E402
    apply_arbitration,
    parent_and_features,
)
from src.train_pretrained_fusion import (  # noqa: E402
    EncodedTextDataset,
    encode_split,
    move_batch as move_pretrained_batch,
)
from src.utils import resolve_device, save_json, seed_everything  # noqa: E402


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)
CONDITIONS = ("full", "text", "audio", "vision")
MEMBER_FILES = (
    "01_exp027_best_model.pt",
    "02_exp029_best_macro_model.pt",
    "03_exp020_best_model.pt",
    "04_exp019_best_model.pt",
    "05_exp022_best_model.pt",
    "06_exp001_best_model.pt",
    "07_exp063_best_model.pt",
)


def clone_arrays(arrays: SplitArrays) -> SplitArrays:
    return SplitArrays(
        features={name: value.copy() for name, value in arrays.features.items()},
        masks={name: value.copy() for name, value in arrays.masks.items()},
        ids=arrays.ids.copy(),
        raw_text=arrays.raw_text.copy(),
        class_labels=None,
        regression_labels=None,
    )


def intervention(arrays: SplitArrays, condition: str) -> SplitArrays:
    result = clone_arrays(arrays)
    if condition == "full":
        return result
    if condition not in MODALITIES:
        raise ValueError(f"Unknown intervention: {condition}")
    result.features[condition].fill(0.0)
    if condition == "text":
        result.raw_text[:] = ""
    return result


def prediction_frame(ids: list[str], probability: np.ndarray, regression: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ids,
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
        }
    )


@torch.no_grad()
def predict_pretrained(
    checkpoint_path: Path,
    arrays_by_condition: Mapping[str, SplitArrays],
    device: torch.device,
) -> dict[str, pd.DataFrame]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = copy.deepcopy(checkpoint["config"])
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    model = PretrainedTextFusionNet(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    del checkpoint
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    max_length = int(cfg["data"].get("max_length", 128))
    context_window = int(cfg["data"].get("context_window", 0))
    batch_size = int(cfg["training"].get("eval_batch_size", 8))
    result: dict[str, pd.DataFrame] = {}
    for condition, arrays in arrays_by_condition.items():
        tokenized = encode_split(tokenizer, arrays, max_length, context_window)
        loader = DataLoader(
            EncodedTextDataset(arrays, tokenized),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        probabilities: list[np.ndarray] = []
        regression: list[np.ndarray] = []
        ids: list[str] = []
        for cpu_batch in loader:
            batch = move_pretrained_batch(cpu_batch, device)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(batch)
            probabilities.append(outputs["class_probabilities"].float().cpu().numpy())
            regression.append(outputs["regression"].float().cpu().numpy())
            ids.extend([str(value) for value in cpu_batch["id"]])
        result[condition] = prediction_frame(
            ids, np.concatenate(probabilities), np.concatenate(regression)
        )
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def predict_hafusion(
    checkpoint_path: Path,
    raw_arrays: SplitArrays,
    device: torch.device,
    batch_size: int,
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, pd.DataFrame]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = copy.deepcopy(checkpoint["config"])
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    normalized = normalizer.transform(raw_arrays)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    del checkpoint
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    result: dict[str, pd.DataFrame] = {}
    for condition in conditions:
        arrays = intervention(normalized, condition)
        loader = DataLoader(
            MultimodalDataset(arrays),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        probabilities: list[np.ndarray] = []
        regression: list[np.ndarray] = []
        ids: list[str] = []
        for cpu_batch in loader:
            batch = {
                name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for name, value in cpu_batch.items()
            }
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(batch)
            probabilities.append(outputs["class_probabilities"].float().cpu().numpy())
            regression.append(outputs["regression"].float().cpu().numpy())
            ids.extend([str(value) for value in cpu_batch["id"]])
        result[condition] = prediction_frame(
            ids, np.concatenate(probabilities), np.concatenate(regression)
        )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def predict_nli(
    checkpoint_path: Path,
    arrays_by_condition: Mapping[str, SplitArrays],
    device: torch.device,
) -> dict[str, pd.DataFrame]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = copy.deepcopy(checkpoint["config"])
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    hypotheses = [str(value) for value in model_cfg["hypotheses"]]
    model = NLILabelSemanticExpert(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    del checkpoint
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    max_length = int(cfg["data"].get("max_length", 128))
    context_window = int(cfg["data"].get("context_window", 0))
    batch_size = int(cfg["training"].get("eval_batch_size", 8))
    result: dict[str, pd.DataFrame] = {}
    for condition, arrays in arrays_by_condition.items():
        tokenized = encode_label_pairs(
            tokenizer, arrays, hypotheses, max_length, context_window
        )
        loader = DataLoader(
            LabelPairDataset(arrays, tokenized),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        probabilities: list[np.ndarray] = []
        regression: list[np.ndarray] = []
        ids: list[str] = []
        for cpu_batch in loader:
            batch = move_nli_batch(cpu_batch, device)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(batch)
            probabilities.append(outputs["class_probabilities"].float().cpu().numpy())
            regression.append(outputs["regression"].float().cpu().numpy())
            ids.extend([str(value) for value in cpu_batch["id"]])
        result[condition] = prediction_frame(
            ids, np.concatenate(probabilities), np.concatenate(regression)
        )
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def align_member_frames(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    reference = frames[0]["id"].astype(str).tolist()
    if len(reference) != len(set(reference)):
        raise ValueError("Duplicate IDs in first member")
    aligned: list[pd.DataFrame] = []
    for index, frame in enumerate(frames):
        local = frame.copy()
        local["id"] = local["id"].astype(str)
        if local["id"].duplicated().any() or set(local["id"]) != set(reference):
            raise ValueError(f"Member {index + 1} ID mismatch")
        aligned.append(local.set_index("id").loc[reference].reset_index())
    return aligned


def deploy(
    frames: list[pd.DataFrame], deployment: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    aligned = align_member_frames(frames)
    report = json.loads((deployment / "final_metrics.json").read_text("utf-8"))
    threshold = float(report["deployment_authenticity_threshold"])
    neutral_bias = float(report["neutral_bias"])
    model = joblib.load(deployment / str(report["deployment_model"]))
    parent, features, _ = parent_and_features(aligned, neutral_bias)
    parent_neutral = parent.argmax(axis=1) == 1
    authenticity = np.ones(len(parent), dtype=np.float64)
    if parent_neutral.any():
        authenticity[parent_neutral] = model.predict_proba(features[parent_neutral])[:, 1]
    probability, rejected = apply_arbitration(parent, authenticity, threshold)
    regression = np.stack(
        [frame["predicted_intensity"].to_numpy(np.float64) for frame in aligned], axis=1
    ).mean(axis=1)
    output = prediction_frame(aligned[0]["id"].astype(str).tolist(), probability, regression)
    output["neutral_authenticity"] = authenticity
    output["neutral_rejected"] = rejected.astype(np.int64)
    return output, parent, features


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen EXP183 Attachment-4 inference")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hafusion-batch-size", type=int, default=32)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    member_output = args.output / "member_predictions"
    member_output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(20260924)

    raw_attachment = load_feature_source(args.data, split_name="test")
    attachment_arrays = parse_split(
        raw_attachment, "test", "text_shared", require_labels=False
    )
    training_arrays = parse_split(
        load_pickle(args.training_data), "train", "text_shared", require_labels=True
    )
    pretrained_normalizer = FeatureNormalizer(
        normalize_text=False, clip_value=10.0
    ).fit(training_arrays)
    normalized_attachment = pretrained_normalizer.transform(attachment_arrays)
    pretrained_conditions = {
        condition: intervention(normalized_attachment, condition)
        for condition in CONDITIONS
    }

    checkpoint_paths = [args.checkpoint_dir / name for name in MEMBER_FILES]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing EXP183 checkpoints: {missing}")

    all_predictions: dict[str, list[pd.DataFrame]] = {name: [] for name in CONDITIONS}
    for member_index, checkpoint_path in enumerate(checkpoint_paths, start=1):
        print(f"[{member_index}/7] {checkpoint_path.name}", flush=True)
        if member_index <= 5:
            predictions = predict_pretrained(
                checkpoint_path, pretrained_conditions, device
            )
        elif member_index == 6:
            predictions = predict_hafusion(
                checkpoint_path,
                attachment_arrays,
                device,
                args.hafusion_batch_size,
            )
        else:
            predictions = predict_nli(
                checkpoint_path, pretrained_conditions, device
            )
        for condition, frame in predictions.items():
            all_predictions[condition].append(frame)
            condition_dir = member_output / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(
                condition_dir / f"member_{member_index:02d}.csv",
                index=False,
                encoding="utf-8-sig",
            )

    deployed: dict[str, pd.DataFrame] = {}
    for condition in CONDITIONS:
        frame, _, _ = deploy(all_predictions[condition], args.deployment)
        deployed[condition] = frame
        frame.to_csv(
            args.output / f"exp183_{condition}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    full = deployed["full"].copy()
    full_probability = full.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    predicted = full_probability.argmax(axis=1)
    row = np.arange(len(full))
    full_confidence = full_probability[row, predicted]
    effects: list[np.ndarray] = []
    signed_confidence: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        changed = deployed[modality]
        changed_probability = changed.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
        confidence_delta = full_confidence - changed_probability[row, predicted]
        regression_delta = (
            full["predicted_intensity"].to_numpy(np.float64)
            - changed["predicted_intensity"].to_numpy(np.float64)
        )
        effect = 0.5 * (np.abs(confidence_delta) + np.abs(regression_delta) / 3.0)
        effects.append(effect)
        signed_confidence[modality] = confidence_delta
        full[f"{modality}_confidence_delta"] = confidence_delta
        full[f"{modality}_intensity_delta"] = regression_delta
    effect_matrix = np.stack(effects, axis=1)
    totals = effect_matrix.sum(axis=1, keepdims=True)
    importance = np.divide(
        effect_matrix,
        totals,
        out=np.full_like(effect_matrix, 1.0 / 3.0),
        where=totals > 1e-12,
    )
    for index, modality in enumerate(MODALITIES):
        full[f"{modality}_importance"] = importance[:, index]
    full["dominant_modality"] = [MODALITIES[index] for index in importance.argmax(axis=1)]
    full.insert(0, "sample_id", full.pop("id"))
    full.rename(
        columns={
            "predicted_label": "predicted_class",
            "negative_probability": "p_negative",
            "neutral_probability": "p_neutral",
            "positive_probability": "p_positive",
        },
        inplace=True,
    )

    video_root = args.video_root
    if video_root is None and (args.data / "videos").is_dir():
        video_root = args.data / "videos"
    mapping = build_video_mapping(full["sample_id"].astype(str).tolist(), video_root)
    mapping.rename(columns={"id": "sample_id"}, inplace=True)
    full = full.merge(mapping, on="sample_id", how="left", validate="one_to_one")
    full.to_csv(
        args.output / "attachment4_predictions.csv", index=False, encoding="utf-8-sig"
    )
    full.to_excel(args.output / "attachment4_predictions.xlsx", index=False)

    summary = {
        "scope": "unlabeled_attachment4_frozen_exp183_inference",
        "sample_count": int(len(full)),
        "labels_used": False,
        "metrics_computed": False,
        "checkpoint_files": [str(path.resolve()) for path in checkpoint_paths],
        "deployment": str(args.deployment.resolve()),
        "data": str(args.data.resolve()),
        "video_root": None if video_root is None else str(video_root.resolve()),
        "interventions": {
            "text": "blank raw_text and zero the aligned 50x768 text representation",
            "audio": "zero the aligned 50x74 acoustic representation",
            "vision": "zero the aligned 50x35 visual representation",
            "importance": "normalize half absolute frozen-class confidence change plus half absolute intensity change / 3",
        },
        "class_counts": {
            name: int((full["predicted_class"] == name).sum()) for name in CLASS_NAMES
        },
        "dominant_modality_counts": {
            name: int((full["dominant_modality"] == name).sum()) for name in MODALITIES
        },
        "mean_modality_importance": {
            name: float(full[f"{name}_importance"].mean()) for name in MODALITIES
        },
        "video_mapping_found": int(full["video_found"].sum()),
        "local_evidence_status": "pending pipeline-level evidence-deletion pass",
        "outputs": {
            "csv": "attachment4_predictions.csv",
            "xlsx": "attachment4_predictions.xlsx",
            "member_predictions": "member_predictions/",
            "condition_predictions": [
                f"exp183_{condition}_predictions.csv" for condition in CONDITIONS
            ],
        },
    }
    save_json(summary, args.output / "inference_manifest.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
