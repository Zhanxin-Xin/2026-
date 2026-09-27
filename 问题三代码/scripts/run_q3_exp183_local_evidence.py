"""Pipeline-level local evidence for the frozen EXP183 deployment.

The final EXP183 system is heterogeneous and has no shared internal attention
tensor.  Local evidence is therefore defined on the actual deployment graph by
leave-one-aligned-position-out perturbations.  A second, joint-deletion pass
checks whether deleting high-impact positions changes the frozen prediction
more than deleting low-impact or deterministic-random positions.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict_attachment4_exp183 import (  # noqa: E402
    MEMBER_FILES,
    MODALITIES,
    PROBABILITY_COLUMNS,
    deploy,
    predict_nli,
    predict_pretrained,
)
from src.data import (  # noqa: E402
    FeatureNormalizer,
    MultimodalDataset,
    SplitArrays,
    load_feature_source,
    load_pickle,
    parse_split,
)
from src.explain import (  # noqa: E402
    find_video,
    map_segment_to_words,
    proportional_time,
    select_segments,
    video_duration,
)
from src.model import build_model  # noqa: E402
from src.utils import resolve_device, save_json, seed_everything  # noqa: E402


def subset_arrays(arrays: SplitArrays, indices: np.ndarray) -> SplitArrays:
    return SplitArrays(
        features={name: values[indices].copy() for name, values in arrays.features.items()},
        masks={name: values[indices].copy() for name, values in arrays.masks.items()},
        ids=arrays.ids[indices].copy(),
        raw_text=arrays.raw_text[indices].copy(),
        class_labels=None if arrays.class_labels is None else arrays.class_labels[indices].copy(),
        regression_labels=(
            None
            if arrays.regression_labels is None
            else arrays.regression_labels[indices].copy()
        ),
    )


def _word_indices_for_positions(
    raw_text: str, positions: Iterable[int], valid_positions: list[int]
) -> set[int]:
    words = re.findall(r"\S+", str(raw_text).strip())
    if not words or not valid_positions:
        return set()
    indices: set[int] = set()
    for position in positions:
        if position not in valid_positions:
            continue
        rank = valid_positions.index(position)
        if len(valid_positions) == len(words):
            indices.add(min(rank, len(words) - 1))
        elif len(valid_positions) == len(words) + 2:
            # Position zero and the last valid position are [CLS]/[SEP].
            if 0 < rank < len(valid_positions) - 1:
                indices.add(rank - 1)
        else:
            indices.add(min(len(words) - 1, math.floor(rank * len(words) / len(valid_positions))))
    return indices


def delete_text_positions(
    raw_text: str, positions: Iterable[int], valid_positions: list[int]
) -> str:
    words = re.findall(r"\S+", str(raw_text).strip())
    deleted = _word_indices_for_positions(raw_text, positions, valid_positions)
    return " ".join(word for index, word in enumerate(words) if index not in deleted)


def build_variants(
    base: SplitArrays,
    plans: list[dict[str, list[int]]],
    plan_ids: list[str],
) -> SplitArrays:
    if len(plans) != len(plan_ids):
        raise ValueError("plans and plan_ids differ in length")
    sample_indices = np.asarray([int(plan_id.split("::", 1)[0]) for plan_id in plan_ids])
    result = SplitArrays(
        features={name: values[sample_indices].copy() for name, values in base.features.items()},
        masks={name: values[sample_indices].copy() for name, values in base.masks.items()},
        ids=np.asarray(plan_ids, dtype=object),
        raw_text=base.raw_text[sample_indices].copy(),
        class_labels=None,
        regression_labels=None,
    )
    for variant_index, (sample_index, plan) in enumerate(zip(sample_indices, plans)):
        valid_positions = np.flatnonzero(base.masks["text"][sample_index]).tolist()
        for modality, positions in plan.items():
            if modality not in MODALITIES:
                raise ValueError(f"Unknown modality: {modality}")
            for position in positions:
                if position < 0 or position >= result.features[modality].shape[1]:
                    raise ValueError(f"Invalid position {position}")
                result.features[modality][variant_index, position, :] = 0.0
            if modality == "text":
                result.raw_text[variant_index] = delete_text_positions(
                    str(base.raw_text[sample_index]), positions, valid_positions
                )
    return result


@torch.no_grad()
def predict_hafusion_variants(
    checkpoint_path: Path,
    raw_base: SplitArrays,
    plans: list[dict[str, list[int]]],
    plan_ids: list[str],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = copy.deepcopy(checkpoint["config"])
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    normalized_base = normalizer.transform(raw_base)
    variants = build_variants(normalized_base, plans, plan_ids)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    del checkpoint
    loader = DataLoader(
        MultimodalDataset(variants),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
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
    probability = np.concatenate(probabilities)
    frame = pd.DataFrame(
        {
            "id": ids,
            "predicted_label": np.asarray(("Negative", "Neutral", "Positive"))[probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": np.concatenate(regression),
        }
    )
    del model, variants, normalized_base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return frame


def make_single_position_plans(base: SplitArrays) -> tuple[list[dict[str, list[int]]], list[str], pd.DataFrame]:
    plans: list[dict[str, list[int]]] = []
    plan_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(base.ids.astype(str)):
        for modality in MODALITIES:
            valid = np.flatnonzero(base.masks[modality][sample_index]).tolist()
            for position in valid:
                variant_id = f"{sample_index}::single::{modality}::{position}"
                plans.append({modality: [position]})
                plan_ids.append(variant_id)
                rows.append(
                    {
                        "variant_id": variant_id,
                        "sample_index": sample_index,
                        "sample_id": sample_id,
                        "modality": modality,
                        "position_zero_based": position,
                        "position": position + 1,
                    }
                )
    return plans, plan_ids, pd.DataFrame(rows)


def run_variants(
    raw_base: SplitArrays,
    train_arrays: SplitArrays,
    plans: list[dict[str, list[int]]],
    plan_ids: list[str],
    checkpoint_dir: Path,
    deployment: Path,
    output: Path,
    device: torch.device,
    hafusion_batch_size: int,
) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    normalizer = FeatureNormalizer(normalize_text=False, clip_value=10.0).fit(train_arrays)
    normalized_base = normalizer.transform(raw_base)
    normalized_variants = build_variants(normalized_base, plans, plan_ids)
    frames: list[pd.DataFrame] = []
    for member_index, filename in enumerate(MEMBER_FILES, start=1):
        checkpoint = checkpoint_dir / filename
        print(f"[{member_index}/7] {filename}: {len(plan_ids)} variants", flush=True)
        if member_index <= 5:
            prediction = predict_pretrained(
                checkpoint, {"variants": normalized_variants}, device
            )["variants"]
        elif member_index == 6:
            prediction = predict_hafusion_variants(
                checkpoint,
                raw_base,
                plans,
                plan_ids,
                device,
                hafusion_batch_size,
            )
        else:
            prediction = predict_nli(
                checkpoint, {"variants": normalized_variants}, device
            )["variants"]
        frames.append(prediction)
        prediction.to_csv(output / f"member_{member_index:02d}.csv", index=False)
    deployed, _, _ = deploy(frames, deployment)
    deployed.to_csv(output / "deployed_predictions.csv", index=False)
    return deployed


def load_and_align_full_members(paths: list[Path], ids: list[str]) -> list[pd.DataFrame]:
    result = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"id": str})
        frame["id"] = frame["id"].astype(str)
        if all(str(value).isdigit() for value in ids):
            width = max(len(str(value)) for value in ids)
            frame["id"] = frame["id"].str.zfill(width)
        result.append(frame.set_index("id").loc[ids].reset_index())
    return result


def local_effect_table(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    full: pd.DataFrame,
) -> pd.DataFrame:
    merged = metadata.merge(
        predictions,
        left_on="variant_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    )
    full = full.copy()
    full["sample_id"] = full["id"].astype(str)
    full_probability = full.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    full_predicted = full_probability.argmax(1)
    full_lookup = {sample_id: index for index, sample_id in enumerate(full["sample_id"])}
    confidence_delta = []
    intensity_delta = []
    effect = []
    for _, row in merged.iterrows():
        index = full_lookup[str(row["sample_id"])]
        predicted_class = full_predicted[index]
        changed_probability = float(row[PROBABILITY_COLUMNS[predicted_class]])
        delta_c = float(full_probability[index, predicted_class] - changed_probability)
        delta_r = float(full.iloc[index]["predicted_intensity"] - row["predicted_intensity"])
        confidence_delta.append(delta_c)
        intensity_delta.append(delta_r)
        effect.append(0.5 * (abs(delta_c) + abs(delta_r) / 3.0))
    merged["frozen_class_confidence_delta"] = confidence_delta
    merged["intensity_delta"] = intensity_delta
    merged["local_effect"] = effect
    return merged.drop(columns=["id"])


def _format_positions(segments: list[dict[str, Any]]) -> str:
    values = []
    for segment in segments:
        start, end = int(segment["start"]) + 1, int(segment["end"]) + 1
        values.append(str(start) if start == end else f"{start}-{end}")
    return "; ".join(values)


def _format_time_ranges(segments: list[dict[str, Any]]) -> str:
    values = []
    for segment in segments:
        if segment.get("start_sec") is not None:
            values.append(f"{segment['start_sec']:.3f}-{segment['end_sec']:.3f}s")
    return "; ".join(values)


def extract_keyframe(video: Path, timestamp: float, destination: Path) -> bool:
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(video))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(destination), frame))
    except (ImportError, OSError):
        return False


def build_summary(
    base: SplitArrays,
    local: pd.DataFrame,
    full: pd.DataFrame,
    video_root: Path | None,
    keyframe_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full = full.copy()
    full["id"] = full["id"].astype(str)
    full = full.set_index("id")
    for sample_index, sample_id in enumerate(base.ids.astype(str)):
        duration = video_duration(find_video(video_root, sample_id))
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "predicted_class": full.loc[sample_id, "predicted_label"],
            "predicted_intensity": float(full.loc[sample_id, "predicted_intensity"]),
            "video_duration_sec": duration,
            "time_mapping_rule": (
                "aligned-position proportional mapping to video duration"
                if duration is not None
                else "aligned position only; original timestamp unavailable"
            ),
        }
        valid = np.flatnonzero(base.masks["text"][sample_index]).tolist()
        for modality in MODALITIES:
            values = np.zeros(base.features[modality].shape[1], dtype=float)
            selection = local[
                (local["sample_id"].astype(str) == sample_id)
                & (local["modality"] == modality)
            ]
            values[selection["position_zero_based"].to_numpy(int)] = selection[
                "local_effect"
            ].to_numpy(float)
            segments = select_segments(
                values,
                base.masks[modality][sample_index],
                cumulative_mass=0.60,
                max_segments=3,
                merge_gap=0,
            )
            for segment in segments:
                mapped = proportional_time(
                    int(segment["start"]), int(segment["end"]), valid, duration
                )
                segment["start_sec"] = None if mapped is None else mapped[0]
                segment["end_sec"] = None if mapped is None else mapped[1]
            row[f"{modality}_key_positions"] = _format_positions(segments)
            row[f"{modality}_local_effect_sum"] = float(values.sum())
            if modality == "text":
                snippets = [
                    map_segment_to_words(
                        str(base.raw_text[sample_index]),
                        int(segment["start"]),
                        int(segment["end"]),
                        valid,
                    )["text"]
                    for segment in segments
                ]
                row["key_text"] = " | ".join(value for value in snippets if value)
            elif modality == "audio":
                row["audio_time_range"] = _format_time_ranges(segments)
            else:
                row["visual_time_range"] = _format_time_ranges(segments)
                generated = []
                video = find_video(video_root, sample_id)
                if video is not None:
                    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
                    for rank, segment in enumerate(segments, start=1):
                        if segment.get("start_sec") is None:
                            continue
                        timestamp = 0.5 * (
                            float(segment["start_sec"]) + float(segment["end_sec"])
                        )
                        destination = keyframe_dir / f"{safe_id}_vision_{rank}.jpg"
                        if extract_keyframe(video, timestamp, destination):
                            generated.append(str(destination.resolve()))
                row["visual_key_frame"] = "; ".join(generated)
        rows.append(row)
    return pd.DataFrame(rows)


def make_joint_plans(
    base: SplitArrays, local: pd.DataFrame
) -> tuple[list[dict[str, list[int]]], list[str], pd.DataFrame]:
    plans: list[dict[str, list[int]]] = []
    ids: list[str] = []
    rows = []
    rng = np.random.default_rng(20260924)
    for sample_index, sample_id in enumerate(base.ids.astype(str)):
        for modality in MODALITIES:
            selection = local[
                (local["sample_id"].astype(str) == sample_id)
                & (local["modality"] == modality)
            ].sort_values("local_effect", ascending=False)
            positions = selection["position_zero_based"].to_numpy(int)
            if len(positions) == 0:
                continue
            random_order = rng.permutation(positions)
            for ratio in (0.10, 0.20, 0.30):
                count = max(1, int(math.ceil(len(positions) * ratio)))
                strategies = {
                    "top": positions[:count],
                    "bottom": positions[-count:],
                    "random": random_order[:count],
                }
                for strategy, chosen in strategies.items():
                    variant_id = (
                        f"{sample_index}::joint::{modality}::{int(ratio*100)}::{strategy}"
                    )
                    plans.append({modality: [int(value) for value in chosen]})
                    ids.append(variant_id)
                    rows.append(
                        {
                            "variant_id": variant_id,
                            "sample_index": sample_index,
                            "sample_id": sample_id,
                            "modality": modality,
                            "deletion_ratio": ratio,
                            "strategy": strategy,
                            "deleted_count": count,
                            "deleted_positions": ";".join(str(int(value) + 1) for value in chosen),
                        }
                    )
    return plans, ids, pd.DataFrame(rows)


def joint_effect_table(metadata: pd.DataFrame, prediction: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    merged = metadata.merge(
        prediction,
        left_on="variant_id",
        right_on="id",
        validate="one_to_one",
    )
    full = full.copy()
    full["id"] = full["id"].astype(str)
    full = full.set_index("id")
    deltas = []
    intensity = []
    flips = []
    for _, row in merged.iterrows():
        reference = full.loc[str(row["sample_id"])]
        class_index = ("Negative", "Neutral", "Positive").index(reference["predicted_label"])
        deltas.append(
            float(reference[PROBABILITY_COLUMNS[class_index]] - row[PROBABILITY_COLUMNS[class_index]])
        )
        intensity.append(float(reference["predicted_intensity"] - row["predicted_intensity"]))
        flips.append(int(reference["predicted_label"] != row["predicted_label"]))
    merged["frozen_class_confidence_delta"] = deltas
    merged["intensity_delta"] = intensity
    merged["prediction_flipped"] = flips
    return merged.drop(columns=["id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--full-member", action="append", type=Path, required=True)
    parser.add_argument("--base-table", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hafusion-batch-size", type=int, default=64)
    parser.add_argument("--joint-deletion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.full_member) != len(MEMBER_FILES):
        raise ValueError(f"Expected {len(MEMBER_FILES)} --full-member inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(20260924)
    data = load_feature_source(args.data, split_name=args.split)
    base = parse_split(
        data, args.split, "text_shared", require_labels=not args.unlabeled
    )
    if args.sample_id:
        lookup = {str(value): index for index, value in enumerate(base.ids)}
        missing = [value for value in args.sample_id if str(value) not in lookup]
        if missing:
            raise ValueError(f"Unknown sample ids: {missing}")
        base = subset_arrays(
            base, np.asarray([lookup[str(value)] for value in args.sample_id], dtype=int)
        )
    training = parse_split(
        load_pickle(args.training_data), "train", "text_shared", require_labels=True
    )
    sample_ids = base.ids.astype(str).tolist()
    full_members = load_and_align_full_members(args.full_member, sample_ids)
    full, _, _ = deploy(full_members, args.deployment)
    full.to_csv(args.output / "frozen_full_predictions.csv", index=False)

    plans, plan_ids, metadata = make_single_position_plans(base)
    local_predictions = run_variants(
        base,
        training,
        plans,
        plan_ids,
        args.checkpoint_dir,
        args.deployment,
        args.output / "single_position_members",
        device,
        args.hafusion_batch_size,
    )
    local = local_effect_table(metadata, local_predictions, full)
    local.to_csv(args.output / "local_evidence.csv", index=False, encoding="utf-8-sig")
    summary = build_summary(
        base,
        local,
        full,
        args.video_root,
        args.output / "keyframes",
    )
    summary.to_csv(args.output / "explanation_summary.csv", index=False, encoding="utf-8-sig")

    deletion_summary = None
    if args.joint_deletion:
        joint_plans, joint_ids, joint_metadata = make_joint_plans(base, local)
        joint_predictions = run_variants(
            base,
            training,
            joint_plans,
            joint_ids,
            args.checkpoint_dir,
            args.deployment,
            args.output / "joint_deletion_members",
            device,
            args.hafusion_batch_size,
        )
        joint = joint_effect_table(joint_metadata, joint_predictions, full)
        joint.to_csv(args.output / "joint_deletion_predictions.csv", index=False)
        deletion_summary = (
            joint.groupby(["modality", "deletion_ratio", "strategy"], as_index=False)
            .agg(
                mean_confidence_delta=("frozen_class_confidence_delta", "mean"),
                mean_absolute_intensity_delta=("intensity_delta", lambda x: float(np.abs(x).mean())),
                prediction_flip_rate=("prediction_flipped", "mean"),
                sample_count=("sample_id", "count"),
            )
        )
        deletion_summary.to_csv(args.output / "evidence_deletion_results.csv", index=False)

    enriched_path = None
    if args.base_table is not None:
        base_table = (
            pd.read_excel(args.base_table)
            if args.base_table.suffix.lower() in (".xlsx", ".xls")
            else pd.read_csv(args.base_table)
        )
        id_column = "sample_id" if "sample_id" in base_table.columns else "id"
        base_table[id_column] = base_table[id_column].astype(str).str.zfill(2) if args.unlabeled else base_table[id_column].astype(str)
        summary["sample_id"] = summary["sample_id"].astype(str).str.zfill(2) if args.unlabeled else summary["sample_id"].astype(str)
        enriched = base_table.merge(summary, left_on=id_column, right_on="sample_id", how="left", suffixes=("", "_local"))
        if "sample_id_local" in enriched.columns:
            enriched.drop(columns=["sample_id_local"], inplace=True)
        enriched_path = args.output / "predictions_with_local_explanations.csv"
        enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
        enriched.to_excel(args.output / "predictions_with_local_explanations.xlsx", index=False)

    manifest = {
        "scope": "frozen_exp183_pipeline_local_evidence",
        "sample_count": int(base.size),
        "single_position_variant_count": int(len(metadata)),
        "labels_used": not args.unlabeled,
        "test_used_for_model_selection": False,
        "local_evidence_definition": (
            "0.5*abs(frozen-class confidence change) + "
            "0.5*abs(predicted-intensity change)/3 after deleting one aligned position"
        ),
        "text_deletion": "zero aligned text position and delete its proportionally mapped word when applicable",
        "time_mapping": "proportional to video duration; not an original frame timestamp",
        "joint_deletion_completed": bool(args.joint_deletion),
        "outputs": {
            "local_evidence": "local_evidence.csv",
            "explanation_summary": "explanation_summary.csv",
            "evidence_deletion": (
                None if deletion_summary is None else "evidence_deletion_results.csv"
            ),
            "enriched_predictions": None if enriched_path is None else enriched_path.name,
        },
    }
    save_json(manifest, args.output / "local_evidence_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
