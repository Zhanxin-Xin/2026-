from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .data import CLASS_NAMES, MODALITIES, PAIR_MODALITY_INDICES, PAIR_NAMES
from .losses import predicted_modality_importance


def _normalize(values: Tensor, dim: int, eps: float = 1e-8) -> Tensor:
    values = values.abs()
    return values / values.sum(dim=dim, keepdim=True).clamp_min(eps)


def local_importance(outputs: Mapping[str, Tensor]) -> Tensor:
    """Return [B,3,L] joint classification-regression evidence importance."""
    reg_raw = outputs["local_regression_contributions"].abs()
    reg_sum = reg_raw.sum(dim=2, keepdim=True)
    fallback = outputs["temporal_weights"]
    reg = torch.where(reg_sum > 1e-8, reg_raw / reg_sum.clamp_min(1e-8), fallback)
    probs = outputs["class_probabilities"]
    top2 = torch.topk(probs, k=2, dim=1).indices
    local_cls = outputs["local_classification_contributions"]
    b, m, length, _ = local_cls.shape
    pred_idx = top2[:, 0, None, None, None].expand(b, m, length, 1)
    runner_idx = top2[:, 1, None, None, None].expand(b, m, length, 1)
    pred = local_cls.gather(3, pred_idx).squeeze(-1)
    runner = local_cls.gather(3, runner_idx).squeeze(-1)
    cls_raw = (pred - runner).abs()
    cls_sum = cls_raw.sum(dim=2, keepdim=True)
    cls = torch.where(cls_sum > 1e-8, cls_raw / cls_sum.clamp_min(1e-8), fallback)
    importance = 0.5 * (reg + cls)
    importance_sum = importance.sum(dim=2, keepdim=True)
    return torch.where(
        importance_sum > 1e-8,
        importance / importance_sum.clamp_min(1e-8),
        fallback,
    )


def pair_local_importance(outputs: Mapping[str, Tensor]) -> Optional[Tensor]:
    """Return [B,3,L] importance for the three explicit pair interactions."""
    if "pair_local_regression_contributions" not in outputs:
        return None
    reg_raw = outputs["pair_local_regression_contributions"].abs()
    reg_sum = reg_raw.sum(dim=2, keepdim=True)
    fallback = outputs["pair_temporal_weights"]
    reg = torch.where(reg_sum > 1e-8, reg_raw / reg_sum.clamp_min(1e-8), fallback)
    probabilities = outputs["class_probabilities"]
    top2 = torch.topk(probabilities, k=2, dim=1).indices
    local_cls = outputs["pair_local_classification_contributions"]
    b, p, length, _ = local_cls.shape
    pred_idx = top2[:, 0, None, None, None].expand(b, p, length, 1)
    runner_idx = top2[:, 1, None, None, None].expand(b, p, length, 1)
    margin = local_cls.gather(3, pred_idx).squeeze(-1) - local_cls.gather(
        3, runner_idx
    ).squeeze(-1)
    cls_sum = margin.abs().sum(dim=2, keepdim=True)
    cls = torch.where(
        cls_sum > 1e-8, margin.abs() / cls_sum.clamp_min(1e-8), fallback
    )
    importance = 0.5 * (reg + cls)
    return importance / importance.sum(dim=2, keepdim=True).clamp_min(1e-8)


def select_segments(
    importance: np.ndarray,
    valid_mask: np.ndarray,
    cumulative_mass: float = 0.70,
    max_segments: int = 3,
    merge_gap: int = 1,
) -> List[Dict[str, Any]]:
    scores = np.asarray(importance, dtype=np.float64).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    scores[~valid_mask] = 0.0
    total = scores.sum()
    if total <= 0:
        valid = np.flatnonzero(valid_mask)
        return [] if len(valid) == 0 else [{"start": int(valid[0]), "end": int(valid[0]), "importance": 0.0}]
    scores /= total
    order = np.argsort(-scores)
    chosen: List[int] = []
    mass = 0.0
    for idx in order:
        if not valid_mask[idx] or scores[idx] <= 0:
            continue
        chosen.append(int(idx))
        mass += float(scores[idx])
        if mass >= cumulative_mass:
            break
    chosen.sort()
    groups: List[List[int]] = []
    for idx in chosen:
        if not groups or idx - groups[-1][-1] > merge_gap + 1:
            groups.append([idx])
        else:
            groups[-1].append(idx)
    segments = [
        {
            "start": int(group[0]),
            "end": int(group[-1]),
            "importance": float(scores[group[0] : group[-1] + 1].sum()),
        }
        for group in groups
    ]
    segments.sort(key=lambda x: x["importance"], reverse=True)
    return segments[:max_segments]


def map_segment_to_words(
    raw_text: str,
    start: int,
    end: int,
    valid_positions: Sequence[int],
) -> Dict[str, Any]:
    words = re.findall(r"\S+", raw_text.strip())
    if not words or not valid_positions:
        return {"text": "", "word_start": None, "word_end": None, "mapping": "unavailable"}
    positions = list(valid_positions)
    try:
        rank_start = positions.index(start)
        rank_end = positions.index(end)
    except ValueError:
        rank_start, rank_end = 0, 0
    valid_len = len(positions)
    if valid_len == len(words):
        word_start, word_end, method = rank_start, rank_end, "direct"
    elif valid_len == len(words) + 2:
        word_start = max(0, rank_start - 1)
        word_end = min(len(words) - 1, rank_end - 1)
        method = "direct_with_special_tokens"
    else:
        word_start = min(len(words) - 1, math.floor(rank_start * len(words) / valid_len))
        word_end = min(
            len(words) - 1,
            max(word_start, math.ceil((rank_end + 1) * len(words) / valid_len) - 1),
        )
        method = "proportional"
    return {
        "text": " ".join(words[word_start : word_end + 1]),
        "word_start": int(word_start),
        "word_end": int(word_end),
        "mapping": method,
    }


def load_timestamps(path: Optional[str | Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    table = pd.read_csv(path)
    required = {"id", "position", "start_sec", "end_sec"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Timestamp CSV is missing columns: {sorted(missing)}")
    table["id"] = table["id"].astype(str)
    return table


def segment_time_from_timestamps(
    timestamps: Optional[pd.DataFrame], sample_id: str, start: int, end: int
) -> Optional[tuple[float, float]]:
    if timestamps is None:
        return None
    rows = timestamps[
        (timestamps["id"] == str(sample_id))
        & (timestamps["position"] >= start)
        & (timestamps["position"] <= end)
    ]
    if rows.empty:
        return None
    return float(rows["start_sec"].min()), float(rows["end_sec"].max())


def proportional_time(start: int, end: int, valid_positions: Sequence[int], duration: Optional[float]):
    if duration is None or not valid_positions:
        return None
    positions = list(valid_positions)
    try:
        rank_start = positions.index(start)
        rank_end = positions.index(end)
    except ValueError:
        return None
    return (
        float(duration * rank_start / len(positions)),
        float(duration * (rank_end + 1) / len(positions)),
    )


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def _numeric_stem(value: str) -> Optional[int]:
    return int(value) if value.isdigit() else None


@lru_cache(maxsize=4096)
def _find_video_cached(root_text: str, sample_id: str) -> Optional[Path]:
    root = Path(root_text)
    if not root.is_dir():
        return None
    sample_id = str(sample_id)
    direct_candidates: List[Path] = []
    target_stems = {sample_id.casefold()}
    pair: Optional[tuple[str, str]] = None
    if "$_$" in sample_id:
        video_id, clip_id = sample_id.split("$_$", 1)
        pair = (video_id, clip_id)
        target_stems.update(
            {
                clip_id.casefold(),
                f"{video_id}_{clip_id}".casefold(),
                f"{video_id}-{clip_id}".casefold(),
            }
        )
        for extension in VIDEO_EXTENSIONS:
            direct_candidates.extend(
                [
                    root / video_id / f"{clip_id}{extension}",
                    root / f"{sample_id}{extension}",
                    root / f"{video_id}_{clip_id}{extension}",
                ]
            )
    else:
        for extension in VIDEO_EXTENSIONS:
            direct_candidates.append(root / f"{sample_id}{extension}")
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    numeric_id = _numeric_stem(sample_id)
    matches: List[tuple[int, str, Path]] = []
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.casefold() not in VIDEO_EXTENSIONS:
            continue
        stem = candidate.stem.casefold()
        score: Optional[int] = None
        if stem == sample_id.casefold():
            score = 0
        elif pair is not None:
            video_id, clip_id = pair
            if candidate.parent.name.casefold() == video_id.casefold() and stem == clip_id.casefold():
                score = 0
            elif stem in target_stems:
                score = 1
        elif numeric_id is not None and _numeric_stem(candidate.stem) == numeric_id:
            score = 2
        if score is not None:
            matches.append((score, str(candidate).casefold(), candidate))
    return min(matches, default=(0, "", None))[2]


def find_video(video_root: Optional[str | Path], sample_id: str) -> Optional[Path]:
    """Locate a sample video across common extensions and nested directory layouts."""
    if video_root is None:
        return None
    root = Path(video_root).resolve()
    return _find_video_cached(str(root), str(sample_id))


def video_duration(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=20
            )
            duration = float(result.stdout.strip())
            if math.isfinite(duration) and duration > 0:
                return duration
        except (subprocess.SubprocessError, ValueError, OSError):
            pass

    # Windows competition environments often have OpenCV but no ffprobe.
    # Keep the fallback local so importing this module does not make cv2 a
    # mandatory dependency for training-only workflows.
    try:
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        try:
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if math.isfinite(frame_count) and math.isfinite(fps) and frame_count > 0 and fps > 0:
            return frame_count / fps
    except (ImportError, OSError):
        pass
    return None


def build_video_mapping(
    ids: Sequence[str], video_root: Optional[str | Path]
) -> pd.DataFrame:
    rows = []
    for sample_id in ids:
        sample_id = str(sample_id)
        path = find_video(video_root, sample_id)
        rows.append(
            {
                "id": sample_id,
                "video_found": path is not None,
                "source_video": "" if path is None else str(path.resolve()),
                "video_duration_sec": video_duration(path),
            }
        )
    return pd.DataFrame(rows)


def extract_vision_keyframes(
    summary: pd.DataFrame,
    video_root: Optional[str | Path],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Extract representative frames for located visual evidence using ffmpeg."""
    summary = summary.copy()
    keyframes: List[str] = []
    ffmpeg = shutil.which("ffmpeg")
    output_dir = Path(output_dir)
    if ffmpeg is not None and video_root is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in summary.iterrows():
        generated: List[str] = []
        video = find_video(video_root, str(row["id"]))
        if ffmpeg is not None and video is not None:
            try:
                segments = json.loads(row["vision_evidence"])
            except (TypeError, json.JSONDecodeError):
                segments = []
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["id"]))
            for rank, segment in enumerate(segments, start=1):
                if segment.get("start_sec") is None or segment.get("end_sec") is None:
                    continue
                timestamp = 0.5 * (float(segment["start_sec"]) + float(segment["end_sec"]))
                destination = output_dir / f"{safe_id}_vision_{rank}.jpg"
                command = [
                    ffmpeg,
                    "-y",
                    "-ss",
                    f"{timestamp:.4f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(destination),
                ]
                try:
                    subprocess.run(command, check=True, capture_output=True, timeout=30)
                    generated.append(destination.name)
                except (subprocess.SubprocessError, OSError):
                    continue
        keyframes.append(json.dumps(generated, ensure_ascii=False))
    summary["vision_keyframes"] = keyframes
    return summary


def build_explanation_rows(
    ids: Sequence[str],
    raw_texts: Sequence[str],
    masks: np.ndarray,
    outputs: Mapping[str, Tensor],
    explanation_cfg: Mapping[str, Any],
    timestamps: Optional[pd.DataFrame] = None,
    video_root: Optional[str | Path] = None,
    ablation_importance: Optional[np.ndarray] = None,
    include_local_table: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    probs = outputs["class_probabilities"].detach().cpu().numpy()
    regression = outputs["regression"].detach().cpu().numpy()
    gates = outputs["modality_gates"].detach().cpu().numpy()
    reg_contrib = outputs["regression_contributions"].detach().cpu().numpy()
    importance = predicted_modality_importance(outputs).detach().cpu().numpy()
    local = local_importance(outputs).detach().cpu().numpy()
    local_signed = outputs["local_regression_contributions"].detach().cpu().numpy()
    pair_local_tensor = pair_local_importance(outputs)
    pair_local = (
        None if pair_local_tensor is None else pair_local_tensor.detach().cpu().numpy()
    )
    pair_local_signed = (
        None
        if "pair_local_regression_contributions" not in outputs
        else outputs["pair_local_regression_contributions"].detach().cpu().numpy()
    )
    pair_reg_contrib = (
        None
        if "pair_regression_contributions" not in outputs
        else outputs["pair_regression_contributions"].detach().cpu().numpy()
    )
    component_gates = (
        None
        if "component_gates" not in outputs
        else outputs["component_gates"].detach().cpu().numpy()
    )
    conflict_scores = (
        None
        if "conflict_scores" not in outputs
        else outputs["conflict_scores"].detach().cpu().numpy()
    )
    pair_reliability = (
        None
        if "pair_reliability" not in outputs
        else outputs["pair_reliability"].detach().cpu().numpy()
    )
    rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    for i, sample_id in enumerate(ids):
        sample_id = str(sample_id)
        pred = int(probs[i].argmax())
        duration = video_duration(find_video(video_root, sample_id))
        all_segments: Dict[str, List[Dict[str, Any]]] = {}
        for m_idx, modality in enumerate(MODALITIES):
            valid_positions = np.flatnonzero(masks[i, m_idx]).tolist()
            segments = select_segments(
                local[i, m_idx],
                masks[i, m_idx],
                cumulative_mass=float(explanation_cfg.get("cumulative_mass", 0.70)),
                max_segments=int(explanation_cfg.get("max_segments_per_modality", 3)),
                merge_gap=int(explanation_cfg.get("merge_gap", 1)),
            )
            for segment in segments:
                start, end = segment["start"], segment["end"]
                word_info = map_segment_to_words(raw_texts[i], start, end, valid_positions)
                time_range = segment_time_from_timestamps(timestamps, sample_id, start, end)
                mapping = "timestamp_csv"
                if time_range is None:
                    time_range = proportional_time(start, end, valid_positions, duration)
                    mapping = "proportional_video_time" if time_range is not None else "unavailable"
                segment.update(
                    {
                        "position_start": start + 1,
                        "position_end": end + 1,
                        "signed_regression_contribution": float(
                            local_signed[i, m_idx, start : end + 1].sum()
                        ),
                        "text": word_info["text"] if modality == "text" else "",
                        "start_sec": None if time_range is None else round(time_range[0], 4),
                        "end_sec": None if time_range is None else round(time_range[1], 4),
                        "time_mapping": mapping,
                    }
                )
            all_segments[modality] = segments
            if include_local_table:
                for position in valid_positions:
                    long_rows.append(
                        {
                            "id": sample_id,
                            "modality": modality,
                            "evidence_type": "modality",
                            "position": position + 1,
                            "importance": float(local[i, m_idx, position]),
                            "signed_regression_contribution": float(
                                local_signed[i, m_idx, position]
                            ),
                        }
                    )

        pair_segments: Dict[str, List[Dict[str, Any]]] = {}
        if pair_local is not None and pair_local_signed is not None:
            for pair_idx, (left_idx, right_idx) in enumerate(PAIR_MODALITY_INDICES):
                pair_name = PAIR_NAMES[pair_idx]
                pair_mask = masks[i, left_idx] & masks[i, right_idx]
                valid_positions = np.flatnonzero(pair_mask).tolist()
                segments = select_segments(
                    pair_local[i, pair_idx],
                    pair_mask,
                    cumulative_mass=float(explanation_cfg.get("cumulative_mass", 0.70)),
                    max_segments=int(explanation_cfg.get("max_segments_per_modality", 3)),
                    merge_gap=int(explanation_cfg.get("merge_gap", 1)),
                )
                for segment in segments:
                    start, end = segment["start"], segment["end"]
                    word_info = map_segment_to_words(
                        raw_texts[i], start, end, valid_positions
                    )
                    time_range = segment_time_from_timestamps(
                        timestamps, sample_id, start, end
                    )
                    mapping = "timestamp_csv"
                    if time_range is None:
                        time_range = proportional_time(
                            start, end, valid_positions, duration
                        )
                        mapping = (
                            "proportional_video_time"
                            if time_range is not None
                            else "unavailable"
                        )
                    segment.update(
                        {
                            "position_start": start + 1,
                            "position_end": end + 1,
                            "signed_regression_contribution": float(
                                pair_local_signed[i, pair_idx, start : end + 1].sum()
                            ),
                            "text": word_info["text"]
                            if left_idx == 0 or right_idx == 0
                            else "",
                            "start_sec": None
                            if time_range is None
                            else round(time_range[0], 4),
                            "end_sec": None
                            if time_range is None
                            else round(time_range[1], 4),
                            "time_mapping": mapping,
                        }
                    )
                pair_segments[pair_name] = segments
                if include_local_table:
                    for position in valid_positions:
                        long_rows.append(
                            {
                                "id": sample_id,
                                "modality": pair_name,
                                "evidence_type": "pair_interaction",
                                "position": position + 1,
                                "importance": float(pair_local[i, pair_idx, position]),
                                "signed_regression_contribution": float(
                                    pair_local_signed[i, pair_idx, position]
                                ),
                            }
                        )

        main_idx = int(np.argmax(importance[i]))
        row: Dict[str, Any] = {
            "id": sample_id,
            "predicted_label": CLASS_NAMES[pred],
            "negative_probability": float(probs[i, 0]),
            "neutral_probability": float(probs[i, 1]),
            "positive_probability": float(probs[i, 2]),
            "predicted_intensity": float(regression[i]),
            "main_modality": MODALITIES[main_idx],
            "text_importance": float(importance[i, 0]),
            "audio_importance": float(importance[i, 1]),
            "vision_importance": float(importance[i, 2]),
            "text_gate": float(gates[i, 0]),
            "audio_gate": float(gates[i, 1]),
            "vision_gate": float(gates[i, 2]),
            "text_signed_contribution": float(reg_contrib[i, 0]),
            "audio_signed_contribution": float(reg_contrib[i, 1]),
            "vision_signed_contribution": float(reg_contrib[i, 2]),
            "text_evidence": json.dumps(all_segments["text"], ensure_ascii=False),
            "audio_evidence": json.dumps(all_segments["audio"], ensure_ascii=False),
            "vision_evidence": json.dumps(all_segments["vision"], ensure_ascii=False),
        }
        if pair_reg_contrib is not None:
            for pair_idx, pair_name in enumerate(PAIR_NAMES):
                pair_mask = masks[i, PAIR_MODALITY_INDICES[pair_idx][0]] & masks[
                    i, PAIR_MODALITY_INDICES[pair_idx][1]
                ]
                denominator = max(int(pair_mask.sum()), 1)
                row[f"{pair_name}_signed_contribution"] = float(
                    pair_reg_contrib[i, pair_idx]
                )
                row[f"{pair_name}_gate"] = float(component_gates[i, 3 + pair_idx])
                row[f"{pair_name}_conflict"] = float(
                    conflict_scores[i, pair_idx][pair_mask].sum() / denominator
                )
                row[f"{pair_name}_reliability"] = float(
                    pair_reliability[i, pair_idx][pair_mask].sum() / denominator
                )
                row[f"{pair_name}_evidence"] = json.dumps(
                    pair_segments.get(pair_name, []), ensure_ascii=False
                )
        if "regression_bias" in outputs:
            regression_bias = outputs["regression_bias"].detach().cpu().numpy()
            row["regression_conservation_error"] = float(
                abs(
                    outputs["regression_raw"].detach().cpu().numpy()[i]
                    - regression_bias[i]
                    - reg_contrib[i].sum()
                )
            )
        if "class_bias" in outputs:
            class_logits_np = outputs["class_logits"].detach().cpu().numpy()
            class_bias_np = outputs["class_bias"].detach().cpu().numpy()
            class_contrib_np = outputs["classification_contributions"].detach().cpu().numpy()
            row["classification_conservation_error"] = float(
                np.max(
                    np.abs(
                        class_logits_np[i]
                        - class_bias_np[i]
                        - class_contrib_np[i].sum(axis=0)
                    )
                )
            )
        if "class_probability_std" in outputs:
            cls_std = outputs["class_probability_std"].detach().cpu().numpy()
            row["predicted_class_probability_std"] = float(cls_std[i, pred])
        if "regression_std" in outputs:
            reg_std = outputs["regression_std"].detach().cpu().numpy()
            row["predicted_intensity_std"] = float(reg_std[i])
        if ablation_importance is not None:
            for m_idx, modality in enumerate(MODALITIES):
                row[f"{modality}_ablation_importance"] = float(ablation_importance[i, m_idx])
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(long_rows)


def plot_explanation_cards(
    summary: pd.DataFrame,
    long_table: pd.DataFrame,
    output_dir: str | Path,
    limit: int = 20,
) -> None:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in summary.head(limit).iterrows():
        sample_id = str(row["id"])
        sample_long = long_table[long_table["id"] == sample_id]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        values = [row[f"{m}_importance"] for m in MODALITIES]
        axes[0].bar(MODALITIES, values, color=["#4776E6", "#E6A147", "#4AA564"])
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("joint importance")
        axes[0].set_title(
            f"{row['predicted_label']} / intensity={row['predicted_intensity']:.3f}"
        )
        for modality in MODALITIES:
            part = sample_long[sample_long["modality"] == modality]
            axes[1].plot(part["position"], part["importance"], label=modality)
        axes[1].set_xlabel("aligned sequence position")
        axes[1].set_ylabel("local importance")
        axes[1].legend()
        axes[1].set_title("Local evidence distribution")
        # MOSEI identifiers contain '$', which Matplotlib otherwise parses as math text.
        fig.suptitle(sample_id.replace("$", r"\$"))
        fig.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
        fig.savefig(output_dir / f"{safe_name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
