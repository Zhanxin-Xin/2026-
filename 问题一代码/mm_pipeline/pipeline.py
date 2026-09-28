from __future__ import annotations

import json
import logging
import math
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audio_features import extract_audio_features
from .common import atomic_json, sample_key, stable_hash, write_csv
from .media import extract_wav, probe_video
from .text_features import aggregate_text, force_align, roberta_word_embeddings, save_word_alignment
from .visual_features import extract_visual

LOG = logging.getLogger(__name__)
PIPELINE_REVISION = "3-container-duration-verified-by-decoded-counts"


def _signature(row: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    p = Path(row["video_path"])
    stat = p.stat()
    relevant = {k: cfg[k] for k in ("time", "audio", "text", "visual")}
    return stable_hash({"pipeline_revision": PIPELINE_REVISION, "config": relevant, "video_size": stat.st_size, "video_mtime_ns": stat.st_mtime_ns, "text": row["text"]})


def _window_data(duration: float, cfg: Dict[str, Any]):
    import numpy as np
    step, maximum = cfg["time"]["window_seconds"], cfg["time"]["max_windows"]
    length = min(maximum, int(math.ceil(duration / step)))
    starts = np.arange(maximum, dtype=np.float32) * step
    ends = starts + step
    padding = np.arange(maximum) < length
    return length, starts, ends, padding


def process_one(row: Dict[str, Any], cfg: Dict[str, Any], deps: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    out = Path(cfg["paths"]["output_dir"])
    key, video = row["sample_key"], Path(row["video_path"])
    sample_dir = out / "samples" / key
    status_path = sample_dir / "status.json"
    sig = _signature(row, cfg)
    if status_path.exists() and not cfg["runtime"]["overwrite"]:
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "success" and old.get("signature") == sig and (sample_dir / "features.npz").exists():
            old["resumed"] = True
            return old
    sample_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Any] = {"sample_key": key, "video_id": row["video_id"], "clip_id": row["clip_id"], "source_row": row["source_row"], "video_path": str(video), "signature": sig, "status": "running", "warnings": [], "errors": [], "stages": {}, "tool_versions": deps}
    atomic_json(status_path, status)
    try:
        media = probe_video(video, cfg)
        status["stages"]["probe"] = "success"
        duration = media["duration_seconds"]
        if duration <= 0: raise RuntimeError(f"invalid duration {duration}")
        if duration > cfg["time"]["window_seconds"] * cfg["time"]["max_windows"]:
            status["warnings"].append("duration exceeds configured maximum; tail is truncated")
        length, starts, ends, padding_mask = _window_data(duration, cfg)
        wav = sample_dir / "audio.wav"
        if not media["has_audio"]:
            raise RuntimeError("video has no audio stream; text alignment and audio extraction cannot run")
        extract_wav(video, wav, cfg)
        status["stages"]["audio_decode"] = "success"

        aligned, align_meta = force_align(wav, str(row["text"] or ""), duration, cfg)
        save_word_alignment(sample_dir / "word_alignment.csv", aligned)
        word_vec, supplied_words = roberta_word_embeddings(str(row["text"] or ""), cfg)
        text_feat, text_mask = aggregate_text(word_vec, aligned, duration, cfg)
        status["stages"]["text"] = "success" if align_meta["status"] == "ok" else align_meta["status"]
        if align_meta.get("unaligned_words"):
            status["warnings"].append(f"{len(align_meta['unaligned_words'])} supplied words lack timestamps")

        audio_feat, audio_mask, audio_meta = extract_audio_features(wav, duration, cfg)
        status["stages"]["audio"] = "success"
        with tempfile.TemporaryDirectory(prefix=f"openface_{key}_", dir=str(out / "work")) as td:
            visual_feat, visual_mask, visual_meta, frames = extract_visual(video, duration, Path(td), cfg)
        status["stages"]["visual"] = "success"

        valid = padding_mask
        text_mask &= valid; audio_mask &= valid; visual_mask &= valid
        np.savez_compressed(sample_dir / "features.npz", text=text_feat, audio=audio_feat, visual=visual_feat,
            text_mask=text_mask, audio_mask=audio_mask, visual_mask=visual_mask, padding_mask=padding_mask,
            window_start=starts, window_end=ends, valid_length=np.int32(length), duration_seconds=np.float32(duration),
            video_id=np.array(row["video_id"]), clip_id=np.array(row["clip_id"]), source_row=np.int32(row["source_row"]))
        atomic_json(sample_dir / "timeline.json", {"sample_key": key, "video_path": str(video), "window_seconds": cfg["time"]["window_seconds"], "duration_seconds": duration, "words": aligned, "frames": frames, "audio_silence_windows": audio_meta["silence_windows"]})
        atomic_json(sample_dir / "metadata.json", {"media": media, "alignment": align_meta, "text": {"model": cfg["text"]["roberta_model"], "source_words": supplied_words, "feature_dim": int(text_feat.shape[1]), "overlap_rule": cfg["text"]["overlap_rule"]}, "audio": audio_meta, "visual": visual_meta, "labels_read_only": {"intensity": row["intensity"], "polarity": row["polarity"]}, "padding_rule": "zero-filled to max_windows; padding_mask distinguishes real timeline; modality masks distinguish valid observations"})
        wav.unlink(missing_ok=True)
        status.update({"status": "success", "duration_seconds": duration, "valid_length": length, "text_dim": int(text_feat.shape[1]), "audio_dim": int(audio_feat.shape[1]), "visual_dim": int(visual_feat.shape[1]), "text_valid_windows": int(text_mask.sum()), "audio_valid_windows": int(audio_mask.sum()), "visual_valid_windows": int(visual_mask.sum())})
    except Exception as e:
        status["status"] = "failed"
        status["errors"].append({"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()})
        LOG.error("sample %s failed: %s", key, e)
    atomic_json(status_path, status)
    return status


def run_all(rows: List[Dict[str, Any]], cfg: Dict[str, Any], deps: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
    work = Path(cfg["paths"]["output_dir"]) / "work"
    work.mkdir(parents=True, exist_ok=True)
    statuses = []
    for i, row in enumerate(rows if limit is None else rows[:limit], 1):
        LOG.info("processing %d/%d %s", i, len(rows if limit is None else rows[:limit]), row["sample_key"])
        statuses.append(process_one(row, cfg, deps))
    return statuses


SUMMARY_FIELDS = ["source_row", "video_id", "clip_id", "sample_key", "video_path", "intensity", "polarity", "status", "duration_seconds", "alignment_granularity_seconds", "max_windows", "valid_length", "text_dim", "audio_dim", "visual_dim", "text_valid_windows", "audio_valid_windows", "visual_valid_windows", "warnings", "errors"]
MODALITY_SUMMARY_FIELDS = ["source_row", "video_id", "clip_id", "sample_key", "modality", "status", "original_duration_seconds", "alignment_granularity_seconds", "max_windows", "valid_length", "feature_dimension", "valid_modality_windows", "feature_file", "mask_field", "source_video"]


def write_summary(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = Path(cfg["paths"]["output_dir"]); result=[]
    for row in rows:
        p = out / "samples" / row["sample_key"] / "status.json"
        status = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "not_run", "warnings": [], "errors": []}
        merged = {k: row.get(k, status.get(k, "")) for k in SUMMARY_FIELDS}
        for k in ("status", "duration_seconds", "valid_length", "text_dim", "audio_dim", "visual_dim", "text_valid_windows", "audio_valid_windows", "visual_valid_windows"):
            merged[k] = status.get(k, merged.get(k, ""))
        merged["alignment_granularity_seconds"] = cfg["time"]["window_seconds"]
        merged["max_windows"] = cfg["time"]["max_windows"]
        merged["warnings"] = json.dumps(status.get("warnings", []), ensure_ascii=False)
        merged["errors"] = json.dumps(status.get("errors", []), ensure_ascii=False)
        result.append(merged)
    write_csv(out / "summary.csv", result, SUMMARY_FIELDS)
    modality_rows = []
    for row in result:
        for modality in ("text", "audio", "visual"):
            modality_rows.append({
                "source_row": row["source_row"], "video_id": row["video_id"],
                "clip_id": row["clip_id"], "sample_key": row["sample_key"],
                "modality": modality, "status": row["status"],
                "original_duration_seconds": row["duration_seconds"],
                "alignment_granularity_seconds": row["alignment_granularity_seconds"],
                "max_windows": row["max_windows"], "valid_length": row["valid_length"],
                "feature_dimension": row.get(f"{modality}_dim", ""),
                "valid_modality_windows": row.get(f"{modality}_valid_windows", ""),
                "feature_file": str((out / "samples" / row["sample_key"] / "features.npz").resolve()),
                "mask_field": f"{modality}_mask", "source_video": row["video_path"],
            })
    write_csv(out / "modality_summary.csv", modality_rows, MODALITY_SUMMARY_FIELDS)
    return result
