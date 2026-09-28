from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .common import command_path, run_checked


def probe_video(video: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    exe = command_path(cfg["paths"]["ffprobe"])
    if not exe:
        raise RuntimeError(f"ffprobe unavailable: configured={cfg['paths']['ffprobe']!r}")
    cp = run_checked([exe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)])
    raw = json.loads(cp.stdout)
    streams = raw.get("streams", [])
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    aus = [s for s in streams if s.get("codec_type") == "audio"]
    if vs is None:
        raise RuntimeError("no video stream")
    container_duration = float(raw.get("format", {}).get("duration") or 0)
    stream_ends = []
    for stream in streams:
        if stream.get("codec_type") not in ("video", "audio") or stream.get("duration") is None:
            continue
        stream_ends.append(float(stream.get("start_time") or 0) + float(stream["duration"]))
    # The supplied clips have inconsistent per-stream duration metadata (for
    # example 261 frames at 30 fps but a stream duration of only 5.5 s). The
    # container duration agrees with decoded frame/audio counts and forced
    # alignment, so it is authoritative; stream_end is retained for auditing.
    duration = container_duration if container_duration > 0 else max(stream_ends, default=0.0)
    rate = vs.get("avg_frame_rate", "0/1")
    num, den = (float(x) for x in rate.split("/"))
    return {
        "duration_seconds": duration, "container_duration_seconds": container_duration,
        "reported_stream_end_seconds": max(stream_ends, default=None),
        "duration_rule": "container duration; per-stream duration is retained but not trusted because it conflicts with decoded frame counts",
        "video_codec": vs.get("codec_name"), "width": vs.get("width"),
        "height": vs.get("height"), "frame_rate": num / den if den else None, "has_audio": bool(aus),
        "audio_streams": [{"codec": s.get("codec_name"), "sample_rate": s.get("sample_rate"), "channels": s.get("channels")} for s in aus],
    }


def extract_wav(video: Path, wav: Path, cfg: Dict[str, Any]) -> None:
    exe = command_path(cfg["paths"]["ffmpeg"])
    if not exe:
        raise RuntimeError(f"ffmpeg unavailable: configured={cfg['paths']['ffmpeg']!r}")
    wav.parent.mkdir(parents=True, exist_ok=True)
    run_checked([exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vn", "-ac", str(cfg["audio"]["channels"]), "-ar", str(cfg["audio"]["sample_rate"]), "-c:a", "pcm_s16le", str(wav)])
