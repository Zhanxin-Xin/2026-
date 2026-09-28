#!/usr/bin/env python3
"""Create a verifiable typical-sample package with real frames and window table."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def words_in_window(words, start, end):
    return " ".join(str(w.get("word", "")).strip() for w in words
                    if w.get("valid") and w.get("start") is not None
                    and float(w["end"]) > start and float(w["start"]) < end)


def main() -> None:
    p = argparse.ArgumentParser(description="Build one traceable multimodal alignment example")
    p.add_argument("sample_dir", type=Path, help="outputs/samples/<video_id>__<clip_id>")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--frames", type=int, default=6, help="number of real frame thumbnails")
    args = p.parse_args()
    import matplotlib.pyplot as plt
    import numpy as np

    sample_dir = args.sample_dir.resolve()
    package = (args.output_dir or (sample_dir / "typical_alignment")).resolve()
    frame_dir = package / "frames"; frame_dir.mkdir(parents=True, exist_ok=True)
    timeline = json.loads((sample_dir / "timeline.json").read_text(encoding="utf-8"))
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    video = Path(timeline["video_path"])
    with np.load(sample_dir / "features.npz", allow_pickle=False) as z:
        masks = {name: z[f"{name}_mask"].astype(bool) for name in ("text", "audio", "visual")}
        starts, ends = z["window_start"].copy(), z["window_end"].copy()
        length = int(z["valid_length"])

    frame_by_window = {int(x["window"]): x for x in timeline["frames"]}
    table = []
    for i in range(length):
        vf = frame_by_window.get(i, {})
        table.append({
            "window_index": i, "window_start": float(starts[i]), "window_end": float(ends[i]),
            "text_fragment": words_in_window(timeline["words"], float(starts[i]), float(ends[i])),
            "audio_interval": f"[{starts[i]:.3f}, {min(ends[i], timeline['duration_seconds']):.3f})",
            "video_frame_time": vf.get("time", ""),
            "text_valid": int(masks["text"][i]), "audio_valid": int(masks["audio"][i]),
            "visual_valid": int(masks["visual"][i]), "face_confidence": vf.get("confidence", ""),
            "feature_file": str((sample_dir / "features.npz").resolve()), "feature_row": i,
        })
    fields = list(table[0])
    with (package / "alignment_windows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(table)

    # Decode sequentially because these MP4 clips have inconsistent stream
    # duration metadata and random seeking can incorrectly stop early.
    subprocess.run([args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(video), "-vf", "fps=2", "-vsync", "0", "-q:v", "2",
                    str(frame_dir / "decoded_%03d.jpg")], check=True)
    decoded = sorted(frame_dir.glob("decoded_*.jpg"))
    if not decoded:
        raise RuntimeError("sequential 2fps decode produced no frames")
    available = min(length, len(decoded))
    count = max(1, min(args.frames, available))
    selected = sorted(set(np.linspace(0, available - 1, count, dtype=int).tolist()))
    images = []
    for i in selected:
        timestamp = float(frame_by_window.get(i, {}).get("time", starts[i]))
        path = frame_dir / f"decoded_{i+1:03d}.jpg"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"sequential 2fps decode produced no frame for window {i}")
        images.append((i, timestamp, path, plt.imread(path)))

    fig = plt.figure(figsize=(15, 7))
    grid = fig.add_gridspec(2, len(images), height_ratios=[2.2, 1.5], hspace=.38)
    for col, (i, timestamp, path, img) in enumerate(images):
        ax = fig.add_subplot(grid[0, col]); ax.imshow(img); ax.axis("off")
        ax.set_title(f"window {i}\nt={timestamp:.2f}s", fontsize=9)
        fragment = table[i]["text_fragment"] or "(no aligned word)"
        ax.text(.5, -.08, fragment, transform=ax.transAxes, ha="center", va="top", fontsize=7, wrap=True)
    ax = fig.add_subplot(grid[1, :])
    colors = {"text": "#4C78A8", "audio": "#F58518", "visual": "#54A24B"}
    for y, name in enumerate(("text", "audio", "visual"), 1):
        for t in starts[:length][masks[name][:length]]:
            ax.broken_barh([(float(t), .5)], (y - .3, .6), facecolors=colors[name])
    for i, timestamp, _, _ in images: ax.axvline(timestamp, color="black", alpha=.25, lw=.8)
    ax.set(yticks=[1, 2, 3], yticklabels=["text", "audio", "visual"], xlabel="original video time (s)",
           title=f"{timeline['sample_key']} — 0.5 s aligned modality masks")
    ax.set_xlim(0, timeline["duration_seconds"]); ax.grid(axis="x", alpha=.2)
    fig.savefig(package / "alignment_with_frames.png", dpi=180, bbox_inches="tight")
    (package / "README.json").write_text(json.dumps({
        "sample_key": timeline["sample_key"], "source_video": str(video),
        "window_seconds": timeline["window_seconds"], "feature_file": str((sample_dir / "features.npz").resolve()),
        "word_alignment_file": str((sample_dir / "word_alignment.csv").resolve()),
        "table": "alignment_windows.csv", "figure": "alignment_with_frames.png",
        "frame_rule": "displayed frames are the real sequential 2 fps samples used by the visual pipeline; windows without a decoded frame remain invalid",
        "feature_dimensions": {"text": 768, "audio": len(metadata["audio"]["feature_names"]), "visual": len(metadata["visual"]["feature_names"])},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(package)


if __name__ == "__main__": main()
