from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .data import MODALITIES, load_feature_source, parse_split
from .explain import build_video_mapping
from .utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check attachment-4 sample IDs, source videos, durations and FFmpeg"
    )
    parser.add_argument("--data", required=True, help="Attachment-4 aligned directory")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mask-strategy", default="text_shared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    video_root = Path(args.video_root) if args.video_root else data_path / "videos"
    raw = load_feature_source(data_path, split_name=args.split)
    if args.split not in raw and all(modality in raw for modality in MODALITIES):
        raw = {args.split: raw}
    arrays = parse_split(
        raw, args.split, mask_strategy=args.mask_strategy, require_labels=False
    )
    mapping = build_video_mapping([str(value) for value in arrays.ids], video_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(
        output / "attachment4_video_mapping.csv", index=False, encoding="utf-8-sig"
    )
    missing = mapping.loc[~mapping["video_found"], "id"].astype(str).tolist()
    duration_available = int(mapping["video_duration_sec"].notna().sum())
    report = {
        "passed": len(missing) == 0,
        "data": str(data_path.resolve()),
        "video_root": str(video_root.resolve()),
        "samples": int(len(mapping)),
        "videos_found": int(mapping["video_found"].sum()),
        "videos_missing": len(missing),
        "missing_ids": missing,
        "durations_available": duration_available,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "notes": [],
    }
    if shutil.which("ffmpeg") is None:
        report["notes"].append("FFmpeg is missing; keyframes cannot be extracted")
    if shutil.which("ffprobe") is None:
        report["notes"].append(
            "ffprobe is missing; proportional video-time mapping cannot be computed"
        )
    if missing:
        report["notes"].append(
            "Rename videos to match PKL stems such as 01.mp4, or keep video_id/clip_id.mp4 layout"
        )
    save_json(report, output / "attachment4_video_check.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
