from __future__ import annotations

import csv
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import command_path, run_checked


def extract_visual(video: Path, duration: float, work_dir: Path, cfg: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any], List[Dict[str, Any]]]:
    import numpy as np
    ffmpeg = command_path(cfg["paths"]["ffmpeg"])
    openface = command_path(cfg["paths"]["openface_feature_extraction"])
    if not ffmpeg or not openface:
        raise RuntimeError(f"visual tools unavailable: ffmpeg={ffmpeg}, OpenFace FeatureExtraction={openface}")
    frames, result = work_dir / "frames", work_dir / "openface"
    frames.mkdir(parents=True, exist_ok=True); result.mkdir(parents=True, exist_ok=True)
    # fps=2 samples at deterministic 0.0,0.5,... timestamps in the original timeline.
    run_checked([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"fps={cfg['time']['frame_fps']}", "-vsync", "0", str(frames / "frame_%06d.png")])
    args = [openface, "-fdir", str(frames), "-out_dir", str(result), "-aus", "-pose", "-gaze", "-2Dfp", "-q"] + [str(x) for x in cfg["visual"]["openface_extra_args"]]
    run_checked(args)
    csvs = sorted(result.glob("*.csv"))
    if not csvs: raise RuntimeError("OpenFace produced no CSV")
    rows = []
    for csv_path in csvs:
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            part = list(csv.DictReader(f))
        # FeatureExtraction may emit one CSV per sampled image or one multi-row CSV.
        rows.extend(part)
    excluded = {"frame", "face_id", "timestamp", "confidence", "success"}
    names = [x.strip() for x in (rows[0].keys() if rows else []) if x.strip() not in excluded]
    max_w = cfg["time"]["max_windows"]
    feat = np.zeros((max_w, len(names)), np.float32); mask = np.zeros(max_w, bool); timeline=[]
    for i, row in enumerate(rows[:max_w]):
        clean = {str(k).strip(): v for k, v in row.items()}
        confidence = float(clean.get("confidence", 0)); success = int(float(clean.get("success", 0))) == 1
        values = np.array([float(clean[x]) for x in names], np.float32)
        valid = success and confidence >= cfg["visual"]["confidence_threshold"] and np.isfinite(values).all()
        if valid: feat[i] = values; mask[i] = True
        timeline.append({"window": i, "time": i / cfg["time"]["frame_fps"], "frame_file": f"frame_{i+1:06d}.png", "confidence": confidence, "success": success, "valid": bool(valid)})
    return feat, mask, {"feature_names": names, "sample_fps": cfg["time"]["frame_fps"], "confidence_threshold": cfg["visual"]["confidence_threshold"]}, timeline
