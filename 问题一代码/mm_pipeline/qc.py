from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def check_outputs(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    out = Path(cfg["paths"]["output_dir"]); problems=[]; checked=0
    word_total = word_valid = frame_total = frame_valid = 0
    for row in rows:
        d = out / "samples" / row["sample_key"]
        status_path, feature_path = d / "status.json", d / "features.npz"
        if not status_path.exists(): continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "success": continue
        checked += 1
        if not feature_path.exists(): problems.append(f"{row['sample_key']}: success without features.npz"); continue
        with np.load(feature_path, allow_pickle=False) as z:
            maximum = cfg["time"]["max_windows"]
            for name in ("text", "audio", "visual"):
                if z[name].shape[0] != maximum: problems.append(f"{row['sample_key']}: {name} shape {z[name].shape}")
                if not np.isfinite(z[name]).all(): problems.append(f"{row['sample_key']}: {name} has non-finite values")
                mask = z[f"{name}_mask"]
                if np.any(mask & ~z["padding_mask"]): problems.append(f"{row['sample_key']}: {name} valid in padding")
                if np.any(z[name][~mask] != 0): problems.append(f"{row['sample_key']}: {name} invalid slots not zero")
            length = int(z["valid_length"])
            if int(z["padding_mask"].sum()) != length: problems.append(f"{row['sample_key']}: valid_length mismatch")
            if np.any(z["window_start"][:length] >= float(z["duration_seconds"]) + cfg["time"]["window_seconds"]): problems.append(f"{row['sample_key']}: timestamp beyond duration")
        timeline_path = d / "timeline.json"
        if not timeline_path.exists() or not (d / "word_alignment.csv").exists():
            problems.append(f"{row['sample_key']}: trace files missing")
        else:
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            duration = float(timeline["duration_seconds"])
            for word in timeline.get("words", []):
                word_total += 1
                if word.get("valid"):
                    word_valid += 1
                    start, end = float(word["start"]), float(word["end"])
                    if not (0 <= start <= end <= duration + 1e-3):
                        problems.append(f"{row['sample_key']}: word timestamp {start}-{end} outside {duration}")
            for frame in timeline.get("frames", []):
                frame_total += 1
                frame_valid += int(bool(frame.get("valid")))
                timestamp = float(frame["time"])
                if not (0 <= timestamp < duration + cfg["time"]["window_seconds"] + 1e-3):
                    problems.append(f"{row['sample_key']}: frame timestamp {timestamp} outside {duration}")
    expected = cfg["runtime"]["expected_samples"]
    complete = len(rows) == expected and checked == len(rows)
    return {"manifest_samples": len(rows), "expected_samples": expected, "successful_samples_checked": checked,
            "structural_checks_passed": not problems, "extraction_complete": complete,
            "aligned_words_valid": word_valid, "aligned_words_total": word_total,
            "visual_frames_valid": frame_valid, "visual_frames_total": frame_total,
            "problems": problems, "passed": not problems and complete}
