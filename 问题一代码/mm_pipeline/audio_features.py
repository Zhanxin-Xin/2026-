from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Tuple


def extract_audio_features(wav: Path, duration: float, cfg: Dict[str, Any]) -> Tuple[Any, Any, Dict[str, Any]]:
    import numpy as np
    try:
        import opensmile
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(f"openSMILE/soundfile dependency unavailable: {e}") from e
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    frame = smile.process_file(str(wav)).reset_index()
    feature_cols = list(smile.feature_names)
    max_w, step = cfg["time"]["max_windows"], cfg["time"]["window_seconds"]
    feat = np.zeros((max_w, len(feature_cols) * 2), np.float32)
    mask = np.zeros(max_w, dtype=bool)
    starts = frame["start"].map(lambda x: x.total_seconds()).to_numpy()
    data = frame[feature_cols].to_numpy(dtype=np.float32)
    samples, sr = sf.read(str(wav), always_2d=False)
    if samples.ndim > 1: samples = samples.mean(axis=1)
    silence = []
    n = min(max_w, int(math.ceil(duration / step)))
    for k in range(n):
        chosen = data[(starts >= k * step) & (starts < (k + 1) * step)]
        a, b = int(k * step * sr), min(len(samples), int((k + 1) * step * sr))
        rms = float(np.sqrt(np.mean(np.square(samples[a:b], dtype=np.float64)))) if b > a else 0.0
        is_silent = rms < cfg["audio"]["silence_rms_threshold"]
        silence.append(is_silent)
        if len(chosen) and not is_silent and np.isfinite(chosen).all():
            feat[k, :len(feature_cols)] = chosen.mean(0)
            feat[k, len(feature_cols):] = chosen.std(0)
            mask[k] = True
    meta = {"feature_names": [f"{x}__mean" for x in feature_cols] + [f"{x}__std" for x in feature_cols], "source_sample_rate": int(sr), "silence_windows": silence, "statistics": ["mean", "std"]}
    return feat, mask, meta
