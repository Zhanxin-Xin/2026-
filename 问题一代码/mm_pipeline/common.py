from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = path.resolve().parent
    for key in ("dataset_dir", "labels_file", "output_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else (base / p).resolve())
    return cfg


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def sample_key(video_id: str, clip_id: str) -> str:
    return f"{video_id}__{clip_id}"


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def command_path(value: str) -> Optional[str]:
    p = Path(value)
    if p.parent != Path(".") or p.is_absolute():
        return str(p.resolve()) if p.is_file() and os.access(p, os.X_OK) else None
    return shutil.which(value)


def run_checked(args: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)


def setup_logging(output: Path, level: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(output / "pipeline.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def dependency_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import importlib
    import importlib.metadata

    report: Dict[str, Any] = {"python": sys.version, "platform": platform.platform(), "commands": {}, "packages": {}}
    for name in ("ffmpeg", "ffprobe", "openface_feature_extraction"):
        configured = cfg["paths"][name]
        resolved = command_path(configured)
        item: Dict[str, Any] = {"configured": configured, "path": resolved, "available": bool(resolved)}
        if resolved and name != "openface_feature_extraction":
            try:
                cp = subprocess.run([resolved, "-version"], text=True, capture_output=True, timeout=15)
                item["version_output"] = (cp.stdout or cp.stderr).splitlines()[:2]
            except Exception as e:
                item["version_error"] = repr(e)
        elif resolved:
            # OpenFace 2.2 has no cheap --version action: unknown flags load all
            # detector models and can hang in headless environments. The source
            # tree is pinned to the OpenFace_2.2.0 tag during installation.
            item["version_output"] = ["OpenFace 2.2.0 (pinned source build)"]
        report["commands"][name] = item
    for name in ("numpy", "yaml", "torch", "transformers", "whisperx", "opensmile", "pandas", "openpyxl", "soundfile", "matplotlib"):
        try:
            mod = importlib.import_module(name)
            try:
                version = importlib.metadata.version("PyYAML" if name == "yaml" else name)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(mod, "__version__", "unknown")
            report["packages"][name] = {"available": True, "version": version}
        except Exception as e:
            report["packages"][name] = {"available": False, "error": f"{type(e).__name__}: {e}"}
    return report
