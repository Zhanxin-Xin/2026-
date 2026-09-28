#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mm_pipeline.common import atomic_json, dependency_report, load_config, seed_all, setup_logging
from mm_pipeline.manifest import build_manifest, save_manifest
from mm_pipeline.pipeline import run_all, write_summary


def main() -> int:
    p = argparse.ArgumentParser(description="Traceable 0.5 s multimodal feature pipeline")
    p.add_argument("command", choices=["scan", "preflight", "run", "qc"])
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    p.add_argument("--limit", type=int, help="process only first N manifest rows (smoke test)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config); cfg["runtime"]["overwrite"] |= args.overwrite
    out = Path(cfg["paths"]["output_dir"]); setup_logging(out, cfg["runtime"]["log_level"]); seed_all(cfg["runtime"]["seed"])
    rows, manifest_report = build_manifest(cfg)
    save_manifest(out / "manifest.csv", rows); atomic_json(out / "manifest_report.json", manifest_report)
    if args.command == "scan":
        print(json.dumps(manifest_report, ensure_ascii=False, indent=2)); return 0 if not (manifest_report["duplicate_keys"] or manifest_report["missing_videos"] or manifest_report["unlabelled_videos"]) else 2
    deps = dependency_report(cfg); atomic_json(out / "environment.json", deps)
    if args.command == "preflight":
        print(json.dumps(deps, ensure_ascii=False, indent=2)); return 0
    if args.command == "run":
        run_all(rows, cfg, deps, args.limit); write_summary(rows, cfg); return 0
    from mm_pipeline.qc import check_outputs
    report = check_outputs(rows, cfg); atomic_json(out / "qc_report.json", report); write_summary(rows, cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0 if report["passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
