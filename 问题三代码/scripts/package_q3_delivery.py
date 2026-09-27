"""Assemble the complete, non-destructive SAN2 Question-3 delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


DOCS = (
    "README_FIRST_Q3_中文.md",
    "PROJECT_AUDIT_Q3.md",
    "MODEL_CODE_MAPPING_Q3.md",
    "EXPERIMENT_PLAN_Q3.md",
    "EXPERIMENT_RESULTS_Q3.md",
    "MODEL_ITERATION_LOG_Q3.md",
    "FINAL_Q3_AUDIT.md",
    "TODO_Q3_EXPERIMENTS.md",
    "问题三完整操作手册.md",
    "experiments.md",
    "RESEARCH_ARCHITECTURE_LOG.md",
)
Q3_SCRIPTS = (
    "analyze_q3_dataset.py",
    "predict_attachment4_exp183.py",
    "run_q3_exp183_test_occlusion.py",
    "run_q3_exp183_local_evidence.py",
    "build_q3_paper_assets.py",
    "package_q3_delivery.py",
)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree_filtered(source: Path, destination: Path, suffixes: set[str] | None = None) -> None:
    for path in source.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        copy_file(path, destination / path.relative_to(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--update-existing", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.destination.resolve()
    if destination.exists() and not args.update_existing:
        raise FileExistsError(f"Refusing to overwrite existing delivery: {destination}")
    destination.mkdir(parents=True, exist_ok=args.update_existing)

    for name in DOCS:
        copy_file(root / name, destination / "docs" / name)
    copy_file(root / "README_FIRST_Q3_中文.md", destination / "README_FIRST_Q3_中文.md")
    copy_file(root / "verify_q3_delivery.py", destination / "verify_q3_delivery.py")
    copy_file(root / "environment_exp183.yml", destination / "environment.yml")
    copy_file(root / "requirements.txt", destination / "requirements.txt")

    copy_tree_filtered(root / "src", destination / "src", {".py"})
    for name in Q3_SCRIPTS:
        copy_file(root / "scripts" / name, destination / "scripts" / name)
    copy_tree_filtered(root / "tests", destination / "tests", {".py"})

    # Main configs only: one base config and the seven actual final member configs.
    copy_file(root / "configs/aligned_hafusion.yaml", destination / "configs/aligned_hafusion.yaml")
    config_source = root / "delivery/SAN2_EXP183_complete/models/configs"
    for path in sorted(config_source.glob("*.json")):
        copy_file(path, destination / "configs/final_members" / path.name)

    model_source = root / "delivery/SAN2_EXP183_complete/models"
    for path in sorted(model_source.glob("*.pt")):
        copy_file(path, destination / "models" / path.name)
    copy_tree_filtered(
        root / "delivery/SAN2_EXP183_complete/deployment",
        destination / "deployment",
    )

    copy_file(root / "data/aligned_50.pkl", destination / "data/aligned_50.pkl")
    copy_file(root / "data/label.xlsx", destination / "data/label.xlsx")
    copy_tree_filtered(root / "data/附件4/对齐版本", destination / "data/附件4/对齐版本")

    copy_tree_filtered(root / "paper_results/q3", destination / "paper_results/q3")

    # Make the two public Attachment-4 tables portable when the delivery folder
    # is moved to another machine.  The source project keeps absolute audit
    # paths; the package uses paths relative to its own root.
    packaged_attachment_dir = destination / "paper_results/q3/attachment4"
    packaged_csv = packaged_attachment_dir / "attachment4_predictions.csv"
    packaged_xlsx = packaged_attachment_dir / "attachment4_predictions.xlsx"
    attachment = pd.read_csv(packaged_csv, dtype={"sample_id": str})
    attachment["sample_id"] = attachment["sample_id"].str.zfill(2)
    attachment["source_video"] = attachment["sample_id"].map(
        lambda value: f"data/附件4/对齐版本/videos/{value}.mp4"
    )
    attachment["visual_key_frame"] = attachment["visual_key_frame"].fillna("").map(
        lambda value: "; ".join(
            f"paper_results/q3/attachment4/keyframes/{Path(part.strip()).name}"
            for part in str(value).split(";")
            if part.strip()
        )
    )
    attachment.to_csv(packaged_csv, index=False, encoding="utf-8-sig")
    attachment.to_excel(packaged_xlsx, index=False)
    copy_tree_filtered(
        root / "runs/final_test/exp183_neutral_authenticity_arbitrator",
        destination / "results/formal_exp183_test",
    )
    copy_tree_filtered(
        root / "runs/calibration/exp183_neutral_authenticity_valid_threshold_locked",
        destination / "results/selection_audit",
    )
    member_sources = [
        root / "runs/final_test_sources/exp183/01_exp027/test_predictions.csv",
        root / "runs/final_test_sources/exp183/02_exp029_macro/test_predictions.csv",
        root / "runs/final_test_sources/exp183/03_exp020/test_predictions.csv",
        root / "runs/final_test_sources/exp183/04_exp019/test_predictions.csv",
        root / "runs/final_test_sources/exp183/05_exp022/test_predictions.csv",
        root / "runs/experiments/exp001_text_vision_seed_20260924/test_predictions.csv",
        root / "runs/final_test_sources/exp183/07_exp063/test_predictions.csv",
    ]
    for index, path in enumerate(member_sources, start=1):
        copy_file(path, destination / "results/member_test_predictions" / f"member_{index:02d}.csv")

    manifest_rows = []
    for path in sorted(value for value in destination.rglob("*") if value.is_file()):
        manifest_rows.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "delivery": "SAN2 Question-3 complete delivery",
        "formal_model": "EXP183",
        "formal_test_accuracy": 0.7331499312242091,
        "formal_test_macro_f1": 0.6903469465620291,
        "config_policy": "one base config plus seven final member resolved configs only",
        "protected_source_modified": False,
        "file_count": len(manifest_rows),
        "total_bytes": sum(row["bytes"] for row in manifest_rows),
        "files": manifest_rows,
    }
    (destination / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.zip:
        zip_path = destination.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", allowZip64=True) as archive:
            for path in sorted(value for value in destination.rglob("*") if value.is_file()):
                suffix = path.suffix.lower()
                compression = (
                    zipfile.ZIP_STORED
                    if suffix in {".pt", ".pkl", ".mp4", ".xlsx", ".png", ".jpg", ".pdf"}
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(
                    path,
                    (destination.name / path.relative_to(destination)).as_posix(),
                    compress_type=compression,
                )
        checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
        checksum_path.write_text(f"{sha256(zip_path)}  {zip_path.name}\n", encoding="ascii")
        print(json.dumps({"delivery": str(destination), "zip": str(zip_path), "sha256": str(checksum_path)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"delivery": str(destination), "manifest": str(destination / 'DELIVERY_MANIFEST.json')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
