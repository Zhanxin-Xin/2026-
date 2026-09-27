"""Verify Question-3 paper assets and Attachment-4 final outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ATTACHMENT_COLUMNS = {
    "sample_id",
    "predicted_class",
    "p_negative",
    "p_neutral",
    "p_positive",
    "predicted_intensity",
    "text_importance",
    "audio_importance",
    "vision_importance",
    "dominant_modality",
    "text_key_positions",
    "audio_key_positions",
    "vision_key_positions",
    "key_text",
    "audio_time_range",
    "visual_time_range",
    "visual_key_frame",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    q3 = root / "paper_results" / "q3"
    errors: list[str] = []

    official_path = root / "runs/final_test/exp183_neutral_authenticity_arbitrator/final_test_metrics.json"
    if not official_path.is_file():
        official_path = root / "results/formal_exp183_test/final_test_metrics.json"
    official = json.loads(official_path.read_text("utf-8"))["test"]
    main_results = pd.read_csv(q3 / "main/main_test_results.csv")
    formal = main_results[main_results["model"] == "EXP183 frozen deployment"].iloc[0]
    for key in ("accuracy", "macro_f1", "mae", "pearson"):
        if not np.isclose(float(formal[key]), float(official[key]), atol=1e-12):
            errors.append(f"Main test metric mismatch: {key}")
    if set(main_results["evaluation_split"]) != {"test"}:
        errors.append("Main results contain a non-test row")

    attachment_csv = q3 / "attachment4/attachment4_predictions.csv"
    attachment_xlsx = q3 / "attachment4/attachment4_predictions.xlsx"
    csv = pd.read_csv(attachment_csv, dtype={"sample_id": str})
    xlsx = pd.read_excel(attachment_xlsx, dtype={"sample_id": str})
    csv["sample_id"] = csv["sample_id"].str.zfill(2)
    xlsx["sample_id"] = xlsx["sample_id"].str.zfill(2)
    if len(csv) != 20 or csv["sample_id"].nunique() != 20:
        errors.append("Attachment 4 must have 20 unique rows")
    missing = REQUIRED_ATTACHMENT_COLUMNS - set(csv.columns)
    if missing:
        errors.append(f"Attachment 4 missing columns: {sorted(missing)}")
    probabilities = csv[["p_negative", "p_neutral", "p_positive"]].to_numpy(float)
    importance = csv[["text_importance", "audio_importance", "vision_importance"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-8):
        errors.append("Attachment probabilities are invalid")
    if not np.isfinite(importance).all() or not np.allclose(importance.sum(1), 1, atol=1e-8):
        errors.append("Attachment modality importance is invalid")
    if not np.isfinite(csv["predicted_intensity"].to_numpy(float)).all():
        errors.append("Attachment intensities are invalid")
    if not ((csv["predicted_intensity"].astype(float) >= -3) & (csv["predicted_intensity"].astype(float) <= 3)).all():
        errors.append("Attachment intensities leave [-3,3]")
    if set(csv["predicted_class"]) - {"Negative", "Neutral", "Positive"}:
        errors.append("Attachment contains an unknown class")
    if set(csv["dominant_modality"]) - {"text", "audio", "vision"}:
        errors.append("Attachment contains an unknown dominant modality")
    for column in ("text_key_positions", "audio_key_positions", "vision_key_positions", "key_text"):
        if csv[column].fillna("").astype(str).str.strip().eq("").any():
            errors.append(f"Attachment local explanation has blanks: {column}")
    if list(csv.columns) != list(xlsx.columns) or csv["sample_id"].tolist() != xlsx["sample_id"].tolist():
        errors.append("Attachment CSV/XLSX structure mismatch")
    for path_text in csv["source_video"].fillna(""):
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not path_text or not candidate.is_file():
            errors.append(f"Missing Attachment video: {path_text}")
            break
    keyframes = list((q3 / "attachment4/keyframes").glob("*.jpg"))
    if len(keyframes) < 20:
        errors.append(f"Too few keyframes: {len(keyframes)}")

    figure_stems = {path.stem for path in (q3 / "figures").glob("*.pdf")}
    for stem in figure_stems:
        for suffix in (".pdf", ".svg", ".png"):
            if not (q3 / "figures" / f"{stem}{suffix}").is_file():
                errors.append(f"Missing figure variant: {stem}{suffix}")
    if len(figure_stems) != 18:
        errors.append(f"Expected 18 figure groups, found {len(figure_stems)}")
    tables = list((q3 / "tables").glob("tab_q3_*.tex"))
    if len(tables) != 15:
        errors.append(f"Expected 15 LaTeX tables, found {len(tables)}")

    required_docs = [
        "PROJECT_AUDIT_Q3.md",
        "MODEL_CODE_MAPPING_Q3.md",
        "EXPERIMENT_PLAN_Q3.md",
        "EXPERIMENT_RESULTS_Q3.md",
        "MODEL_ITERATION_LOG_Q3.md",
        "FINAL_Q3_AUDIT.md",
        "TODO_Q3_EXPERIMENTS.md",
        "问题三完整操作手册.md",
        "README_FIRST_Q3_中文.md",
    ]
    docs_root = root if (root / "PROJECT_AUDIT_Q3.md").is_file() else root / "docs"
    for name in required_docs:
        if not (docs_root / name).is_file():
            errors.append(f"Missing document: {name}")
    if not (q3 / "latex/question3_complete.tex").is_file():
        errors.append("Missing question3_complete.tex")
    if not (q3 / "latex/question3_standalone.pdf").is_file():
        errors.append("Missing compiled Question-3 PDF")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "formal_test": {key: official[key] for key in ("accuracy", "macro_f1", "mae", "pearson")},
        "attachment4_rows": int(len(csv)),
        "attachment4_unique_ids": int(csv["sample_id"].nunique()),
        "keyframe_count": len(keyframes),
        "figure_groups": len(figure_stems),
        "latex_tables": len(tables),
    }
    (q3 / "Q3_VERIFICATION_REPORT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
