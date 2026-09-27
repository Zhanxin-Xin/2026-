from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.metrics import compute_metrics
from src.build_q3_results import collect_seed_metrics
from src.q3_reporting import export_attachment4_deliverables


CLASS_NAMES = np.asarray(["Negative", "Neutral", "Positive"])


def labeled_predictions(offset: int = 0) -> tuple[pd.DataFrame, dict]:
    target = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    predicted = np.asarray([0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2])
    if offset:
        predicted = np.roll(predicted, offset)
    probabilities = np.full((len(target), 3), 0.08, dtype=np.float64)
    probabilities[np.arange(len(target)), predicted] = 0.84
    true_intensity = np.asarray([-2.5, 0.0, 2.4, -1.8, 0.0, 1.5, -0.8, 0.0, 0.7, -2.8, 0.0, 2.8])
    predicted_intensity = true_intensity * 0.82 + np.linspace(-0.12, 0.12, len(target))
    frame = pd.DataFrame(
        {
            "id": [f"sample_{index:02d}" for index in range(len(target))],
            "predicted_label": CLASS_NAMES[predicted],
            "negative_probability": probabilities[:, 0],
            "neutral_probability": probabilities[:, 1],
            "positive_probability": probabilities[:, 2],
            "predicted_intensity": predicted_intensity,
            "true_label": CLASS_NAMES[target],
            "true_intensity": true_intensity,
        }
    )
    metrics = compute_metrics(target, probabilities, true_intensity, predicted_intensity)
    return frame, metrics


def attachment4_predictions() -> pd.DataFrame:
    rows = []
    for index in range(6):
        label_index = index % 3
        probabilities = np.full(3, 0.1)
        probabilities[label_index] = 0.8
        importance = np.roll(np.asarray([0.55, 0.30, 0.15]), index % 3)
        main_modality = ("text", "audio", "vision")[int(importance.argmax())]
        row = {
            "id": f"{index + 1:02d}",
            "predicted_label": CLASS_NAMES[label_index],
            "negative_probability": probabilities[0],
            "neutral_probability": probabilities[1],
            "positive_probability": probabilities[2],
            "predicted_intensity": float(-2.0 + 0.8 * index),
            "main_modality": main_modality,
            "text_importance": importance[0],
            "audio_importance": importance[1],
            "vision_importance": importance[2],
            "text_signed_contribution": 0.1,
            "audio_signed_contribution": -0.05,
            "vision_signed_contribution": 0.02,
            "text_ablation_importance": importance[0],
            "audio_ablation_importance": importance[1],
            "vision_ablation_importance": importance[2],
            "text_evidence": "[]",
            "audio_evidence": "[]",
            "vision_evidence": "[]",
            "predicted_class_probability_std": 0.01,
            "predicted_intensity_std": 0.04,
            "regression_conservation_error": 1e-7,
            "classification_conservation_error": 1e-7,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_complete_q3_result_package() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runs_root = root / "runs"
        for index, seed in enumerate((20260924, 20260925)):
            run = runs_root / f"seed_{seed}"
            run.mkdir(parents=True)
            predictions, metrics = labeled_predictions(offset=index)
            predictions.to_csv(run / "best_valid_predictions.csv", index=False)
            predictions.to_csv(run / "test_predictions.csv", index=False)
            report = {
                "best_epoch": 10 + index,
                "best_selection_score": 0.8 - 0.1 * index,
                "valid": metrics,
                "test": metrics,
            }
            (run / "final_metrics.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

        attachment_dir = root / "attachment4"
        attachment_dir.mkdir()
        attachment = attachment4_predictions()
        attachment.to_csv(
            attachment_dir / "attachment4_predictions_and_explanations.csv", index=False
        )
        pd.DataFrame(
            {
                "id": ["01"],
                "modality": ["text"],
                "evidence_type": ["modality"],
                "position": [1],
                "importance": [1.0],
                "signed_regression_contribution": [0.2],
            }
        ).to_csv(attachment_dir / "attachment4_local_evidence.csv", index=False)
        export_attachment4_deliverables(attachment, attachment_dir, make_plots=False)

        direct_metrics, _ = collect_seed_metrics(runs_root / "seed_20260924")
        assert len(direct_metrics) == 1
        assert direct_metrics.iloc[0]["run"] == "seed_20260924"

        output = root / "problem3_package"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.build_q3_results",
                "--runs-root",
                str(runs_root),
                "--attachment4-dir",
                str(attachment_dir),
                "--output",
                str(output),
                "--bootstrap-samples",
                "40",
                "--top-errors",
                "5",
            ],
            cwd=PROJECT,
            check=True,
        )
        assert (output / "问题三结果汇总.xlsx").is_file()
        assert (output / "问题三成果说明.md").is_file()
        assert (output / "tables" / "best_validation_metric_ci95.csv").is_file()
        assert (output / "figures" / "validation_confusion_matrix.png").is_file()
        check = json.loads((output / "问题三成果检查.json").read_text(encoding="utf-8"))
        assert check["passed"] is True
        assert check["attachment4_samples"] == 6


if __name__ == "__main__":
    test_complete_q3_result_package()
    print("All problem-3 reporting tests passed.")
