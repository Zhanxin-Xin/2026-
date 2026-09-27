from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import CLASS_NAMES, MODALITIES
from .utils import save_json


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)

REQUIRED_ATTACHMENT4_COLUMNS = (
    "id",
    "predicted_label",
    "predicted_intensity",
    *PROBABILITY_COLUMNS,
    "main_modality",
    "text_importance",
    "audio_importance",
    "vision_importance",
    "text_evidence",
    "audio_evidence",
    "vision_evidence",
)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def add_prediction_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add confidence and entropy without changing any model prediction."""
    require_columns(frame, PROBABILITY_COLUMNS, "prediction table")
    result = frame.copy()
    probabilities = result.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    result["prediction_confidence"] = probabilities.max(axis=1)
    result["prediction_entropy"] = -(probabilities * np.log(probabilities)).sum(axis=1)
    return result


def export_labeled_prediction_views(
    frame: pd.DataFrame, output_dir: str | Path, split: str
) -> pd.DataFrame:
    """Write explicit classification and regression files for an attachment-2 split."""
    require_columns(
        frame,
        (
            "id",
            "true_label",
            "predicted_label",
            "true_intensity",
            "predicted_intensity",
            *PROBABILITY_COLUMNS,
        ),
        f"attachment-2 {split} predictions",
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = add_prediction_diagnostics(frame)
    result["classification_correct"] = result["true_label"] == result["predicted_label"]
    result["signed_regression_error"] = (
        result["predicted_intensity"].astype(float)
        - result["true_intensity"].astype(float)
    )
    result["absolute_regression_error"] = result["signed_regression_error"].abs()
    result.loc[
        :,
        [
            "id",
            "true_label",
            "predicted_label",
            *PROBABILITY_COLUMNS,
            "prediction_confidence",
            "prediction_entropy",
            "classification_correct",
        ],
    ].to_csv(
        output / f"{split}_classification_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.loc[
        :,
        [
            "id",
            "true_intensity",
            "predicted_intensity",
            "signed_regression_error",
            "absolute_regression_error",
        ],
    ].to_csv(
        output / f"{split}_regression_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def _finite_stats(values: Sequence[float] | np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def attachment4_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    require_columns(frame, REQUIRED_ATTACHMENT4_COLUMNS, "attachment-4 predictions")
    class_counts = frame["predicted_label"].value_counts()
    modality_counts = frame["main_modality"].value_counts()
    summary: Dict[str, Any] = {
        "samples": int(len(frame)),
        "class_distribution": {
            name: int(class_counts.get(name, 0)) for name in CLASS_NAMES
        },
        "predicted_intensity": _finite_stats(frame["predicted_intensity"]),
        "main_modality_distribution": {
            modality: int(modality_counts.get(modality, 0)) for modality in MODALITIES
        },
        "mean_modality_importance": {
            modality: float(frame[f"{modality}_importance"].mean())
            for modality in MODALITIES
        },
    }
    for column in (
        "prediction_confidence",
        "prediction_entropy",
        "predicted_class_probability_std",
        "predicted_intensity_std",
        "regression_conservation_error",
        "classification_conservation_error",
    ):
        if column in frame.columns:
            summary[column] = _finite_stats(frame[column])
    return summary


def modality_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample_count = max(len(frame), 1)
    for modality in MODALITIES:
        importance = frame[f"{modality}_importance"].astype(float)
        row: Dict[str, Any] = {
            "modality": modality,
            "mean_importance": float(importance.mean()),
            "std_importance": float(importance.std(ddof=1)) if len(frame) > 1 else 0.0,
            "median_importance": float(importance.median()),
            "main_modality_count": int((frame["main_modality"] == modality).sum()),
            "main_modality_ratio": float((frame["main_modality"] == modality).sum() / sample_count),
        }
        signed = f"{modality}_signed_contribution"
        ablation = f"{modality}_ablation_importance"
        if signed in frame.columns:
            row["mean_signed_contribution"] = float(frame[signed].astype(float).mean())
            row["mean_abs_signed_contribution"] = float(
                frame[signed].astype(float).abs().mean()
            )
        if ablation in frame.columns:
            row["mean_ablation_importance"] = float(
                frame[ablation].astype(float).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_attachment4_summary(
    frame: pd.DataFrame, modality_table: pd.DataFrame, output_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    class_counts = [int((frame["predicted_label"] == name).sum()) for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(CLASS_NAMES, class_counts, color=["#C94C4C", "#8A8A8A", "#3E8E67"])
    ax.set_ylabel("sample count")
    ax.set_title("Attachment 4 predicted class distribution")
    for index, value in enumerate(class_counts):
        ax.text(index, value, str(value), ha="center", va="bottom")
    destination = output_dir / "attachment4_class_distribution.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(destination))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(frame["predicted_intensity"].astype(float), bins=15, color="#4776E6", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlim(-3.0, 3.0)
    ax.set_xlabel("predicted sentiment intensity")
    ax.set_ylabel("sample count")
    ax.set_title("Attachment 4 regression output distribution")
    destination = output_dir / "attachment4_intensity_distribution.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(destination))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(
        modality_table["modality"],
        modality_table["mean_importance"],
        yerr=modality_table["std_importance"],
        capsize=4,
        color=["#4776E6", "#E6A147", "#4AA564"],
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean joint importance")
    ax.set_title("Attachment 4 modality importance")
    destination = output_dir / "attachment4_modality_importance.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(destination))
    return written


def export_attachment4_deliverables(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Export explicit problem-3 classification, regression and explanation files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = add_prediction_diagnostics(frame)
    require_columns(frame, REQUIRED_ATTACHMENT4_COLUMNS, "attachment-4 predictions")

    required_path = output / "attachment4_required_results.csv"
    classification_path = output / "attachment4_classification_predictions.csv"
    regression_path = output / "attachment4_regression_predictions.csv"
    modality_path = output / "attachment4_modality_explanations.csv"
    class_distribution_path = output / "attachment4_class_distribution.csv"
    modality_summary_path = output / "attachment4_modality_summary.csv"
    summary_path = output / "attachment4_descriptive_summary.json"

    optional_required = [
        column
        for column in (
            "prediction_confidence",
            "prediction_entropy",
            "predicted_class_probability_std",
            "predicted_intensity_std",
            "text_ablation_importance",
            "audio_ablation_importance",
            "vision_ablation_importance",
            "video_found",
            "source_video",
            "video_duration_sec",
            "vision_keyframes",
        )
        if column in frame.columns
    ]
    frame.loc[:, [*REQUIRED_ATTACHMENT4_COLUMNS, *optional_required]].to_csv(
        required_path, index=False, encoding="utf-8-sig"
    )
    frame.loc[
        :,
        [
            "id",
            "predicted_label",
            *PROBABILITY_COLUMNS,
            "prediction_confidence",
            "prediction_entropy",
            *(
                ["predicted_class_probability_std"]
                if "predicted_class_probability_std" in frame.columns
                else []
            ),
        ],
    ].to_csv(classification_path, index=False, encoding="utf-8-sig")
    regression_columns = ["id", "predicted_intensity"]
    if "predicted_intensity_std" in frame.columns:
        regression_columns.append("predicted_intensity_std")
    frame.loc[:, regression_columns].to_csv(
        regression_path, index=False, encoding="utf-8-sig"
    )

    modality_columns = [
        "id",
        "main_modality",
        *[f"{modality}_importance" for modality in MODALITIES],
        *[
            column
            for modality in MODALITIES
            for column in (
                f"{modality}_signed_contribution",
                f"{modality}_ablation_importance",
            )
            if column in frame.columns
        ],
        *[f"{modality}_evidence" for modality in MODALITIES],
    ]
    frame.loc[:, modality_columns].to_csv(
        modality_path, index=False, encoding="utf-8-sig"
    )

    class_distribution = pd.DataFrame(
        {
            "predicted_label": CLASS_NAMES,
            "count": [int((frame["predicted_label"] == name).sum()) for name in CLASS_NAMES],
        }
    )
    class_distribution["ratio"] = class_distribution["count"] / max(len(frame), 1)
    class_distribution.to_csv(class_distribution_path, index=False, encoding="utf-8-sig")
    modality_table = modality_summary_table(frame)
    modality_table.to_csv(modality_summary_path, index=False, encoding="utf-8-sig")
    descriptive = attachment4_summary(frame)
    save_json(descriptive, summary_path)

    plot_paths = (
        _plot_attachment4_summary(frame, modality_table, output / "summary_figures")
        if make_plots
        else []
    )
    manifest: Dict[str, Any] = {
        "purpose": "Problem 3 attachment-4 required outputs",
        "sample_count": int(len(frame)),
        "files": {
            "full_prediction_and_explanation": "attachment4_predictions_and_explanations.csv",
            "required_results": required_path.name,
            "classification_output": classification_path.name,
            "regression_output": regression_path.name,
            "modality_and_evidence": modality_path.name,
            "local_evidence": "attachment4_local_evidence.csv",
            "class_distribution": class_distribution_path.name,
            "modality_summary": modality_summary_path.name,
            "descriptive_summary": summary_path.name,
            "explanation_cards": "explanation_cards/",
            "vision_keyframes": "vision_keyframes/",
            "video_mapping": "attachment4_video_mapping.csv",
            "summary_figures": [str(Path(path).relative_to(output)) for path in plot_paths],
        },
        "competition_metric_note": (
            "Attachment 4 is unlabeled, so Accuracy, F1, MAE and Pearson must be "
            "reported on the labeled attachment-2 validation/test split, not attachment 4."
        ),
    }
    save_json(manifest, output / "problem3_output_manifest.json")
    return {"frame": frame, "summary": descriptive, "manifest": manifest}


def load_json(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
