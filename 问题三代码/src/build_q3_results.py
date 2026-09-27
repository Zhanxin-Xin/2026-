from __future__ import annotations

import argparse
import json
from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from .data import CLASS_NAMES, MODALITIES
from .metrics import compute_metrics
from .q3_reporting import (
    PROBABILITY_COLUMNS,
    REQUIRED_ATTACHMENT4_COLUMNS,
    export_attachment4_deliverables,
    load_json,
    modality_summary_table,
    require_columns,
)
from .utils import save_json


REQUIRED_METRICS = ("accuracy", "macro_f1", "mae", "pearson")
SUPPLEMENTAL_METRICS = (
    "weighted_f1",
    "balanced_accuracy",
    "rmse",
    "regression_bias",
)
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"id": str}, keep_default_na=False)


def _metric_value(report: Mapping[str, Any], split: str, metric: str) -> float:
    values = report.get(split, {})
    if metric == "macro_f1" and "macro_f1" not in values:
        value = values.get("f1", np.nan)
    else:
        value = values.get(metric, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_seed_metrics(runs_root: str | Path) -> tuple[pd.DataFrame, Dict[str, Mapping[str, Any]]]:
    root = Path(runs_root)
    direct = root / "final_metrics.json"
    files = [direct] if direct.is_file() else sorted(root.glob("seed_*/final_metrics.json"))
    if not files:
        # This fallback also supports a debug/single-model parent directory.
        files = sorted(root.glob("*/final_metrics.json"))
    if not files:
        raise FileNotFoundError(
            f"No final_metrics.json, seed_*/final_metrics.json or */final_metrics.json under {runs_root}"
        )
    rows = []
    reports: Dict[str, Mapping[str, Any]] = {}
    for path in files:
        report = load_json(path)
        run = path.parent.name
        reports[run] = report
        row: Dict[str, Any] = {
            "run": run,
            "run_dir": str(path.parent.resolve()),
            "best_epoch": report.get("best_epoch"),
            "selection_score": report.get("best_selection_score"),
        }
        for split in ("valid", "test"):
            for metric in (*REQUIRED_METRICS, *SUPPLEMENTAL_METRICS):
                row[f"{split}_{metric}"] = _metric_value(report, split, metric)
        rows.append(row)
    return pd.DataFrame(rows), reports


def summarize_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("valid", "test"):
        for metric in (*REQUIRED_METRICS, *SUPPLEMENTAL_METRICS):
            column = f"{split}_{metric}"
            if column not in seed_metrics:
                continue
            values = pd.to_numeric(seed_metrics[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "runs": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _class_indices(values: Iterable[str], field: str) -> np.ndarray:
    labels = []
    for value in values:
        label = str(value)
        if label not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown {field} label: {label!r}; expected {CLASS_NAMES}")
        labels.append(LABEL_TO_INDEX[label])
    return np.asarray(labels, dtype=np.int64)


def metrics_from_predictions(frame: pd.DataFrame) -> Dict[str, Any]:
    required = (
        "true_label",
        "predicted_label",
        "true_intensity",
        "predicted_intensity",
        *PROBABILITY_COLUMNS,
    )
    require_columns(frame, required, "labeled prediction table")
    target = _class_indices(frame["true_label"], "true")
    probabilities = frame.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy()
    regression_target = frame["true_intensity"].astype(float).to_numpy()
    regression_prediction = frame["predicted_intensity"].astype(float).to_numpy()
    return compute_metrics(target, probabilities, regression_target, regression_prediction)


def bootstrap_required_metrics(
    frame: pd.DataFrame, samples: int, seed: int = 20260924
) -> pd.DataFrame:
    require_columns(
        frame,
        ("true_label", "predicted_label", "true_intensity", "predicted_intensity"),
        "labeled prediction table",
    )
    if samples <= 0:
        return pd.DataFrame(columns=["metric", "estimate", "ci95_low", "ci95_high"])
    rng = np.random.default_rng(seed)
    n = len(frame)
    if n < 2:
        raise ValueError("At least two validation samples are required for bootstrap intervals")
    true_class = _class_indices(frame["true_label"], "true")
    pred_class = _class_indices(frame["predicted_label"], "predicted")
    true_reg = frame["true_intensity"].astype(float).to_numpy()
    pred_reg = frame["predicted_intensity"].astype(float).to_numpy()

    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    values: Dict[str, list[float]] = {metric: [] for metric in REQUIRED_METRICS}
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        values["accuracy"].append(float(accuracy_score(true_class[indices], pred_class[indices])))
        values["macro_f1"].append(
            float(
                f1_score(
                    true_class[indices],
                    pred_class[indices],
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            )
        )
        values["mae"].append(float(mean_absolute_error(true_reg[indices], pred_reg[indices])))
        true_boot = true_reg[indices]
        pred_boot = pred_reg[indices]
        if np.std(true_boot) < 1e-12 or np.std(pred_boot) < 1e-12:
            values["pearson"].append(0.0)
        else:
            values["pearson"].append(float(np.corrcoef(true_boot, pred_boot)[0, 1]))

    point_metrics = metrics_from_predictions(frame)
    rows = []
    for metric in REQUIRED_METRICS:
        distribution = np.asarray(values[metric], dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "estimate": float(point_metrics[metric]),
                "ci95_low": float(np.quantile(distribution, 0.025)),
                "ci95_high": float(np.quantile(distribution, 0.975)),
                "bootstrap_samples": int(samples),
            }
        )
    return pd.DataFrame(rows)


def per_class_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for name in CLASS_NAMES:
        row = {"class": name}
        row.update(metrics["per_class"][name])
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_validation_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["classification_correct"] = result["true_label"] == result["predicted_label"]
    result["signed_regression_error"] = (
        result["predicted_intensity"].astype(float) - result["true_intensity"].astype(float)
    )
    result["absolute_regression_error"] = result["signed_regression_error"].abs()
    probabilities = result.loc[:, PROBABILITY_COLUMNS].astype(float).to_numpy()
    result["prediction_confidence"] = probabilities.max(axis=1)
    result["prediction_entropy"] = -(
        np.clip(probabilities, 1e-12, 1.0)
        * np.log(np.clip(probabilities, 1e-12, 1.0))
    ).sum(axis=1)
    return result


def error_attribution_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = enrich_validation_predictions(frame)
    by_class = (
        enriched.groupby("true_label", observed=False)
        .agg(
            samples=("id", "size"),
            classification_accuracy=("classification_correct", "mean"),
            mean_absolute_error=("absolute_regression_error", "mean"),
            mean_signed_error=("signed_regression_error", "mean"),
            mean_confidence=("prediction_confidence", "mean"),
        )
        .reindex(CLASS_NAMES)
        .reset_index()
    )
    bins = [-3.000001, -1.5, -0.5, 0.5, 1.5, 3.000001]
    labels = ["[-3,-1.5)", "[-1.5,-0.5)", "[-0.5,0.5)", "[0.5,1.5)", "[1.5,3]"]
    enriched["true_intensity_bin"] = pd.cut(
        enriched["true_intensity"].astype(float), bins=bins, labels=labels, include_lowest=True
    )
    by_intensity = (
        enriched.groupby("true_intensity_bin", observed=False)
        .agg(
            samples=("id", "size"),
            classification_accuracy=("classification_correct", "mean"),
            mean_absolute_error=("absolute_regression_error", "mean"),
            mean_signed_error=("signed_regression_error", "mean"),
        )
        .reset_index()
    )
    return by_class, by_intensity


def _plot_validation_results(
    frame: pd.DataFrame, metrics: Mapping[str, Any], output_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Validation confusion matrix")
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(3):
        for column in range(3):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    destination = output_dir / "validation_confusion_matrix.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(destination))

    true_value = frame["true_intensity"].astype(float)
    predicted_value = frame["predicted_intensity"].astype(float)
    fig, ax = plt.subplots(figsize=(5.7, 5.2))
    ax.scatter(true_value, predicted_value, s=18, alpha=0.65, color="#4776E6")
    ax.plot([-3, 3], [-3, 3], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("true sentiment intensity")
    ax.set_ylabel("predicted sentiment intensity")
    ax.set_title(f"Validation regression: MAE={metrics['mae']:.3f}, r={metrics['pearson']:.3f}")
    destination = output_dir / "validation_regression_scatter.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(destination))

    residual = predicted_value - true_value
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(residual, bins=20, color="#E6A147", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("prediction - ground truth")
    ax.set_ylabel("sample count")
    ax.set_title("Validation regression residuals")
    destination = output_dir / "validation_regression_residuals.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(str(destination))
    return written


def _flatten_best_metrics(metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for metric in (*REQUIRED_METRICS, *SUPPLEMENTAL_METRICS):
        if metric in metrics:
            rows.append({"metric": metric, "value": float(metrics[metric])})
    return pd.DataFrame(rows)


def _safe_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        dtype = result[column].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            result[column] = result[column].map(
                lambda value: (
                    value[:32700]
                    if isinstance(value, str) and len(value) > 32700
                    else value
                )
            )
    return result


def _write_excel(
    path: Path,
    tables: Mapping[str, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            _safe_excel_frame(frame).to_excel(writer, sheet_name=sheet_name[:31], index=False)
        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                cell.font = font
            for column_cells in worksheet.columns:
                width = min(
                    max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                    + 2,
                    40,
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = max(width, 10)


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_markdown_report(
    best_run: str,
    best_metrics: Mapping[str, Any],
    confidence_intervals: pd.DataFrame,
    seed_summary: pd.DataFrame,
    attachment_summary: Mapping[str, Any],
    output: Path,
) -> None:
    required_summary = seed_summary[
        (seed_summary["split"] == "valid")
        & (seed_summary["metric"].isin(REQUIRED_METRICS))
    ]
    lines = [
        "# 问题三成果自动汇总",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 1. 题目规定的验证集指标",
        "",
        f"代表模型：`{best_run}`。F1采用三分类Macro-F1。",
        "",
        "| 指标 | 代表模型 | 95%置信区间下限 | 95%置信区间上限 |",
        "|---|---:|---:|---:|",
    ]
    ci_by_metric = confidence_intervals.set_index("metric")
    for metric in REQUIRED_METRICS:
        row = ci_by_metric.loc[metric]
        lines.append(
            f"| {metric} | {best_metrics[metric]:.4f} | {row['ci95_low']:.4f} | {row['ci95_high']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 2. 多随机种子稳定性",
            "",
            *_markdown_table(required_summary, ("metric", "runs", "mean", "std", "min", "max")),
            "",
            "## 3. 附件4全量预测概况",
            "",
            f"附件4样本数：{attachment_summary['samples']}。附件4无标签，因此不能计算Accuracy、F1、MAE和Pearson。",
            "",
            "预测类别分布：",
            "",
        ]
    )
    for name, count in attachment_summary["class_distribution"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "平均模态作用程度：", ""])
    for modality, value in attachment_summary["mean_modality_importance"].items():
        lines.append(f"- {modality}: {value:.4f}")
    lines.extend(
        [
            "",
            "## 4. 题目要求与文件对应关系",
            "",
            "- 情感极性：`attachment4_classification_predictions.csv`",
            "- 情感强度回归：`attachment4_regression_predictions.csv`",
            "- 主要模态、作用程度和关键片段：`attachment4_modality_explanations.csv`",
            "- 附件4全量规定结果：`attachment4_required_results.csv`",
            "- 全字段结果：`attachment4_predictions_and_explanations.csv`",
            "- 局部证据：`attachment4_local_evidence.csv`",
            "- 典型样本解释卡：`explanation_cards/`",
            "- 视觉关键帧：`vision_keyframes/`",
            "- 验证集错误归因：本成果包的 `tables/validation_error_by_class.csv` 和相关图表。",
            "",
            "## 5. 图表",
            "",
            "![混淆矩阵](figures/validation_confusion_matrix.png)",
            "",
            "![回归散点图](figures/validation_regression_scatter.png)",
            "",
            "![附件4模态作用程度](figures/attachment4_modality_importance.png)",
            "",
            "本报告为程序自动汇总材料。论文正文仍需解释模型原理、训练策略、创新点、消融实验和典型案例。",
        ]
    )
    (output / "问题三成果说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a complete problem-3 metrics, figures and deliverables package"
    )
    parser.add_argument(
        "--runs-root",
        required=True,
        help="A single run directory, a directory containing seed_*/, or a parent containing run directories",
    )
    parser.add_argument(
        "--attachment4-dir",
        required=True,
        help="Inference output containing attachment4_predictions_and_explanations.csv",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--top-errors", type=int, default=50)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    attachment_dir = Path(args.attachment4_dir)
    output = Path(args.output)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_metrics, reports = collect_seed_metrics(runs_root)
    seed_summary = summarize_metrics(seed_metrics)
    seed_metrics.to_csv(tables_dir / "required_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(tables_dir / "required_metrics_summary.csv", index=False, encoding="utf-8-sig")

    scores = pd.to_numeric(seed_metrics["selection_score"], errors="coerce")
    best_index = int(scores.fillna(-np.inf).to_numpy().argmax())
    best_run = str(seed_metrics.iloc[best_index]["run"])
    best_run_dir = Path(str(seed_metrics.iloc[best_index]["run_dir"]))
    valid_path = best_run_dir / "best_valid_predictions.csv"
    if not valid_path.is_file():
        raise FileNotFoundError(f"Missing best validation predictions: {valid_path}")
    validation = _read_csv(valid_path)
    validation_metrics = metrics_from_predictions(validation)
    confidence_intervals = bootstrap_required_metrics(
        validation, args.bootstrap_samples
    )
    per_class = per_class_table(validation_metrics)
    confusion = pd.DataFrame(
        validation_metrics["confusion_matrix"],
        index=[f"true_{name}" for name in CLASS_NAMES],
        columns=[f"pred_{name}" for name in CLASS_NAMES],
    ).reset_index(names="true_class")
    enriched_validation = enrich_validation_predictions(validation)
    top_errors = enriched_validation.sort_values(
        ["classification_correct", "absolute_regression_error"],
        ascending=[True, False],
    ).head(max(args.top_errors, 0))
    error_by_class, error_by_intensity = error_attribution_tables(validation)

    _flatten_best_metrics(validation_metrics).to_csv(
        tables_dir / "best_validation_metrics.csv", index=False, encoding="utf-8-sig"
    )
    confidence_intervals.to_csv(
        tables_dir / "best_validation_metric_ci95.csv", index=False, encoding="utf-8-sig"
    )
    per_class.to_csv(tables_dir / "validation_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(tables_dir / "validation_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    enriched_validation.to_csv(
        tables_dir / "validation_predictions_with_errors.csv", index=False, encoding="utf-8-sig"
    )
    top_errors.to_csv(tables_dir / "validation_top_errors.csv", index=False, encoding="utf-8-sig")
    error_by_class.to_csv(
        tables_dir / "validation_error_by_class.csv", index=False, encoding="utf-8-sig"
    )
    error_by_intensity.to_csv(
        tables_dir / "validation_error_by_intensity.csv", index=False, encoding="utf-8-sig"
    )
    validation_plots = _plot_validation_results(validation, validation_metrics, figures_dir)

    attachment_path = attachment_dir / "attachment4_predictions_and_explanations.csv"
    if not attachment_path.is_file():
        raise FileNotFoundError(f"Missing attachment-4 predictions: {attachment_path}")
    attachment4 = _read_csv(attachment_path)
    attachment_exports = export_attachment4_deliverables(attachment4, attachment_dir)
    attachment4 = attachment_exports["frame"]
    attachment_required = attachment4.loc[:, list(REQUIRED_ATTACHMENT4_COLUMNS)]
    attachment_modality = modality_summary_table(attachment4)
    attachment_classes = pd.DataFrame(
        {
            "predicted_label": CLASS_NAMES,
            "count": [int((attachment4["predicted_label"] == name).sum()) for name in CLASS_NAMES],
        }
    )
    attachment_classes["ratio"] = attachment_classes["count"] / max(len(attachment4), 1)
    attachment_required.to_csv(
        tables_dir / "attachment4_required_results.csv", index=False, encoding="utf-8-sig"
    )
    attachment_modality.to_csv(
        tables_dir / "attachment4_modality_summary.csv", index=False, encoding="utf-8-sig"
    )
    attachment_classes.to_csv(
        tables_dir / "attachment4_class_distribution.csv", index=False, encoding="utf-8-sig"
    )

    # Copy the three attachment-4 overview figures into the consolidated package.
    for source in (attachment_dir / "summary_figures").glob("*.png"):
        destination = figures_dir / source.name
        destination.write_bytes(source.read_bytes())

    excel_tables = {
        "metrics_by_seed": seed_metrics,
        "metrics_summary": seed_summary,
        "best_valid_metrics": _flatten_best_metrics(validation_metrics),
        "metric_ci95": confidence_intervals,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "validation_errors": top_errors,
        "error_by_class": error_by_class,
        "error_by_intensity": error_by_intensity,
        "attachment4_results": attachment_required,
        "attachment4_modality": attachment_modality,
        "attachment4_classes": attachment_classes,
    }
    _write_excel(output / "问题三结果汇总.xlsx", excel_tables)

    build_markdown_report(
        best_run,
        validation_metrics,
        confidence_intervals,
        seed_summary,
        attachment_exports["summary"],
        output,
    )

    errors = []
    warnings = []
    if not np.isfinite([validation_metrics[name] for name in REQUIRED_METRICS]).all():
        errors.append("One or more required validation metrics are not finite")
    local_evidence_path = attachment_dir / "attachment4_local_evidence.csv"
    if not local_evidence_path.is_file():
        errors.append("attachment4_local_evidence.csv is missing")
    explanation_cards = list((attachment_dir / "explanation_cards").glob("*.png"))
    if not explanation_cards:
        warnings.append("No explanation cards were found")
    keyframes = list((attachment_dir / "vision_keyframes").glob("*.jpg"))
    if not keyframes:
        warnings.append("No vision keyframes were found; install FFmpeg and check video IDs")
    output_check = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "best_run": best_run,
        "required_validation_metrics": {
            name: float(validation_metrics[name]) for name in REQUIRED_METRICS
        },
        "attachment4_samples": int(len(attachment4)),
        "explanation_cards": len(explanation_cards),
        "vision_keyframes": len(keyframes),
    }
    save_json(output_check, output / "问题三成果检查.json")
    save_json(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "best_run": best_run,
            "runs_root": str(runs_root.resolve()),
            "attachment4_dir": str(attachment_dir.resolve()),
            "output": str(output.resolve()),
            "required_outputs": {
                "metrics": "tables/required_metrics_summary.csv",
                "metrics_ci95": "tables/best_validation_metric_ci95.csv",
                "error_analysis": "tables/validation_top_errors.csv",
                "attachment4_results": "tables/attachment4_required_results.csv",
                "excel_summary": "问题三结果汇总.xlsx",
                "written_report": "问题三成果说明.md",
                "check": "问题三成果检查.json",
            },
            "figures": [
                str(Path(path).relative_to(output)) for path in validation_plots
            ]
            + [
                str(path.relative_to(output)) for path in sorted(figures_dir.glob("attachment4_*.png"))
            ],
        },
        output / "问题三成果清单.json",
    )
    print(json.dumps(output_check, ensure_ascii=False, indent=2))
    print(f"Problem-3 result package written to {output.resolve()}")


if __name__ == "__main__":
    main()
