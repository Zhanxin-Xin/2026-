"""Build the traceable Question-3 result tables, figures, and attachment exports.

All primary performance graphics use the frozen EXP183 test predictions.  The
validation split is read only for the documented model-selection audit and the
classification/regression consistency threshold; it is never presented as the
final performance result.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


CLASS_NAMES = ("Negative", "Neutral", "Positive")
MODALITIES = ("text", "audio", "vision")
COLORS = {"text": "#3264A8", "audio": "#D9822B", "vision": "#2D8A62"}
CLASS_COLORS = {"Negative": "#C44E52", "Neutral": "#777777", "Positive": "#4C72B0"}
METRIC_COLUMNS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "mae",
    "rmse",
    "regression_bias",
    "pearson",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(name: str, metrics: dict[str, Any], source: str, note: str = "") -> dict[str, Any]:
    return {
        "model": name,
        "evaluation_split": "test",
        **{key: metrics.get(key) for key in METRIC_COLUMNS},
        "source": source,
        "note": note,
    }


def save_figure(fig: plt.Figure, figures: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            figures / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.7)


def add_bar_labels(ax: plt.Axes, decimals: int = 3) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if math.isfinite(height):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height,
                f"{height:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def latex_table(
    frame: pd.DataFrame,
    caption: str,
    label: str,
    path: Path,
    columns: list[str] | None = None,
    headers: list[str] | None = None,
    float_columns: Iterable[str] = (),
) -> None:
    table = frame if columns is None else frame.loc[:, columns]
    headers = list(table.columns) if headers is None else headers
    float_columns = set(float_columns)
    align = "l" + "r" * (len(table.columns) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        values = []
        for column in table.columns:
            value = row[column]
            if pd.isna(value):
                values.append("--")
            elif column in float_columns:
                values.append(f"{float(value):.4f}")
            else:
                values.append(latex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def draw_framework(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    labels = [
        "Aligned inputs\nT 50x768 / A 50x74\nV 50x35",
        "7 heterogeneous\nfrozen members",
        "Vote-probability\nconsensus",
        "Neutral-authenticity\narbitrator",
        "Class + intensity\nprediction",
        "Occlusion + deletion\nexplanation",
    ]
    colors = ["#E8EFF8", "#DDEBE5", "#F6E7D5", "#F2DFE7", "#E6E1F2", "#FFF3C7"]
    xs = np.linspace(0.2, 11.8, len(labels))
    for index, (x, label, color) in enumerate(zip(xs, labels, colors)):
        box = FancyBboxPatch(
            (x, 1.35),
            1.85,
            1.25,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + 0.925, 1.975, label, ha="center", va="center", fontsize=9)
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1], 1.975),
                xytext=(x + 1.85, 1.975),
                arrowprops=dict(arrowstyle="->", color="#444444", linewidth=1.2),
            )
    ax.text(7, 3.35, "Frozen EXP183 prediction-and-explanation pipeline", ha="center", fontsize=13, weight="bold")
    ax.text(7, 0.55, "Validation freezes the arbitrator threshold; test is used once for reporting and post-freeze explanation only.", ha="center", fontsize=9, color="#555555")
    save_figure(fig, figures, "fig_q3_framework")


def draw_architecture(figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
    axes[0].set_title("Base explainable C³-HAFusion", fontsize=12, weight="bold")
    left = [
        ("Aligned T/A/V", 8.8),
        ("Linear + LN + GELU + dropout", 7.5),
        ("Multi-scale temporal encoder", 6.2),
        ("Sparse local evidence pooling", 4.9),
        ("Cross-context and conflict features", 3.6),
        ("Reliability-modulated adaptive fusion", 2.3),
        ("Classification / bounded regression", 1.0),
    ]
    for index, (label, y) in enumerate(left):
        box = FancyBboxPatch((1.2, y - 0.42), 7.6, 0.84, boxstyle="round,pad=0.03", facecolor="#EAF1F8", edgecolor="#315A7D")
        axes[0].add_patch(box)
        axes[0].text(5, y, label, ha="center", va="center", fontsize=9)
        if index < len(left) - 1:
            axes[0].annotate("", xy=(5, left[index + 1][1] + 0.42), xytext=(5, y - 0.42), arrowprops=dict(arrowstyle="->", color="#555555"))
    axes[1].set_title("Final EXP183 deployment graph", fontsize=12, weight="bold")
    member_labels = ["DeBERTa routers ×4", "Twitter-RoBERTa fusion", "C³-HAFusion", "NLI semantic expert"]
    member_y = [8.5, 7.3, 6.1, 4.9]
    for label, y in zip(member_labels, member_y):
        box = FancyBboxPatch((0.5, y - 0.38), 4.1, 0.76, boxstyle="round,pad=0.03", facecolor="#DDEBE5", edgecolor="#2D6A4F")
        axes[1].add_patch(box)
        axes[1].text(2.55, y, label, ha="center", va="center", fontsize=9)
        axes[1].annotate("", xy=(5.4, 6.7), xytext=(4.6, y), arrowprops=dict(arrowstyle="->", color="#666666"))
    for label, y, color in [
        ("Equal vote + probability consensus", 6.7, "#F6E7D5"),
        ("Neutral-authenticity arbitrator", 4.7, "#F2DFE7"),
        ("Final class probabilities", 2.7, "#E6E1F2"),
        ("Mean member intensity", 1.2, "#FFF3C7"),
    ]:
        box = FancyBboxPatch((5.4, y - 0.42), 4.1, 0.84, boxstyle="round,pad=0.03", facecolor=color, edgecolor="#555555")
        axes[1].add_patch(box)
        axes[1].text(7.45, y, label, ha="center", va="center", fontsize=9)
    axes[1].annotate("", xy=(7.45, 5.12), xytext=(7.45, 6.28), arrowprops=dict(arrowstyle="->", color="#555555"))
    axes[1].annotate("", xy=(7.45, 3.12), xytext=(7.45, 4.28), arrowprops=dict(arrowstyle="->", color="#555555"))
    axes[1].annotate("", xy=(7.45, 1.62), xytext=(7.45, 2.28), arrowprops=dict(arrowstyle="->", color="#555555"))
    fig.suptitle("Architecture boundary: the base network and the deployed ensemble are distinct systems", fontsize=13)
    save_figure(fig, figures, "fig_q3_architecture")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    q3 = root / "paper_results" / "q3"
    figures = q3 / "figures"
    tables = q3 / "tables"
    for directory in (
        q3 / "main",
        q3 / "modalities",
        q3 / "fusion",
        q3 / "ablation",
        q3 / "multitask",
        q3 / "seeds",
        q3 / "sensitivity",
        q3 / "interpretability",
        q3 / "cases",
        q3 / "attachment4",
        figures,
        tables,
        q3 / "latex",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    labels = pd.read_excel(root / "data" / "label.xlsx")
    labels["sample_id"] = labels["video_id"].astype(str) + "$_$" + labels["clip_id"].astype(str)
    labels["mode"] = labels["mode"].astype(str).str.lower()
    labels["annotation"] = labels["annotation"].astype(str).str.title()

    exp183_report = read_json(root / "runs" / "final_test" / "exp183_neutral_authenticity_arbitrator" / "final_test_metrics.json")
    exp183 = exp183_report["test"]
    parent = exp183_report["test_parent"]
    final_predictions = pd.read_csv(root / "runs" / "final_test" / "exp183_neutral_authenticity_arbitrator" / "test_predictions.csv")
    final_predictions["id"] = final_predictions["id"].astype(str)
    valid_predictions = pd.read_csv(root / "delivery" / "SAN2_EXP183_complete" / "deployment" / "valid_predictions.csv")

    # Main test-only performance table.
    baseline_report = read_json(root / "runs" / "c3_hafusion_single" / "seed_20260924" / "final_metrics.json")
    main_results = pd.DataFrame(
        [
            metric_row("Single C3-HAFusion", baseline_report["test"], "runs/c3_hafusion_single/seed_20260924/final_metrics.json"),
            metric_row("EXP169 parent consensus", parent, "EXP183 final_test_metrics.json:test_parent", "Post-freeze descriptive reference; not selected by test"),
            metric_row("EXP183 frozen deployment", exp183, "EXP183 final_test_metrics.json:test", "Formal final result frozen before test"),
        ]
    )
    main_results.to_csv(q3 / "main" / "main_test_results.csv", index=False)

    # Validation-only selection audit is deliberately separate from the main table.
    calibration = read_json(root / "runs" / "calibration" / "exp183_neutral_authenticity_valid_threshold_locked" / "final_metrics.json")
    selection_rows = []
    for split_name, payload in (("train_grouped_oof", calibration["oof"]), ("validation", calibration["valid"]), ("test", exp183)):
        selection_rows.append({"split": split_name, **{key: payload.get(key) for key in METRIC_COLUMNS}})
    pd.DataFrame(selection_rows).to_csv(q3 / "main" / "selection_and_final_audit.csv", index=False)

    # Per-class and confusion matrix, strictly on test.
    per_class = pd.DataFrame(
        [{"class": name, **exp183["per_class"][name]} for name in CLASS_NAMES]
    )
    per_class.insert(0, "evaluation_split", "test")
    per_class.to_csv(q3 / "main" / "test_per_class_metrics.csv", index=False)
    confusion = pd.DataFrame(exp183["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
    confusion.to_csv(q3 / "main" / "test_confusion_matrix.csv", index_label="true_class")

    # Valid-fitted epsilon for the auxiliary classification-regression consistency audit.
    label_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    valid_true = valid_predictions["true_label"].map(label_index).to_numpy(int)
    valid_reg = valid_predictions["predicted_intensity"].to_numpy(float)
    candidates = np.linspace(0.0, 1.0, 1001)
    scores = []
    for epsilon in candidates:
        inferred = np.where(valid_reg < -epsilon, 0, np.where(valid_reg > epsilon, 2, 1))
        scores.append(float(np.mean(inferred == valid_true)))
    best_score = max(scores)
    epsilon = float(candidates[next(index for index, value in enumerate(scores) if value == best_score)])
    test_reg = final_predictions["predicted_intensity"].to_numpy(float)
    regression_class = np.where(test_reg < -epsilon, 0, np.where(test_reg > epsilon, 2, 1))
    classification_class = final_predictions["predicted_label"].map(label_index).to_numpy(int)
    test_true = final_predictions["true_label"].map(label_index).to_numpy(int)
    consistency = {
        "epsilon_selected_on_validation": epsilon,
        "validation_regression_implied_class_accuracy": best_score,
        "test_classification_regression_consistency_rate": float(np.mean(regression_class == classification_class)),
        "test_regression_implied_class_accuracy": float(np.mean(regression_class == test_true)),
        "test_used_to_select_epsilon": False,
    }
    (q3 / "main" / "classification_regression_consistency.json").write_text(json.dumps(consistency, ensure_ascii=False, indent=2), encoding="utf-8")

    # Baseline modality combinations, evaluated on test after the checkpoint was frozen.
    combo_json = read_json(q3 / "modalities" / "baseline_exp000_test_modality_combinations.json")
    combo_names = {
        "text": "T",
        "audio": "A",
        "vision": "V",
        "text_audio": "T+A",
        "text_vision": "T+V",
        "audio_vision": "A+V",
        "all": "T+A+V",
    }
    modality_results = pd.DataFrame(
        [
            {
                "modalities": combo_names[key],
                "evaluation_split": "test",
                **{metric: payload.get(metric) for metric in METRIC_COLUMNS},
                "source": "post-freeze intervention on EXP000 HAFusion",
            }
            for key, payload in combo_json.items()
        ]
    )
    modality_results.to_csv(q3 / "modalities" / "modality_results.csv", index=False)

    # Real, already-run architecture/fusion routes. These are descriptive rather than a
    # perfectly matched-capacity benchmark, and that limitation is retained in the table.
    route_specs = [
        ("Single C3-HAFusion", root / "runs/c3_hafusion_single/seed_20260924/final_metrics.json", "test"),
        ("Dynamic-router trimodal", root / "runs/final_test_sources/exp183/01_exp027/final_metrics.json", "test"),
        ("Dynamic-router macro checkpoint", root / "runs/final_test_sources/exp183/02_exp029_macro/final_metrics.json", "test"),
        ("Trimodal prototype fusion", root / "runs/final_test_sources/exp183/03_exp020/final_metrics.json", "test"),
        ("Context trimodal prototype", root / "runs/final_test_sources/exp183/04_exp019/final_metrics.json", "test"),
        ("Twitter-RoBERTa multiview", root / "runs/final_test_sources/exp183/05_exp022/final_metrics.json", "test"),
        ("Text-vision C3-HAFusion", root / "runs/experiments/exp001_text_vision_seed_20260924/final_metrics.json", "test"),
        ("NLI semantic expert", root / "runs/final_test_sources/exp183/07_exp063/final_metrics.json", "test"),
        ("EXP009 probability average", root / "runs/ensembles/exp009_ensemble_001_005_007/final_metrics.json", "test"),
    ]
    route_rows = []
    for name, path, key in route_specs:
        payload = read_json(path)
        if key in payload:
            route_rows.append(metric_row(name, payload[key], str(path.relative_to(root)), "Historical frozen run"))
    route_rows.append(metric_row("EXP183 heterogeneous consensus + arbitrator", exp183, "EXP183 frozen test", "Formal final system"))
    fusion_results = pd.DataFrame(route_rows)
    fusion_results.to_csv(q3 / "fusion" / "fusion_results.csv", index=False)

    # Test-set component interventions for the actual final deployment.
    occlusion_dir = q3 / "interpretability" / "exp183_test_occlusion"
    occlusion = pd.read_csv(occlusion_dir / "occlusion_results.csv")
    importance = pd.read_csv(occlusion_dir / "test_modality_importance.csv")
    occlusion.to_csv(q3 / "interpretability" / "occlusion_results.csv", index=False)
    importance.to_csv(q3 / "interpretability" / "modality_importance.csv", index=False)
    ablation_rows = [metric_row("Full EXP183", exp183, "Frozen test")]
    ablation_rows.append(metric_row("Without Neutral arbitrator (parent)", parent, "Frozen parent test", "Descriptive; not chosen after test"))
    for _, row in occlusion[occlusion["condition"] != "full"].iterrows():
        ablation_rows.append(
            {
                "model": row["condition"].replace("without_", "Without ").title(),
                "evaluation_split": "test",
                **{key: row.get(key) for key in METRIC_COLUMNS},
                "source": "Frozen EXP183 intervention",
                "note": "Post-freeze zero/blank occlusion",
            }
        )
    ablation_results = pd.DataFrame(ablation_rows)
    ablation_results.to_csv(q3 / "ablation" / "ablation_results.csv", index=False)

    # Multi-task evidence: exact full versus the already-run classification-focused loss.
    exp003 = read_json(root / "runs/experiments/exp003_cls_focused_seed_20260924/final_metrics.json")["test"]
    multitask = pd.DataFrame(
        [
            metric_row("Balanced classification + regression", baseline_report["test"], "Single C3-HAFusion"),
            metric_row("Classification-focused joint loss", exp003, "EXP003", "Regression weight 0.5, Pearson weight 0.1; not pure classification-only"),
        ]
    )
    multitask.to_csv(q3 / "multitask" / "multitask_results.csv", index=False)

    seeds = pd.read_csv(root / "runs/multi_seed/focal_gamma1/multi_seed_metrics.csv")
    seed_results = seeds[[column for column in seeds.columns if column.startswith("test_") or column in ("run", "best_epoch")]].copy()
    seed_results.insert(1, "evaluation_split", "test")
    seed_results.to_csv(q3 / "seeds" / "seed_results.csv", index=False)

    sensitivity_specs = [
        ("Base T+V", "EXP001", root / "runs/experiments/exp001_text_vision_seed_20260924/final_metrics.json"),
        ("Dropout 0.30", "EXP004", root / "runs/experiments/exp004_dropout030_seed_20260924/final_metrics.json"),
        ("Class-weight power 0.75", "EXP005", root / "runs/experiments/exp005_class_weight075_seed_20260924/final_metrics.json"),
        ("Focal gamma 1.0", "EXP008", root / "runs/experiments/exp008_focal_gamma1_seed_20260924/final_metrics.json"),
    ]
    sensitivity = pd.DataFrame(
        [
            {
                "setting": name,
                "experiment": experiment,
                "evaluation_split": "test",
                **{key: read_json(path)["test"].get(key) for key in METRIC_COLUMNS},
            }
            for name, experiment, path in sensitivity_specs
        ]
    )
    sensitivity.to_csv(q3 / "sensitivity" / "sensitivity_results.csv", index=False)

    # Test-conditioned modality summaries.
    class_importance = importance.groupby("true_label")[[f"{m}_importance" for m in MODALITIES]].mean().reindex(CLASS_NAMES).reset_index()
    class_importance.to_csv(q3 / "interpretability" / "modality_importance_by_true_class.csv", index=False)
    bin_edges = [-np.inf, -1.0, -1e-12, 1e-12, 1.0, np.inf]
    bin_labels = ["Strong negative", "Weak negative", "Neutral", "Weak positive", "Strong positive"]
    importance["intensity_bin"] = pd.cut(importance["true_intensity"], bins=bin_edges, labels=bin_labels, include_lowest=True)
    intensity_importance = importance.groupby("intensity_bin", observed=False)[[f"{m}_importance" for m in MODALITIES]].mean().reset_index()
    intensity_importance.to_csv(q3 / "interpretability" / "modality_importance_by_true_intensity.csv", index=False)
    dominant = importance["dominant_modality"].value_counts().reindex(MODALITIES, fill_value=0).rename_axis("modality").reset_index(name="count")
    dominant["fraction"] = dominant["count"] / len(importance)
    dominant.to_csv(q3 / "interpretability" / "dominant_modality_distribution.csv", index=False)

    # Error analysis on frozen test predictions.
    errors = final_predictions.copy()
    errors["classification_correct"] = errors["predicted_label"] == errors["true_label"]
    errors["absolute_regression_error"] = np.abs(errors["predicted_intensity"] - errors["true_intensity"])
    errors["error_type"] = "Correct"
    wrong = ~errors["classification_correct"]
    errors.loc[wrong & ((errors["true_label"] == "Neutral") | (errors["predicted_label"] == "Neutral")), "error_type"] = "Neutral boundary"
    errors.loc[wrong & (errors["true_label"] == "Negative") & (errors["predicted_label"] == "Positive"), "error_type"] = "Polarity reversal"
    errors.loc[wrong & (errors["true_label"] == "Positive") & (errors["predicted_label"] == "Negative"), "error_type"] = "Polarity reversal"
    errors.to_csv(q3 / "main" / "test_error_attribution.csv", index=False)
    error_summary = errors.groupby("error_type", as_index=False).agg(count=("id", "count"), mean_absolute_regression_error=("absolute_regression_error", "mean"))
    error_summary["fraction"] = error_summary["count"] / len(errors)
    error_summary.to_csv(q3 / "main" / "test_error_summary.csv", index=False)

    # Attachment 4: keep the earlier global-only file and publish the complete local export.
    attachment_source = q3 / "attachment4" / "exp183_frozen"
    global_only = attachment_source / "attachment4_predictions.csv"
    complete_local = attachment_source / "local_evidence" / "explanation_summary.csv"
    attachment = pd.read_csv(global_only, dtype={"sample_id": str})
    attachment["sample_id"] = attachment["sample_id"].str.zfill(2)
    local_summary = pd.read_csv(complete_local, dtype={"sample_id": str})
    local_summary["sample_id"] = local_summary["sample_id"].str.zfill(2)
    local_columns = [
        "sample_id",
        "video_duration_sec",
        "time_mapping_rule",
        "text_key_positions",
        "key_text",
        "audio_key_positions",
        "audio_time_range",
        "vision_key_positions",
        "visual_time_range",
        "visual_key_frame",
    ]
    if "video_duration_sec" in attachment.columns:
        attachment.drop(columns=["video_duration_sec"], inplace=True)
    attachment_complete = attachment.merge(local_summary[local_columns], on="sample_id", how="left", validate="one_to_one")
    attachment_complete.to_csv(q3 / "attachment4" / "attachment4_predictions.csv", index=False, encoding="utf-8-sig")
    attachment_complete.to_excel(q3 / "attachment4" / "attachment4_predictions.xlsx", index=False)
    shutil.copy2(global_only, q3 / "attachment4" / "attachment4_predictions_global_only.csv")
    shutil.copytree(attachment_source / "local_evidence" / "keyframes", q3 / "attachment4" / "keyframes", dirs_exist_ok=True)
    attachment_complete.head(10).to_csv(q3 / "attachment4" / "attachment4_prediction_examples.csv", index=False)

    # Selected test cases and their deletion audit.
    case_dir = q3 / "cases" / "exp183_test_cases"
    cases = pd.read_csv(case_dir / "predictions_with_local_explanations.csv")
    local_case_ids = pd.read_csv(case_dir / "explanation_summary.csv")["sample_id"].astype(str).tolist()
    cases = cases[cases["id"].astype(str).isin(local_case_ids)].copy()
    case_ids = cases["id"].astype(str).tolist()
    case_info = labels[labels["sample_id"].isin(case_ids)][["sample_id", "text"]]
    cases = cases.merge(case_info, left_on="id", right_on="sample_id", how="left")
    cases = cases.merge(importance[["id", "text_importance", "audio_importance", "vision_importance", "dominant_modality"]], on="id", how="left", suffixes=("", "_test"))
    if "dominant_modality_test" not in cases.columns and "dominant_modality" in cases.columns:
        cases.rename(columns={"dominant_modality": "dominant_modality_test"}, inplace=True)
    cases.to_csv(q3 / "cases" / "explanation_cases.csv", index=False, encoding="utf-8-sig")
    case_deletion = pd.read_csv(case_dir / "evidence_deletion_results.csv")
    case_deletion.insert(0, "evaluation_split", "test_selected_cases")
    case_deletion.to_csv(q3 / "interpretability" / "evidence_deletion_results.csv", index=False)

    # Paper figures.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    draw_framework(figures)
    draw_architecture(figures)

    class_counts = labels.groupby(["mode", "annotation"]).size().unstack(fill_value=0).reindex(index=["train", "valid", "test"], columns=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(class_counts.index)); width = 0.24
    for index, name in enumerate(CLASS_NAMES):
        ax.bar(x + (index - 1) * width, class_counts[name], width, label=name, color=CLASS_COLORS[name])
    ax.set_xticks(x, [value.title() for value in class_counts.index]); ax.set_ylabel("Number of samples"); ax.set_title("Class distribution by split"); ax.legend(frameon=False); style_axes(ax)
    save_figure(fig, figures, "fig_q3_class_distribution")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-3, 3, 19)
    for split, color in zip(("train", "valid", "test"), ("#4C72B0", "#DD8452", "#55A868")):
        ax.hist(labels.loc[labels["mode"] == split, "label"], bins=bins, alpha=0.45, label=split.title(), color=color, edgecolor="white")
    ax.set_xlabel("Ground-truth sentiment intensity"); ax.set_ylabel("Count"); ax.set_title("Continuous sentiment-label distribution"); ax.legend(frameon=False); style_axes(ax)
    save_figure(fig, figures, "fig_q3_intensity_distribution")

    history = read_json(root / "runs" / "c3_hafusion_single" / "seed_20260924" / "history.json")
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], color="#4C72B0", label="Train total loss")
    axes[0].plot(epochs, [row["train"]["classification"] for row in history], color="#C44E52", label="Classification")
    axes[0].plot(epochs, [row["train"]["regression"] for row in history], color="#55A868", label="Regression")
    axes[0].set_xlabel("Epoch"); axes[0].set_title("Training objectives"); axes[0].legend(frameon=False, fontsize=8); style_axes(axes[0])
    axes[1].plot(epochs, [row["valid"]["accuracy"] for row in history], label="Validation Accuracy", color="#4C72B0")
    axes[1].plot(epochs, [row["valid"]["macro_f1"] for row in history], label="Validation Macro-F1", color="#C44E52")
    axes[1].set_xlabel("Epoch"); axes[1].set_title("Selection diagnostics (not final performance)"); axes[1].legend(frameon=False, fontsize=8); style_axes(axes[1])
    save_figure(fig, figures, "fig_q3_training_curve")

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(confusion.to_numpy(), cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(confusion.iloc[i, j]), ha="center", va="center", color="white" if confusion.iloc[i, j] > confusion.to_numpy().max() / 2 else "black")
    ax.set_xticks(range(3), CLASS_NAMES); ax.set_yticks(range(3), CLASS_NAMES); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("Frozen EXP183 confusion matrix (test)"); fig.colorbar(image, ax=ax, fraction=0.046)
    save_figure(fig, figures, "fig_q3_confusion_matrix")

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(final_predictions["true_intensity"], final_predictions["predicted_intensity"], s=15, alpha=0.45, color="#4C72B0", edgecolors="none")
    ax.plot([-3, 3], [-3, 3], linestyle="--", color="#444444", linewidth=1)
    ax.set_xlim(-3.1, 3.1); ax.set_ylim(-3.1, 3.1); ax.set_xlabel("Ground truth"); ax.set_ylabel("Prediction"); ax.set_title(f"Regression on test (r={exp183['pearson']:.3f}, MAE={exp183['mae']:.3f})"); style_axes(ax)
    save_figure(fig, figures, "fig_q3_regression_scatter")

    residual = final_predictions["predicted_intensity"] - final_predictions["true_intensity"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(final_predictions["true_intensity"], residual, s=15, alpha=0.45, color="#D9822B", edgecolors="none"); ax.axhline(0, linestyle="--", color="#444444")
    ax.set_xlabel("Ground-truth intensity"); ax.set_ylabel("Residual (prediction - truth)"); ax.set_title("Regression residuals on frozen test"); style_axes(ax)
    save_figure(fig, figures, "fig_q3_residual")

    means = [importance[f"{m}_importance"].mean() for m in MODALITIES]
    fig, ax = plt.subplots(figsize=(5.7, 4.3)); ax.bar([m.title() for m in MODALITIES], means, color=[COLORS[m] for m in MODALITIES]); ax.set_ylim(0, max(means) * 1.2); ax.set_ylabel("Mean normalized perturbation importance"); ax.set_title("Global modality importance on test"); add_bar_labels(ax); style_axes(ax)
    save_figure(fig, figures, "fig_q3_modality_global")

    fig, ax = plt.subplots(figsize=(7.5, 4.5)); x = np.arange(3); width = 0.24
    for index, modality in enumerate(MODALITIES):
        ax.bar(x + (index - 1) * width, class_importance[f"{modality}_importance"], width, label=modality.title(), color=COLORS[modality])
    ax.set_xticks(x, CLASS_NAMES); ax.set_ylim(0, 1); ax.set_ylabel("Mean importance"); ax.set_title("Modality importance by true class (test)"); ax.legend(frameon=False); style_axes(ax)
    save_figure(fig, figures, "fig_q3_modality_class")

    fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(intensity_importance)); width = 0.24
    for index, modality in enumerate(MODALITIES):
        ax.bar(x + (index - 1) * width, intensity_importance[f"{modality}_importance"], width, label=modality.title(), color=COLORS[modality])
    ax.set_xticks(x, intensity_importance["intensity_bin"].astype(str), rotation=15); ax.set_ylim(0, 1); ax.set_ylabel("Mean importance"); ax.set_title("Modality importance by true-intensity interval (test)"); ax.legend(frameon=False); style_axes(ax)
    save_figure(fig, figures, "fig_q3_modality_intensity")

    fig, ax = plt.subplots(figsize=(5.7, 4.3)); ax.bar([m.title() for m in dominant["modality"]], dominant["fraction"], color=[COLORS[m] for m in dominant["modality"]]); ax.set_ylim(0, 1); ax.set_ylabel("Fraction of test samples"); ax.set_title("Dominant modality distribution (test)"); add_bar_labels(ax); style_axes(ax)
    save_figure(fig, figures, "fig_q3_dominant_modality")

    case_local = pd.read_csv(case_dir / "local_evidence.csv")
    heat_ids = case_ids[:4]
    fig, axes = plt.subplots(len(heat_ids), 1, figsize=(10, 6.8), sharex=True)
    for ax, sample_id in zip(np.atleast_1d(axes), heat_ids):
        matrix = np.zeros((3, 50))
        selected = case_local[case_local["sample_id"].astype(str) == sample_id]
        for modality_index, modality in enumerate(MODALITIES):
            rows = selected[selected["modality"] == modality]
            matrix[modality_index, rows["position_zero_based"].to_numpy(int)] = rows["local_effect"].to_numpy(float)
        image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(range(3), ["T", "A", "V"]); ax.set_title(sample_id.replace("$", r"\$"), loc="left", fontsize=8); fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)
    axes[-1].set_xlabel("Aligned position (1-50)"); fig.suptitle("Pipeline-level local deletion evidence on selected test cases")
    save_figure(fig, figures, "fig_q3_evidence_heatmap")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    occ = occlusion.copy(); labels_occ = [value.replace("without_", "w/o ").title() for value in occ["condition"]]
    axes[0].bar(labels_occ, occ["accuracy"], color="#4C72B0"); axes[0].tick_params(axis="x", rotation=20); axes[0].set_ylim(0, 0.8); axes[0].set_ylabel("Accuracy"); axes[0].set_title("Frozen test Accuracy")
    axes[1].bar(labels_occ, occ["macro_f1"], color="#C44E52"); axes[1].tick_params(axis="x", rotation=20); axes[1].set_ylim(0, 0.8); axes[1].set_ylabel("Macro-F1"); axes[1].set_title("Frozen test Macro-F1")
    for ax in axes: style_axes(ax)
    save_figure(fig, figures, "fig_q3_occlusion")

    text_deletion = case_deletion[(case_deletion["modality"] == "text")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, color in zip(("top", "random", "bottom"), ("#C44E52", "#4C72B0", "#777777")):
        subset = text_deletion[text_deletion["strategy"] == strategy]
        ax.plot(subset["deletion_ratio"] * 100, subset["mean_confidence_delta"], marker="o", label=strategy.title(), color=color)
    ax.set_xlabel("Deleted text positions (%)"); ax.set_ylabel("Mean frozen-class confidence drop"); ax.set_title("Evidence deletion on six selected test cases"); ax.legend(frameon=False); style_axes(ax)
    save_figure(fig, figures, "fig_q3_evidence_deletion")

    fig, ax = plt.subplots(figsize=(9, 4.6)); x = np.arange(len(ablation_results)); width = 0.38
    ax.bar(x - width/2, ablation_results["accuracy"], width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width/2, ablation_results["macro_f1"], width, label="Macro-F1", color="#C44E52")
    ax.set_xticks(x, ablation_results["model"], rotation=20, ha="right"); ax.set_ylim(0, 0.82); ax.set_title("Frozen test component interventions"); ax.legend(frameon=False); style_axes(ax)
    save_figure(fig, figures, "fig_q3_ablation")

    fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(sensitivity)); width = 0.36
    ax.bar(x - width/2, sensitivity["accuracy"], width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width/2, sensitivity["macro_f1"], width, label="Macro-F1", color="#C44E52")
    ax.set_xticks(x, sensitivity["setting"], rotation=15, ha="right"); ax.set_ylim(0.55, 0.72); ax.set_title("Historical one-factor settings evaluated on test"); ax.legend(frameon=False); style_axes(ax)
    save_figure(fig, figures, "fig_q3_sensitivity")

    card = cases.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.5)); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes, boxstyle="round,pad=0.02", facecolor="#FAFAFA", edgecolor="#444444"))
    lines = [
        f"Test sample: {card['id']}",
        f"Truth / prediction: {card['true_label']} / {card['predicted_label']}",
        f"True / predicted intensity: {float(card['true_intensity']):.3f} / {float(card['predicted_intensity']):.3f}",
        f"Dominant modality: {card['dominant_modality_test']}",
        f"T/A/V importance: {float(card['text_importance']):.3f} / {float(card['audio_importance']):.3f} / {float(card['vision_importance']):.3f}",
        f"Key text: {str(card.get('key_text', ''))[:150]}",
        f"Aligned key positions T/A/V: {card.get('text_key_positions','')} | {card.get('audio_key_positions','')} | {card.get('vision_key_positions','')}",
        "Local evidence is measured by final-pipeline deletion, not by a single member's attention.",
    ]
    for index, line in enumerate(lines):
        ax.text(0.06, 0.88 - index * 0.105, line.replace("$", r"\$"), transform=ax.transAxes, fontsize=10 if index else 13, weight="bold" if index == 0 else "normal", va="top")
    save_figure(fig, figures, "fig_q3_explanation_card")

    # LaTeX booktabs tables.
    feature_table = pd.DataFrame([
        {"Modality": "Text", "Symbol": "X(T)", "Length": 50, "Dimension": 768, "Description": "Competition-provided aligned textual representation"},
        {"Modality": "Audio", "Symbol": "X(A)", "Length": 50, "Dimension": 74, "Description": "Competition-provided aligned acoustic representation"},
        {"Modality": "Vision", "Symbol": "X(V)", "Length": 50, "Dimension": 35, "Description": "Competition-provided aligned visual representation"},
    ])
    latex_table(feature_table, "三模态输入特征", "tab:q3_input_features", tables / "tab_q3_01_input_features.tex")
    split_table = class_counts.reset_index().rename(columns={"mode": "Split"}); split_table["Total"] = split_table[list(CLASS_NAMES)].sum(axis=1)
    latex_table(split_table, "数据集划分及类别分布", "tab:q3_dataset_split", tables / "tab_q3_02_dataset_split.tex")
    stats = labels.groupby("mode")["label"].agg(["mean", "std", "min", "median", "max"]).reindex(["train", "valid", "test"]).reset_index().rename(columns={"mode": "Split"})
    latex_table(stats, "连续标签统计", "tab:q3_label_statistics", tables / "tab_q3_03_label_statistics.tex", float_columns=["mean", "std", "min", "median", "max"])
    config = read_json(root / "runs/c3_hafusion_single/seed_20260924/resolved_config.json")
    hyper = pd.DataFrame([
        ("Input", "aligned_50.pkl"), ("Hidden dimension", config["model"]["d_model"]), ("Heads / layers", f"{config['model']['n_heads']} / {config['model']['n_layers']}"),
        ("Batch size", config["training"]["batch_size"]), ("Learning rate", config["training"]["learning_rate"]), ("Weight decay", config["training"]["weight_decay"]),
        ("Dropout", config["model"]["dropout"]), ("Gradient clipping", config["training"]["grad_clip"]), ("Seed", config["seed"]),
        ("Final members", 7), ("Neutral threshold", exp183_report["authenticity_threshold"]), ("Device", "CUDA / RTX 4060 8GB"),
    ], columns=["Parameter", "Value"])
    latex_table(hyper, "模型与训练主要参数", "tab:q3_hyperparameters", tables / "tab_q3_04_hyperparameters.tex")
    latex_table(main_results, "冻结测试集主模型性能", "tab:q3_main_performance", tables / "tab_q3_05_main_performance.tex", columns=["model", *METRIC_COLUMNS], headers=["Model", "Acc.", "Macro-F1", "Weighted-F1", "Bal. Acc.", "MAE", "RMSE", "Bias", "Pearson"], float_columns=METRIC_COLUMNS)
    latex_table(modality_results, "单模态与模态组合的测试集结果", "tab:q3_modality_combinations", tables / "tab_q3_06_modality_combinations.tex", columns=["modalities", "accuracy", "macro_f1", "mae", "pearson"], headers=["Modalities", "Acc.", "Macro-F1", "MAE", "Pearson"], float_columns=["accuracy", "macro_f1", "mae", "pearson"])
    latex_table(fusion_results, "实际实现的结构与融合路线测试集对比", "tab:q3_fusion_comparison", tables / "tab_q3_07_fusion_comparison.tex", columns=["model", "accuracy", "macro_f1", "mae", "pearson"], headers=["Route", "Acc.", "Macro-F1", "MAE", "Pearson"], float_columns=["accuracy", "macro_f1", "mae", "pearson"])
    latex_table(ablation_results, "冻结部署组件干预测试", "tab:q3_ablation", tables / "tab_q3_08_ablation.tex", columns=["model", "accuracy", "macro_f1", "mae", "pearson"], headers=["Intervention", "Acc.", "Macro-F1", "MAE", "Pearson"], float_columns=["accuracy", "macro_f1", "mae", "pearson"])
    latex_table(multitask, "联合任务损失的真实对比", "tab:q3_multitask", tables / "tab_q3_09_multitask.tex", columns=["model", "accuracy", "macro_f1", "mae", "pearson"], headers=["Loss setting", "Acc.", "Macro-F1", "MAE", "Pearson"], float_columns=["accuracy", "macro_f1", "mae", "pearson"])
    latex_table(seed_results, "随机种子稳定性测试", "tab:q3_seed_stability", tables / "tab_q3_10_seed_stability.tex", columns=["run", "test_accuracy", "test_macro_f1", "test_mae", "test_pearson"], headers=["Run", "Acc.", "Macro-F1", "MAE", "Pearson"], float_columns=["test_accuracy", "test_macro_f1", "test_mae", "test_pearson"])
    latex_table(per_class, "冻结测试集分类别性能", "tab:q3_per_class", tables / "tab_q3_11_per_class.tex", columns=["class", "precision", "recall", "f1", "support"], headers=["Class", "Precision", "Recall", "F1", "Support"], float_columns=["precision", "recall", "f1"])
    latex_table(occlusion, "冻结测试集模态遮挡结果", "tab:q3_occlusion", tables / "tab_q3_12_occlusion.tex", columns=["condition", "accuracy", "macro_f1", "mae", "pearson", "prediction_flip_rate"], headers=["Condition", "Acc.", "Macro-F1", "MAE", "Pearson", "Flip rate"], float_columns=["accuracy", "macro_f1", "mae", "pearson", "prediction_flip_rate"])
    latex_table(case_deletion, "选定测试案例的局部证据删除结果", "tab:q3_evidence_deletion", tables / "tab_q3_13_evidence_deletion.tex", columns=["modality", "deletion_ratio", "strategy", "mean_confidence_delta", "prediction_flip_rate"], headers=["Modality", "Ratio", "Strategy", "Confidence drop", "Flip rate"], float_columns=["deletion_ratio", "mean_confidence_delta", "prediction_flip_rate"])
    latex_table(cases, "典型测试样本解释结果", "tab:q3_cases", tables / "tab_q3_14_cases.tex", columns=["id", "true_label", "predicted_label", "predicted_intensity", "dominant_modality_test", "key_text"], headers=["ID", "Truth", "Prediction", "Intensity", "Dominant", "Key text"], float_columns=["predicted_intensity"])
    latex_table(attachment_complete.head(10), "附件4预测结果示例", "tab:q3_attachment4", tables / "tab_q3_15_attachment4.tex", columns=["sample_id", "predicted_class", "predicted_intensity", "dominant_modality", "key_text"], headers=["ID", "Class", "Intensity", "Dominant", "Key text"], float_columns=["predicted_intensity"])

    # Machine-readable inventory used by the final audit.
    inventory = {
        "reporting_policy": "primary performance tables and figures use frozen test results",
        "validation_role": "checkpoint/threshold selection only",
        "formal_test_metrics": {key: exp183[key] for key in METRIC_COLUMNS},
        "consistency": consistency,
        "figures": sorted(path.name for path in figures.glob("*")),
        "tables": sorted(path.name for path in tables.glob("*.tex")),
        "attachment4_rows": int(len(attachment_complete)),
        "attachment4_labels_used": False,
    }
    (q3 / "Q3_RESULTS_MANIFEST.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
