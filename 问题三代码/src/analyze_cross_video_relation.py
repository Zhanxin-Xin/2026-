from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


LABEL_INDEX = {"Negative": 0, "Neutral": 1, "Positive": 2}


def reconstruct_parent(frame: pd.DataFrame) -> np.ndarray:
    child = frame[
        ["negative_probability", "neutral_probability", "positive_probability"]
    ].to_numpy(np.float64)
    shift = frame["cross_video_neutral_shift"].to_numpy(np.float64)
    child_neutral = np.clip(child[:, 1], 1e-8, 1.0 - 1e-8)
    parent_neutral = 1.0 / (
        1.0
        + np.exp(-(
            np.log(child_neutral / (1.0 - child_neutral)) - shift
        ))
    )
    positive_within_polar = child[:, 2] / np.maximum(
        child[:, 0] + child[:, 2], 1e-8
    )
    return np.column_stack(
        [
            (1.0 - parent_neutral) * (1.0 - positive_within_polar),
            parent_neutral,
            (1.0 - parent_neutral) * positive_within_polar,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions")
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions)
    target = frame["true_label"].map(LABEL_INDEX).to_numpy(np.int64)
    child = frame[
        ["negative_probability", "neutral_probability", "positive_probability"]
    ].to_numpy(np.float64)
    parent = reconstruct_parent(frame)
    child_prediction = child.argmax(axis=1)
    parent_prediction = parent.argmax(axis=1)
    for name, prediction in (("parent", parent_prediction), ("child", child_prediction)):
        print(
            name,
            {
                "accuracy": float(accuracy_score(target, prediction)),
                "macro_f1": float(f1_score(target, prediction, average="macro")),
                "confusion": confusion_matrix(target, prediction).tolist(),
            },
        )
    changed = child_prediction != parent_prediction
    print(
        "changes",
        {
            "count": int(changed.sum()),
            "corrected": int(((child_prediction == target) & (parent_prediction != target)).sum()),
            "broken": int(((child_prediction != target) & (parent_prediction == target)).sum()),
        },
    )
    shift = frame["cross_video_neutral_shift"].to_numpy(np.float64)
    print(
        "shift",
        {
            "quantiles": np.quantile(
                shift, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
            ).tolist(),
            "negative_count": int((shift < 0.0).sum()),
        },
    )
    for side, mask in (("left", target != 2), ("right", target != 0)):
        binary_target = target[mask] == 1
        print(side, {"samples": int(mask.sum())})
        for quantity in ("logit", "residual"):
            column = f"cross_video_{side}_{quantity}"
            print(column, float(roc_auc_score(binary_target, frame.loc[mask, column])))
        for modality in ("semantic", "audio", "vision"):
            column = f"cross_video_{modality}_{side}_margin"
            print(column, float(roc_auc_score(binary_target, frame.loc[mask, column])))


if __name__ == "__main__":
    main()
