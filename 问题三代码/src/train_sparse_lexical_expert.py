from __future__ import annotations

"""Group-safe word/character lexical expert for domain-specific sentiment cues."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier

from .data import CLASS_NAMES, attach_labels_from_excel, load_pickle, parse_split
from .metrics import compute_metrics
from .train_pretrained_oof import video_groups


def build_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
    )
    character = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
    )
    return word, character


def fit_predict(
    train_text: np.ndarray,
    train_class: np.ndarray,
    train_intensity: np.ndarray,
    heldout_text: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    word, character = build_vectorizers()
    train_values = [str(value) for value in train_text]
    heldout_values = [str(value) for value in heldout_text]
    word_train = word.fit_transform(train_values)
    character_train = character.fit_transform(train_values)
    word_heldout = word.transform(heldout_values)
    character_heldout = character.transform(heldout_values)
    train_matrix = sparse.hstack(
        [word_train, character_train], format="csr", dtype=np.float64
    )
    heldout_matrix = sparse.hstack(
        [word_heldout, character_heldout], format="csr", dtype=np.float64
    )
    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=2.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=600,
            random_state=seed,
        )
    )
    classifier.fit(train_matrix, train_class)
    probability = classifier.predict_proba(heldout_matrix)
    ordered_probability = np.zeros((len(heldout_values), 3), dtype=np.float64)
    ordered_probability[:, classifier.classes_.astype(np.int64)] = probability

    regressor = Ridge(alpha=10.0, solver="lsqr")
    regressor.fit(train_matrix, train_intensity)
    regression = np.clip(regressor.predict(heldout_matrix), -3.0, 3.0)
    vocabulary = {
        "word_features": int(word_train.shape[1]),
        "character_features": int(character_train.shape[1]),
        "total_features": int(train_matrix.shape[1]),
    }
    return ordered_probability, regression, vocabulary


def prediction_frame(
    identifiers: np.ndarray,
    probability: np.ndarray,
    regression: np.ndarray,
    true_class: np.ndarray,
    true_intensity: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": identifiers.astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": [CLASS_NAMES[index] for index in true_class],
            "true_intensity": true_intensity,
        }
    )


def grouped_oof(arrays: Any, seed: int, folds: int) -> tuple[dict[str, Any], pd.DataFrame]:
    groups = video_groups(arrays.ids)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    chunks: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for fold, (train_index, heldout_index) in enumerate(
        splitter.split(np.zeros(arrays.size), arrays.class_labels, groups)
    ):
        overlap = set(groups[train_index]).intersection(groups[heldout_index])
        if overlap:
            raise RuntimeError(f"Fold {fold} has group leakage")
        probability, regression, vocabulary = fit_predict(
            arrays.raw_text[train_index],
            arrays.class_labels[train_index],
            arrays.regression_labels[train_index],
            arrays.raw_text[heldout_index],
            seed + fold,
        )
        metrics = compute_metrics(
            arrays.class_labels[heldout_index],
            probability,
            arrays.regression_labels[heldout_index],
            regression,
        )
        frame = prediction_frame(
            arrays.ids[heldout_index],
            probability,
            regression,
            arrays.class_labels[heldout_index],
            arrays.regression_labels[heldout_index],
        )
        frame.insert(0, "source_index", heldout_index)
        frame.insert(1, "fold", fold)
        chunks.append(frame)
        reports.append(
            {
                "fold": fold,
                "train_samples": int(len(train_index)),
                "heldout_samples": int(len(heldout_index)),
                "group_overlap": 0,
                "vocabulary": vocabulary,
                "metrics": metrics,
            }
        )
        print(
            f"fold={fold} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
    oof = pd.concat(chunks, ignore_index=True).sort_values("source_index")
    if not np.array_equal(oof["source_index"].to_numpy(), np.arange(arrays.size)):
        raise RuntimeError("OOF rows do not cover every train sample exactly once")
    probability = oof[
        ["negative_probability", "neutral_probability", "positive_probability"]
    ].to_numpy(np.float64)
    metrics = compute_metrics(
        arrays.class_labels,
        probability,
        arrays.regression_labels,
        oof["predicted_intensity"].to_numpy(np.float64),
    )
    return {"oof": metrics, "fold_reports": reports}, oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--locked-valid", action="store_true")
    args = parser.parse_args()
    if args.folds < 3:
        raise ValueError("At least three grouped folds are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    loaded = load_pickle(args.data)
    split_names = ["train", "valid"] if args.locked_valid else ["train"]
    raw = attach_labels_from_excel({name: loaded[name] for name in split_names}, args.labels)
    train = parse_split(raw, "train", "text_shared", require_labels=True)

    if not args.locked_valid:
        report, predictions = grouped_oof(train, args.seed, args.folds)
        predictions.to_csv(
            output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report.update(
            {
                "scope": "train_only_grouped_oof_no_valid_or_test_access",
                "architecture": "word_bigram_plus_character_3_5gram_sparse_expert",
                "seed": args.seed,
                "folds": args.folds,
                "grouping": "sample id prefix before $_$",
                "elapsed_minutes": (time.time() - started) / 60.0,
            }
        )
        path = output / "manifest.json"
    else:
        valid = parse_split(raw, "valid", "text_shared", require_labels=True)
        probability, regression, vocabulary = fit_predict(
            train.raw_text,
            train.class_labels,
            train.regression_labels,
            valid.raw_text,
            args.seed,
        )
        metrics = compute_metrics(
            valid.class_labels,
            probability,
            valid.regression_labels,
            regression,
        )
        predictions = prediction_frame(
            valid.ids,
            probability,
            regression,
            valid.class_labels,
            valid.regression_labels,
        )
        predictions.to_csv(
            output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report = {
            "scope": "full_train_fixed_sparse_model_then_single_locked_valid_no_test_access",
            "architecture": "word_bigram_plus_character_3_5gram_sparse_expert",
            "seed": args.seed,
            "vocabulary": vocabulary,
            "valid": metrics,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        path = output / "final_metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
