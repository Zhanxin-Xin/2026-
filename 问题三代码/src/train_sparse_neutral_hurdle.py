from __future__ import annotations

"""Cross-fitted sparse-semantic Neutral hurdle with a polar-odds invariant."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer

from .build_cross_fitted_ensemble import PROBABILITY_COLUMNS, load_aligned
from .data import CLASS_NAMES, attach_labels_from_excel, load_pickle, parse_split
from .metrics import compute_metrics
from .train_frozen_minilm_prototype_expert import encode_texts
from .train_pretrained_oof import video_groups


def vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    return (
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        ),
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=50000,
            sublinear_tf=True,
        ),
    )


def numeric_features(frames: list[pd.DataFrame]) -> np.ndarray:
    blocks: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    regressions: list[np.ndarray] = []
    for frame in frames:
        probability = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
        probability = np.clip(probability, 1e-7, 1.0)
        probability /= probability.sum(axis=1, keepdims=True)
        regression = frame["predicted_intensity"].to_numpy(np.float64)[:, None]
        entropy = -(probability * np.log(probability)).sum(axis=1, keepdims=True)
        sorted_probability = np.sort(probability, axis=1)
        margin = (sorted_probability[:, -1] - sorted_probability[:, -2])[:, None]
        neutral_logit = (
            np.log(probability[:, 1])
            - np.log(probability[:, 0] + probability[:, 2])
        )[:, None]
        blocks.append(
            np.concatenate(
                [probability, np.log(probability), regression, entropy, margin, neutral_logit],
                axis=1,
            )
        )
        probabilities.append(probability)
        regressions.append(regression[:, 0])
    probability_stack = np.stack(probabilities, axis=1)
    regression_stack = np.stack(regressions, axis=1)
    blocks.extend(
        [
            probability_stack.mean(axis=1),
            probability_stack.std(axis=1),
            regression_stack.mean(axis=1, keepdims=True),
            regression_stack.std(axis=1, keepdims=True),
        ]
    )
    return np.concatenate(blocks, axis=1)


def fit_hurdle(
    train_text: np.ndarray,
    train_numeric: np.ndarray,
    train_target: np.ndarray,
    heldout_text: np.ndarray,
    heldout_numeric: np.ndarray,
    seed: int,
    train_selection: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    if train_selection is not None:
        selection = np.asarray(train_selection, dtype=bool)
        if selection.shape != (len(train_text),):
            raise ValueError("train_selection must align with train_text")
        if np.unique(train_target[selection]).size != 2:
            raise ValueError("selected hurdle training rows need both classes")
        train_text = train_text[selection]
        train_numeric = train_numeric[selection]
        train_target = train_target[selection]
    word, character = vectorizers()
    train_values = [str(value) for value in train_text]
    heldout_values = [str(value) for value in heldout_text]
    word_train = word.fit_transform(train_values)
    char_train = character.fit_transform(train_values)
    word_heldout = word.transform(heldout_values)
    char_heldout = character.transform(heldout_values)
    mean = train_numeric.mean(axis=0, keepdims=True)
    scale = train_numeric.std(axis=0, keepdims=True).clip(min=1e-6)
    numeric_train = sparse.csr_matrix((train_numeric - mean) / scale)
    numeric_heldout = sparse.csr_matrix((heldout_numeric - mean) / scale)
    train_matrix = sparse.hstack(
        [word_train, char_train, numeric_train], format="csr", dtype=np.float64
    )
    heldout_matrix = sparse.hstack(
        [word_heldout, char_heldout, numeric_heldout], format="csr", dtype=np.float64
    )
    model = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        solver="liblinear",
        max_iter=600,
        random_state=seed,
    )
    model.fit(train_matrix, train_target)
    probability = model.predict_proba(heldout_matrix)[:, list(model.classes_).index(1)]
    return probability, {
        "word_features": int(word_train.shape[1]),
        "character_features": int(char_train.shape[1]),
        "numeric_features": int(train_numeric.shape[1]),
        "total_features": int(train_matrix.shape[1]),
    }


def bounded_neutral_correction(
    parent_probability: np.ndarray,
    hurdle_neutral_probability: np.ndarray,
    maximum_shift: float = 0.75,
    protect_positive_to_neutral: bool = False,
    polar_rejector: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.clip(parent_probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
    parent /= parent.sum(axis=1, keepdims=True)
    hurdle = np.clip(hurdle_neutral_probability, 1e-7, 1.0 - 1e-7)
    parent_neutral_logit = np.log(parent[:, 1]) - np.log1p(-parent[:, 1])
    hurdle_logit = np.log(hurdle) - np.log1p(-hurdle)
    residual = float(maximum_shift) * np.tanh(hurdle_logit - parent_neutral_logit)
    corrected_neutral = 1.0 / (1.0 + np.exp(-(parent_neutral_logit + residual)))
    polar_total = (parent[:, 0] + parent[:, 2]).clip(min=1e-7)
    positive_within_polar = parent[:, 2] / polar_total
    corrected = np.stack(
        [
            (1.0 - corrected_neutral) * (1.0 - positive_within_polar),
            corrected_neutral,
            (1.0 - corrected_neutral) * positive_within_polar,
        ],
        axis=1,
    )
    corrected /= corrected.sum(axis=1, keepdims=True)
    if polar_rejector:
        # A selective rejector is allowed to send an existing polar decision
        # to Neutral, but it cannot perturb samples already accepted as
        # Neutral.  This turns the branch into a one-sided error corrector.
        accepted_neutral = parent.argmax(axis=1) == 1
        corrected[accepted_neutral] = parent[accepted_neutral]
        residual[accepted_neutral] = 0.0
    if protect_positive_to_neutral:
        protected = (parent.argmax(axis=1) == 2) & (corrected.argmax(axis=1) == 1)
        corrected[protected] = parent[protected]
        residual[protected] = 0.0
    return corrected, residual


def output_frame(
    parent: pd.DataFrame,
    probability: np.ndarray,
    residual: np.ndarray,
    include_fold: bool,
    folds: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": parent["id"].astype(str),
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": parent["predicted_intensity"].to_numpy(np.float64),
            "true_label": parent["true_label"].astype(str),
            "true_intensity": parent["true_intensity"].to_numpy(np.float64),
            "neutral_logit_residual": residual,
        }
    )
    if include_fold:
        frame.insert(
            1,
            "fold",
            parent["fold"].to_numpy(np.int64) if folds is None else folds,
        )
    return frame


def metrics(parent: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    label_map = {name: index for index, name in enumerate(CLASS_NAMES)}
    return compute_metrics(
        parent["true_label"].map(label_map).to_numpy(np.int64),
        probability,
        parent["true_intensity"].to_numpy(np.float64),
        parent["predicted_intensity"].to_numpy(np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--parent-oof", required=True)
    parser.add_argument("--parent-valid")
    parser.add_argument("--oof-source", action="append", required=True)
    parser.add_argument("--valid-source", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--meta-fold-seed", type=int)
    parser.add_argument("--maximum-shift", type=float, default=0.75)
    parser.add_argument("--protect-positive-to-neutral", action="store_true")
    parser.add_argument(
        "--polar-rejector",
        action="store_true",
        help=(
            "Train the Neutral hurdle only on parent-polar decisions and keep "
            "all parent-Neutral decisions invariant"
        ),
    )
    parser.add_argument("--locked-valid", action="store_true")
    parser.add_argument("--semantic-model")
    parser.add_argument("--semantic-max-length", type=int, default=128)
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.locked_valid and (
        not args.parent_valid or len(args.valid_source) != len(args.oof_source)
    ):
        raise ValueError("Locked valid requires parent-valid and matching valid sources")
    if not 0.0 <= args.maximum_shift <= 1.5:
        raise ValueError("maximum-shift must lie in [0, 1.5]")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    loaded = load_pickle(args.data)
    split_names = ["train", "valid"] if args.locked_valid else ["train"]
    raw = attach_labels_from_excel({name: loaded[name] for name in split_names}, args.labels)
    train_arrays = parse_split(raw, "train", "text_shared", require_labels=True)
    parent_oof = load_aligned([args.parent_oof], require_fold=True)[0]
    source_oof = load_aligned(args.oof_source, require_fold=True)
    if not np.array_equal(parent_oof["id"].astype(str), train_arrays.ids.astype(str)):
        raise RuntimeError("Parent OOF IDs do not align with train text")
    numeric_oof = numeric_features(source_oof)
    semantic_encoder = None
    semantic_tokenizer = None
    device = torch.device(args.device)
    if args.semantic_model:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for semantic encoder but unavailable")
        semantic_tokenizer = AutoTokenizer.from_pretrained(args.semantic_model)
        semantic_encoder = AutoModel.from_pretrained(args.semantic_model).to(device)
        semantic_encoder.eval()
        for parameter in semantic_encoder.parameters():
            parameter.requires_grad_(False)
        train_semantic = encode_texts(
            train_arrays.raw_text,
            semantic_tokenizer,
            semantic_encoder,
            device,
            args.semantic_max_length,
            args.semantic_batch_size,
        )
        numeric_oof = np.concatenate([numeric_oof, train_semantic], axis=1)
    target = parent_oof["true_label"].astype(str).eq("Neutral").to_numpy(np.int64)

    if not args.locked_valid:
        probability = np.zeros((len(parent_oof), 3), dtype=np.float64)
        residual = np.zeros(len(parent_oof), dtype=np.float64)
        fold_reports: list[dict[str, Any]] = []
        meta_folds = parent_oof["fold"].to_numpy(np.int64).copy()
        if args.meta_fold_seed is not None:
            meta_folds.fill(-1)
            groups = video_groups(train_arrays.ids)
            splitter = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=args.meta_fold_seed
            )
            for fold, (_, heldout_index) in enumerate(
                splitter.split(np.zeros(train_arrays.size), train_arrays.class_labels, groups)
            ):
                meta_folds[heldout_index] = fold
            if (meta_folds < 0).any():
                raise RuntimeError("Prospective meta folds do not cover every sample")
        for fold in sorted(np.unique(meta_folds)):
            heldout = meta_folds == int(fold)
            fit = ~heldout
            fit_parent_probability = parent_oof.loc[
                fit, PROBABILITY_COLUMNS
            ].to_numpy(np.float64)
            hurdle, vocabulary = fit_hurdle(
                train_arrays.raw_text[fit],
                numeric_oof[fit],
                target[fit],
                train_arrays.raw_text[heldout],
                numeric_oof[heldout],
                args.seed + int(fold),
                train_selection=(
                    fit_parent_probability.argmax(axis=1) != 1
                    if args.polar_rejector
                    else None
                ),
            )
            parent_probability = parent_oof.loc[heldout, PROBABILITY_COLUMNS].to_numpy(
                np.float64
            )
            probability[heldout], residual[heldout] = bounded_neutral_correction(
                parent_probability,
                hurdle,
                maximum_shift=args.maximum_shift,
                protect_positive_to_neutral=args.protect_positive_to_neutral,
                polar_rejector=args.polar_rejector,
            )
            fold_metrics = metrics(parent_oof.loc[heldout].reset_index(drop=True), probability[heldout])
            fold_reports.append(
                {"fold": int(fold), "vocabulary": vocabulary, "metrics": fold_metrics}
            )
            print(
                f"fold={fold} accuracy={fold_metrics['accuracy']:.4f} "
                f"macro_f1={fold_metrics['macro_f1']:.4f}", flush=True
            )
        predictions = output_frame(
            parent_oof, probability, residual, True, folds=meta_folds
        )
        predictions.to_csv(
            output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report = {
            "scope": "cross_fitted_sparse_semantic_neutral_hurdle_no_valid_or_test_access",
            "architecture": (
                "counterfactual_sparse_semantic_polar_to_neutral_rejector"
                if args.polar_rejector
                else "sparse_word_character_plus_deep_disagreement_neutral_hurdle"
            ),
            "polar_odds_invariant": True,
            "maximum_neutral_logit_shift": args.maximum_shift,
            "meta_fold_seed": args.meta_fold_seed,
            "protect_positive_to_neutral": args.protect_positive_to_neutral,
            "polar_rejector": args.polar_rejector,
            "semantic_model": args.semantic_model,
            "parent_oof": metrics(
                parent_oof,
                parent_oof.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64),
            ),
            "oof": metrics(parent_oof, probability),
            "fold_reports": fold_reports,
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        path = output / "manifest.json"
    else:
        valid_arrays = parse_split(raw, "valid", "text_shared", require_labels=True)
        parent_valid = load_aligned([args.parent_valid], require_fold=False)[0]
        source_valid = load_aligned(args.valid_source, require_fold=False)
        if not np.array_equal(parent_valid["id"].astype(str), valid_arrays.ids.astype(str)):
            raise RuntimeError("Parent valid IDs do not align with valid text")
        numeric_valid = numeric_features(source_valid)
        if semantic_encoder is not None and semantic_tokenizer is not None:
            valid_semantic = encode_texts(
                valid_arrays.raw_text,
                semantic_tokenizer,
                semantic_encoder,
                device,
                args.semantic_max_length,
                args.semantic_batch_size,
            )
            numeric_valid = np.concatenate(
                [numeric_valid, valid_semantic], axis=1
            )
        hurdle, vocabulary = fit_hurdle(
            train_arrays.raw_text,
            numeric_oof,
            target,
            valid_arrays.raw_text,
            numeric_valid,
            args.seed,
            train_selection=(
                parent_oof.loc[:, PROBABILITY_COLUMNS]
                .to_numpy(np.float64)
                .argmax(axis=1)
                != 1
                if args.polar_rejector
                else None
            ),
        )
        parent_probability = parent_valid.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
        probability, residual = bounded_neutral_correction(
            parent_probability,
            hurdle,
            maximum_shift=args.maximum_shift,
            protect_positive_to_neutral=args.protect_positive_to_neutral,
            polar_rejector=args.polar_rejector,
        )
        predictions = output_frame(parent_valid, probability, residual, False)
        predictions.to_csv(
            output / "valid_predictions.csv", index=False, encoding="utf-8-sig"
        )
        report = {
            "scope": "full_train_sparse_semantic_hurdle_then_single_locked_valid_no_test_access",
            "architecture": (
                "counterfactual_sparse_semantic_polar_to_neutral_rejector"
                if args.polar_rejector
                else "sparse_word_character_plus_deep_disagreement_neutral_hurdle"
            ),
            "polar_odds_invariant": True,
            "maximum_neutral_logit_shift": args.maximum_shift,
            "protect_positive_to_neutral": args.protect_positive_to_neutral,
            "polar_rejector": args.polar_rejector,
            "semantic_model": args.semantic_model,
            "vocabulary": vocabulary,
            "valid": metrics(parent_valid, probability),
            "elapsed_minutes": (time.time() - started) / 60.0,
        }
        path = output / "final_metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
