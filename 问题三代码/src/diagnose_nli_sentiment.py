from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import attach_labels_from_excel, load_pickle, parse_split
from .metrics import CLASS_NAMES, compute_metrics
from .utils import save_json


DEFAULT_HYPOTHESES = (
    "The speaker expresses negative sentiment.",
    "The speaker expresses neutral sentiment.",
    "The speaker expresses positive sentiment.",
)
PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)


def _entailment_index(config: Any) -> int:
    """Resolve the entailment logit without assuming a model-specific label order."""
    labels: dict[int, str] = {}
    for index, value in dict(getattr(config, "id2label", {}) or {}).items():
        labels[int(index)] = str(value).strip().lower()
    for value, index in dict(getattr(config, "label2id", {}) or {}).items():
        labels.setdefault(int(index), str(value).strip().lower())
    matches = [index for index, value in labels.items() if "entail" in value]
    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely identify the entailment label from model config: "
            f"{labels}"
        )
    return matches[0]


def _contradiction_index(config: Any) -> int:
    """Resolve the contradiction logit without assuming a label order."""
    labels: dict[int, str] = {}
    for index, value in dict(getattr(config, "id2label", {}) or {}).items():
        labels[int(index)] = str(value).strip().lower()
    for value, index in dict(getattr(config, "label2id", {}) or {}).items():
        labels.setdefault(int(index), str(value).strip().lower())
    matches = [index for index, value in labels.items() if "contradict" in value]
    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely identify the contradiction label from model config: "
            f"{labels}"
        )
    return matches[0]


@torch.inference_mode()
def predict_nli(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: list[str],
    hypotheses: tuple[str, str, str],
    entailment_index: int,
    device: torch.device,
    batch_size: int,
    max_length: int,
    amp: bool,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        premises = [text for text in batch_texts for _ in hypotheses]
        paired_hypotheses = list(hypotheses) * len(batch_texts)
        encoded = tokenizer(
            premises,
            paired_hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.amp.autocast(
            "cuda", enabled=amp and device.type == "cuda", dtype=torch.float16
        ):
            logits = model(**encoded).logits
        entailment = logits[:, entailment_index].float().reshape(len(batch_texts), 3)
        chunks.append(torch.softmax(entailment, dim=-1).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _load_reference_predictions(
    path: str | Path,
    ids: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"id", "true_label", *PROBABILITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Reference predictions are missing columns: {sorted(missing)}")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError("Reference predictions contain duplicate ids")
    frame = frame.set_index(frame["id"].astype(str), drop=False)
    expected_ids = [str(value) for value in ids]
    missing_ids = sorted(set(expected_ids).difference(frame.index))
    extra_ids = sorted(set(frame.index).difference(expected_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            "Reference/validation id mismatch: "
            f"missing={missing_ids[:5]}, extra={extra_ids[:5]}"
        )
    aligned = frame.loc[expected_ids]
    expected_labels = np.asarray([CLASS_NAMES[index] for index in targets])
    observed_labels = aligned["true_label"].astype(str).to_numpy()
    if not np.array_equal(expected_labels, observed_labels):
        raise ValueError("Reference true labels do not match the validation split")
    probability = aligned.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("Reference probabilities are invalid")
    regression = aligned["predicted_intensity"].to_numpy(np.float64)
    return probability, regression


def _complementarity(
    targets: np.ndarray,
    reference_probability: np.ndarray,
    nli_probability: np.ndarray,
) -> dict[str, Any]:
    reference_prediction = reference_probability.argmax(axis=1)
    nli_prediction = nli_probability.argmax(axis=1)
    reference_correct = reference_prediction == targets
    nli_correct = nli_prediction == targets
    result: dict[str, Any] = {
        "reference_correct_nli_correct": int((reference_correct & nli_correct).sum()),
        "reference_correct_nli_wrong": int((reference_correct & ~nli_correct).sum()),
        "reference_wrong_nli_correct": int((~reference_correct & nli_correct).sum()),
        "reference_wrong_nli_wrong": int((~reference_correct & ~nli_correct).sum()),
        "prediction_disagreement_count": int((reference_prediction != nli_prediction).sum()),
        "prediction_disagreement_rate": float(np.mean(reference_prediction != nli_prediction)),
        # This is a truth-informed diagnostic ceiling, never a deployable selector.
        "oracle_any_correct_accuracy": float(np.mean(reference_correct | nli_correct)),
        "oracle_is_diagnostic_only": True,
        "by_true_class": {},
    }
    for index, name in enumerate(CLASS_NAMES):
        mask = targets == index
        result["by_true_class"][name] = {
            "support": int(mask.sum()),
            "reference_errors": int((mask & ~reference_correct).sum()),
            "nli_rescues": int((mask & ~reference_correct & nli_correct).sum()),
            "nli_harms": int((mask & reference_correct & ~nli_correct).sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only zero-shot NLI sentiment audit and error-complementarity "
            "analysis. This command never reads or evaluates the test split."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--reference-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--hypothesis",
        action="append",
        default=None,
        help=(
            "Ordered Negative/Neutral/Positive hypothesis. Repeat exactly three "
            "times; defaults to the canonical sentiment descriptions."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_length < 8:
        raise ValueError("batch-size must be positive and max-length must be at least 8")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    loaded = load_pickle(args.data)
    # Restrict the object before label attachment so this validation audit does
    # not even materialize targets for the held-out test split.
    raw = attach_labels_from_excel({"valid": loaded["valid"]}, args.labels)
    valid = parse_split(raw, "valid", "text_shared", require_labels=True)
    load_kwargs = {
        "revision": args.revision,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, **load_kwargs
    ).to(device)
    entailment_index = _entailment_index(model.config)

    hypotheses = tuple(args.hypothesis or DEFAULT_HYPOTHESES)
    if len(hypotheses) != 3:
        raise ValueError("--hypothesis must be omitted or repeated exactly three times")
    probability = predict_nli(
        model=model,
        tokenizer=tokenizer,
        texts=[str(value) for value in valid.raw_text],
        hypotheses=hypotheses,
        entailment_index=entailment_index,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        amp=not args.no_amp,
    )
    reference_probability, reference_regression = _load_reference_predictions(
        args.reference_predictions, valid.ids, valid.class_labels
    )
    nli_regression = 3.0 * (probability[:, 2] - probability[:, 0])
    nli_metrics = compute_metrics(
        valid.class_labels,
        probability,
        valid.regression_labels,
        nli_regression,
    )
    reference_metrics = compute_metrics(
        valid.class_labels,
        reference_probability,
        valid.regression_labels,
        reference_regression,
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prediction = probability.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "id": [str(value) for value in valid.ids],
            "predicted_label": [CLASS_NAMES[index] for index in prediction],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": nli_regression,
            "true_label": [CLASS_NAMES[index] for index in valid.class_labels],
            "true_intensity": valid.regression_labels,
        }
    )
    frame.to_csv(output / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    report = {
        "scope": "validation_only_no_test_access",
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_commit_hash": getattr(model.config, "_commit_hash", None),
        "samples": valid.size,
        "entailment_index": entailment_index,
        "hypotheses_in_class_order": list(hypotheses),
        "nli_zero_shot": nli_metrics,
        "reference": {
            "predictions": str(Path(args.reference_predictions)),
            "metrics_recomputed": reference_metrics,
        },
        "error_complementarity": _complementarity(
            valid.class_labels, reference_probability, probability
        ),
        "regression_note": (
            "NLI predicted_intensity is only a diagnostic polarity expectation "
            "3*(P(positive)-P(negative)); it is not a trained regression head."
        ),
    }
    save_json(report, output / "final_metrics.json")
    print(
        "NLI valid: "
        f"accuracy={nli_metrics['accuracy']:.4f}, "
        f"macro_f1={nli_metrics['macro_f1']:.4f}, "
        f"reference_errors_rescued="
        f"{report['error_complementarity']['reference_wrong_nli_correct']}, "
        f"oracle_any_correct="
        f"{report['error_complementarity']['oracle_any_correct_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
