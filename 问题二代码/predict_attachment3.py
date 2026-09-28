#!/usr/bin/env python3
"""Frozen TASP-MSA inference on unlabeled Attachment 3."""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import BertModel

from utils.checkpoint import load_checkpoint
from tasp_msa.model import MODALITIES, TASPMsa

NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "datasets/attachment3/aligned"
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_model.pth")
    parser.add_argument("--bert-path", type=Path, default=ROOT / ".hf_model/bert-base-uncased")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "results/attachment3_predictions.csv")
    parser.add_argument(
        "--detailed-output", type=Path,
        default=ROOT / "results/attachment3_predictions_detailed.csv",
    )
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = TASPMsa(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    bert = BertModel.from_pretrained(args.bert_path, local_files_only=True).to(device).eval()

    files = sorted(args.data_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no pkl files found under {args.data_dir}")
    names, ids, attention, token_types, audio, vision = [], [], [], [], [], []
    for path in files:
        with path.open("rb") as handle:
            root = pickle.load(handle)
        if set(root) != {"test"}:
            raise ValueError(f"unexpected outer keys in {path}: {list(root)}")
        sample = root["test"]
        if set(sample) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"unexpected fields in {path}: {list(sample)}")
        text_bert = np.asarray(sample["text_bert"])
        if text_bert.shape != (1, 3, 50):
            raise ValueError(f"{path}: text_bert shape {text_bert.shape}")
        names.append(path.stem)
        ids.append(text_bert[0, 0])
        attention.append(text_bert[0, 1])
        token_types.append(text_bert[0, 2])
        audio.append(np.asarray(sample["audio"])[0])
        vision.append(np.asarray(sample["vision"])[0])
    ids_t = torch.as_tensor(np.stack(ids), dtype=torch.long, device=device)
    attention_t = torch.as_tensor(np.stack(attention), dtype=torch.long, device=device)
    types_t = torch.as_tensor(np.stack(token_types), dtype=torch.long, device=device)
    audio_t = torch.as_tensor(np.stack(audio), dtype=torch.float32, device=device)
    vision_t = torch.as_tensor(np.stack(vision), dtype=torch.float32, device=device)
    text_t = bert(
        input_ids=ids_t, attention_mask=attention_t, token_type_ids=types_t
    ).last_hidden_state
    valid = attention_t.bool() & (ids_t != 101) & (ids_t != 102)
    masks = {
        "text": torch.ones_like(valid),
        "audio": ~(audio_t == 0).all(-1) | ~valid,
        "vision": ~(vision_t == 0).all(-1) | ~valid,
    }
    output = model(
        text=text_t, audio=audio_t, vision=vision_t,
        valid_mask_text=valid, valid_mask_audio=valid, valid_mask_vision=valid,
        missing_mask_text=masks["text"], missing_mask_audio=masks["audio"],
        missing_mask_vision=masks["vision"],
    )
    probability = output["class_probabilities"].cpu()
    prediction = probability.argmax(-1)
    regression = output["regression"].squeeze(-1).cpu()
    rows, details = [], []
    for index, name in enumerate(names):
        base = {
            "id": name,
            "predicted_polarity": NAMES[int(prediction[index])],
            "predicted_intensity": float(regression[index]),
            "prob_negative": float(probability[index, 0]),
            "prob_neutral": float(probability[index, 1]),
            "prob_positive": float(probability[index, 2]),
        }
        rows.append(base)
        missing = [m for m in MODALITIES if bool((valid[index] & ~masks[m][index]).any())]
        detail = {
            **base,
            "detected_missing_type": "+".join(missing) if missing else "None",
            "reliability_audio": float(output["reliability"]["audio"][index]),
            "reliability_vision": float(output["reliability"]["vision"][index]),
        }
        count = max(1, int(valid[index].sum()))
        for modality in MODALITIES:
            detail[f"{modality}_missing_ratio"] = float(
                (valid[index] & ~masks[modality][index]).sum()
            ) / count
        details.append(detail)
    for path, content in ((args.output, rows), (args.detailed_output, details)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    print(
        f"samples={len(rows)} checkpoint={args.checkpoint} "
        f"prediction={args.output} detailed={args.detailed_output}"
    )


if __name__ == "__main__":
    main()
