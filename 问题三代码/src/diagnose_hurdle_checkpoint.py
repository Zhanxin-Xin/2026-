from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import FeatureNormalizer, attach_labels_from_excel, load_pickle, parse_split
from .pretrained_fusion import PretrainedTextFusionNet
from .train_pretrained_fusion import EncodedTextDataset, encode_split, move_batch
from .utils import save_json


def hurdle_probabilities(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    neutral = torch.sigmoid(outputs["neutral_logit"].float())
    positive_given_polar = torch.sigmoid(outputs["polarity_logit"].float())
    polar = 1.0 - neutral
    return torch.stack(
        [polar * (1.0 - positive_given_polar), neutral, polar * positive_given_polar],
        dim=-1,
    )


@torch.no_grad()
def predict(
    model: PretrainedTextFusionNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, hurdle, target = [], [], []
    model.eval()
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        outputs = model(batch)
        base.append(outputs["class_probabilities"].float().cpu().numpy())
        hurdle.append(hurdle_probabilities(outputs).cpu().numpy())
        target.append(batch["class_label"].cpu().numpy())
    return np.concatenate(base), np.concatenate(hurdle), np.concatenate(target)


def metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    accuracy = float(accuracy_score(target, prediction))
    macro_f1 = float(f1_score(target, prediction, average="macro"))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "selection_score": 0.4 * accuracy + 0.6 * macro_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a trained neutral-vs-polar hurdle head without using test labels"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg: dict[str, Any] = copy.deepcopy(checkpoint["config"])
    # Keep the saved base posterior intact; the diagnostic reconstructs the
    # hurdle posterior separately from the already-trained auxiliary logits.
    cfg["model"]["use_hurdle"] = False
    model = PretrainedTextFusionNet(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    raw = attach_labels_from_excel(load_pickle(args.data), args.labels)
    arrays = {
        split: parse_split(raw, split, "text_shared", require_labels=True)
        for split in ("train", "valid")
    }
    normalizer = FeatureNormalizer(normalize_text=False, clip_value=10.0).fit(
        arrays["train"]
    )
    arrays = {name: normalizer.transform(value) for name, value in arrays.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg["model"]["pretrained_model"]),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
    )
    max_length = int(cfg["data"].get("max_length", 128))
    context_window = int(cfg["data"].get("context_window", 0))
    loaders = {
        name: DataLoader(
            EncodedTextDataset(
                value,
                encode_split(tokenizer, value, max_length, context_window),
            ),
            batch_size=16,
            shuffle=False,
            num_workers=0,
        )
        for name, value in arrays.items()
    }
    predictions = {
        name: predict(model, loader, device) for name, loader in loaders.items()
    }
    train_base, train_hurdle, train_target = predictions["train"]
    candidates = []
    for mix in np.linspace(0.0, 1.0, 21):
        probability = (1.0 - mix) * train_base + mix * train_hurdle
        candidates.append((metrics(train_target, probability)["selection_score"], mix))
    _, selected_mix = max(candidates)

    result: dict[str, Any] = {"selected_on": "train", "selected_mix": selected_mix}
    for split, (base, hurdle, target) in predictions.items():
        mixed = (1.0 - selected_mix) * base + selected_mix * hurdle
        result[split] = {
            "base": metrics(target, base),
            "hurdle": metrics(target, hurdle),
            "train_selected_mix": metrics(target, mixed),
        }
    save_json(result, Path(args.output))
    print(result)


if __name__ == "__main__":
    main()
