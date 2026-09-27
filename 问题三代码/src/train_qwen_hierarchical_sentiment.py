from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from .data import CLASS_NAMES, SplitArrays, attach_labels_from_excel, compute_class_weights, load_pickle, parse_split
from .metrics import compute_metrics
from .utils import apply_overrides, load_config, make_logger, resolve_device, save_json, seed_everything


PROMPT_PREFIX = (
    "Classify the sentiment of the following utterance as Negative, Neutral, or Positive.\n"
    "Utterance: "
)
PROMPT_SUFFIX = "\nSentiment:"


class PromptDataset(Dataset):
    def __init__(self, arrays: SplitArrays, encoded: Mapping[str, Tensor]) -> None:
        self.arrays = arrays
        self.encoded = dict(encoded)

    def __len__(self) -> int:
        return self.arrays.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {name: value[index] for name, value in self.encoded.items()}
        item.update(
            class_label=torch.tensor(self.arrays.class_labels[index], dtype=torch.long),
            regression_label=torch.tensor(self.arrays.regression_labels[index], dtype=torch.float32),
            id=str(self.arrays.ids[index]),
        )
        return item


def encode(tokenizer: Any, arrays: SplitArrays, max_length: int) -> dict[str, Tensor]:
    texts = [f"{PROMPT_PREFIX}{str(value)}{PROMPT_SUFFIX}" for value in arrays.raw_text]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return dict(encoded)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def initialize_verbalizer_head(model: Any, tokenizer: Any) -> None:
    score = getattr(model, "score", None)
    if score is None or not hasattr(score, "weight") or score.weight.shape[0] != 3:
        raise ValueError("Qwen sequence classifier does not expose a three-row score head")
    embedding = model.get_input_embeddings().weight.detach()
    target_norm = score.weight.detach().float().norm(dim=1).mean().clamp_min(1e-3)
    rows = []
    for label in ("negative", "neutral", "positive"):
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"Tokenizer produced no verbalizer tokens for {label}")
        vector = embedding[token_ids].float().mean(dim=0)
        rows.append(F.normalize(vector, dim=0) * target_norm)
    with torch.no_grad():
        score.weight.copy_(torch.stack(rows).to(dtype=score.weight.dtype, device=score.weight.device))


def hierarchical_loss(
    logits: Tensor,
    target: Tensor,
    class_weights: Tensor,
    neutral_pos_weight: Tensor,
    cfg: Mapping[str, float],
) -> tuple[Tensor, dict[str, Tensor]]:
    logits = logits.float()
    classification = F.cross_entropy(
        logits,
        target,
        weight=class_weights.float(),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    neutral_logit = logits[:, 1] - torch.logsumexp(logits[:, [0, 2]], dim=-1)
    neutral_target = (target == 1).float()
    neutral_hurdle = F.binary_cross_entropy_with_logits(
        neutral_logit,
        neutral_target,
        pos_weight=neutral_pos_weight.float(),
    )
    polar = target != 1
    if polar.any():
        polarity_logit = logits[polar, 2] - logits[polar, 0]
        polarity_target = (target[polar] == 2).float()
        polarity = F.binary_cross_entropy_with_logits(polarity_logit, polarity_target)
    else:
        polarity = classification.new_zeros(())
    components = {
        "classification": classification,
        "neutral_hurdle": neutral_hurdle,
        "polarity": polarity,
    }
    total = sum(float(cfg.get(name, 0.0)) * value for name, value in components.items())
    return total, components


@torch.inference_mode()
def evaluate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    probability_chunks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    regression_targets: list[np.ndarray] = []
    identifiers: list[str] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        encoded = {
            name: value
            for name, value in batch.items()
            if name in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.amp.autocast(
            "cuda", enabled=amp and device.type == "cuda", dtype=torch.float16
        ):
            logits = model(**encoded).logits.float()
        probability_chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        targets.append(batch["class_label"].cpu().numpy())
        regression_targets.append(batch["regression_label"].float().cpu().numpy())
        identifiers.extend([str(value) for value in cpu_batch["id"]])
    probability = np.concatenate(probability_chunks)
    target = np.concatenate(targets)
    regression_target = np.concatenate(regression_targets)
    regression = 3.0 * (probability[:, 2] - probability[:, 0])
    metrics = compute_metrics(target, probability, regression_target, regression)
    frame = pd.DataFrame(
        {
            "id": identifiers,
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(axis=1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": [CLASS_NAMES[index] for index in target],
            "true_intensity": regression_target,
        }
    )
    return metrics, frame


def build_model(cfg: Mapping[str, Any], tokenizer: Any, device: torch.device) -> Any:
    base = AutoModelForSequenceClassification.from_pretrained(
        str(cfg["pretrained_model"]),
        revision=cfg.get("revision"),
        local_files_only=bool(cfg.get("local_files_only", False)),
        num_labels=3,
        id2label={index: name for index, name in enumerate(CLASS_NAMES)},
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    if bool(cfg.get("verbalizer_initialize", True)):
        initialize_verbalizer_head(base, tokenizer)
    if bool(cfg.get("gradient_checkpointing", True)):
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(cfg.get("lora_rank", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
        bias="none",
    )
    model = get_peft_model(base, lora).to(device)
    # Mixed-precision PEFT keeps the frozen 1.5B backbone in FP16, while LoRA
    # matrices and the newly initialized score head must accumulate FP32 grads.
    # GradScaler intentionally rejects FP16 leaf gradients during unscale.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5 hierarchical sentiment LoRA trainer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    if args.evaluate_test:
        raise ValueError("Test access is disabled until the validation and multi-seed gates pass")

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed_everything(int(cfg["seed"]))
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "resolved_config.json")
    logger = make_logger(output, name="qwen_hierarchical_sentiment")
    logger.info("Device: %s", device)

    loaded = load_pickle(args.data)
    raw = attach_labels_from_excel(
        {name: loaded[name] for name in ("train", "valid")}, args.labels
    )
    arrays = {
        name: parse_split(raw, name, "text_shared", require_labels=True)
        for name in ("train", "valid")
    }
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_length = int(cfg["data"].get("max_length", 128))
    datasets = {
        name: PromptDataset(split, encode(tokenizer, split, max_length))
        for name, split in arrays.items()
    }
    training_cfg = cfg["training"]
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    eval_loaders = {
        name: DataLoader(
            dataset,
            batch_size=int(training_cfg.get("eval_batch_size", 4)),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }
    model = build_model(model_cfg, tokenizer, device)
    logger.info(
        "Trainable parameters: %s",
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}",
    )
    class_weights = compute_class_weights(
        arrays["train"].class_labels,
        power=float(training_cfg.get("class_weight_power", 0.5)),
        max_weight=float(training_cfg.get("class_weight_max", 3.0)),
    ).to(device)
    neutral_count = int((arrays["train"].class_labels == 1).sum())
    polar_count = arrays["train"].size - neutral_count
    neutral_pos_weight = torch.tensor(
        (polar_count / neutral_count) ** float(cfg["loss"].get("neutral_pos_weight_power", 0.35)),
        device=device,
    )
    classifier_parameters = []
    adapter_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "score" in name or "modules_to_save" in name:
            classifier_parameters.append(parameter)
        else:
            adapter_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": float(training_cfg["adapter_learning_rate"])},
            {"params": classifier_parameters, "lr": float(training_cfg["classifier_learning_rate"])},
        ],
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    accumulation = int(training_cfg.get("gradient_accumulation", 1))
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    total_updates = int(training_cfg["epochs"]) * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * float(training_cfg.get("warmup_ratio", 0.08))),
        num_training_steps=total_updates,
    )
    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score = -float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, Any]] = []
    checkpoint_dir = output / "best_adapter"
    started = time.time()
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        progress = tqdm(train_loader, desc=f"qwen-hierarchical {epoch:02d}", leave=False)
        for step, cpu_batch in enumerate(progress, start=1):
            batch = move_batch(cpu_batch, device)
            encoded_batch = {
                name: value
                for name, value in batch.items()
                if name in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with torch.amp.autocast("cuda", enabled=amp, dtype=torch.float16):
                logits = model(**encoded_batch).logits
                loss, _ = hierarchical_loss(
                    logits,
                    batch["class_label"],
                    class_weights,
                    neutral_pos_weight,
                    cfg["loss"],
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_cfg.get("grad_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += float(loss.detach())
            progress.set_postfix(loss=f"{running / step:.4f}")
        valid_metrics, valid_predictions = evaluate(model, eval_loaders["valid"], device, amp)
        selection_score = (
            float(cfg["selection"].get("accuracy", 0.4)) * valid_metrics["accuracy"]
            + float(cfg["selection"].get("macro_f1", 0.6)) * valid_metrics["macro_f1"]
        )
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "valid": valid_metrics,
            "selection_score": selection_score,
        }
        history.append(record)
        save_json(history, output / "history.json")
        logger.info(
            "Epoch %02d | loss %.4f | valid accuracy %.4f macro-F1 %.4f | score %.5f",
            epoch,
            record["train_loss"],
            valid_metrics["accuracy"],
            valid_metrics["macro_f1"],
            selection_score,
        )
        if selection_score > best_score + float(training_cfg.get("min_delta", 0.0)):
            best_score = selection_score
            best_epoch = epoch
            patience = 0
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
            valid_predictions.to_csv(
                output / "best_valid_predictions.csv", index=False, encoding="utf-8-sig"
            )
        else:
            patience += 1
            if patience >= int(training_cfg.get("early_stopping_patience", 2)):
                logger.info("Early stopping at epoch %d", epoch)
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    base = AutoModelForSequenceClassification.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=True,
        num_labels=3,
        id2label={index: name for index, name in enumerate(CLASS_NAMES)},
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, checkpoint_dir).to(device)
    final: dict[str, Any] = {
        "scope": "train_and_validation_only_no_test_access",
        "architecture": "qwen25_lora_verbalizer_hierarchical_sentiment",
        "model": str(model_cfg["pretrained_model"]),
        "revision": model_cfg.get("revision"),
        "license": "apache-2.0",
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    for name, loader in eval_loaders.items():
        metrics, predictions = evaluate(model, loader, device, amp)
        final[name] = metrics
        predictions.to_csv(
            output / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    save_json(final, output / "final_metrics.json")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
