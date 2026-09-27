from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from .data import (
    SplitArrays,
    attach_labels_from_excel,
    compute_class_weights,
    load_pickle,
    parse_split,
)
from .diagnose_nli_sentiment import _complementarity, _load_reference_predictions
from .metrics import CLASS_NAMES, compute_metrics
from .train_prompt_neutral_router import binary_metrics, hurdle_probability
from .utils import (
    apply_overrides,
    atomic_torch_save,
    load_config,
    make_logger,
    resolve_device,
    save_json,
    seed_everything,
)


class EncodedSubjectivityDataset(Dataset):
    def __init__(
        self,
        arrays: SplitArrays,
        tokenized: Mapping[str, Tensor],
    ) -> None:
        self.arrays = arrays
        self.tokenized = dict(tokenized)

    def __len__(self) -> int:
        return self.arrays.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            name: value[index] for name, value in self.tokenized.items()
        }
        item.update(
            class_label=torch.tensor(
                self.arrays.class_labels[index], dtype=torch.long
            ),
            regression_label=torch.tensor(
                self.arrays.regression_labels[index], dtype=torch.float32
            ),
            id=str(self.arrays.ids[index]),
        )
        return item


class SubjectivityResidualNeutralExpert(nn.Module):
    """Adapt objective/subjective evidence into a supervised Neutral hurdle.

    The pretrained OBJ-vs-SUBJ margin supplies an identifiable prior. A zero-initialized
    residual head corrects domain mismatch, while a three-class auxiliary head keeps the
    LoRA representation sentiment-aware without taking over the final polar decision.
    """

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_name = str(cfg["pretrained_model"])
        base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=cfg.get("revision"),
            local_files_only=bool(cfg.get("local_files_only", False)),
        )
        labels = {
            int(index): str(label).upper()
            for index, label in base.config.id2label.items()
        }
        self.prior_mode = str(cfg.get("prior_mode", "objective_subjective"))
        if self.prior_mode == "objective_subjective":
            if labels.get(0) != "OBJ" or labels.get(1) != "SUBJ":
                raise ValueError(
                    "objective_subjective prior requires label order "
                    f"{{0: OBJ, 1: SUBJ}}; received {base.config.id2label}"
                )
            self.prior_label_index = 0
        elif self.prior_mode == "neutral_label":
            indices = [index for index, label in labels.items() if label == "NEUTRAL"]
            if len(indices) != 1:
                raise ValueError(
                    "neutral_label prior requires exactly one Neutral checkpoint label; "
                    f"received {base.config.id2label}"
                )
            self.prior_label_index = indices[0]
        else:
            raise ValueError(
                "model.prior_mode must be 'objective_subjective' or 'neutral_label'"
            )
        if bool(cfg.get("gradient_checkpointing", True)):
            base.gradient_checkpointing_enable()
        base.config.use_cache = False
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(cfg.get("lora_rank", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.08)),
            target_modules=list(cfg.get("lora_target_modules", ["query_proj", "value_proj"])),
            bias="none",
        )
        self.encoder = get_peft_model(base, lora)
        hidden_size = int(base.config.hidden_size)
        residual_hidden = int(cfg.get("residual_hidden", 64))
        dropout = float(cfg.get("dropout", 0.10))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, residual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, 1),
        )
        self.sentiment_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, residual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, 3),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.use_prototype_head = bool(cfg.get("use_prototype_head", False))
        if self.use_prototype_head:
            prototype_dimension = int(cfg.get("prototype_dimension", 128))
            self.prototype_projection = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, prototype_dimension),
                nn.GELU(),
            )
            self.prototypes = nn.Parameter(torch.empty(2, prototype_dimension))
            nn.init.normal_(self.prototypes, std=0.02)
            self.prototype_temperature = float(
                cfg.get("prototype_temperature", 0.20)
            )
            if self.prototype_temperature <= 0.0:
                raise ValueError("prototype_temperature must be positive")
            self.prototype_scale_raw = nn.Parameter(
                torch.tensor(float(cfg.get("prototype_scale_raw", -1.5)))
            )
        self.log_temperature = nn.Parameter(
            torch.tensor(float(cfg.get("log_temperature", 0.0)))
        )

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        encoded = {
            name: value
            for name, value in batch.items()
            if name in {"input_ids", "attention_mask", "token_type_ids"}
        }
        outputs = self.encoder(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        )
        pooled = outputs.hidden_states[-1][:, 0]
        if self.prior_mode == "objective_subjective":
            subjectivity_margin = (
                outputs.logits.float()[:, 0] - outputs.logits.float()[:, 1]
            )
        else:
            # Multi-label emotion checkpoints expose a calibrated Neutral log-odds
            # directly; subtracting all other ontology logits would scale with the
            # number of emotion labels and destroy that calibration.
            subjectivity_margin = outputs.logits.float()[:, self.prior_label_index]
        correction = self.residual_head(pooled).squeeze(-1).float()
        prototype_margin = correction.new_zeros(correction.shape)
        prototype_scale = correction.new_zeros(())
        if self.use_prototype_head:
            embedding = F.normalize(self.prototype_projection(pooled).float(), dim=-1)
            prototypes = F.normalize(self.prototypes.float(), dim=-1)
            prototype_logits = embedding @ prototypes.transpose(0, 1)
            prototype_margin = (
                prototype_logits[:, 1] - prototype_logits[:, 0]
            ) / self.prototype_temperature
            prototype_scale = F.softplus(self.prototype_scale_raw.float())
        temperature = self.log_temperature.float().exp().clamp(0.25, 4.0)
        neutral_logit = (
            subjectivity_margin + correction + prototype_scale * prototype_margin
        ) / temperature
        return {
            "neutral_logit": neutral_logit,
            "subjectivity_margin": subjectivity_margin,
            "residual_correction": correction,
            "prototype_margin": prototype_margin,
            "prototype_scale": prototype_scale,
            "sentiment_logits": self.sentiment_head(pooled).float(),
            "temperature": temperature,
        }


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for name, value in batch.items()
    }


def trainable_state_dict(model: nn.Module) -> dict[str, Tensor]:
    state = model.state_dict()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing = sorted(trainable_names.difference(state))
    if missing:
        raise RuntimeError(f"Trainable parameters absent from state_dict: {missing[:5]}")
    return {
        name: state[name].detach().cpu().clone()
        for name in sorted(trainable_names)
    }


def load_trainable_state(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != trainable_names:
        missing = sorted(trainable_names.difference(state))
        extra = sorted(set(state).difference(trainable_names))
        raise RuntimeError(
            f"Checkpoint/trainable-state mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    model.load_state_dict(dict(state), strict=False)


def make_prediction_frame(
    arrays: SplitArrays,
    probability: np.ndarray,
    regression: np.ndarray,
    neutral_probability: np.ndarray,
    subjectivity_margin: np.ndarray,
    residual_correction: np.ndarray,
    auxiliary_probability: np.ndarray,
) -> pd.DataFrame:
    prediction = probability.argmax(axis=1)
    return pd.DataFrame(
        {
            "id": [str(value) for value in arrays.ids],
            "predicted_label": [CLASS_NAMES[index] for index in prediction],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression,
            "true_label": [CLASS_NAMES[index] for index in arrays.class_labels],
            "true_intensity": arrays.regression_labels,
            "subjectivity_neutral_probability": neutral_probability,
            "subjectivity_margin": subjectivity_margin,
            "residual_correction": residual_correction,
            "aux_negative_probability": auxiliary_probability[:, 0],
            "aux_neutral_probability": auxiliary_probability[:, 1],
            "aux_positive_probability": auxiliary_probability[:, 2],
        }
    )


@torch.inference_mode()
def predict_expert(
    model: SubjectivityResidualNeutralExpert,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    neutral: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    corrections: list[np.ndarray] = []
    auxiliary: list[np.ndarray] = []
    identifiers: list[str] = []
    temperatures: list[float] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.amp.autocast(
            "cuda", enabled=amp and device.type == "cuda", dtype=torch.float16
        ):
            outputs = model(batch)
        neutral.append(torch.sigmoid(outputs["neutral_logit"]).cpu().numpy())
        margins.append(outputs["subjectivity_margin"].cpu().numpy())
        corrections.append(outputs["residual_correction"].cpu().numpy())
        auxiliary.append(
            torch.softmax(outputs["sentiment_logits"], dim=-1).cpu().numpy()
        )
        identifiers.extend([str(value) for value in cpu_batch["id"]])
        temperatures.append(float(outputs["temperature"].cpu()))
    return {
        "ids": np.asarray(identifiers, dtype=str),
        "neutral_probability": np.concatenate(neutral),
        "subjectivity_margin": np.concatenate(margins),
        "residual_correction": np.concatenate(corrections),
        "auxiliary_probability": np.concatenate(auxiliary),
        "temperature": float(np.mean(temperatures)),
    }


def evaluate_split(
    model: SubjectivityResidualNeutralExpert,
    loader: DataLoader,
    arrays: SplitArrays,
    reference_probability: np.ndarray,
    reference_regression: np.ndarray,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    expert = predict_expert(model, loader, device, amp)
    expected_ids = np.asarray([str(value) for value in arrays.ids], dtype=str)
    if not np.array_equal(expected_ids, expert["ids"]):
        raise ValueError("Evaluation loader changed sample order or IDs")
    probability = hurdle_probability(
        expert["neutral_probability"], reference_probability
    )
    metrics = compute_metrics(
        arrays.class_labels,
        probability,
        arrays.regression_labels,
        reference_regression,
    )
    metrics["neutral_binary"] = binary_metrics(
        arrays.class_labels == 1, expert["neutral_probability"]
    )
    metrics["temperature"] = expert["temperature"]
    metrics["subjectivity_margin"] = {
        "mean": float(expert["subjectivity_margin"].mean()),
        "std": float(expert["subjectivity_margin"].std()),
    }
    metrics["residual_correction"] = {
        "mean": float(expert["residual_correction"].mean()),
        "std": float(expert["residual_correction"].std()),
        "min": float(expert["residual_correction"].min()),
        "max": float(expert["residual_correction"].max()),
    }
    metrics["complementarity"] = _complementarity(
        arrays.class_labels, reference_probability, probability
    )
    return metrics, make_prediction_frame(
        arrays,
        probability,
        reference_regression,
        expert["neutral_probability"],
        expert["subjectivity_margin"],
        expert["residual_correction"],
        expert["auxiliary_probability"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LoRA subjectivity-aware Neutral residual expert on train/valid "
            "only. The held-out test split is deliberately inaccessible."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed_everything(int(cfg["seed"]))
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output / "resolved_config.json")
    logger = make_logger(output, name="subjectivity_neutral_expert")
    logger.info("Device: %s", device)

    loaded = load_pickle(args.data)
    # Deliberately discard test before labels are attached or parsed.
    raw = attach_labels_from_excel(
        {name: loaded[name] for name in ("train", "valid")}, args.labels
    )
    arrays = {
        name: parse_split(raw, name, "text_shared", require_labels=True)
        for name in ("train", "valid")
    }
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["tokenizer"]),
        revision=model_cfg.get("tokenizer_revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    max_length = int(cfg["data"].get("max_length", 128))
    datasets = {
        name: EncodedSubjectivityDataset(
            split,
            tokenizer(
                [str(value) for value in split.raw_text],
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ),
        )
        for name, split in arrays.items()
    }
    references = {
        name: _load_reference_predictions(
            Path(args.reference) / f"{name}_predictions.csv",
            split.ids,
            split.class_labels,
        )
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
            batch_size=int(
                training_cfg.get("eval_batch_size", training_cfg["batch_size"])
            ),
            shuffle=False,
            num_workers=int(cfg["data"].get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    model = SubjectivityResidualNeutralExpert(model_cfg).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder_parameters = [
        parameter for parameter in trainable if id(parameter) in encoder_ids
    ]
    head_parameters = [
        parameter for parameter in trainable if id(parameter) not in encoder_ids
    ]
    logger.info(
        "Trainable parameters: %s / %s",
        f"{sum(parameter.numel() for parameter in trainable):,}",
        f"{sum(parameter.numel() for parameter in model.parameters()):,}",
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(training_cfg["adapter_learning_rate"]),
            },
            {
                "params": head_parameters,
                "lr": float(training_cfg["head_learning_rate"]),
            },
        ],
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    accumulation = int(training_cfg.get("gradient_accumulation", 1))
    updates_per_epoch = max(1, math.ceil(len(train_loader) / accumulation))
    total_updates = int(training_cfg["epochs"]) * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_updates * float(training_cfg.get("warmup_ratio", 0.1))
        ),
        num_training_steps=total_updates,
    )
    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    class_weights = compute_class_weights(
        arrays["train"].class_labels,
        power=float(training_cfg.get("class_weight_power", 0.5)),
        max_weight=float(training_cfg.get("class_weight_max", 3.0)),
    ).to(device)
    neutral_count = int((arrays["train"].class_labels == 1).sum())
    polar_count = arrays["train"].size - neutral_count
    neutral_pos_weight = (polar_count / max(1, neutral_count)) ** float(
        cfg["loss"].get("neutral_pos_weight_power", 0.5)
    )
    neutral_pos_weight_tensor = torch.tensor(
        neutral_pos_weight, dtype=torch.float32, device=device
    )
    logger.info("Neutral BCE positive weight: %.6f", neutral_pos_weight)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    patience = 0
    started = time.time()
    checkpoint_path = output / "best_model.pt"
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {"total": 0.0, "neutral": 0.0, "auxiliary": 0.0}
        progress = tqdm(
            train_loader, desc=f"subjectivity-neutral {epoch:02d}", leave=False
        )
        for step, cpu_batch in enumerate(progress, start=1):
            batch = move_batch(cpu_batch, device)
            with torch.amp.autocast(
                "cuda", enabled=amp and device.type == "cuda", dtype=torch.float16
            ):
                outputs = model(batch)
                neutral_loss = F.binary_cross_entropy_with_logits(
                    outputs["neutral_logit"].float(),
                    batch["class_label"].eq(1).float(),
                    pos_weight=neutral_pos_weight_tensor,
                )
                auxiliary_loss = F.cross_entropy(
                    outputs["sentiment_logits"].float(),
                    batch["class_label"],
                    weight=class_weights.float(),
                    label_smoothing=float(cfg["loss"].get("label_smoothing", 0.0)),
                )
                correction_penalty = outputs["residual_correction"].square().mean()
                loss = (
                    float(cfg["loss"].get("neutral_binary", 1.0)) * neutral_loss
                    + float(cfg["loss"].get("auxiliary_classification", 0.15))
                    * auxiliary_loss
                    + float(cfg["loss"].get("correction_penalty", 0.001))
                    * correction_penalty
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(training_cfg.get("grad_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running["total"] += float(loss.detach())
            running["neutral"] += float(neutral_loss.detach())
            running["auxiliary"] += float(auxiliary_loss.detach())

        valid_metrics, valid_predictions = evaluate_split(
            model,
            eval_loaders["valid"],
            arrays["valid"],
            *references["valid"],
            device,
            amp,
        )
        selection_cfg = cfg["selection"]
        score = (
            float(selection_cfg.get("accuracy", 0.4)) * valid_metrics["accuracy"]
            + float(selection_cfg.get("macro_f1", 0.6))
            * valid_metrics["macro_f1"]
        )
        record = {
            "epoch": epoch,
            "train_loss": {
                key: value / max(1, len(train_loader))
                for key, value in running.items()
            },
            "valid": valid_metrics,
            "selection_score": score,
        }
        history.append(record)
        save_json(history, output / "history.json")
        logger.info(
            "Epoch %02d | loss %.4f | valid accuracy %.4f macro-F1 %.4f "
            "Neutral-F1 %.4f binary-F1 %.4f | score %.5f",
            epoch,
            record["train_loss"]["total"],
            valid_metrics["accuracy"],
            valid_metrics["macro_f1"],
            valid_metrics["per_class"]["Neutral"]["f1"],
            valid_metrics["neutral_binary"]["neutral_f1"],
            score,
        )
        if score > best_score + float(training_cfg.get("min_delta", 0.0001)):
            best_score = score
            patience = 0
            atomic_torch_save(
                {
                    "trainable_state": trainable_state_dict(model),
                    "config": copy.deepcopy(cfg),
                    "epoch": epoch,
                    "valid_metrics": valid_metrics,
                    "selection_score": score,
                    "resolved_model_commit_hash": getattr(
                        model.encoder.config, "_commit_hash", model_cfg.get("revision")
                    ),
                    "resolved_tokenizer_commit_hash": getattr(
                        tokenizer, "_commit_hash", model_cfg.get("tokenizer_revision")
                    ),
                },
                checkpoint_path,
            )
            valid_predictions.to_csv(
                output / "best_valid_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
        else:
            patience += 1
            if patience >= int(training_cfg.get("early_stopping_patience", 2)):
                logger.info("Early stopping at epoch %d", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_trainable_state(model, checkpoint["trainable_state"])
    final: dict[str, Any] = {
        "scope": "train_and_validation_only_no_test_access",
        "architecture": "subjectivity_residual_neutral_hurdle",
        "best_epoch": checkpoint["epoch"],
        "best_selection_score": checkpoint["selection_score"],
        "elapsed_minutes": (time.time() - started) / 60.0,
        "reference": str(args.reference),
        "neutral_pos_weight": neutral_pos_weight,
        "checkpoint_trainable_parameter_tensors": len(checkpoint["trainable_state"]),
    }
    for name, split in arrays.items():
        metrics, predictions = evaluate_split(
            model,
            eval_loaders[name],
            split,
            *references[name],
            device,
            amp,
        )
        final[name] = metrics
        predictions.to_csv(
            output / f"{name}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    save_json(final, output / "final_metrics.json")
    logger.info("Final metrics: %s", final)


if __name__ == "__main__":
    main()

