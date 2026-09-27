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
from .diagnose_nli_sentiment import _contradiction_index, _entailment_index
from .metrics import CLASS_NAMES, compute_metrics
from .utils import (
    apply_overrides,
    atomic_torch_save,
    load_config,
    make_logger,
    resolve_device,
    save_json,
    seed_everything,
)


def encode_label_pairs(
    tokenizer: Any,
    arrays: SplitArrays,
    hypotheses: list[str],
    max_length: int,
    context_window: int = 0,
) -> dict[str, Tensor]:
    texts = [str(value) for value in arrays.raw_text]

    def encode_view(premise_texts: list[str]) -> dict[str, Tensor]:
        premises = [text for text in premise_texts for _ in hypotheses]
        paired_hypotheses = hypotheses * len(premise_texts)
        encoded = tokenizer(
            premises,
            paired_hypotheses,
            padding="max_length",
            truncation="longest_first",
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            name: value.reshape(len(texts), len(hypotheses), max_length)
            for name, value in encoded.items()
        }

    current = encode_view(texts)
    if context_window <= 0:
        return current

    contexts = build_previous_contexts(arrays, context_window)
    contextual_texts = [
        (
            f"Previous utterances: {context}\nCurrent utterance: {text}"
            if context.strip()
            else text
        )
        for text, context in zip(texts, contexts)
    ]
    contextual = encode_view(contextual_texts)
    dual_view = {
        name: torch.stack([current[name], contextual[name]], dim=1)
        for name in current
    }
    dual_view["context_available"] = torch.tensor(
        [bool(value.strip()) for value in contexts], dtype=torch.bool
    )
    return dual_view


def build_previous_contexts(arrays: SplitArrays, window: int) -> list[str]:
    """Build causal, split-local dialogue context without using any targets."""

    contexts = [""] * arrays.size
    groups: dict[str, list[tuple[int, int]]] = {}
    for index, identifier in enumerate(arrays.ids):
        text_id = str(identifier)
        try:
            video_id, clip_id = text_id.rsplit("$_$", 1)
            order = int(clip_id)
        except (ValueError, TypeError):
            video_id, order = text_id, index
        groups.setdefault(video_id, []).append((order, index))
    for group in groups.values():
        ordered = [index for _, index in sorted(group)]
        for position, index in enumerate(ordered):
            previous = ordered[max(0, position - window) : position]
            contexts[index] = " ".join(
                str(arrays.raw_text[item]) for item in previous
            )
    return contexts


class LabelPairDataset(Dataset):
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
        item["id"] = str(self.arrays.ids[index])
        if self.arrays.class_labels is not None:
            item["class_label"] = torch.tensor(
                self.arrays.class_labels[index], dtype=torch.long
            )
        if self.arrays.regression_labels is not None:
            item["regression_label"] = torch.tensor(
                self.arrays.regression_labels[index], dtype=torch.float32
            )
        return item


class NLILabelSemanticExpert(nn.Module):
    """Classify sentiment by scoring text against shared natural-language labels."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        model_name = str(cfg["pretrained_model"])
        load_kwargs = {
            "revision": cfg.get("revision"),
            "local_files_only": bool(cfg.get("local_files_only", False)),
        }
        encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        )
        self.entailment_index = _entailment_index(encoder.config)
        self.contradiction_index = _contradiction_index(encoder.config)
        self.semantic_mode = str(cfg.get("semantic_mode", "label_hypotheses"))
        if self.semantic_mode not in {
            "label_hypotheses",
            "dual_polarity",
            "multi_verbalizer",
            "neutral_residual_verbalizer",
        }:
            raise ValueError(
                "model.semantic_mode must be 'label_hypotheses', "
                "'dual_polarity', 'multi_verbalizer', or "
                "'neutral_residual_verbalizer'"
            )
        if bool(cfg.get("gradient_checkpointing", True)):
            encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=int(cfg.get("lora_rank", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=list(
                cfg.get("lora_target_modules", ["query_proj", "value_proj"])
            ),
            bias="none",
        )
        self.encoder = get_peft_model(encoder, lora)
        self.relation_class_count = int(encoder.config.num_labels)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(cfg.get("log_temperature", 0.0)))
        )
        self.class_bias = nn.Parameter(torch.zeros(3))
        if self.semantic_mode == "multi_verbalizer":
            verbalizer_classes = [
                int(value) for value in cfg["verbalizer_class_indices"]
            ]
            if set(verbalizer_classes) != {0, 1, 2}:
                raise ValueError(
                    "multi_verbalizer requires at least one verbalizer for each class"
                )
            self.register_buffer(
                "verbalizer_class_indices",
                torch.tensor(verbalizer_classes, dtype=torch.long),
            )
            prior_logits = cfg.get(
                "verbalizer_prior_logits", [0.0] * len(verbalizer_classes)
            )
            if len(prior_logits) != len(verbalizer_classes):
                raise ValueError(
                    "verbalizer_prior_logits must match verbalizer_class_indices"
                )
            self.verbalizer_logits = nn.Parameter(
                torch.tensor([float(value) for value in prior_logits])
            )
        if self.semantic_mode == "neutral_residual_verbalizer":
            self.neutral_prompt_gate_type = str(
                cfg.get("neutral_prompt_gate_type", "learned")
            )
            if self.neutral_prompt_gate_type not in {"learned", "competitive"}:
                raise ValueError(
                    "neutral_prompt_gate_type must be 'learned' or 'competitive'"
                )
            self.neutral_prompt_mix_logit = nn.Parameter(
                torch.tensor(float(cfg.get("neutral_prompt_mix_logit", -2.0)))
            )
            if self.neutral_prompt_gate_type == "learned":
                prompt_gate_hidden = int(cfg.get("neutral_prompt_gate_hidden", 16))
                self.neutral_prompt_gate = nn.Sequential(
                    nn.LayerNorm(4 * self.relation_class_count),
                    nn.Linear(
                        4 * self.relation_class_count,
                        prompt_gate_hidden,
                    ),
                    nn.GELU(),
                    nn.Dropout(float(cfg.get("dropout", 0.10))),
                    nn.Linear(prompt_gate_hidden, 1),
                )
                nn.init.zeros_(self.neutral_prompt_gate[-1].weight)
                nn.init.zeros_(self.neutral_prompt_gate[-1].bias)
            else:
                self.neutral_competition_scale = nn.Parameter(
                    torch.tensor(float(cfg.get("neutral_competition_scale", 1.0)))
                )
                self.neutral_competition_temperature = float(
                    cfg.get("neutral_competition_temperature", 1.0)
                )
        self.context_dual_view = bool(cfg.get("context_dual_view", False))
        if self.context_dual_view:
            context_hidden = int(cfg.get("context_gate_hidden", 16))
            self.context_mix_logit = nn.Parameter(
                torch.tensor(float(cfg.get("context_mix_logit", -2.0)))
            )
            self.context_gate = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, context_hidden),
                nn.GELU(),
                nn.Dropout(float(cfg.get("dropout", 0.10))),
                nn.Linear(context_hidden, 3),
            )
            nn.init.zeros_(self.context_gate[-1].weight)
            nn.init.zeros_(self.context_gate[-1].bias)
        self.use_relation_adapter = bool(cfg.get("use_relation_adapter", False))
        if self.use_relation_adapter:
            if self.semantic_mode != "label_hypotheses":
                raise ValueError(
                    "use_relation_adapter is supported only in label_hypotheses mode"
                )
            relation_hidden = int(cfg.get("relation_adapter_hidden", 24))
            self.relation_adapter = nn.Sequential(
                nn.LayerNorm(3 * self.relation_class_count),
                nn.Linear(3 * self.relation_class_count, relation_hidden),
                nn.GELU(),
                nn.Dropout(float(cfg.get("dropout", 0.10))),
                nn.Linear(relation_hidden, 3),
            )
            nn.init.zeros_(self.relation_adapter[-1].weight)
            nn.init.zeros_(self.relation_adapter[-1].bias)
        self.use_neutral_hurdle = bool(cfg.get("use_neutral_hurdle", False))
        self.neutral_hurdle_residual = bool(
            cfg.get("neutral_hurdle_residual", False)
        )
        if self.use_neutral_hurdle:
            if self.semantic_mode != "label_hypotheses":
                raise ValueError(
                    "use_neutral_hurdle is supported only in label_hypotheses mode"
                )
            hurdle_hidden = int(cfg.get("neutral_hurdle_hidden", 24))
            self.neutral_hurdle = nn.Sequential(
                nn.LayerNorm(3 * self.relation_class_count),
                nn.Linear(3 * self.relation_class_count, hurdle_hidden),
                nn.GELU(),
                nn.Dropout(float(cfg.get("dropout", 0.10))),
                nn.Linear(hurdle_hidden, 1),
            )
            nn.init.zeros_(self.neutral_hurdle[-1].weight)
            nn.init.constant_(
                self.neutral_hurdle[-1].bias,
                0.0
                if self.neutral_hurdle_residual
                else float(cfg.get("neutral_prior_logit", -1.25)),
            )
        evidence_dimension = 2 if self.semantic_mode == "dual_polarity" else 3
        if self.semantic_mode == "dual_polarity":
            self.neutral_offset = nn.Parameter(
                torch.tensor(float(cfg.get("neutral_offset", 0.0)))
            )
            correction_hidden = int(cfg.get("semantic_correction_hidden", 16))
            self.semantic_correction = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, correction_hidden),
                nn.GELU(),
                nn.Dropout(float(cfg.get("dropout", 0.10))),
                nn.Linear(correction_hidden, 3),
            )
            nn.init.zeros_(self.semantic_correction[-1].weight)
            nn.init.zeros_(self.semantic_correction[-1].bias)
        regression_hidden = int(cfg.get("regression_hidden", 16))
        self.regressor = nn.Sequential(
            nn.LayerNorm(evidence_dimension),
            nn.Linear(evidence_dimension, regression_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.get("dropout", 0.10))),
            nn.Linear(regression_hidden, 1),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        input_ids = batch["input_ids"]
        if input_ids.ndim == 3:
            batch_size, class_count, sequence_length = input_ids.shape
            view_count = 1
        elif input_ids.ndim == 4:
            batch_size, view_count, class_count, sequence_length = input_ids.shape
        else:
            raise ValueError(
                "NLI input_ids must have [batch, class, length] or "
                "[batch, view, class, length] shape"
            )
        encoder_inputs = {
            name: batch[name].reshape(
                batch_size * view_count * class_count, sequence_length
            )
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in batch
        }
        relation_logits = self.encoder(**encoder_inputs).logits.float().reshape(
            batch_size, view_count, class_count, -1
        )
        effective_context_gate: Tensor | None = None
        neutral_prompt_gate: Tensor | None = None
        if view_count == 1:
            fused_relation = relation_logits[:, 0]
        else:
            if not self.context_dual_view or view_count != 2 or class_count != 3:
                raise ValueError(
                    "dual-view inputs require context_dual_view=true, two views, "
                    "and three label hypotheses"
                )
            current_relation = relation_logits[:, 0]
            contextual_relation = relation_logits[:, 1]
            current_entailment = current_relation[:, :, self.entailment_index]
            contextual_entailment = contextual_relation[:, :, self.entailment_index]
            gate_features = torch.cat(
                [current_entailment, contextual_entailment - current_entailment],
                dim=-1,
            )
            context_gate = torch.sigmoid(
                self.context_mix_logit.float()
                + self.context_gate(gate_features).float()
            )
            available = batch["context_available"].float().unsqueeze(-1)
            effective_context_gate = available * context_gate
            fused_relation = current_relation + effective_context_gate.unsqueeze(-1) * (
                contextual_relation - current_relation
            )
        entailment = fused_relation[:, :, self.entailment_index]
        temperature = self.log_temperature.float().clamp(-2.0, 2.0).exp()
        if self.semantic_mode == "label_hypotheses":
            if class_count != 3:
                raise ValueError("label_hypotheses mode requires three hypotheses")
            semantic_evidence = entailment
            class_logits = entailment / temperature + self.class_bias.float()
            if self.use_relation_adapter:
                relation_features = fused_relation.reshape(batch_size, -1)
                class_logits = class_logits + self.relation_adapter(
                    relation_features
                ).float()
            if self.use_neutral_hurdle:
                relation_features = fused_relation.reshape(batch_size, -1)
                neutral_correction = self.neutral_hurdle(
                    relation_features
                ).squeeze(-1).float()
                if self.neutral_hurdle_residual:
                    base_neutral_logit = class_logits[:, 1] - torch.logsumexp(
                        class_logits[:, [0, 2]], dim=-1
                    )
                    neutral_logit = base_neutral_logit + neutral_correction
                else:
                    neutral_logit = neutral_correction
                polar_log_probability = torch.log_softmax(
                    class_logits[:, [0, 2]], dim=-1
                )
                neutral_log_probability = F.logsigmoid(neutral_logit)
                polar_mass_log_probability = F.logsigmoid(-neutral_logit)
                class_logits = torch.stack(
                    [
                        polar_mass_log_probability + polar_log_probability[:, 0],
                        neutral_log_probability,
                        polar_mass_log_probability + polar_log_probability[:, 1],
                    ],
                    dim=-1,
                )
        elif self.semantic_mode == "multi_verbalizer":
            if class_count != len(self.verbalizer_class_indices):
                raise ValueError(
                    "input hypothesis count does not match verbalizer_class_indices"
                )
            class_evidence: list[Tensor] = []
            verbalizer_weights = torch.zeros_like(self.verbalizer_logits.float())
            for class_index in range(3):
                indices = torch.nonzero(
                    self.verbalizer_class_indices == class_index,
                    as_tuple=False,
                ).squeeze(-1)
                weights = torch.softmax(
                    self.verbalizer_logits.float()[indices], dim=0
                )
                verbalizer_weights = verbalizer_weights.index_copy(
                    0, indices, weights
                )
                class_evidence.append(
                    (entailment[:, indices] * weights.unsqueeze(0)).sum(dim=1)
                )
            semantic_evidence = torch.stack(class_evidence, dim=-1)
            class_logits = semantic_evidence / temperature + self.class_bias.float()
        elif self.semantic_mode == "neutral_residual_verbalizer":
            if class_count != 4:
                raise ValueError(
                    "neutral_residual_verbalizer requires four ordered hypotheses"
                )
            if self.neutral_prompt_gate_type == "learned":
                prompt_features = fused_relation.reshape(batch_size, -1)
                gate_adjustment = self.neutral_prompt_gate(
                    prompt_features
                ).squeeze(-1).float()
            else:
                strongest_polar = torch.maximum(
                    entailment[:, 0], entailment[:, 3]
                )
                factual_competition = torch.tanh(
                    (entailment[:, 2] - strongest_polar)
                    / self.neutral_competition_temperature
                )
                gate_adjustment = (
                    self.neutral_competition_scale.float() * factual_competition
                )
            neutral_prompt_gate = torch.sigmoid(
                self.neutral_prompt_mix_logit.float() + gate_adjustment
            )
            neutral_evidence = entailment[:, 1] + neutral_prompt_gate * (
                entailment[:, 2] - entailment[:, 1]
            )
            semantic_evidence = torch.stack(
                [entailment[:, 0], neutral_evidence, entailment[:, 3]], dim=-1
            )
            class_logits = semantic_evidence / temperature + self.class_bias.float()
        else:
            if class_count != 2:
                raise ValueError("dual_polarity mode requires Negative/Positive hypotheses")
            contradiction = fused_relation[:, :, self.contradiction_index]
            semantic_evidence = entailment - contradiction
            polar = semantic_evidence / temperature
            neutral = self.neutral_offset.float() - torch.logsumexp(polar, dim=-1)
            structured = torch.stack(
                [polar[:, 0], neutral, polar[:, 1]], dim=-1
            )
            correction_features = torch.stack(
                [
                    polar[:, 0],
                    polar[:, 1],
                    polar.max(dim=-1).values,
                    polar.min(dim=-1).values,
                    (polar[:, 0] - polar[:, 1]).abs(),
                    neutral,
                ],
                dim=-1,
            )
            class_logits = (
                structured
                + self.semantic_correction(correction_features).float()
                + self.class_bias.float()
            )
        class_probabilities = torch.softmax(class_logits, dim=-1)
        regression = 3.0 * torch.tanh(
            self.regressor(semantic_evidence).squeeze(-1).float() / 3.0
        )
        return {
            "class_logits": class_logits,
            "class_probabilities": class_probabilities,
            "regression": regression,
            "entailment_logits": entailment,
            "semantic_evidence": semantic_evidence,
            "temperature": temperature,
            **(
                {"verbalizer_weights": verbalizer_weights}
                if self.semantic_mode == "multi_verbalizer"
                else {}
            ),
            **(
                {"neutral_hurdle_logit": neutral_logit}
                if self.use_neutral_hurdle
                else {}
            ),
            **(
                {"neutral_prompt_gate": neutral_prompt_gate}
                if neutral_prompt_gate is not None
                else {}
            ),
            **(
                {"context_gate": effective_context_gate}
                if effective_context_gate is not None
                else {}
            ),
        }


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for name, value in batch.items()
    }


def compute_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    class_weights: Tensor,
    cfg: Mapping[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    classification = F.cross_entropy(
        outputs["class_logits"],
        batch["class_label"],
        weight=class_weights,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    regression = F.smooth_l1_loss(
        outputs["regression"], batch["regression_label"].float()
    )
    neutral_binary = classification.new_zeros(())
    if "neutral_hurdle_logit" in outputs:
        positive_weight = torch.tensor(
            float(cfg.get("neutral_pos_weight", 1.0)),
            dtype=outputs["neutral_hurdle_logit"].dtype,
            device=outputs["neutral_hurdle_logit"].device,
        )
        neutral_binary = F.binary_cross_entropy_with_logits(
            outputs["neutral_hurdle_logit"],
            batch["class_label"].eq(1).float(),
            pos_weight=positive_weight,
        )
    total = (
        float(cfg.get("classification", 1.0)) * classification
        + float(cfg.get("regression", 0.0)) * regression
        + float(cfg.get("neutral_binary", 0.0)) * neutral_binary
    )
    return total, {
        "classification": float(classification.detach()),
        "regression": float(regression.detach()),
        "neutral_binary": float(neutral_binary.detach()),
    }


@torch.inference_mode()
def evaluate(
    model: NLILabelSemanticExpert,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    probabilities: list[np.ndarray] = []
    regression: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    regression_targets: list[np.ndarray] = []
    identifiers: list[str] = []
    temperatures: list[float] = []
    context_gates: list[np.ndarray] = []
    verbalizer_weights: list[np.ndarray] = []
    neutral_prompt_gates: list[np.ndarray] = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            outputs = model(batch)
        probabilities.append(outputs["class_probabilities"].cpu().numpy())
        regression.append(outputs["regression"].cpu().numpy())
        targets.append(batch["class_label"].cpu().numpy())
        regression_targets.append(batch["regression_label"].cpu().numpy())
        identifiers.extend([str(value) for value in cpu_batch["id"]])
        temperatures.append(float(outputs["temperature"].cpu()))
        if "context_gate" in outputs:
            context_gates.append(outputs["context_gate"].cpu().numpy())
        if "verbalizer_weights" in outputs:
            verbalizer_weights.append(outputs["verbalizer_weights"].cpu().numpy())
        if "neutral_prompt_gate" in outputs:
            neutral_prompt_gates.append(
                outputs["neutral_prompt_gate"].cpu().numpy()
            )
    probability = np.concatenate(probabilities)
    regression_prediction = np.concatenate(regression)
    target = np.concatenate(targets)
    regression_target = np.concatenate(regression_targets)
    metrics = compute_metrics(
        target, probability, regression_target, regression_prediction
    )
    metrics["temperature"] = float(np.mean(temperatures))
    if context_gates:
        gate = np.concatenate(context_gates)
        metrics["context_gate_mean_per_class"] = gate.mean(axis=0).tolist()
        metrics["context_available_fraction"] = float(
            np.any(gate > 0.0, axis=1).mean()
        )
    if verbalizer_weights:
        metrics["verbalizer_weights"] = np.mean(
            verbalizer_weights, axis=0
        ).tolist()
    if neutral_prompt_gates:
        gates = np.concatenate(neutral_prompt_gates)
        metrics["neutral_prompt_gate"] = {
            "mean": float(gates.mean()),
            "std": float(gates.std()),
            "min": float(gates.min()),
            "max": float(gates.max()),
        }
    frame = pd.DataFrame(
        {
            "id": identifiers,
            "predicted_label": [CLASS_NAMES[index] for index in probability.argmax(1)],
            "negative_probability": probability[:, 0],
            "neutral_probability": probability[:, 1],
            "positive_probability": probability[:, 2],
            "predicted_intensity": regression_prediction,
            "true_label": [CLASS_NAMES[index] for index in target],
            "true_intensity": regression_target,
        }
    )
    return metrics, frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LoRA NLI label-semantic sentiment expert on train/valid only. "
            "The held-out test split is deliberately inaccessible to this command."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
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
    logger = make_logger(output, name="nli_semantic_expert")
    logger.info("Device: %s", device)

    loaded = load_pickle(args.data)
    # Remove test before attaching labels so test targets are never materialized.
    raw = attach_labels_from_excel(
        {name: loaded[name] for name in ("train", "valid")}, args.labels
    )
    arrays = {
        name: parse_split(raw, name, "text_shared", require_labels=True)
        for name in ("train", "valid")
    }
    model_cfg = cfg["model"]
    hypotheses = [str(value) for value in model_cfg["hypotheses"]]
    semantic_mode = str(model_cfg.get("semantic_mode", "label_hypotheses"))
    expected_hypotheses = {
        "dual_polarity": 2,
        "label_hypotheses": 3,
        "neutral_residual_verbalizer": 4,
    }.get(semantic_mode)
    if expected_hypotheses is not None and len(hypotheses) != expected_hypotheses:
        raise ValueError(
            f"{semantic_mode} requires {expected_hypotheses} ordered hypotheses"
        )
    if semantic_mode == "multi_verbalizer" and len(
        model_cfg.get("verbalizer_class_indices", [])
    ) != len(hypotheses):
        raise ValueError(
            "multi_verbalizer requires one verbalizer_class_indices entry per hypothesis"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    max_length = int(cfg["data"].get("max_length", 128))
    context_window = int(cfg["data"].get("context_window", 0))
    if bool(model_cfg.get("context_dual_view", False)) != (context_window > 0):
        raise ValueError(
            "model.context_dual_view must be true exactly when data.context_window > 0"
        )
    datasets = {
        name: LabelPairDataset(
            split,
            encode_label_pairs(
                tokenizer,
                split,
                hypotheses,
                max_length,
                context_window=context_window,
            ),
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

    model = NLILabelSemanticExpert(model_cfg).to(device)
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
    loss_cfg = cfg["loss"]
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_macro = -float("inf")
    patience = 0
    started = time.time()
    checkpoint_path = output / "best_model.pt"
    macro_checkpoint_path = output / "best_macro_model.pt"

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"nli-semantic {epoch:02d}", leave=False)
        for step, cpu_batch in enumerate(progress, start=1):
            batch = move_batch(cpu_batch, device)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(batch)
                loss, _ = compute_loss(outputs, batch, class_weights, loss_cfg)
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
            running_loss += float(loss.detach())
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        valid_metrics, valid_predictions = evaluate(
            model, eval_loaders["valid"], device, amp
        )
        score = (
            float(cfg["selection"].get("accuracy", 0.4))
            * valid_metrics["accuracy"]
            + float(cfg["selection"].get("macro_f1", 0.6))
            * valid_metrics["macro_f1"]
        )
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "valid": valid_metrics,
            "selection_score": score,
        }
        history.append(record)
        save_json(history, output / "history.json")
        logger.info(
            "Epoch %02d | loss %.4f | valid accuracy %.4f macro-F1 %.4f "
            "neutral-F1 %.4f MAE %.4f | score %.5f",
            epoch,
            record["train_loss"],
            valid_metrics["accuracy"],
            valid_metrics["macro_f1"],
            valid_metrics["per_class"]["Neutral"]["f1"],
            valid_metrics["mae"],
            score,
        )
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "config": copy.deepcopy(cfg),
            "epoch": epoch,
            "valid_metrics": valid_metrics,
            "selection_score": score,
            "resolved_commit_hash": getattr(
                model.encoder.config, "_commit_hash", None
            ),
        }
        if valid_metrics["macro_f1"] > best_macro:
            best_macro = float(valid_metrics["macro_f1"])
            macro_payload = dict(checkpoint_payload)
            macro_payload["selection_metric"] = "macro_f1"
            atomic_torch_save(macro_payload, macro_checkpoint_path)
            valid_predictions.to_csv(
                output / "best_macro_valid_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )
        if score > best_score + float(training_cfg.get("min_delta", 0.0)):
            best_score = score
            patience = 0
            atomic_torch_save(checkpoint_payload, checkpoint_path)
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
    model.load_state_dict(checkpoint["model_state"], strict=True)
    final: dict[str, Any] = {
        "scope": "train_and_validation_only_no_test_access",
        "best_epoch": checkpoint["epoch"],
        "best_selection_score": checkpoint["selection_score"],
        "elapsed_minutes": (time.time() - started) / 60.0,
        "macro_checkpoint_valid": torch.load(
            macro_checkpoint_path, map_location="cpu", weights_only=False
        )["valid_metrics"],
    }
    for split in ("train", "valid"):
        metrics, predictions = evaluate(model, eval_loaders[split], device, amp)
        final[split] = metrics
        predictions.to_csv(
            output / f"{split}_predictions.csv", index=False, encoding="utf-8-sig"
        )
    save_json(final, output / "final_metrics.json")
    logger.info("Final metrics: %s", final)


if __name__ == "__main__":
    main()
