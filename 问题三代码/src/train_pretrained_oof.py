from __future__ import annotations

"""Generate leakage-safe grouped OOF predictions for a pretrained fusion model.

The held-out fold is never used for checkpoint selection.  Every fold is trained
for the same, pre-declared number of epochs, and videos (rather than individual
utterances) are the grouping unit.  This makes the resulting table suitable as
level-one evidence for a stacking/router model.
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import (
    FeatureNormalizer,
    SplitArrays,
    attach_labels_from_excel,
    compute_class_weights,
    load_pickle,
    parse_split,
)
from .metrics import compute_metrics
from .pretrained_fusion import PretrainedTextFusionNet
from .train_pretrained_fusion import (
    EncodedTextDataset,
    add_video_group_dro_loss,
    attach_emotion_view,
    build_cross_video_relation_references,
    build_video_group_dro,
    compute_loss,
    encode_split,
    evaluate,
    hierarchical_best_view_consistency,
    hierarchical_rdrop_consistency,
    move_batch,
    resolve_loss_config,
)
from .utils import apply_overrides, load_config, resolve_device, save_json, seed_everything


PROBABILITY_COLUMNS = (
    "negative_probability",
    "neutral_probability",
    "positive_probability",
)


def video_groups(identifiers: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.rsplit("$_$", 1)[0] if "$_$" in value else value
            for value in map(str, identifiers)
        ],
        dtype=object,
    )


def subset_arrays(arrays: SplitArrays, indices: np.ndarray) -> SplitArrays:
    indices = np.asarray(indices, dtype=np.int64)
    return SplitArrays(
        features={name: value[indices].copy() for name, value in arrays.features.items()},
        masks={name: value[indices].copy() for name, value in arrays.masks.items()},
        ids=arrays.ids[indices].copy(),
        raw_text=arrays.raw_text[indices].copy(),
        class_labels=(
            None if arrays.class_labels is None else arrays.class_labels[indices].copy()
        ),
        regression_labels=(
            None
            if arrays.regression_labels is None
            else arrays.regression_labels[indices].copy()
        ),
    )


def _loader(
    arrays: SplitArrays,
    tokenized: Mapping[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
    seed: int,
    *,
    include_video_pair_supervision: bool = False,
    cross_video_references: Mapping[str, np.ndarray] | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        EncodedTextDataset(
            arrays,
            tokenized,
            include_video_pair_supervision=include_video_pair_supervision,
            cross_video_references=cross_video_references,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _optimize_fixed_epochs(
    model: PretrainedTextFusionNet,
    cfg: Mapping[str, Any],
    train_loader: DataLoader,
    class_labels: np.ndarray,
    device: torch.device,
    fixed_epochs: int,
    stage: str,
) -> list[dict[str, float | str]]:
    training_cfg = cfg["training"]
    trainable_module_prefixes = tuple(
        str(value).strip()
        for value in training_cfg.get("trainable_module_prefixes", [])
        if str(value).strip()
    )
    if trainable_module_prefixes:
        for prefix in trainable_module_prefixes:
            model.get_submodule(prefix)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                any(
                    name == prefix or name.startswith(f"{prefix}.")
                    for prefix in trainable_module_prefixes
                )
            )
    class_weights = compute_class_weights(
        class_labels,
        power=float(training_cfg.get("class_weight_power", 0.5)),
        max_weight=float(training_cfg.get("class_weight_max", 3.0)),
    ).to(device)
    loss_cfg = resolve_loss_config(cfg["loss"], class_labels)
    all_encoder_parameters = list(model.text_encoder.parameters())
    encoder_parameters = [
        parameter for parameter in all_encoder_parameters if parameter.requires_grad
    ]
    encoder_ids = {id(parameter) for parameter in all_encoder_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(training_cfg["encoder_learning_rate"]),
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
    scheduled_epochs = int(training_cfg["epochs"])
    total_updates = scheduled_epochs * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_updates * float(training_cfg.get("warmup_ratio", 0.1))
        ),
        num_training_steps=total_updates,
    )
    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    video_group_dro, video_group_dro_coefficient = build_video_group_dro(
        training_cfg,
        loss_cfg,
        train_loader.dataset,
        device,
    )
    history: list[dict[str, float | str]] = []
    for epoch in range(1, fixed_epochs + 1):
        if trainable_module_prefixes:
            encoder_frozen = True
            model.eval()
            for prefix in trainable_module_prefixes:
                model.get_submodule(prefix).train()
        else:
            encoder_frozen = epoch <= int(
                training_cfg.get("freeze_encoder_epochs", 0)
            )
            model.set_text_encoder_trainable(not encoder_frozen)
            model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        rdrop_sum = 0.0
        for step, cpu_batch in enumerate(train_loader, start=1):
            batch = move_batch(cpu_batch, device)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(batch)
                loss, _ = compute_loss(outputs, batch, class_weights, loss_cfg)
                rdrop_coefficient = float(
                    loss_cfg.get("hierarchical_rdrop_consistency", 0.0)
                )
                best_view_coefficient = float(
                    loss_cfg.get("hierarchical_best_view_consistency", 0.0)
                )
                if rdrop_coefficient > 0.0 and best_view_coefficient > 0.0:
                    raise ValueError(
                        "symmetric R-Drop and best-view consistency are mutually exclusive"
                    )
                if rdrop_coefficient > 0.0 or best_view_coefficient > 0.0:
                    second_outputs = model(batch)
                    second_loss, _ = compute_loss(
                        second_outputs, batch, class_weights, loss_cfg
                    )
                    loss = 0.5 * (loss + second_loss)
                    if rdrop_coefficient > 0.0:
                        consistency_loss, _ = hierarchical_rdrop_consistency(
                            outputs, second_outputs, loss_cfg
                        )
                        consistency_coefficient = rdrop_coefficient
                    else:
                        consistency_loss, _ = hierarchical_best_view_consistency(
                            outputs, second_outputs, batch, loss_cfg
                        )
                        consistency_coefficient = best_view_coefficient
                    loss = loss + consistency_coefficient * consistency_loss
                    group_outputs = dict(outputs)
                    group_outputs["class_logits"] = 0.5 * (
                        outputs["class_logits"] + second_outputs["class_logits"]
                    )
                    rdrop_sum += float(consistency_loss.detach())
                else:
                    group_outputs = outputs
                loss, _ = add_video_group_dro_loss(
                    loss,
                    group_outputs,
                    batch,
                    class_weights,
                    loss_cfg,
                    video_group_dro,
                    video_group_dro_coefficient,
                )
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite OOF loss at stage={stage}, epoch={epoch}, step={step}"
                )
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
            loss_sum += float(loss.detach())
        video_group_diagnostics = (
            video_group_dro.finish_epoch()
            if video_group_dro is not None
            else {}
        )
        history.append(
            {
                "stage": stage,
                "epoch": float(epoch),
                "train_loss": loss_sum / max(1, len(train_loader)),
                "encoder_frozen": float(encoder_frozen),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "hierarchical_rdrop_loss": rdrop_sum
                / max(1, len(train_loader)),
                **video_group_diagnostics,
                **(
                    {
                        "neutral_energy_positive_weight": float(
                            loss_cfg["_neutral_energy_positive_weight"]
                        )
                    }
                    if "_neutral_energy_positive_weight" in loss_cfg
                    else {}
                ),
            }
        )
    del optimizer, scheduler, scaler
    return history


def train_fixed_epochs(
    cfg: Mapping[str, Any],
    train_arrays: SplitArrays,
    heldout_arrays: SplitArrays,
    tokenizer: Any,
    device: torch.device,
    seed: int,
    fixed_epochs: int,
    parent_cfg: Mapping[str, Any] | None = None,
    parent_fixed_epochs: int = 0,
    intermediate_cfg: Mapping[str, Any] | None = None,
    intermediate_fixed_epochs: int = 0,
    emotion_tokenizer: Any | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    seed_everything(seed)
    normalizer = FeatureNormalizer(normalize_text=False, clip_value=10.0).fit(
        train_arrays
    )
    train_arrays = normalizer.transform(train_arrays)
    heldout_arrays = normalizer.transform(heldout_arrays)
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    max_length = int(data_cfg.get("max_length", 128))
    context_window = int(data_cfg.get("context_window", 0))
    context_mode = str(data_cfg.get("context_mode", "previous"))
    emotion_max_length = int(cfg["model"].get("emotion_max_length", max_length))
    train_tokens = attach_emotion_view(
        encode_split(
            tokenizer,
            train_arrays,
            max_length,
            context_window,
            context_mode=context_mode,
        ),
        emotion_tokenizer,
        train_arrays,
        emotion_max_length,
    )
    heldout_tokens = attach_emotion_view(
        encode_split(
            tokenizer,
            heldout_arrays,
            max_length,
            context_window,
            context_mode=context_mode,
        ),
        emotion_tokenizer,
        heldout_arrays,
        emotion_max_length,
    )
    cross_video_train_references = None
    cross_video_heldout_references = None
    if bool(cfg["model"].get("use_cross_video_bilateral_relation", False)):
        references_per_class = int(
            cfg["model"].get("cross_video_references_per_class", 3)
        )
        cross_video_train_references = build_cross_video_relation_references(
            train_arrays, train_arrays, references_per_class
        )
        cross_video_heldout_references = build_cross_video_relation_references(
            train_arrays, heldout_arrays, references_per_class
        )
    heldout_loader = _loader(
        heldout_arrays,
        heldout_tokens,
        int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        False,
        seed,
        include_video_pair_supervision=False,
        cross_video_references=cross_video_heldout_references,
    )

    history: list[dict[str, Any]] = []
    transfer_reports: list[dict[str, Any]] = []
    transfer_report: dict[str, Any] = {
        "phased": parent_cfg is not None,
        "stage_count": 1 if parent_cfg is None else (3 if intermediate_cfg else 2),
        "loaded_parameters": 0,
        "missing_parameters": [],
        "transfers": transfer_reports,
    }

    def optimize_stage(
        stage_model: PretrainedTextFusionNet,
        stage_cfg: Mapping[str, Any],
        stage_epochs: int,
        stage_name: str,
    ) -> None:
        # Deployment warm starts are separate training invocations. Rebuilding
        # the loader resets its seeded shuffle stream at every stage as those
        # invocations do, rather than silently continuing the previous stream.
        stage_loader = _loader(
            train_arrays,
            train_tokens,
            int(stage_cfg["training"]["batch_size"]),
            True,
            seed,
            include_video_pair_supervision=True,
            cross_video_references=(
                cross_video_train_references
                if bool(
                    stage_cfg["model"].get(
                        "use_cross_video_bilateral_relation", False
                    )
                )
                else None
            ),
        )
        history.extend(
            _optimize_fixed_epochs(
                stage_model,
                stage_cfg,
                stage_loader,
                train_arrays.class_labels,
                device,
                stage_epochs,
                stage_name,
            )
        )
        del stage_loader

    def extract_state(
        source_model: PretrainedTextFusionNet,
    ) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in source_model.state_dict().items()
        }

    def build_from_state(
        source_state: Mapping[str, torch.Tensor],
        target_cfg: Mapping[str, Any],
        source_stage: str,
        target_stage: str,
    ) -> PretrainedTextFusionNet:
        target_model = PretrainedTextFusionNet(target_cfg["model"]).to(device)
        target_state = target_model.state_dict()
        compatible = {
            name: value
            for name, value in source_state.items()
            if name in target_state and target_state[name].shape == value.shape
        }
        load_result = target_model.load_state_dict(compatible, strict=False)
        report = {
            "source_stage": source_stage,
            "target_stage": target_stage,
            "loaded_parameters": len(compatible),
            "missing_parameters": list(load_result.missing_keys),
            "unexpected_parameters": list(load_result.unexpected_keys),
        }
        transfer_reports.append(report)
        del target_state, compatible
        return target_model

    if parent_cfg is not None:
        parent_model = PretrainedTextFusionNet(parent_cfg["model"]).to(device)
        optimize_stage(
            parent_model,
            parent_cfg,
            parent_fixed_epochs,
            "parent",
        )
        parent_state = extract_state(parent_model)
        del parent_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if intermediate_cfg is not None:
            intermediate_model = build_from_state(
                parent_state,
                intermediate_cfg,
                "parent",
                "intermediate",
            )
            del parent_state
            optimize_stage(
                intermediate_model,
                intermediate_cfg,
                intermediate_fixed_epochs,
                "intermediate",
            )
            intermediate_state = extract_state(intermediate_model)
            del intermediate_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model = build_from_state(
                intermediate_state,
                cfg,
                "intermediate",
                "final",
            )
            del intermediate_state
        else:
            model = build_from_state(parent_state, cfg, "parent", "child")
            del parent_state
        transfer_report.update(
            {
                "loaded_parameters": int(
                    sum(item["loaded_parameters"] for item in transfer_reports)
                ),
                "missing_parameters": [
                    item["missing_parameters"] for item in transfer_reports
                ],
            }
        )
    else:
        model = PretrainedTextFusionNet(cfg["model"]).to(device)
    optimize_stage(
        model,
        cfg,
        fixed_epochs,
        "final" if intermediate_cfg is not None else (
            "child" if parent_cfg is not None else "single"
        ),
    )

    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    metrics, predictions = evaluate(model, heldout_loader, device, amp)
    del model, heldout_loader
    del train_tokens, heldout_tokens
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, predictions, history, transfer_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped fixed-epoch OOF training for pretrained fusion"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fixed-epochs", type=int, required=True)
    parser.add_argument("--parent-config", default=None)
    parser.add_argument("--parent-fixed-epochs", type=int, default=0)
    parser.add_argument("--intermediate-config", default=None)
    parser.add_argument("--intermediate-fixed-epochs", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    if args.folds < 3:
        raise ValueError("At least three grouped folds are required")

    cfg = apply_overrides(load_config(args.config), args.overrides)
    cfg["seed"] = args.seed
    scheduled_epochs = int(cfg["training"]["epochs"])
    if not 1 <= args.fixed_epochs <= scheduled_epochs:
        raise ValueError("fixed-epochs must be within the configured training schedule")
    parent_cfg = None
    intermediate_cfg = None
    if args.parent_config:
        parent_cfg = apply_overrides(load_config(args.parent_config), args.overrides)
        parent_cfg["seed"] = args.seed
        parent_scheduled_epochs = int(parent_cfg["training"]["epochs"])
        if not 1 <= args.parent_fixed_epochs <= parent_scheduled_epochs:
            raise ValueError(
                "parent-fixed-epochs must be within the parent training schedule"
            )
        if str(parent_cfg["model"]["pretrained_model"]) != str(
            cfg["model"]["pretrained_model"]
        ):
            raise ValueError("Parent and child pretrained backbones must match")
    elif args.parent_fixed_epochs != 0:
        raise ValueError("parent-fixed-epochs requires parent-config")
    if args.intermediate_config:
        if parent_cfg is None:
            raise ValueError("intermediate-config requires parent-config")
        intermediate_cfg = apply_overrides(
            load_config(args.intermediate_config), args.overrides
        )
        intermediate_cfg["seed"] = args.seed
        intermediate_scheduled_epochs = int(intermediate_cfg["training"]["epochs"])
        if not 1 <= args.intermediate_fixed_epochs <= intermediate_scheduled_epochs:
            raise ValueError(
                "intermediate-fixed-epochs must be within the intermediate schedule"
            )
        if str(intermediate_cfg["model"]["pretrained_model"]) != str(
            cfg["model"]["pretrained_model"]
        ):
            raise ValueError("Intermediate and final pretrained backbones must match")
    elif args.intermediate_fixed_epochs != 0:
        raise ValueError("intermediate-fixed-epochs requires intermediate-config")
    stage_cfgs = [stage for stage in (parent_cfg, intermediate_cfg) if stage]
    for stage_cfg in stage_cfgs:
        for key, default in (
            ("max_length", 128),
            ("context_window", 0),
            ("context_mode", "previous"),
        ):
            if stage_cfg["data"].get(key, default) != cfg["data"].get(key, default):
                raise ValueError(f"All phased configs must share data.{key}")
    device = resolve_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "child": cfg,
            "parent": parent_cfg,
            "parent_fixed_epochs": args.parent_fixed_epochs,
            "intermediate": intermediate_cfg,
            "intermediate_fixed_epochs": args.intermediate_fixed_epochs,
            "child_fixed_epochs": args.fixed_epochs,
        }
        if parent_cfg is not None
        else cfg,
        output / "resolved_config.json",
    )

    loaded = load_pickle(args.data)
    # Hard boundary: this program never materializes valid or test data.
    raw = attach_labels_from_excel({"train": loaded["train"]}, args.labels)
    arrays = parse_split(raw, "train", "text_shared", require_labels=True)
    groups = video_groups(arrays.ids)
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    folds = list(splitter.split(np.zeros(arrays.size), arrays.class_labels, groups))
    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_cfg["pretrained_model"]),
        revision=model_cfg.get("revision"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    emotion_tokenizer = None
    if bool(model_cfg.get("use_emotion_evidence", False)):
        emotion_tokenizer = AutoTokenizer.from_pretrained(
            str(model_cfg["emotion_pretrained_model"]),
            revision=model_cfg.get("emotion_revision"),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
        )

    started = time.time()
    chunks: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for fold, (train_index, heldout_index) in enumerate(folds):
        train_groups = set(groups[train_index])
        heldout_groups = set(groups[heldout_index])
        overlap = train_groups.intersection(heldout_groups)
        if overlap:
            raise RuntimeError(f"Fold {fold} has group leakage: {sorted(overlap)[:3]}")
        metrics, frame, history, transfer_report = train_fixed_epochs(
            cfg,
            subset_arrays(arrays, train_index),
            subset_arrays(arrays, heldout_index),
            tokenizer,
            device,
            args.seed + fold,
            args.fixed_epochs,
            parent_cfg=parent_cfg,
            parent_fixed_epochs=args.parent_fixed_epochs,
            intermediate_cfg=intermediate_cfg,
            intermediate_fixed_epochs=args.intermediate_fixed_epochs,
            emotion_tokenizer=emotion_tokenizer,
        )
        frame.insert(0, "source_index", heldout_index)
        frame.insert(1, "fold", fold)
        chunks.append(frame)
        fold_report = {
            "fold": fold,
            "seed": args.seed + fold,
            "train_samples": int(len(train_index)),
            "heldout_samples": int(len(heldout_index)),
            "train_groups": int(len(train_groups)),
            "heldout_groups": int(len(heldout_groups)),
            "group_overlap": 0,
            "metrics": metrics,
            "history": history,
            "transfer": transfer_report,
        }
        fold_reports.append(fold_report)
        save_json(fold_report, output / f"fold_{fold}_report.json")
        print(
            f"fold={fold} heldout={len(heldout_index)} "
            f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )

    oof = pd.concat(chunks, ignore_index=True).sort_values("source_index")
    if not np.array_equal(oof["source_index"].to_numpy(), np.arange(arrays.size)):
        raise RuntimeError("OOF rows do not cover every train sample exactly once")
    if not np.array_equal(oof["id"].astype(str).to_numpy(), arrays.ids.astype(str)):
        raise RuntimeError("OOF row IDs do not preserve the original train order")
    probability = oof.loc[:, PROBABILITY_COLUMNS].to_numpy(np.float64)
    oof_metrics = compute_metrics(
        arrays.class_labels,
        probability,
        arrays.regression_labels,
        oof["predicted_intensity"].to_numpy(np.float64),
    )
    oof.to_csv(output / "train_oof_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "scope": "train_only_grouped_oof_no_valid_or_test_access",
        "architecture": "pretrained_trimodal_fusion_fixed_epoch_oof",
        "base_config": str(Path(args.config)),
        "pretrained_model": str(model_cfg["pretrained_model"]),
        "revision": model_cfg.get("revision"),
        "seed": args.seed,
        "folds": args.folds,
        "fixed_epochs": args.fixed_epochs,
        "parent_config": args.parent_config,
        "parent_fixed_epochs": args.parent_fixed_epochs,
        "intermediate_config": args.intermediate_config,
        "intermediate_fixed_epochs": args.intermediate_fixed_epochs,
        "checkpoint_selection_on_heldout_fold": False,
        "grouping": "sample id prefix before $_$",
        "samples": arrays.size,
        "groups": int(len(set(groups))),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "oof_metrics": oof_metrics,
        "fold_reports": fold_reports,
        "artifact": "train_oof_predictions.csv",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
