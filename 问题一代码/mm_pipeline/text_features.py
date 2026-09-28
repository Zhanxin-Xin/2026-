from __future__ import annotations

import csv
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

LOG = logging.getLogger(__name__)
_ALIGN_MODELS: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_ROBERTA_MODELS: Dict[Tuple[str, bool], Tuple[Any, Any]] = {}


def _words(text: str) -> List[str]:
    return re.findall(r"\S+", text)


def _normal(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def force_align(wav: Path, transcript: str, duration: float, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """WhisperX forced alignment only: no ASR transcript is generated or used."""
    if not transcript.strip():
        return [], {"status": "empty_text", "unaligned_words": []}
    try:
        import torch
        import whisperx
    except ImportError as e:
        raise RuntimeError(f"WhisperX alignment dependency unavailable: {e}") from e
    device = cfg["text"]["whisperx_device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = whisperx.load_audio(str(wav))
    align_key = (cfg["text"]["whisperx_language"], device)
    if align_key not in _ALIGN_MODELS:
        _ALIGN_MODELS[align_key] = whisperx.load_align_model(
            language_code=cfg["text"]["whisperx_language"], device=device
        )
    model, metadata = _ALIGN_MODELS[align_key]
    # The supplied competition transcript is the sole text passed to align().
    result = whisperx.align([{"text": transcript, "start": 0.0, "end": duration}], model, metadata, audio, device, return_char_alignments=False)
    aligned = []
    for i, w in enumerate(result.get("word_segments", [])):
        item = {"aligned_index": i, "word": str(w.get("word", "")).strip(), "start": w.get("start"), "end": w.get("end"), "score": w.get("score")}
        item["valid"] = item["start"] is not None and item["end"] is not None
        aligned.append(item)
    supplied = _words(transcript)
    # Trace supplied word indices onto WhisperX output conservatively; unmatched words receive no fabricated time.
    cursor = 0
    for item in aligned:
        item["source_word_index"] = None
        target = _normal(item["word"])
        for j in range(cursor, min(len(supplied), cursor + 5)):
            if _normal(supplied[j]) == target:
                item["source_word_index"] = j
                cursor = j + 1
                break
    mapped = {x["source_word_index"] for x in aligned if x["source_word_index"] is not None and x["valid"]}
    return aligned, {"status": "ok", "source_word_count": len(supplied), "aligned_word_count": len(mapped), "unaligned_words": [{"index": i, "word": w} for i, w in enumerate(supplied) if i not in mapped], "device": device}


def roberta_word_embeddings(transcript: str, cfg: Dict[str, Any]) -> Tuple[Any, List[str]]:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    words = _words(transcript)
    model_name = cfg["text"]["roberta_model"]
    local_only = bool(cfg["text"]["local_files_only"])
    model_key = (model_name, local_only)
    try:
        if model_key not in _ROBERTA_MODELS:
            tok = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                add_prefix_space=True,
                local_files_only=local_only,
            )
            model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
            model.eval()
            _ROBERTA_MODELS[model_key] = (tok, model)
        tok, model = _ROBERTA_MODELS[model_key]
    except Exception as e:
        raise RuntimeError(f"cannot load {model_name!r} (local_files_only={local_only}); obtain/cache the fixed model explicitly: {e}") from e
    encoded = tok(words, is_split_into_words=True, return_tensors="pt", truncation=False)
    if encoded["input_ids"].shape[1] > model.config.max_position_embeddings:
        raise RuntimeError(f"tokenized transcript too long: {encoded['input_ids'].shape[1]}")
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state[0].cpu().numpy()
    ids = encoded.word_ids(0)
    out = np.zeros((len(words), hidden.shape[1]), dtype=np.float32)
    counts = np.zeros(len(words), dtype=np.int32)
    for vec, idx in zip(hidden, ids):
        if idx is not None:
            out[idx] += vec
            counts[idx] += 1
    out /= np.maximum(counts[:, None], 1)
    return out, words


def aggregate_text(word_vectors: Any, aligned: List[Dict[str, Any]], duration: float, cfg: Dict[str, Any]) -> Tuple[Any, Any]:
    import numpy as np
    n = min(cfg["time"]["max_windows"], int(math.ceil(duration / cfg["time"]["window_seconds"])))
    dim = word_vectors.shape[1] if word_vectors.ndim == 2 else 0
    sums = np.zeros((cfg["time"]["max_windows"], dim), np.float32)
    weights = np.zeros(cfg["time"]["max_windows"], np.float32)
    step = cfg["time"]["window_seconds"]
    for w in aligned:
        idx, start, end = w.get("source_word_index"), w.get("start"), w.get("end")
        if idx is None or not w.get("valid") or idx >= len(word_vectors) or end <= start:
            continue
        for k in range(max(0, int(start // step)), min(n, int(math.ceil(end / step)))):
            overlap = max(0.0, min(end, (k + 1) * step) - max(start, k * step))
            if overlap:
                sums[k] += word_vectors[idx] * overlap
                weights[k] += overlap
    mask = weights > 0
    sums[mask] /= weights[mask, None]
    return sums, mask


def save_word_alignment(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["aligned_index", "source_word_index", "word", "start", "end", "score", "valid"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
