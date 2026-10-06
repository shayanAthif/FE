"""
FinBERT Sentiment Engine — Phase 2

Runs ProsusAI/finbert on sentence / segment text and returns:
  - positive_prob
  - negative_prob
  - neutral_prob
  - sentiment_label   ("positive" | "negative" | "neutral")

Design:
  - Singleton model loader (load once, reuse)
  - Supports CPU and GPU automatically
  - Batch inference with configurable batch_size
  - max_length=256 tokens to stay within FinBERT limits
  - Explicit memory cleanup between large batches
  - Model cached locally in models/finbert/ on first download
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from src.config import get_config, PROJECT_ROOT
from src.logger import get_logger

logger = get_logger("finbert_engine")

_MODEL_ID = "ProsusAI/finbert"
_LOCAL_CACHE = PROJECT_ROOT / "models" / "finbert"

# ──────────────────────────────────────────────────────────────────────────────
# Singleton state
# ──────────────────────────────────────────────────────────────────────────────
_tokenizer = None
_model = None
_device: Optional[torch.device] = None
_label_map: Dict[int, str] = {}  # populated on load


def _resolve_device() -> torch.device:
    """Resolve the configured device, falling back to CPU if unavailable."""
    configured = get_config().processing.device.strip().lower()
    if configured == "gpu1":
        configured = "cuda:1"

    requested = torch.device(configured)
    if requested.type == "cuda":
        if torch.cuda.is_available():
            device_index = requested.index if requested.index is not None else 0
            if device_index < torch.cuda.device_count():
                logger.info(f"FinBERT: using configured CUDA device {requested}.")
                return requested
            logger.warning(
                f"FinBERT: configured CUDA device {requested} is unavailable "
                f"(only {torch.cuda.device_count()} device(s) detected); using CPU."
            )
        else:
            logger.warning(
                f"FinBERT: configured device {requested} is unavailable; using CPU."
            )
    elif requested.type != "cpu":
        raise ValueError(
            f"Unsupported processing.device '{configured}'. Use 'cpu' or 'cuda:N'."
        )

    logger.info("FinBERT: using CPU.")
    return torch.device("cpu")


def load_finbert(force_reload: bool = False) -> None:
    """
    Load FinBERT tokenizer + model into module-level singletons.

    Downloads to models/finbert/ on first run; subsequent runs reuse cache.
    Thread-safe if called before spawning threads (standard pattern).
    """
    global _tokenizer, _model, _device, _label_map

    if _model is not None and not force_reload:
        return  # already loaded

    # Lazy import to avoid overhead when module is imported but model not needed
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    _device = _resolve_device()
    cache_dir = str(_LOCAL_CACHE)
    _LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading FinBERT from '{_MODEL_ID}' (cache: {cache_dir}) …")
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, cache_dir=cache_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID, cache_dir=cache_dir
    )
    _model.to(_device)
    _model.eval()

    # Build label map from model config (FinBERT uses 0=positive,1=negative,2=neutral)
    id2label = _model.config.id2label  # e.g. {0: 'positive', 1: 'negative', 2: 'neutral'}
    _label_map = {k: v.lower() for k, v in id2label.items()}
    logger.info(f"FinBERT loaded. Label map: {_label_map}. Device: {_device}")


def unload_finbert() -> None:
    """Free model from memory (useful after a processing stage completes)."""
    global _tokenizer, _model, _device
    _model = None
    _tokenizer = None
    _device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("FinBERT unloaded from memory.")


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def _run_batch(texts: List[str], max_length: int = 256) -> List[Dict]:
    """
    Internal: run a single batch of texts through FinBERT.
    Assumes model is already loaded. Returns list of result dicts.
    """
    if _model is None or _tokenizer is None:
        raise RuntimeError("FinBERT not loaded. Call load_finbert() first.")

    encoding = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding = {k: v.to(_device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = _model(**encoding)

    probs = _softmax(outputs.logits).cpu().numpy()  # shape: (batch, 3)

    results = []
    for row in probs:
        # Build label -> prob mapping from label_map
        label_probs = {_label_map[i]: float(row[i]) for i in range(len(row))}
        pred_label = max(label_probs, key=label_probs.get)
        results.append(
            {
                "positive_prob": round(label_probs.get("positive", 0.0), 4),
                "negative_prob": round(label_probs.get("negative", 0.0), 4),
                "neutral_prob": round(label_probs.get("neutral", 0.0), 4),
                "sentiment_label": pred_label,
            }
        )
    return results


def run_sentiment_batch(
    texts: List[str],
    batch_size: int = 16,
    max_length: int = 256,
) -> List[Dict]:
    """
    Public API: run FinBERT on a list of texts in mini-batches.

    Parameters
    ----------
    texts       : list of sentences / segments
    batch_size  : inference mini-batch size (default 16; reduce if OOM)
    max_length  : token truncation limit (default 256)

    Returns
    -------
    List of dicts with keys:
      positive_prob, negative_prob, neutral_prob, sentiment_label
    """
    load_finbert()  # no-op if already loaded

    if not texts:
        return []

    # Replace None/empty with a placeholder so indexing stays aligned
    safe_texts = [t if t and t.strip() else "[PAD]" for t in texts]

    all_results: List[Dict] = []
    n = len(safe_texts)
    for start in range(0, n, batch_size):
        chunk = safe_texts[start : start + batch_size]
        batch_results = _run_batch(chunk, max_length=max_length)
        all_results.extend(batch_results)

        # Periodic GPU cache clear (no-op on CPU)
        if torch.cuda.is_available() and (start // batch_size) % 10 == 0:
            torch.cuda.empty_cache()

    return all_results


def run_sentiment_single(text: str, max_length: int = 256) -> Dict:
    """Convenience wrapper for a single sentence."""
    results = run_sentiment_batch([text], batch_size=1, max_length=max_length)
    return results[0] if results else _empty_sentiment()


def _empty_sentiment() -> Dict:
    return {
        "positive_prob": 0.0,
        "negative_prob": 0.0,
        "neutral_prob": 0.0,
        "sentiment_label": "neutral",
    }
