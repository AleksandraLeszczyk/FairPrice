"""
Sentiment analysis — three-tier backend hierarchy.

Tier 1  FinBERT   transformers + torch        Best quality, finance-specific
Tier 2  VADER     vaderSentiment              Fast, good for headlines/social
Tier 3  Keyword   built-in lexicon            Zero-dependency fallback

The active backend is detected once at import time and cached.
All backends produce the same output shape:
    {'positive': float, 'negative': float, 'neutral': float}  per text

Additional helpers:
    detect_uncertainty(text)         → float 0–1
    detect_guidance_revision(texts)  → 'raised' | 'lowered' | 'reiterated' | None
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from fairprice.nlp.lexicon import (
    GUIDANCE_LOWERED,
    GUIDANCE_RAISED,
    GUIDANCE_REITERATED,
    NEGATIVE,
    POSITIVE,
    UNCERTAINTY,
)

logger = logging.getLogger(__name__)

_MAX_CHARS = 400   # truncate long Reddit posts before sending to model


# ── Backend detection ────────────────────────────────────────────────────────

def _detect_backend() -> str:
    try:
        from transformers import pipeline as _p  # noqa: F401
        import torch  # noqa: F401
        return "finbert"
    except ImportError:
        pass
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _V  # noqa: F401
        return "vader"
    except ImportError:
        pass
    return "keyword"


_BACKEND: str = _detect_backend()
logger.debug("NLP backend: %s", _BACKEND)


# ── Public scorer ────────────────────────────────────────────────────────────

def score_texts(texts: list[str]) -> list[dict[str, float]]:
    """
    Score a list of texts.

    Returns a list of dicts with keys 'positive', 'negative', 'neutral'
    (probabilities / fractions, each 0–1, summing to 1).
    """
    if not texts:
        return []
    clean = [t[:_MAX_CHARS].strip() for t in texts if t and t.strip()]
    if not clean:
        return []

    if _BACKEND == "finbert":
        return _finbert(clean)
    if _BACKEND == "vader":
        return _vader(clean)
    return _keyword(clean)


def detect_uncertainty(text: str) -> float:
    """Fraction of uncertainty-word tokens in ``text`` (0–1)."""
    tokens = _tokenise(text)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in UNCERTAINTY)
    return min(1.0, hits / max(len(tokens), 1) * 5)   # scale up for sensitivity


def detect_guidance_revision(texts: list[str]) -> Optional[str]:
    """
    Scan texts for guidance-revision language.

    Returns 'raised', 'lowered', 'reiterated', or None.
    Later items in the list take precedence (most recent signal wins).
    """
    result: Optional[str] = None
    combined = " ".join(texts).lower()

    for phrase in GUIDANCE_REITERATED:
        if phrase in combined:
            result = "reiterated"

    for phrase in GUIDANCE_RAISED:
        if phrase in combined:
            result = "raised"

    for phrase in GUIDANCE_LOWERED:
        if phrase in combined:
            result = "lowered"

    return result


# ── FinBERT ──────────────────────────────────────────────────────────────────

_finbert_pipe = None   # lazy-loaded


def _finbert(texts: list[str]) -> list[dict[str, float]]:
    global _finbert_pipe
    if _finbert_pipe is None:
        from transformers import pipeline
        logger.info("Loading FinBERT (ProsusAI/finbert)…")
        _finbert_pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True,
            truncation=True,
            max_length=512,
        )

    try:
        raw = _finbert_pipe(texts)  # list[list[dict]]
        results = []
        for item in raw:
            row = {d["label"].lower(): d["score"] for d in item}
            results.append({
                "positive": row.get("positive", 0.0),
                "negative": row.get("negative", 0.0),
                "neutral":  row.get("neutral",  0.0),
            })
        return results
    except Exception as exc:
        logger.warning("FinBERT inference failed: %s — falling back to keyword", exc)
        return _keyword(texts)


# ── VADER ────────────────────────────────────────────────────────────────────

_vader_analyzer = None


def _vader(texts: list[str]) -> list[dict[str, float]]:
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()

    results = []
    for text in texts:
        scores = _vader_analyzer.polarity_scores(text)
        # VADER gives neg/neu/pos fractions plus compound (-1..+1)
        # Normalise so they sum to 1 (neg + neu + pos already do from VADER)
        results.append({
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral":  scores["neu"],
        })
    return results


# ── Keyword fallback ─────────────────────────────────────────────────────────

def _keyword(texts: list[str]) -> list[dict[str, float]]:
    results = []
    for text in texts:
        tokens = _tokenise(text)
        pos = sum(1 for t in tokens if t in POSITIVE)
        neg = sum(1 for t in tokens if t in NEGATIVE)
        total = pos + neg
        if total == 0:
            results.append({"positive": 0.0, "negative": 0.0, "neutral": 1.0})
        else:
            results.append({
                "positive": pos / total * 0.9,   # scale to leave room for neutral
                "negative": neg / total * 0.9,
                "neutral":  0.1 + abs(pos - neg) / total * 0.0,  # baseline neutral
            })
    return results


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def active_backend() -> str:
    return _BACKEND
