"""
NLP / sentiment layer.

Entry point: SentimentClient.get_sentiment(ticker) → SentimentData

Data pipeline
─────────────
  yfinance news        ─┐
  FinViz headlines     ─┼─→ raw texts ─→ SentimentAnalyzer ─→ scores
  Alpaca News API      ─┘                                   ─→ uncertainty
                                                             ─→ guidance revision

Backend hierarchy (auto-detected at import):
  FinBERT (transformers + torch) → VADER (vaderSentiment) → keyword lexicon

Confidence adjustment applied by valuation.py:
  • Sentiment score < −0.3 AND uncertainty > 0.5  →  confidence × 0.90
  • Guidance lowered                               →  confidence × 0.95
  • Sentiment score > +0.4                         →  confidence × 1.05
  • Guidance raised                                →  confidence × 1.03
"""
from __future__ import annotations

import logging
import statistics
from typing import Optional

from fairprice.config import settings
from fairprice.data.base import CachedClient
from fairprice.nlp import sentiment as _sent
from fairprice.nlp import sources as _src
from fairprice.schemas import SentimentData

logger = logging.getLogger(__name__)

_CACHE_TTL = 3_600   # 1 hour


class SentimentClient(CachedClient):

    def __init__(self) -> None:
        super().__init__(settings.cache_dir / "nlp", _CACHE_TTL)

    def get_sentiment(self, ticker: str) -> SentimentData:
        return self._fetch_cached(
            f"sentiment:{ticker}",
            self._fetch_sentiment,
            ticker,
            ttl=_CACHE_TTL,
        )

    # ── Private ──────────────────────────────────────────────────────────────

    def _fetch_sentiment(self, ticker: str) -> SentimentData:
        # --- Gather raw texts from all sources ---
        yf_headlines             = _src.fetch_yfinance_news(ticker)
        fv_headlines, consensus  = _src.fetch_finviz(ticker)
        alpaca_texts             = _src.fetch_alpaca_news(ticker)

        all_texts   = yf_headlines + fv_headlines + alpaca_texts
        n_articles  = len(yf_headlines) + len(fv_headlines)
        n_alpaca    = len(alpaca_texts)

        sources_used: list[str] = []
        if yf_headlines:  sources_used.append("yfinance")
        if fv_headlines:  sources_used.append("finviz")
        if alpaca_texts:  sources_used.append("alpaca")

        logger.info(
            "Sentiment %s: %d yf + %d fv + %d alpaca = %d texts  [%s]",
            ticker, len(yf_headlines), len(fv_headlines), n_alpaca,
            len(all_texts), _sent.active_backend(),
        )

        if not all_texts:
            return _empty(ticker)

        # --- Score ---
        scores = _sent.score_texts(all_texts)

        pos_mean = statistics.mean(s["positive"] for s in scores)
        neg_mean = statistics.mean(s["negative"] for s in scores)
        neu_mean = statistics.mean(s["neutral"]  for s in scores)
        net      = pos_mean - neg_mean

        uncertainty = statistics.mean(
            _sent.detect_uncertainty(t) for t in all_texts
        )
        guidance = _sent.detect_guidance_revision(all_texts)

        # Prefer Alpaca headlines (richest) then FinViz, then yfinance
        top_src = alpaca_texts or fv_headlines or yf_headlines
        top     = [t[:120] for t in top_src[:5]]

        return SentimentData(
            ticker=ticker,
            sentiment_score=round(net, 4),
            positive_ratio=round(pos_mean, 4),
            negative_ratio=round(neg_mean, 4),
            neutral_ratio=round(neu_mean, 4),
            uncertainty_score=round(uncertainty, 4),
            guidance_revision=guidance,
            analyst_consensus=consensus,
            n_articles=n_articles,
            n_alpaca_articles=n_alpaca,
            backend=_sent.active_backend(),
            sources=sources_used,
            top_headlines=top,
        )


def _empty(ticker: str) -> SentimentData:
    return SentimentData(
        ticker=ticker,
        sentiment_score=0.0,
        positive_ratio=0.0,
        negative_ratio=0.0,
        neutral_ratio=1.0,
        uncertainty_score=0.0,
        guidance_revision=None,
        analyst_consensus=None,
        n_articles=0,
        n_alpaca_articles=0,
        backend=_sent.active_backend(),
        sources=[],
        top_headlines=[],
    )
