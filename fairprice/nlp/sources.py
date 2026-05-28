"""
Raw text fetchers for the NLP layer.

yfinance     — headlines, always available, no key needed
FinViz       — scraped headlines + analyst consensus, no key needed
Alpaca News  — headline + summary per article via REST API
               same credentials as MarketView: ALPACA_API_KEY + ALPACA_SECRET

All fetchers return plain strings — sentiment analysis is downstream.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import yfinance as yf

from fairprice.config import settings

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_REQUEST_TIMEOUT = 12

_ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
_ALPACA_LOOKBACK_DAYS = 7


# ── yfinance news ─────────────────────────────────────────────────────────────

def fetch_yfinance_news(ticker: str, limit: int = 20) -> list[str]:
    """Return up to ``limit`` recent headline strings from yfinance."""
    try:
        news = yf.Ticker(ticker).news or []
        return [item.get("title", "") for item in news[:limit] if item.get("title")]
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", ticker, exc)
        return []


# ── Alpaca News ───────────────────────────────────────────────────────────────

def fetch_alpaca_news(ticker: str, limit: int = 30) -> list[str]:
    """
    Fetch recent news articles for ``ticker`` from the Alpaca News API.

    Each article contributes ``headline + '. ' + summary`` so the sentiment
    scorer has more context than a headline alone.

    Requires ALPACA_API_KEY and ALPACA_SECRET in .env (same credentials
    used by MarketView). Returns an empty list silently when absent.
    """
    key = settings.alpaca_api_key
    secret = settings.alpaca_secret
    if not (key and secret):
        logger.debug("Alpaca creds not set — skipping Alpaca news")
        return []

    start = (
        datetime.now(timezone.utc) - timedelta(days=_ALPACA_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "symbols": ticker,
        "limit": min(limit, 50),   # Alpaca caps at 50 per request
        "sort": "desc",
        "start": start,
        "include_content": "false",  # headline + summary is enough
    }
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }

    try:
        resp = requests.get(
            _ALPACA_NEWS_URL, params=params, headers=headers,
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        articles = resp.json().get("news", [])

        texts: list[str] = []
        for article in articles:
            headline = (article.get("headline") or "").strip()
            summary  = (article.get("summary")  or "").strip()
            if headline and summary:
                texts.append(f"{headline}. {summary}")
            elif headline:
                texts.append(headline)

        logger.debug("Alpaca news: fetched %d articles for %s", len(texts), ticker)
        return texts

    except Exception as exc:
        logger.warning("Alpaca news failed for %s: %s", ticker, exc)
        return []


# ── FinViz ────────────────────────────────────────────────────────────────────

def fetch_finviz(ticker: str) -> tuple[list[str], Optional[str]]:
    """
    Scrape FinViz quote page.

    Returns
    -------
    headlines : list[str]
        Up to 20 news headlines.
    analyst_consensus : Optional[str]
        Normalised recommendation: 'buy' | 'hold' | 'sell' | None
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.debug("beautifulsoup4 not installed — FinViz scraping disabled")
        return [], None

    url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
    try:
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return _parse_finviz_news(soup), _parse_finviz_rating(soup)
    except Exception as exc:
        logger.warning("FinViz fetch failed for %s: %s", ticker, exc)
        return [], None


def _parse_finviz_news(soup) -> list[str]:
    headlines: list[str] = []
    table = soup.find("table", id="news-table")
    if not table:
        return headlines
    for row in table.find_all("tr"):
        link = row.find("a")
        if link and link.get_text(strip=True):
            headlines.append(link.get_text(strip=True))
        if len(headlines) >= 20:
            break
    return headlines


def _parse_finviz_rating(soup) -> Optional[str]:
    try:
        cells  = soup.find_all("td", class_="snapshot-td2")
        labels = soup.find_all("td", class_="snapshot-td2-cp")
        snap = {l.get_text(strip=True): c.get_text(strip=True)
                for l, c in zip(labels, cells)}
        return _normalise_rating(snap.get("Recom", ""))
    except Exception:
        return None


def _normalise_rating(raw: str) -> Optional[str]:
    raw = raw.strip().lower()
    if not raw:
        return None
    try:
        score = float(raw)
        if score <= 2.0:  return "buy"
        if score <= 3.5:  return "hold"
        return "sell"
    except ValueError:
        pass
    if any(w in raw for w in ("buy", "outperform", "overweight")):   return "buy"
    if any(w in raw for w in ("hold", "neutral", "market perform")):  return "hold"
    if any(w in raw for w in ("sell", "underperform", "underweight")): return "sell"
    return None
