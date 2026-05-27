"""
Relative valuation using sector peer multiples.

Fetches EV/EBITDA, trailing P/E, and Price/FCF for each peer concurrently
(via yfinance), takes the sector median, and applies it to the target's
TTM metrics to derive an implied per-share value.

Outliers (>3σ above mean, or negative) are dropped before computing the median.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from fairprice.schemas import FinancialStatements, MarketData

logger = logging.getLogger(__name__)

_MAX_PEERS = 8
_PEER_TIMEOUT = 20      # seconds total for concurrent fetch
_WORKERS = 4

# Weights must sum to 1.0 (re-normalised automatically if a multiple is missing)
_MULTIPLE_WEIGHTS: dict[str, float] = {
    "ev_ebitda":     0.35,
    "pe_trailing":   0.30,
    "price_to_fcf":  0.35,
}


def estimate(
    statements: FinancialStatements,
    market: MarketData,
    peers: list[str],
) -> Optional[float]:
    """Return implied per-share value, or None if data is insufficient."""
    if not peers:
        logger.info("Relative skipped for %s: no peers", statements.ticker)
        return None

    target = _target_metrics(statements, market)
    if not any(v is not None and v > 0 for v in target.values()):
        logger.info("Relative skipped for %s: no TTM metrics", statements.ticker)
        return None

    peer_df = _fetch_peer_multiples(peers[:_MAX_PEERS])
    if peer_df.empty:
        return None

    implied: list[float] = []
    used_weights: list[float] = []

    for key, weight in _MULTIPLE_WEIGHTS.items():
        val = _implied_price(key, peer_df, target)
        if val is not None and val > 0:
            implied.append(val)
            used_weights.append(weight)

    if not implied:
        return None

    w = np.array(used_weights) / sum(used_weights)
    return float(np.dot(implied, w))


# ── Target company metrics ───────────────────────────────────────────────────

def _target_metrics(
    statements: FinancialStatements,
    market: MarketData,
) -> dict:
    ttm = statements.ttm
    mc = market.market_cap
    net_debt = _net_debt(statements)
    ev = mc + net_debt

    return {
        "ebitda":     ttm.get("EBITDA"),
        "net_income": ttm.get("Net Income"),
        "fcf":        ttm.get("Free Cash Flow") or ttm.get("Operating Cash Flow"),
        "ev":         ev,
        "market_cap": mc,
        "shares":     market.shares_outstanding,
    }


def _net_debt(statements: FinancialStatements) -> float:
    bs = statements.balance_sheet
    debt, cash = 0.0, 0.0
    for col in ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"]:
        if col in bs.columns:
            try:
                debt = max(0.0, float(bs[col].dropna().iloc[-1]))
                break
            except Exception:
                pass
    for col in ["Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments"]:
        if col in bs.columns:
            try:
                cash = max(0.0, float(bs[col].dropna().iloc[-1]))
                break
            except Exception:
                pass
    return debt - cash


# ── Peer data fetch ──────────────────────────────────────────────────────────

def _fetch_peer_multiples(peers: list[str]) -> pd.DataFrame:
    """Fetch yfinance info for each peer concurrently; clean outliers."""

    def _one(ticker: str) -> Optional[dict]:
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap") or 0
            fcf = info.get("freeCashflow")
            return {
                "ticker":       ticker,
                "ev_ebitda":    info.get("enterpriseToEbitda"),
                "pe_trailing":  info.get("trailingPE"),
                "price_to_fcf": mc / fcf if mc and fcf and fcf > 0 else None,
            }
        except Exception as exc:
            logger.debug("Peer %s failed: %s", ticker, exc)
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_one, t): t for t in peers}
        for fut in as_completed(futures, timeout=_PEER_TIMEOUT):
            result = fut.result()
            if result:
                rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("ticker")

    # Coerce, drop negatives, then cap outliers via IQR (robust to extremes)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        valid = df[col].dropna()
        if len(valid) > 2:
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            iqr = q3 - q1
            upper = (q3 + 3.0 * iqr) if iqr > 0 else valid.median() * 5.0
            df[col] = df[col].where((df[col] > 0) & (df[col] < upper))

    return df


# ── Multiple application ─────────────────────────────────────────────────────

def _implied_price(
    key: str,
    peer_df: pd.DataFrame,
    target: dict,
) -> Optional[float]:
    if key not in peer_df.columns:
        return None

    valid = peer_df[key].dropna()
    if valid.empty:
        return None

    median = float(valid.median())
    shares = target.get("shares") or 0
    if shares <= 0:
        return None

    if key == "ev_ebitda":
        ebitda = target.get("ebitda")
        if ebitda and ebitda > 0:
            implied_ev = median * ebitda
            # Convert EV → equity value → per share
            net_d = target["ev"] - target["market_cap"]
            return max(0.0, (implied_ev - net_d) / shares)

    elif key == "pe_trailing":
        ni = target.get("net_income")
        if ni and ni > 0:
            return max(0.0, median * ni / shares)

    elif key == "price_to_fcf":
        fcf = target.get("fcf")
        if fcf and fcf > 0:
            return max(0.0, median * fcf / shares)

    return None
