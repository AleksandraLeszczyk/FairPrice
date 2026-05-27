"""
Statistical growth forecast model.

Blends two signals to estimate forward earnings/FCF growth:

  1. Historical trend  — OLS on log-revenue gives an annualised organic
                         growth rate that smooths one-off year volatility.
  2. Analyst consensus — yfinance earningsGrowth / revenueGrowth fields
                         (forward-looking; not always available).

A light mean-reversion pull toward 3% long-run growth is applied before the
final estimate is fed into a two-stage Gordon Growth Model to produce an
implied per-share price.

This avoids the need for a cross-sectional training set while still being
more principled than naively using last year's growth.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import yfinance as yf

from fairprice.schemas import FinancialStatements, MacroData, MarketData

logger = logging.getLogger(__name__)

_LONG_RUN_GROWTH = 0.03     # mean-reversion anchor
_REVERSION_WEIGHT = 0.20    # 20% pull toward long-run average
_STAGE1_YEARS = 5


def estimate(
    statements: FinancialStatements,
    market: MarketData,
    macro: MacroData,
) -> Optional[float]:
    """Return implied per-share value, or None if data is insufficient."""
    growth = _forecast_growth(statements, market)
    if growth is None:
        return None

    base_eps = _base_earnings_per_share(statements, market)
    if base_eps is None or base_eps <= 0:
        logger.info("ML skipped for %s: no positive earnings/FCF per share",
                    statements.ticker)
        return None

    return _two_stage_gordon(growth, base_eps, market, macro)


# ── Growth forecast ──────────────────────────────────────────────────────────

def _forecast_growth(
    statements: FinancialStatements,
    market: MarketData,
) -> Optional[float]:
    hist = _historical_trend_growth(statements)
    analyst = _analyst_consensus_growth(statements.ticker)

    if hist is None and analyst is None:
        return None

    if hist is not None and analyst is not None:
        # Analyst estimates are more forward-looking; weight them higher
        blended = 0.35 * hist + 0.65 * analyst
    else:
        blended = analyst if analyst is not None else hist

    # Partial mean-reversion toward long-run growth
    blended = (1 - _REVERSION_WEIGHT) * blended + _REVERSION_WEIGHT * _LONG_RUN_GROWTH
    return float(np.clip(blended, -0.05, 0.35))


def _historical_trend_growth(statements: FinancialStatements) -> Optional[float]:
    """
    Fit OLS on ln(revenue) ~ t.
    Slope ≈ annualised log growth rate (good proxy for compound growth).
    """
    inc = statements.income_statement
    for col in ["Total Revenue", "Revenue"]:
        if col in inc.columns:
            s = inc[col].dropna()
            if len(s) < 2:
                continue
            try:
                y = np.log(s.values.astype(float))
                if not np.all(np.isfinite(y)):
                    continue
                x = np.arange(len(y), dtype=float)
                # OLS: slope = Cov(x, y) / Var(x)
                slope = np.cov(x, y)[0, 1] / np.var(x)
                return float(slope)
            except Exception as exc:
                logger.debug("Trend fit failed: %s", exc)
    return None


def _analyst_consensus_growth(ticker: str) -> Optional[float]:
    """Blend earnings and revenue growth estimates from yfinance info."""
    try:
        info = yf.Ticker(ticker).info
        eps_g = info.get("earningsGrowth")
        rev_g = info.get("revenueGrowth")
        estimates = [g for g in [eps_g, rev_g] if g is not None]
        if estimates:
            return float(np.mean(estimates))
    except Exception as exc:
        logger.debug("Analyst growth unavailable for %s: %s", ticker, exc)
    return None


# ── Valuation ────────────────────────────────────────────────────────────────

def _two_stage_gordon(
    growth: float,
    base_eps: float,
    market: MarketData,
    macro: MacroData,
) -> float:
    """
    Stage 1: grow base_eps at `growth` for 5 years, discount each year.
    Terminal: Gordon Growth at clamped nominal GDP, capitalised and discounted.
    """
    beta = float(np.clip(
        market.beta_1y if 0.1 < market.beta_1y < 4.0 else market.beta_reported,
        0.3, 3.0,
    ))
    discount = macro.risk_free_rate + beta * macro.equity_risk_premium
    g_terminal = float(np.clip(
        macro.real_gdp_growth + macro.inflation_rate, 0.02, 0.04,
    ))

    if discount <= growth:
        discount = growth + 0.03
    if discount <= g_terminal:
        discount = g_terminal + 0.01

    pv, eps = 0.0, base_eps
    for year in range(1, _STAGE1_YEARS + 1):
        eps *= 1 + growth
        pv += eps / (1 + discount) ** year

    tv = eps * (1 + g_terminal) / (discount - g_terminal)
    pv += tv / (1 + discount) ** _STAGE1_YEARS

    return max(0.0, pv)


def _base_earnings_per_share(
    statements: FinancialStatements,
    market: MarketData,
) -> Optional[float]:
    """Prefer TTM FCF/share; fall back to TTM net income/share."""
    shares = market.shares_outstanding
    if shares <= 0:
        return None

    ttm = statements.ttm
    fcf = ttm.get("Free Cash Flow") or ttm.get("Operating Cash Flow")
    if fcf and fcf > 0:
        return fcf / shares

    ni = ttm.get("Net Income")
    if ni and ni > 0:
        return ni / shares

    return None
