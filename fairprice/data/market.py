"""
Market data fetcher: prices, beta, VIX, treasury yields.

Primary source  : yfinance (no key needed)
VIX backup      : CBOE public CSV
Treasury yields : yfinance ^TNX (10Y) — FRED provides more granularity
                  if macro.py is populated.
"""
import logging

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from fairprice.config import settings
from fairprice.schemas import MarketData
from .base import CachedClient

logger = logging.getLogger(__name__)

CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

# yfinance symbols for yield proxies
_YIELD_10Y = "^TNX"   # CBOE 10-Year Treasury Note Yield Index (in %)
_YIELD_30Y = "^TYX"   # 30-year — used as fallback only
_VIX_SYM   = "^VIX"
_BENCHMARK  = "SPY"


class MarketClient(CachedClient):

    def __init__(self) -> None:
        super().__init__(settings.cache_dir / "market", settings.cache_ttl_prices)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_market_data(self, ticker: str) -> MarketData:
        return self._fetch_cached(
            f"market:{ticker}",
            self._fetch_market_data,
            ticker,
            ttl=settings.cache_ttl_prices,
        )

    def get_prices(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """Return OHLCV DataFrame for a single ticker, adjusted for splits/dividends."""
        return self._fetch_cached(
            f"prices:{ticker}:{period}",
            lambda: yf.Ticker(ticker).history(period=period, auto_adjust=True),
            ttl=settings.cache_ttl_prices,
        )

    # ------------------------------------------------------------------ #
    #  Private fetchers                                                    #
    # ------------------------------------------------------------------ #

    def _fetch_market_data(self, ticker: str) -> MarketData:
        t = yf.Ticker(ticker)
        info = t.info

        prices = t.history(
            period=f"{settings.price_history_years}y",
            auto_adjust=True,
        )

        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or (float(prices["Close"].iloc[-1]) if not prices.empty else 0.0)
        )

        beta_1y = _compute_beta(ticker, settings.beta_lookback_days)
        vix = _fetch_vix()
        treasury_10y = _fetch_treasury_10y()

        return MarketData(
            ticker=ticker,
            prices=prices,
            current_price=float(current_price),
            market_cap=float(info.get("marketCap") or 0),
            shares_outstanding=float(info.get("sharesOutstanding") or 0),
            beta_1y=beta_1y,
            beta_reported=float(info.get("beta") or 1.0),
            vix=vix,
            treasury_10y=treasury_10y,
            dividend_yield=float(info.get("dividendYield") or 0.0),
        )


# ------------------------------------------------------------------ #
#  Module-level helpers                                                #
# ------------------------------------------------------------------ #

def _compute_beta(ticker: str, lookback_days: int = 252) -> float:
    """
    OLS beta of ``ticker`` against SPY over the last ``lookback_days`` trading days.
    β = Cov(R_ticker, R_SPY) / Var(R_SPY)
    """
    try:
        raw = yf.download(
            [ticker, _BENCHMARK],
            period=f"{lookback_days + 20}d",
            auto_adjust=True,
            progress=False,
        )
        # yfinance returns multi-level columns when downloading multiple tickers
        closes = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
        if ticker not in closes.columns or _BENCHMARK not in closes.columns:
            return 1.0

        returns = closes.pct_change().dropna()
        returns = returns.tail(lookback_days)
        cov_matrix = returns.cov()
        beta = cov_matrix.loc[ticker, _BENCHMARK] / cov_matrix.loc[_BENCHMARK, _BENCHMARK]
        return round(float(beta), 4)
    except Exception as exc:
        logger.warning("Beta computation failed for %s: %s", ticker, exc)
        return 1.0


def _fetch_vix() -> float:
    """Latest VIX close. CBOE CSV is primary; yfinance is fallback."""
    try:
        df = pd.read_csv(CBOE_VIX_URL, parse_dates=["DATE"])
        return float(df["CLOSE"].iloc[-1])
    except Exception:
        pass
    try:
        hist = yf.Ticker(_VIX_SYM).history(period="5d")
        return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
        return 20.0   # long-run median


def _fetch_treasury_10y() -> float:
    """
    10-year Treasury yield as a decimal (e.g. 0.045 for 4.5%).
    yfinance ^TNX quotes in percent-points (e.g. 4.5).
    """
    try:
        hist = yf.Ticker(_YIELD_10Y).history(period="5d")
        return float(hist["Close"].iloc[-1]) / 100
    except Exception as exc:
        logger.warning("10Y yield fetch failed: %s", exc)
        return 0.045   # reasonable placeholder
