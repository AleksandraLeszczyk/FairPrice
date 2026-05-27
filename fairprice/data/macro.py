"""
Macro-economic data fetcher via FRED (St. Louis Fed).

Requires FRED_API_KEY in .env for live data.
Without a key, returns historically reasonable hard-coded defaults and logs a warning.

Key series used:
  DGS10          — 10-Year Treasury Constant Maturity (% p.a.)
  DGS2           — 2-Year Treasury
  DTB3           — 3-Month T-Bill
  FEDFUNDS       — Effective Federal Funds Rate
  CPIAUCSL       — CPI All Urban Consumers (levels, SA)
  A191RL1Q225SBEA — Real GDP Growth Rate (%, annualized, quarterly)
  UNRATE         — Civilian Unemployment Rate
"""
import logging
from typing import Optional

import pandas as pd

from fairprice.config import settings
from fairprice.schemas import MacroData
from .base import CachedClient

logger = logging.getLogger(__name__)

FRED_SERIES: dict[str, str] = {
    "treasury_10y":   "DGS10",
    "treasury_2y":    "DGS2",
    "treasury_3m":    "DTB3",
    "fed_funds":      "FEDFUNDS",
    "cpi":            "CPIAUCSL",
    "gdp_growth":     "A191RL1Q225SBEA",
    "unemployment":   "UNRATE",
}

# Damodaran's implied ERP is updated monthly at pages.stern.nyu.edu.
# We use the long-run historical average as a stable default.
_HISTORICAL_ERP = 0.055


class MacroClient(CachedClient):

    def __init__(self) -> None:
        super().__init__(settings.cache_dir / "macro", settings.cache_ttl_macro)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_macro_data(self) -> MacroData:
        return self._fetch_cached("macro:snapshot", self._fetch_macro)

    def get_yield_curve(self) -> dict[str, float]:
        """Return {maturity: yield_decimal} for key points on the curve."""
        return self._fetch_cached("macro:yield_curve", self._fetch_yield_curve)

    # ------------------------------------------------------------------ #
    #  Private fetchers                                                    #
    # ------------------------------------------------------------------ #

    def _fetch_macro(self) -> MacroData:
        if not settings.fred_api_key:
            logger.warning(
                "FRED_API_KEY not set — macro data uses hard-coded defaults. "
                "Register at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            return _fallback_macro()

        try:
            from fredapi import Fred
        except ImportError:
            logger.error("fredapi not installed: pip install fredapi")
            return _fallback_macro()

        fred = Fred(api_key=settings.fred_api_key)
        series: dict[str, pd.Series] = {}

        for name, sid in FRED_SERIES.items():
            try:
                series[name] = fred.get_series(sid)
            except Exception as exc:
                logger.warning("FRED series %s (%s) failed: %s", name, sid, exc)

        risk_free = _latest(series.get("treasury_10y"), default=4.5) / 100
        fed_funds = _latest(series.get("fed_funds"), default=5.25) / 100
        inflation = _yoy_change(series.get("cpi"), default=0.03)
        gdp_growth = _latest(series.get("gdp_growth"), default=2.5) / 100
        unemployment = _latest(series.get("unemployment"), default=4.0) / 100

        return MacroData(
            risk_free_rate=risk_free,
            equity_risk_premium=_HISTORICAL_ERP,
            inflation_rate=inflation,
            real_gdp_growth=gdp_growth,
            unemployment_rate=unemployment,
            fed_funds_rate=fed_funds,
            vix=0.0,    # populated by caller from MarketClient if needed
            series=series,
        )

    def _fetch_yield_curve(self) -> dict[str, float]:
        if not settings.fred_api_key:
            return {
                "3m": 0.053, "2y": 0.048, "5y": 0.046,
                "10y": 0.045, "30y": 0.047,
            }

        from fredapi import Fred
        fred = Fred(api_key=settings.fred_api_key)
        curve_series = {
            "3m": "DTB3", "2y": "DGS2", "5y": "DGS5",
            "10y": "DGS10", "30y": "DGS30",
        }
        result = {}
        for label, sid in curve_series.items():
            try:
                s = fred.get_series(sid)
                result[label] = _latest(s, default=0.0) / 100
            except Exception:
                pass
        return result


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _latest(s: Optional[pd.Series], default: float = 0.0) -> float:
    if s is None or s.empty:
        return default
    val = s.dropna()
    return float(val.iloc[-1]) if not val.empty else default


def _yoy_change(s: Optional[pd.Series], default: float = 0.03) -> float:
    """Year-over-year percentage change of the latest observation."""
    if s is None:
        return default
    s = s.dropna()
    if len(s) < 13:
        return default
    return float(s.iloc[-1] / s.iloc[-13] - 1)   # ~12 monthly observations ago


def _fallback_macro() -> MacroData:
    """Conservative mid-cycle defaults when FRED is unavailable."""
    return MacroData(
        risk_free_rate=0.045,
        equity_risk_premium=0.055,
        inflation_rate=0.030,
        real_gdp_growth=0.025,
        unemployment_rate=0.040,
        fed_funds_rate=0.0525,
        vix=20.0,
    )
