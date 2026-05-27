"""
Financial statements fetcher.

Primary source : yfinance (no API key needed)
Supplement     : Financial Modeling Prep (FMP) free tier — richer history,
                 peer lists, key metrics.  Set FMP_API_KEY in .env to enable.
"""
import logging
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from fairprice.config import settings
from fairprice.schemas import CompanyProfile, FinancialStatements
from .base import CachedClient, RateLimiter

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/api/v3"
_fmp_limiter = RateLimiter(calls_per_minute=8)   # conservative within free 250/day cap


class FinancialsClient(CachedClient):

    def __init__(self) -> None:
        super().__init__(settings.cache_dir / "financials", settings.cache_ttl_financials)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_profile(self, ticker: str) -> CompanyProfile:
        return self._fetch_cached(f"profile:{ticker}", self._fetch_profile, ticker)

    def get_statements(self, ticker: str) -> FinancialStatements:
        return self._fetch_cached(f"statements:{ticker}", self._fetch_statements, ticker)

    def get_peers(self, ticker: str) -> list[str]:
        """Return up to 10 sector peers for relative valuation."""
        return self._fetch_cached(f"peers:{ticker}", self._fetch_peers, ticker)

    def get_key_metrics(self, ticker: str) -> pd.DataFrame:
        """FMP key-metrics endpoint: ROIC, EV/EBITDA, etc. (requires FMP key)."""
        return self._fetch_cached(f"metrics:{ticker}", self._fetch_key_metrics, ticker)

    # ------------------------------------------------------------------ #
    #  Private fetchers                                                    #
    # ------------------------------------------------------------------ #

    def _fetch_profile(self, ticker: str) -> CompanyProfile:
        info = yf.Ticker(ticker).info
        return CompanyProfile(
            ticker=ticker,
            name=info.get("longName") or info.get("shortName") or ticker,
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"),
            country=info.get("country", "Unknown"),
            currency=info.get("currency", "USD"),
            exchange=info.get("exchange", "Unknown"),
            description=info.get("longBusinessSummary", ""),
        )

    def _fetch_statements(self, ticker: str) -> FinancialStatements:
        t = yf.Ticker(ticker)

        income = _normalize_yf(t.financials)
        balance = _normalize_yf(t.balance_sheet)
        cashflow = _normalize_yf(t.cashflow)
        quarterly = _normalize_yf(t.quarterly_financials)

        # Supplement annual statements with FMP when key is available
        source = "yfinance"
        if settings.fmp_api_key:
            try:
                income, balance, cashflow = self._merge_fmp(ticker, income, balance, cashflow)
                source = "yfinance+fmp"
            except Exception as exc:
                logger.warning("FMP supplement failed for %s: %s", ticker, exc)

        ttm = _compute_ttm(quarterly, _normalize_yf(t.quarterly_cashflow))

        return FinancialStatements(
            ticker=ticker,
            income_statement=income,
            balance_sheet=balance,
            cash_flow=cashflow,
            quarterly_income=quarterly,
            ttm=ttm,
            source=source,
        )

    def _fetch_peers(self, ticker: str) -> list[str]:
        if not settings.fmp_api_key:
            logger.info("FMP key not set — peer list unavailable")
            return []
        try:
            _fmp_limiter.wait()
            url = f"{FMP_BASE}/stock_peers?symbol={ticker}&apikey={settings.fmp_api_key}"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            return data[0].get("peersList", [])[:10] if data else []
        except Exception as exc:
            logger.warning("Peer fetch failed for %s: %s", ticker, exc)
            return []

    def _fetch_key_metrics(self, ticker: str) -> pd.DataFrame:
        if not settings.fmp_api_key:
            return pd.DataFrame()
        try:
            _fmp_limiter.wait()
            url = (
                f"{FMP_BASE}/key-metrics/{ticker}"
                f"?limit={settings.financial_history_years * 4}"
                f"&apikey={settings.fmp_api_key}"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            if df.empty:
                return df
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date").sort_index()
        except Exception as exc:
            logger.warning("Key metrics fetch failed for %s: %s", ticker, exc)
            return pd.DataFrame()

    def _merge_fmp(
        self,
        ticker: str,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch FMP annual statements and add columns that yfinance omits
        (e.g. EPS, EBITDA, weighted average shares).
        """
        def _get(endpoint: str) -> pd.DataFrame:
            _fmp_limiter.wait()
            url = (
                f"{FMP_BASE}/{endpoint}/{ticker}"
                f"?limit={settings.financial_history_years}"
                f"&apikey={settings.fmp_api_key}"
            )
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            if df.empty:
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date").sort_index()

        fmp_income = _get("income-statement")
        fmp_balance = _get("balance-sheet-statement")
        fmp_cashflow = _get("cash-flow-statement")

        income = _merge_new_cols(income, fmp_income)
        balance = _merge_new_cols(balance, fmp_balance)
        cashflow = _merge_new_cols(cashflow, fmp_cashflow)

        return income, balance, cashflow


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _normalize_yf(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    yfinance returns DataFrames with dates as columns and metrics as rows.
    Transpose so rows = dates, cols = metrics, sorted ascending.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.T.copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out.columns = [str(c) for c in out.columns]
    return out.sort_index()


def _merge_new_cols(base: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    """Add columns from supplement that are not already in base, aligned by date."""
    if supplement.empty:
        return base
    new_cols = [c for c in supplement.columns if c not in base.columns]
    if not new_cols:
        return base
    return base.join(supplement[new_cols], how="left")


def _compute_ttm(quarterly: pd.DataFrame, quarterly_cf: pd.DataFrame) -> dict:
    """Sum the last 4 quarters for flow items; take latest for stock items."""
    ttm: dict = {}
    if quarterly.empty:
        return ttm

    recent_q = quarterly.tail(4)

    flow_items = [
        "Total Revenue",
        "Net Income",
        "Operating Income",
        "Gross Profit",
        "EBIT",
        "EBITDA",
    ]
    for col in flow_items:
        if col in recent_q.columns:
            val = recent_q[col].dropna()
            if not val.empty:
                ttm[col] = float(val.sum())

    if not quarterly_cf.empty:
        recent_cf = quarterly_cf.tail(4)
        cf_items = ["Free Cash Flow", "Capital Expenditure", "Operating Cash Flow"]
        for col in cf_items:
            if col in recent_cf.columns:
                val = recent_cf[col].dropna()
                if not val.empty:
                    ttm[col] = float(val.sum())

    return ttm
