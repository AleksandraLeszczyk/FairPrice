"""
Financial statements fetcher.

Primary source : yfinance (no API key needed)
Supplement     : Financial Modeling Prep (FMP) free tier — richer history,
                 peer lists, key metrics.  Set FMP_API_KEY in .env to enable.

FMP reliability: if FMP fails 2+ times the client switches entirely to yfinance
                 for the lifetime of the instance (self._fmp_disabled = True).
"""
import logging
from typing import Any, Callable, Optional

import pandas as pd
import requests
import yfinance as yf

from fairprice.config import settings
from fairprice.schemas import CompanyProfile, FinancialStatements
from .base import CachedClient, RateLimiter

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"
_fmp_limiter = RateLimiter(calls_per_minute=8)   # conservative within free 250/day cap
_FMP_ERROR_THRESHOLD = 2  # disable FMP after this many cumulative errors


class FinancialsClient(CachedClient):

    def __init__(self) -> None:
        super().__init__(settings.cache_dir / "financials", settings.cache_ttl_financials)
        self._fmp_error_count: int = 0
        self._fmp_disabled: bool = False

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
        """Key metrics: ROIC, EV/EBITDA, etc. FMP when available, yfinance fallback."""
        return self._fetch_cached(f"metrics:{ticker}", self._fetch_key_metrics, ticker)

    # ------------------------------------------------------------------ #
    #  FMP error tracking                                                  #
    # ------------------------------------------------------------------ #

    def _try_fmp(self, label: str, fn: Callable, *args: Any) -> Any:
        """
        Run an FMP call, track failures, and raise on error.
        After _FMP_ERROR_THRESHOLD cumulative errors the client switches
        to yfinance-only mode and every subsequent call raises immediately.
        """
        if self._fmp_disabled:
            raise RuntimeError("FMP disabled — using yfinance fallback")
        try:
            result = fn(*args)
            return result
        except Exception as exc:
            self._fmp_error_count += 1
            if self._fmp_error_count >= _FMP_ERROR_THRESHOLD:
                self._fmp_disabled = True
                logger.warning(
                    "FMP disabled after %d errors (last error in '%s': %s) "
                    "— switching entirely to yfinance for this session",
                    self._fmp_error_count, label, exc,
                )
            raise

    def _fmp_available(self) -> bool:
        return bool(settings.fmp_api_key) and not self._fmp_disabled

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

        source = "yfinance"
        if self._fmp_available():
            try:
                income, balance, cashflow = self._try_fmp(
                    "merge_statements", self._merge_fmp, ticker, income, balance, cashflow
                )
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
        if self._fmp_available():
            try:
                return self._try_fmp("peers", self._fetch_peers_fmp, ticker)
            except Exception as exc:
                logger.warning("FMP peer fetch failed for %s, using yfinance fallback: %s", ticker, exc)
        return self._fetch_peers_yf(ticker)

    def _fetch_key_metrics(self, ticker: str) -> pd.DataFrame:
        if self._fmp_available():
            try:
                return self._try_fmp("key_metrics", self._fetch_key_metrics_fmp, ticker)
            except Exception as exc:
                logger.warning("FMP key metrics failed for %s, using yfinance fallback: %s", ticker, exc)
        return self._fetch_key_metrics_yf(ticker)

    # ------------------------------------------------------------------ #
    #  FMP implementations                                                 #
    # ------------------------------------------------------------------ #

    def _fetch_peers_fmp(self, ticker: str) -> list[str]:
        _fmp_limiter.wait()
        url = f"{FMP_BASE}/stock-peers?symbol={ticker}&apikey={settings.fmp_api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data or []

    def _fetch_key_metrics_fmp(self, ticker: str) -> pd.DataFrame:
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
                f"{FMP_BASE}/{endpoint}?symbol={ticker}"
                f"&limit={settings.financial_history_years}"
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
    #  yfinance fallbacks                                                  #
    # ------------------------------------------------------------------ #

    def _fetch_peers_yf(self, ticker: str) -> list[str]:
        """
        yfinance has no native peers endpoint.
        Returns an empty list — callers should handle gracefully.
        """
        logger.info("Peer list unavailable via yfinance for %s", ticker)
        return []

    def _fetch_key_metrics_yf(self, ticker: str) -> pd.DataFrame:
        """
        Build a single-row key-metrics DataFrame from yfinance .info fields.
        Covers the most commonly used ratios (P/E, P/B, EV/EBITDA, margins, ROE/ROA).
        """
        try:
            info = yf.Ticker(ticker).info
            row = {
                "peRatio":             info.get("trailingPE"),
                "forwardPE":           info.get("forwardPE"),
                "pbRatio":             info.get("priceToBook"),
                "evToEbitda":          info.get("enterpriseToEbitda"),
                "evToRevenue":         info.get("enterpriseToRevenue"),
                "roe":                 info.get("returnOnEquity"),
                "roa":                 info.get("returnOnAssets"),
                "grossProfitMargin":   info.get("grossMargins"),
                "operatingProfitMargin": info.get("operatingMargins"),
                "netProfitMargin":     info.get("profitMargins"),
                "dividendYield":       info.get("dividendYield"),
                "revenuePerShare":     info.get("revenuePerShare"),
                "currentRatio":        info.get("currentRatio"),
                "debtToEquity":        info.get("debtToEquity"),
            }
            row = {k: v for k, v in row.items() if v is not None}
            if not row:
                return pd.DataFrame()
            today = pd.Timestamp.now().normalize()
            df = pd.DataFrame([row], index=pd.DatetimeIndex([today]))
            df.index.name = "date"
            return df
        except Exception as exc:
            logger.warning("yfinance key metrics fallback failed for %s: %s", ticker, exc)
            return pd.DataFrame()


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
