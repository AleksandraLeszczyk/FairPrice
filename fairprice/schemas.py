"""
Output dataclasses shared across data, model, and valuation layers.
All monetary values are in the company's reporting currency unless noted.
Rates / yields are in decimal form (0.045 = 4.5%).
"""
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class CompanyProfile:
    ticker: str
    name: str
    sector: str
    industry: str
    country: str
    currency: str
    exchange: str
    description: str = ""


@dataclass
class FinancialStatements:
    ticker: str
    # Annual statements: rows = dates, cols = line items
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    cash_flow: pd.DataFrame
    # Last 8 quarters for TTM computation
    quarterly_income: pd.DataFrame
    # Key TTM aggregates (revenue, net income, FCF, …)
    ttm: dict = field(default_factory=dict)
    source: str = "yfinance"


@dataclass
class MarketData:
    ticker: str
    prices: pd.DataFrame        # OHLCV, adjusted, indexed by date
    current_price: float
    market_cap: float
    shares_outstanding: float
    beta_1y: float              # computed vs SPY over 252 days
    beta_reported: float        # as reported by data provider
    vix: float
    treasury_10y: float         # 10-year yield, decimal
    dividend_yield: float


@dataclass
class MacroData:
    risk_free_rate: float       # 10-year Treasury (decimal)
    equity_risk_premium: float  # implied ERP (decimal)
    inflation_rate: float       # CPI YoY (decimal)
    real_gdp_growth: float      # annualized (decimal)
    unemployment_rate: float    # decimal
    fed_funds_rate: float       # decimal
    vix: float
    series: dict = field(default_factory=dict)   # raw FRED pd.Series by name


@dataclass
class ValuationResult:
    ticker: str
    current_price: float
    # Three-model estimates
    dcf_value: Optional[float]
    relative_value: Optional[float]
    ml_value: Optional[float]
    # Ensemble
    fair_value_low: float
    fair_value_base: float
    fair_value_high: float
    margin_of_safety: float     # (fair_value_base - price) / fair_value_base
    confidence_score: float     # 0–1, based on data completeness + model agreement
    notes: list[str] = field(default_factory=list)
