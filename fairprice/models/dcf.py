"""
3-Stage Discounted Cash Flow (DCF) valuation.

Stage 1 (years 1-5)  : explicit FCF projections at estimated growth rate
Stage 2 (years 6-10) : linear fade from stage-1 growth toward terminal rate
Terminal             : Gordon Growth Model on year-10 FCF

WACC = w_e × (rf + β × ERP) + w_d × r_d × (1 - t)

Returns (low, base, high) per-share intrinsic value, or None when FCF data
is unavailable or the company has persistently negative free cash flow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fairprice.schemas import FinancialStatements, MacroData, MarketData

logger = logging.getLogger(__name__)

_STAGE1_YEARS = 5
_STAGE2_YEARS = 5
_DEFAULT_TAX_RATE = 0.21    # US statutory rate
_MIN_WACC = 0.05
_MAX_WACC = 0.20
_BEAR_GROWTH_FACTOR = 0.60  # bear = base × 0.6
_BULL_GROWTH_FACTOR = 1.50  # bull = base × 1.5
_WACC_BEAR_BUMP = 0.010     # +100bps WACC in bear
_WACC_BULL_CUT = 0.005      # -50bps WACC in bull


@dataclass
class DCFComponents:
    """Decomposed DCF output for inspection / sensitivity analysis."""
    wacc: float
    growth_stage1: float
    terminal_growth: float
    base_fcf: float
    pv_stage1: float
    pv_stage2: float
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    per_share: float


def estimate(
    statements: FinancialStatements,
    market: MarketData,
    macro: MacroData,
) -> Optional[tuple[float, float, float]]:
    """Return (low, base, high) intrinsic value per share, or None."""
    base_fcf = _base_fcf(statements)
    if base_fcf is None or base_fcf <= 0:
        logger.info("DCF skipped for %s: no positive FCF", statements.ticker)
        return None

    wacc = _compute_wacc(statements, market, macro)
    g_terminal = _terminal_growth(macro)
    g_bear, g_base, g_bull = _growth_estimates(statements)

    low = _scenario(base_fcf, g_bear, min(wacc + _WACC_BEAR_BUMP, _MAX_WACC),
                    g_terminal, statements, market)
    base = _scenario(base_fcf, g_base, wacc, g_terminal, statements, market)
    high = _scenario(base_fcf, g_bull, max(wacc - _WACC_BULL_CUT, _MIN_WACC),
                     g_terminal, statements, market)

    return (max(0.0, low), max(0.0, base), max(0.0, high))


def get_components(
    statements: FinancialStatements,
    market: MarketData,
    macro: MacroData,
) -> Optional[DCFComponents]:
    """Return decomposed base-scenario components for debugging."""
    base_fcf = _base_fcf(statements)
    if base_fcf is None or base_fcf <= 0:
        return None
    wacc = _compute_wacc(statements, market, macro)
    g_terminal = _terminal_growth(macro)
    _, g_base, _ = _growth_estimates(statements)
    return _scenario_detailed(base_fcf, g_base, wacc, g_terminal, statements, market)


# ── WACC ────────────────────────────────────────────────────────────────────

def _compute_wacc(
    statements: FinancialStatements,
    market: MarketData,
    macro: MacroData,
) -> float:
    beta = float(np.clip(
        market.beta_1y if 0.1 < market.beta_1y < 4.0 else market.beta_reported,
        0.3, 3.0,
    ))
    r_e = macro.risk_free_rate + beta * macro.equity_risk_premium
    r_d = _cost_of_debt(statements)
    t = _tax_rate(statements)

    debt = _total_debt(statements)
    equity = market.market_cap
    total = equity + debt

    if total <= 0:
        return float(np.clip(r_e, _MIN_WACC, _MAX_WACC))

    wacc = (equity / total) * r_e + (debt / total) * r_d * (1 - t)
    return float(np.clip(wacc, _MIN_WACC, _MAX_WACC))


def _cost_of_debt(statements: FinancialStatements) -> float:
    inc = statements.income_statement
    bs = statements.balance_sheet
    interest_col = _first_col(inc, ["Interest Expense", "Interest Expense Non Operating"])
    debt_col = _first_col(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation",
                                "Long Term Debt"])
    if interest_col and debt_col:
        try:
            interest = abs(float(inc[interest_col].dropna().iloc[-1]))
            debt = abs(float(bs[debt_col].dropna().iloc[-1]))
            if debt > 0:
                return float(np.clip(interest / debt, 0.02, 0.15))
        except Exception:
            pass
    return 0.05


def _tax_rate(statements: FinancialStatements) -> float:
    inc = statements.income_statement
    tax_col = _first_col(inc, ["Tax Provision", "Income Tax Expense"])
    pretax_col = _first_col(inc, ["Pretax Income", "Income Before Tax"])
    if tax_col and pretax_col:
        try:
            tax = float(inc[tax_col].dropna().iloc[-1])
            pretax = float(inc[pretax_col].dropna().iloc[-1])
            if pretax > 0:
                return float(np.clip(tax / pretax, 0.05, 0.40))
        except Exception:
            pass
    return _DEFAULT_TAX_RATE


# ── Free Cash Flow ───────────────────────────────────────────────────────────

def _base_fcf(statements: FinancialStatements) -> Optional[float]:
    """TTM FCF → most recent annual FCF → operating CF → EBITDA proxy."""
    ttm = statements.ttm
    if ttm.get("Free Cash Flow", 0) > 0:
        return ttm["Free Cash Flow"]

    cf = statements.cash_flow
    for col in ["Free Cash Flow", "Operating Cash Flow"]:
        if col in cf.columns:
            s = cf[col].dropna()
            if not s.empty and float(s.iloc[-1]) > 0:
                return float(s.iloc[-1])

    # Rough proxy: EBITDA × 0.65 (accounts for taxes, capex, WC changes)
    if ttm.get("EBITDA", 0) > 0:
        return ttm["EBITDA"] * 0.65

    return None


def _total_debt(statements: FinancialStatements) -> float:
    bs = statements.balance_sheet
    col = _first_col(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation",
                           "Long Term Debt"])
    if col:
        try:
            return max(0.0, float(bs[col].dropna().iloc[-1]))
        except Exception:
            pass
    return 0.0


def _net_debt(statements: FinancialStatements) -> float:
    bs = statements.balance_sheet
    cash = 0.0
    cash_col = _first_col(bs, ["Cash And Cash Equivalents",
                                "Cash Cash Equivalents And Short Term Investments"])
    if cash_col:
        try:
            cash = max(0.0, float(bs[cash_col].dropna().iloc[-1]))
        except Exception:
            pass
    return _total_debt(statements) - cash


# ── Growth estimation ────────────────────────────────────────────────────────

def _growth_estimates(
    statements: FinancialStatements,
) -> tuple[float, float, float]:
    """Return (bear, base, bull) annual FCF growth rates."""
    cagr = _fcf_cagr(statements)
    if cagr is None:
        cagr = _revenue_cagr(statements)
    if cagr is None:
        cagr = 0.08   # fallback: reasonable growth for a profitable company

    base = float(np.clip(cagr, -0.05, 0.30))
    bear = float(np.clip(base * _BEAR_GROWTH_FACTOR, -0.10, 0.20))
    bull = float(np.clip(base * _BULL_GROWTH_FACTOR, -0.05, 0.40))
    return bear, base, bull


def _fcf_cagr(statements: FinancialStatements) -> Optional[float]:
    cf = statements.cash_flow
    for col in ["Free Cash Flow", "Operating Cash Flow"]:
        if col in cf.columns:
            s = cf[col].dropna()
            if len(s) >= 2:
                first, last = float(s.iloc[0]), float(s.iloc[-1])
                if first > 0 and last > 0:
                    return float((last / first) ** (1 / (len(s) - 1)) - 1)
    return None


def _revenue_cagr(statements: FinancialStatements) -> Optional[float]:
    inc = statements.income_statement
    col = _first_col(inc, ["Total Revenue", "Revenue"])
    if col:
        s = inc[col].dropna()
        if len(s) >= 2 and float(s.iloc[0]) > 0:
            return float((s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1)
    return None


def _terminal_growth(macro: MacroData) -> float:
    """Long-run nominal GDP growth, clamped to [2%, 4%]."""
    return float(np.clip(macro.real_gdp_growth + macro.inflation_rate, 0.02, 0.04))


# ── Scenario runners ─────────────────────────────────────────────────────────

def _scenario(
    base_fcf: float,
    growth: float,
    wacc: float,
    g_terminal: float,
    statements: FinancialStatements,
    market: MarketData,
) -> float:
    wacc = max(wacc, g_terminal + 0.01)
    pv, current_fcf = 0.0, base_fcf

    for year in range(1, _STAGE1_YEARS + 1):
        current_fcf *= 1 + growth
        pv += current_fcf / (1 + wacc) ** year

    for step in range(1, _STAGE2_YEARS + 1):
        year = _STAGE1_YEARS + step
        fade = growth + (g_terminal - growth) * step / _STAGE2_YEARS
        current_fcf *= 1 + fade
        pv += current_fcf / (1 + wacc) ** year

    tv = current_fcf * (1 + g_terminal) / (wacc - g_terminal)
    pv += tv / (1 + wacc) ** (_STAGE1_YEARS + _STAGE2_YEARS)

    equity = pv - _net_debt(statements)
    shares = market.shares_outstanding
    return max(0.0, equity / shares) if shares > 0 else 0.0


def _scenario_detailed(
    base_fcf: float,
    growth: float,
    wacc: float,
    g_terminal: float,
    statements: FinancialStatements,
    market: MarketData,
) -> DCFComponents:
    wacc = max(wacc, g_terminal + 0.01)
    pv1, pv2, current_fcf = 0.0, 0.0, base_fcf

    for year in range(1, _STAGE1_YEARS + 1):
        current_fcf *= 1 + growth
        pv1 += current_fcf / (1 + wacc) ** year

    for step in range(1, _STAGE2_YEARS + 1):
        year = _STAGE1_YEARS + step
        fade = growth + (g_terminal - growth) * step / _STAGE2_YEARS
        current_fcf *= 1 + fade
        pv2 += current_fcf / (1 + wacc) ** year

    tv = current_fcf * (1 + g_terminal) / (wacc - g_terminal)
    pv_tv = tv / (1 + wacc) ** (_STAGE1_YEARS + _STAGE2_YEARS)

    ev = pv1 + pv2 + pv_tv
    nd = _net_debt(statements)
    eq = ev - nd
    ps = max(0.0, eq / market.shares_outstanding) if market.shares_outstanding > 0 else 0.0

    return DCFComponents(
        wacc=wacc, growth_stage1=growth, terminal_growth=g_terminal,
        base_fcf=base_fcf, pv_stage1=pv1, pv_stage2=pv2, pv_terminal=pv_tv,
        enterprise_value=ev, net_debt=nd, equity_value=eq, per_share=ps,
    )


# ── Utility ──────────────────────────────────────────────────────────────────

def _first_col(df, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None
