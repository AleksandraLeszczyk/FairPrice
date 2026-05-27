"""
Ensemble valuation engine.

Runs DCF, relative, and ML models, then produces a weighted average
fair value range and a confidence score.

Confidence is penalised for:
  - Fewer models producing valid estimates
  - High dispersion among model outputs (low agreement)
  - Elevated VIX (market stress → wider uncertainty)
  - Incomplete financial data
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from fairprice.models import dcf, ml_forecast, relative
from fairprice.schemas import (
    FinancialStatements,
    MacroData,
    MarketData,
    ValuationResult,
)

logger = logging.getLogger(__name__)

_BASE_WEIGHTS: dict[str, float] = {
    "dcf":      0.40,
    "relative": 0.35,
    "ml":       0.25,
}

# Symmetric ±% band applied to single-point estimates (relative, ml)
# to convert them into (low, base, high) triples.
_POINT_BAND = 0.15  # ±15 %

# VIX thresholds → confidence multiplier
_VIX_TABLE = [
    (15.0, 1.00),
    (20.0, 0.90),
    (25.0, 0.75),
    (35.0, 0.60),
    (50.0, 0.45),
    (float("inf"), 0.30),
]


def estimate(
    statements: FinancialStatements,
    market: MarketData,
    macro: MacroData,
    peers: Optional[list[str]] = None,
) -> ValuationResult:
    notes: list[str] = []

    dcf_result = _safe_dcf(statements, market, macro, notes)
    rel_result = _safe_relative(statements, market, peers or [], notes)
    ml_result = _safe_ml(statements, market, macro, notes)

    # Unpack to base-only values
    dcf_base = dcf_result[1] if dcf_result else None
    rel_base = rel_result
    ml_base = ml_result

    # Build (low, base, high) triples for each available model
    triples: dict[str, tuple[float, float, float]] = {}

    if dcf_result is not None:
        triples["dcf"] = dcf_result

    if rel_base is not None and rel_base > 0:
        triples["relative"] = (
            rel_base * (1 - _POINT_BAND),
            rel_base,
            rel_base * (1 + _POINT_BAND),
        )

    if ml_base is not None and ml_base > 0:
        triples["ml"] = (
            ml_base * (1 - _POINT_BAND),
            ml_base,
            ml_base * (1 + _POINT_BAND),
        )

    if not triples:
        notes.append("All models failed — returning market price as placeholder")
        logger.warning("No valid valuation for %s", statements.ticker)
        p = market.current_price
        return ValuationResult(
            ticker=statements.ticker,
            current_price=p,
            dcf_value=None,
            relative_value=None,
            ml_value=None,
            fair_value_low=p,
            fair_value_base=p,
            fair_value_high=p,
            margin_of_safety=0.0,
            confidence_score=0.0,
            notes=notes,
        )

    # Normalise weights to models that produced results
    raw = {k: _BASE_WEIGHTS[k] for k in triples}
    total = sum(raw.values())
    w = {k: v / total for k, v in raw.items()}

    fair_low  = sum(triples[k][0] * w[k] for k in triples)
    fair_base = sum(triples[k][1] * w[k] for k in triples)
    fair_high = sum(triples[k][2] * w[k] for k in triples)

    p = market.current_price
    mos = (fair_base - p) / fair_base if fair_base > 0 else 0.0

    confidence = _confidence(
        base_values=[triples[k][1] for k in triples],
        n_models=len(triples),
        vix=market.vix,
        statements=statements,
    )

    return ValuationResult(
        ticker=statements.ticker,
        current_price=p,
        dcf_value=round(dcf_base, 2) if dcf_base else None,
        relative_value=round(rel_base, 2) if rel_base else None,
        ml_value=round(ml_base, 2) if ml_base else None,
        fair_value_low=round(fair_low, 2),
        fair_value_base=round(fair_base, 2),
        fair_value_high=round(fair_high, 2),
        margin_of_safety=round(mos, 4),
        confidence_score=round(confidence, 4),
        notes=notes,
    )


# ── Safe wrappers ────────────────────────────────────────────────────────────

def _safe_dcf(statements, market, macro, notes):
    try:
        result = dcf.estimate(statements, market, macro)
        if result is None:
            notes.append("DCF: no positive FCF — skipped")
        return result
    except Exception as exc:
        notes.append(f"DCF: error — {exc}")
        logger.exception("DCF failed for %s", statements.ticker)
        return None


def _safe_relative(statements, market, peers, notes):
    try:
        result = relative.estimate(statements, market, peers)
        if result is None:
            notes.append("Relative: no peers or insufficient data — skipped")
        return result
    except Exception as exc:
        notes.append(f"Relative: error — {exc}")
        logger.exception("Relative failed for %s", statements.ticker)
        return None


def _safe_ml(statements, market, macro, notes):
    try:
        result = ml_forecast.estimate(statements, market, macro)
        if result is None:
            notes.append("ML: insufficient data — skipped")
        return result
    except Exception as exc:
        notes.append(f"ML: error — {exc}")
        logger.exception("ML failed for %s", statements.ticker)
        return None


# ── Confidence scoring ───────────────────────────────────────────────────────

def _confidence(
    base_values: list[float],
    n_models: int,
    vix: float,
    statements: FinancialStatements,
) -> float:
    count_factor = {1: 0.50, 2: 0.75, 3: 1.00}.get(n_models, 0.50)

    # Agreement: 1 − coefficient of variation (bounded)
    if len(base_values) >= 2:
        mean = np.mean(base_values)
        cv = np.std(base_values) / mean if mean > 0 else 1.0
        agreement = float(np.clip(1.0 - cv, 0.20, 1.00))
    else:
        agreement = 0.60

    # VIX regime
    vix_factor = 0.30
    for threshold, factor in _VIX_TABLE:
        if vix < threshold:
            vix_factor = factor
            break

    # Data completeness (0–1)
    checks = [
        not statements.income_statement.empty,
        not statements.cash_flow.empty,
        not statements.balance_sheet.empty,
        len(statements.ttm) >= 3,
    ]
    data_factor = sum(checks) / len(checks)

    return float(count_factor * agreement * vix_factor * data_factor)
