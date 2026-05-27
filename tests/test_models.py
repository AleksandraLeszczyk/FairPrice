"""
Unit tests for the valuation models.

Uses synthetic data so no network calls are needed.
"""
import pandas as pd
import pytest

from fairprice.schemas import FinancialStatements, MacroData, MarketData


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_income(years=5):
    dates = pd.date_range("2019-12-31", periods=years, freq="YE")
    return pd.DataFrame({
        "Total Revenue":    [100e9, 120e9, 140e9, 160e9, 180e9][:years],
        "Net Income":       [20e9,  25e9,  30e9,  35e9,  40e9][:years],
        "Operating Income": [30e9,  36e9,  42e9,  48e9,  54e9][:years],
        "Gross Profit":     [50e9,  60e9,  70e9,  80e9,  90e9][:years],
        "EBITDA":           [35e9,  42e9,  49e9,  56e9,  63e9][:years],
        "Interest Expense": [1e9,   1e9,   1e9,   1e9,   1e9][:years],
        "Tax Provision":    [5e9,   6e9,   7e9,   8e9,   9e9][:years],
        "Pretax Income":    [25e9,  31e9,  37e9,  43e9,  49e9][:years],
    }, index=pd.Index(dates, name="date"))


def _make_balance(years=5):
    dates = pd.date_range("2019-12-31", periods=years, freq="YE")
    return pd.DataFrame({
        "Total Debt":                        [50e9] * years,
        "Cash And Cash Equivalents":         [30e9] * years,
        "Total Assets":                      [300e9] * years,
        "Total Stockholders Equity":         [100e9] * years,
    }, index=pd.Index(dates, name="date"))


def _make_cashflow(years=5):
    dates = pd.date_range("2019-12-31", periods=years, freq="YE")
    return pd.DataFrame({
        "Operating Cash Flow": [28e9, 34e9, 40e9, 46e9, 52e9][:years],
        "Capital Expenditure": [-5e9, -6e9, -7e9, -8e9, -9e9][:years],
        "Free Cash Flow":      [23e9, 28e9, 33e9, 38e9, 43e9][:years],
    }, index=pd.Index(dates, name="date"))


@pytest.fixture
def statements():
    income = _make_income()
    balance = _make_balance()
    cashflow = _make_cashflow()
    quarterly = _make_income(years=4)   # proxy 4 quarters
    return FinancialStatements(
        ticker="TEST",
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cashflow,
        quarterly_income=quarterly,
        ttm={
            "Total Revenue": 185e9,
            "Net Income": 42e9,
            "Operating Income": 56e9,
            "Gross Profit": 93e9,
            "EBITDA": 65e9,
            "Free Cash Flow": 45e9,
            "Operating Cash Flow": 54e9,
        },
        source="synthetic",
    )


@pytest.fixture
def market():
    return MarketData(
        ticker="TEST",
        prices=pd.DataFrame(),
        current_price=150.0,
        market_cap=2_400e9,
        shares_outstanding=16_000e6,
        beta_1y=1.20,
        beta_reported=1.25,
        vix=18.0,
        treasury_10y=0.045,
        dividend_yield=0.006,
    )


@pytest.fixture
def macro():
    return MacroData(
        risk_free_rate=0.045,
        equity_risk_premium=0.055,
        inflation_rate=0.030,
        real_gdp_growth=0.025,
        unemployment_rate=0.040,
        fed_funds_rate=0.0525,
        vix=18.0,
    )


# ── DCF tests ─────────────────────────────────────────────────────────────────

class TestDCF:
    def test_returns_triple(self, statements, market, macro):
        from fairprice.models.dcf import estimate
        result = estimate(statements, market, macro)
        assert result is not None
        low, base, high = result
        assert low <= base <= high

    def test_values_positive(self, statements, market, macro):
        from fairprice.models.dcf import estimate
        low, base, high = estimate(statements, market, macro)
        assert low > 0
        assert base > 0

    def test_bear_lt_bull(self, statements, market, macro):
        from fairprice.models.dcf import estimate
        low, base, high = estimate(statements, market, macro)
        assert low < high

    def test_returns_none_for_negative_fcf(self, statements, market, macro):
        from fairprice.models.dcf import estimate
        # Wipe FCF
        statements.cash_flow = pd.DataFrame()
        statements.ttm = {}  # no TTM either
        result = estimate(statements, market, macro)
        assert result is None

    def test_get_components(self, statements, market, macro):
        from fairprice.models.dcf import get_components
        comp = get_components(statements, market, macro)
        assert comp is not None
        assert comp.wacc > 0
        assert comp.enterprise_value > 0
        assert comp.per_share > 0
        # Sanity: terminal value should be the largest PV component
        assert comp.pv_terminal > 0

    def test_wacc_bounds(self, statements, market, macro):
        from fairprice.models.dcf import _compute_wacc
        wacc = _compute_wacc(statements, market, macro)
        assert 0.05 <= wacc <= 0.20

    def test_growth_estimates_ordered(self, statements, market):
        from fairprice.models.dcf import _growth_estimates
        bear, base, bull = _growth_estimates(statements)
        assert bear <= base <= bull

    def test_high_beta_increases_wacc(self, statements, market, macro):
        from fairprice.models.dcf import _compute_wacc
        market_hi = MarketData(**{**market.__dict__, "beta_1y": 2.5, "beta_reported": 2.5})
        wacc_lo = _compute_wacc(statements, market, macro)
        wacc_hi = _compute_wacc(statements, market_hi, macro)
        assert wacc_hi > wacc_lo


# ── Relative valuation tests ──────────────────────────────────────────────────

class TestRelative:
    def test_no_peers_returns_none(self, statements, market):
        from fairprice.models.relative import estimate
        assert estimate(statements, market, []) is None

    def test_target_metrics_positive(self, statements, market):
        from fairprice.models.relative import _target_metrics
        m = _target_metrics(statements, market)
        assert m["ebitda"] > 0
        assert m["fcf"] > 0
        assert m["shares"] > 0

    def test_implied_price_ev_ebitda(self, statements, market):
        from fairprice.models.relative import _implied_price, _target_metrics
        target = _target_metrics(statements, market)
        peer_df = pd.DataFrame({"ev_ebitda": [12.0, 13.0, 14.0, 11.0]})
        val = _implied_price("ev_ebitda", peer_df, target)
        assert val is not None
        assert val > 0

    def test_peer_multiples_drops_negatives_and_extreme_outliers(self):
        import pandas as pd
        # Cluster [10, 12, 11, 10] → q1=10, q3=12, IQR=2, upper=18 → 500 dropped.
        # IQR is robust: extreme values don't inflate the cut-off.
        df = pd.DataFrame({
            "ev_ebitda": [10.0, 12.0, 11.0, 10.0, -5.0, 500.0],
        })
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            valid = df[col].dropna()
            if len(valid) > 2:
                q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
                iqr = q3 - q1
                upper = (q3 + 3.0 * iqr) if iqr > 0 else valid.median() * 5.0
                df[col] = df[col].where((df[col] > 0) & (df[col] < upper))
        # -5 (negative) and 500 (IQR outlier) should both be NaN
        assert df["ev_ebitda"].isna().sum() == 2


# ── ML forecast tests ─────────────────────────────────────────────────────────

class TestML:
    def test_returns_positive(self, statements, market, macro):
        from fairprice.models.ml_forecast import estimate
        val = estimate(statements, market, macro)
        assert val is not None
        assert val > 0

    def test_historical_trend_growth(self, statements):
        from fairprice.models.ml_forecast import _historical_trend_growth
        g = _historical_trend_growth(statements)
        assert g is not None
        # Revenue grows ~15% per year in synthetic data → log slope ~0.14
        assert 0.05 < g < 0.25

    def test_no_fcf_no_result(self, statements, market, macro):
        from fairprice.models.ml_forecast import estimate
        statements.ttm = {}
        statements.cash_flow = pd.DataFrame()
        result = estimate(statements, market, macro)
        assert result is None


# ── Ensemble tests ────────────────────────────────────────────────────────────

class TestValuation:
    def test_estimate_runs(self, statements, market, macro):
        from fairprice.valuation import estimate
        result = estimate(statements, market, macro, peers=[])
        assert result.ticker == "TEST"
        assert result.fair_value_base > 0
        assert 0.0 <= result.confidence_score <= 1.0
        assert -2.0 <= result.margin_of_safety <= 2.0

    def test_fair_value_range_ordered(self, statements, market, macro):
        from fairprice.valuation import estimate
        result = estimate(statements, market, macro)
        assert result.fair_value_low <= result.fair_value_base <= result.fair_value_high

    def test_all_models_failed_placeholder(self, statements, market, macro):
        from fairprice.valuation import estimate
        # Zero out everything so all models fail
        statements.cash_flow = pd.DataFrame()
        statements.income_statement = pd.DataFrame()
        statements.balance_sheet = pd.DataFrame()
        statements.ttm = {}
        market.shares_outstanding = 0   # kills ML and relative
        result = estimate(statements, market, macro, peers=[])
        # Should still return a result (placeholder)
        assert result.fair_value_base == market.current_price
        assert result.confidence_score == 0.0

    def test_confidence_increases_with_more_models(self, statements, market, macro):
        from fairprice.valuation import _confidence
        bases_one = [100.0]
        bases_three = [100.0, 105.0, 98.0]
        c1 = _confidence(bases_one, n_models=1, vix=18.0, statements=statements)
        c3 = _confidence(bases_three, n_models=3, vix=18.0, statements=statements)
        assert c3 > c1

    def test_high_vix_lowers_confidence(self, statements, market, macro):
        from fairprice.valuation import _confidence
        c_calm = _confidence([100.0, 102.0], n_models=2, vix=12.0, statements=statements)
        c_panic = _confidence([100.0, 102.0], n_models=2, vix=45.0, statements=statements)
        assert c_panic < c_calm
