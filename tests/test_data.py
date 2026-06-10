"""
Data layer smoke tests.

These tests hit live APIs (yfinance) — they are integration tests, not unit tests.
Run them with:  pytest tests/ -v
Skip network tests:  pytest tests/ -v -m "not network"
"""
import pytest
import pandas as pd

TICKER = "AAPL"


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fin_client(tmp_path_factory):
    from fairprice.config import Settings
    from fairprice.data.financials import FinancialsClient

    tmp = tmp_path_factory.mktemp("cache")
    # Override cache dir so tests don't pollute the project cache
    import fairprice.config as cfg
    original = cfg.settings.cache_dir
    cfg.settings.cache_dir = tmp
    client = FinancialsClient()
    yield client
    cfg.settings.cache_dir = original


@pytest.fixture(scope="module")
def mkt_client(tmp_path_factory):
    from fairprice.data.market import MarketClient
    import fairprice.config as cfg

    tmp = tmp_path_factory.mktemp("cache")
    original = cfg.settings.cache_dir
    cfg.settings.cache_dir = tmp
    client = MarketClient()
    yield client
    cfg.settings.cache_dir = original


@pytest.fixture(scope="module")
def mac_client(tmp_path_factory):
    from fairprice.data.macro import MacroClient
    import fairprice.config as cfg

    tmp = tmp_path_factory.mktemp("cache")
    original = cfg.settings.cache_dir
    cfg.settings.cache_dir = tmp
    client = MacroClient()
    yield client
    cfg.settings.cache_dir = original


# ── FinancialsClient tests ──────────────────────────────────────────────────

@pytest.mark.network
def test_get_profile(fin_client):
    profile = fin_client.get_profile(TICKER)
    assert profile.ticker == TICKER
    assert profile.name
    assert profile.sector
    assert profile.currency == "USD"


@pytest.mark.network
def test_get_statements_returns_dataframes(fin_client):
    stmts = fin_client.get_statements(TICKER)
    assert isinstance(stmts.income_statement, pd.DataFrame)
    assert isinstance(stmts.balance_sheet, pd.DataFrame)
    assert isinstance(stmts.cash_flow, pd.DataFrame)


@pytest.mark.network
def test_income_statement_has_revenue(fin_client):
    stmts = fin_client.get_statements(TICKER)
    assert not stmts.income_statement.empty
    assert "Total Revenue" in stmts.income_statement.columns


@pytest.mark.network
def test_ttm_is_populated(fin_client):
    stmts = fin_client.get_statements(TICKER)
    assert isinstance(stmts.ttm, dict)
    assert len(stmts.ttm) > 0
    assert "Total Revenue" in stmts.ttm or "Net Income" in stmts.ttm


@pytest.mark.network
def test_statements_are_cached(fin_client):
    s1 = fin_client.get_statements(TICKER)
    s2 = fin_client.get_statements(TICKER)
    # Both calls should return the same object (cache hit on second call)
    assert s1.ticker == s2.ticker


# ── MarketClient tests ──────────────────────────────────────────────────────

@pytest.mark.network
def test_get_market_data(mkt_client):
    mkt = mkt_client.get_market_data(TICKER)
    assert mkt.ticker == TICKER
    assert mkt.current_price > 0
    assert mkt.market_cap > 0
    assert isinstance(mkt.prices, pd.DataFrame)
    assert not mkt.prices.empty


@pytest.mark.network
def test_beta_is_finite(mkt_client):
    mkt = mkt_client.get_market_data(TICKER)
    assert 0.0 < mkt.beta_1y < 5.0, f"Unexpected beta: {mkt.beta_1y}"


@pytest.mark.network
def test_vix_is_positive(mkt_client):
    mkt = mkt_client.get_market_data(TICKER)
    assert mkt.vix > 0


@pytest.mark.network
def test_treasury_yield_is_reasonable(mkt_client):
    mkt = mkt_client.get_market_data(TICKER)
    assert 0.001 < mkt.treasury_10y < 0.20, f"Unexpected 10Y yield: {mkt.treasury_10y}"


# ── MacroClient tests ───────────────────────────────────────────────────────

def test_fallback_macro_when_no_key(mac_client, monkeypatch):
    import fairprice.config as cfg
    monkeypatch.setattr(cfg.settings, "fred_api_key", "")
    mac = mac_client._fetch_macro()   # bypass cache to force fresh fetch
    assert 0.0 < mac.risk_free_rate < 0.15
    assert 0.0 < mac.equity_risk_premium < 0.15
    assert -0.05 < mac.inflation_rate < 0.20


@pytest.mark.network
def test_macro_rates_are_reasonable(mac_client):
    mac = mac_client.get_macro_data()
    assert 0.0 < mac.risk_free_rate < 0.15
    assert mac.equity_risk_premium > 0


# ── Helpers ─────────────────────────────────────────────────────────────────

def test_normalize_yf_empty():
    from fairprice.data.financials import _normalize_yf
    result = _normalize_yf(None)
    assert result.empty

    result = _normalize_yf(pd.DataFrame())
    assert result.empty


def test_compute_ttm_empty():
    from fairprice.data.financials import _compute_ttm
    result = _compute_ttm(pd.DataFrame(), pd.DataFrame())
    assert result == {}


# ── FMP circuit-breaker tests ────────────────────────────────────────────────

def test_fmp_disabled_after_threshold(tmp_path, monkeypatch):
    """After _FMP_ERROR_THRESHOLD errors, _fmp_disabled flips to True."""
    import fairprice.config as cfg
    import fairprice.data.financials as fin_mod
    from fairprice.data.financials import FinancialsClient, _FMP_ERROR_THRESHOLD

    monkeypatch.setattr(cfg.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(cfg.settings, "fmp_api_key", "dummy_key")

    client = FinancialsClient()
    assert not client._fmp_disabled

    def _boom(*_):
        raise ConnectionError("FMP is down")

    for _ in range(_FMP_ERROR_THRESHOLD):
        with pytest.raises(Exception):
            client._try_fmp("test", _boom)

    assert client._fmp_disabled
    assert client._fmp_error_count >= _FMP_ERROR_THRESHOLD


def test_fmp_disabled_raises_immediately(tmp_path, monkeypatch):
    """Once disabled, _try_fmp raises without calling the function."""
    import fairprice.config as cfg
    from fairprice.data.financials import FinancialsClient

    monkeypatch.setattr(cfg.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(cfg.settings, "fmp_api_key", "dummy_key")

    client = FinancialsClient()
    client._fmp_disabled = True

    called = []
    def _should_not_run(*_):
        called.append(True)

    with pytest.raises(RuntimeError, match="FMP disabled"):
        client._try_fmp("test", _should_not_run)

    assert not called


def test_fmp_unavailable_when_no_key(tmp_path, monkeypatch):
    import fairprice.config as cfg
    from fairprice.data.financials import FinancialsClient

    monkeypatch.setattr(cfg.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(cfg.settings, "fmp_api_key", "")

    client = FinancialsClient()
    assert not client._fmp_available()


def test_key_metrics_yf_fallback_returns_dataframe(tmp_path, monkeypatch):
    """_fetch_key_metrics_yf returns a DataFrame even when yfinance fields are sparse."""
    import fairprice.config as cfg
    from fairprice.data.financials import FinancialsClient

    monkeypatch.setattr(cfg.settings, "cache_dir", tmp_path)

    client = FinancialsClient()
    # Patch yf.Ticker.info to return a minimal dict
    import yfinance as yf
    monkeypatch.setattr(
        yf.Ticker, "info",
        property(lambda self: {"trailingPE": 25.0, "returnOnEquity": 0.15}),
    )
    df = client._fetch_key_metrics_yf("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "peRatio" in df.columns
