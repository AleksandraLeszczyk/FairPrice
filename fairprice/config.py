from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys (all optional — graceful degradation without them) ---
    fred_api_key: str = ""      # https://fred.stlouisfed.org/docs/api/api_key.html
    fmp_api_key: str = ""       # https://financialmodelingprep.com/developer  (250 req/day free)
    polygon_api_key: str = ""   # https://polygon.io  (5 req/min free, delayed)

    # --- Alpaca (market data + news — same key as MarketView) ---
    alpaca_api_key: str = ""    # ALPACA_API_KEY — https://alpaca.markets
    alpaca_secret: str = ""     # ALPACA_SECRET

    # --- Cache ---
    cache_dir: Path = Path(".cache")
    cache_ttl_prices: int = 300         # 5 min — intraday prices
    cache_ttl_financials: int = 86_400  # 1 day — quarterly statements
    cache_ttl_macro: int = 3_600        # 1 hour — rates / GDP

    # --- Data window defaults ---
    price_history_years: int = 5
    financial_history_years: int = 5
    beta_lookback_days: int = 252       # 1 trading year


settings = Settings()
