# FairPrice

Stock fair value estimator. Combines DCF, peer-relative valuation, and a statistical growth forecast into a single weighted estimate with a margin of safety and confidence score.

## How it works

```
Financial statements  ──┐
Market prices         ──┤─→ DCF model          ──┐
Treasury yields       ──┘                         │
                                                   ├──→ Ensemble → fair value range
Sector peer multiples ──→ Relative model      ────┤          → margin of safety
                                                   │          → confidence score
Analyst consensus     ──→ ML growth forecast  ────┘
Macro (FRED)          ──┘
```

### Models

| Model | Approach | Weight |
|---|---|---|
| **DCF** | 3-stage FCF projection (high growth → fade → terminal), discounted at WACC from CAPM | 40% |
| **Relative** | Median EV/EBITDA, P/E, P/FCF from sector peers applied to TTM metrics | 35% |
| **ML forecast** | OLS trend on log-revenue blended with analyst consensus, fed into two-stage Gordon Growth | 25% |

Weights are renormalised automatically when a model has insufficient data.

The **confidence score** (0–1) reflects how much to trust the output:

- **Model coverage** — how many of the three models produced a result
- **Agreement** — 1 minus the coefficient of variation among model estimates
- **VIX regime** — market stress widens uncertainty; VIX > 35 halves the score
- **Data completeness** — whether income statement, cash flow, and balance sheet are all present

## Data sources

| Layer | Source | Key required |
|---|---|---|
| Financials, prices | Yahoo Finance (`yfinance`) | No |
| VIX | CBOE public CSV | No |
| Macro rates, CPI, GDP | FRED St. Louis Fed (`fredapi`) | Yes — free |
| Detailed financials, peers | Financial Modeling Prep (FMP) | Yes — free (250 req/day) |
| Real-time prices | Polygon.io | Yes — free (5 req/min) |

The app runs without any API keys: yfinance covers prices and financials, and macro falls back to hard-coded mid-cycle defaults with a warning.

## Installation

```bash
git clone <repo>
cd FairPrice
pip install -r requirements.txt

cp .env.example .env
# fill in FRED_API_KEY and FMP_API_KEY
```

## Usage

```bash
# Analyse a single ticker
python main.py AAPL

# Any ticker, with debug logging
python main.py NVDA --log-level DEBUG

# Bypass disk cache (forces fresh API calls)
python main.py MSFT --no-cache
```

Example output:

```
==========================================================
  FairPrice  |  AAPL
==========================================================

  Apple Inc.
  Technology  /  Consumer Electronics  /  United States
  NMS  |  USD

──────────────────────────────────────────  VALUATION

  Model                   Value
  ────────────────────    ──────────
  DCF (base)              $  198.40
  Relative                $  212.60
  ML Forecast             $  205.10

  Current price           $  189.50
  Fair value (low)        $  181.20
  Fair value (base)       $  204.70
  Fair value (high)       $  228.40

  Margin of safety         + 7.4 %
  Confidence score           0.72  (0–1)

  ~  FAIRLY VALUED  (within ±20% of estimated fair value)
```

## Project layout

```
FairPrice/
├── fairprice/
│   ├── config.py          # pydantic-settings; reads .env
│   ├── schemas.py         # shared dataclasses
│   ├── valuation.py       # ensemble engine
│   ├── data/
│   │   ├── base.py        # RateLimiter + CachedClient (diskcache)
│   │   ├── financials.py  # FinancialsClient  — yfinance + FMP
│   │   ├── market.py      # MarketClient      — prices, beta, VIX
│   │   └── macro.py       # MacroClient       — FRED, fallback defaults
│   ├── models/
│   │   ├── dcf.py         # 3-stage DCF
│   │   ├── relative.py    # peer multiples
│   │   └── ml_forecast.py # statistical growth forecast
│   └── nlp/               # sentiment layer (planned)
├── tests/
│   ├── test_data.py       # integration tests (marked @network)
│   └── test_models.py     # unit tests — no network calls needed
├── requirements.txt
└── .env.example
```

## Configuration

Copy `.env.example` to `.env`:

```
FRED_API_KEY=...     # https://fred.stlouisfed.org/docs/api/api_key.html
FMP_API_KEY=...      # https://financialmodelingprep.com/developer
POLYGON_API_KEY=...  # https://polygon.io  (not used yet)
```

Optional overrides:

```
CACHE_TTL_PRICES=300       # seconds; default 5 min
CACHE_TTL_FINANCIALS=86400 # default 1 day
CACHE_TTL_MACRO=3600       # default 1 hour
```

API responses are cached to `.cache/` so repeated runs are fast and free. Pass `--no-cache` to bypass.

## Tests

```bash
# Unit tests only (no network)
pytest tests/test_models.py -v

# All tests including live API calls
pytest tests/ -v -m network
```

## Limitations and caveats

- **Not suitable for banks or insurers** — DCF assumes operating business FCF; financial companies require dividend discount or residual income models.
- **Negative FCF companies** — DCF is skipped; the ensemble falls back to relative and ML only, which lowers the confidence score.
- **Peer quality** — FMP API key required for peer lists. Without it, relative valuation is skipped.
- **Analyst consensus** — yfinance `earningsGrowth` / `revenueGrowth` are often stale or missing for smaller companies.
- **All estimates are point-in-time** — rerun the analysis after each earnings release.
- **Not financial advice.**

## Planned

- Sentiment layer (`fairprice/nlp/`) — FinBERT on news headlines and Reddit threads
- Sector-aware ERP adjustment
- CLI `--compare AAPL MSFT GOOG` for side-by-side output
- Plotly sensitivity tornado chart
