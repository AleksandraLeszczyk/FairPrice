# FairPrice

Stock fair value estimator. Combines DCF, peer-relative valuation, and a statistical
growth forecast into a single weighted estimate with a margin of safety and confidence
score — augmented by a live NLP sentiment layer.

## How it works

```
Financial statements  ──┐
Market prices         ──┤─→ DCF model          ──┐
Treasury yields       ──┘                         │
                                                   ├──→ Ensemble → fair value range
Sector peer multiples ──→ Relative model      ────┤          → margin of safety
                                                   │          → confidence score
Analyst consensus     ──→ ML growth forecast  ────┘               ↑
Macro (FRED)                                                       │
                                                        sentiment adjustment (±10%)
yfinance news  ──┐                                                 │
FinViz headlines ┼──→ NLP sentiment ──→ score / uncertainty ───────┘
Alpaca News    ──┘                  ──→ guidance revision
```

### Valuation models

| Model | Approach | Weight |
|---|---|---|
| **DCF** | 3-stage FCF projection (high growth → fade → terminal), discounted at WACC from CAPM | 40 % |
| **Relative** | Median EV/EBITDA, P/E, P/FCF from sector peers (IQR outlier filter) applied to TTM metrics | 35 % |
| **ML forecast** | OLS trend on log-revenue blended with analyst consensus, fed into two-stage Gordon Growth | 25 % |

Weights are renormalised automatically when a model has insufficient data.

### Confidence score (0–1)

Reflects how much to trust the output. Four factors multiply together:

| Factor | Description |
|---|---|
| **Model coverage** | 1 model → 0.50; 2 → 0.75; 3 → 1.00 |
| **Agreement** | 1 − coefficient of variation among model estimates |
| **VIX regime** | VIX < 15 → ×1.00 … VIX > 50 → ×0.30 |
| **Data completeness** | Fraction of income statement / cash flow / balance sheet present |

Sentiment then applies a small multiplier on top:

| Sentiment signal | Adjustment |
|---|---|
| Bearish score < −0.3 AND uncertainty > 0.5 | × 0.90 |
| Guidance lowered | × 0.95 |
| Bullish score > +0.4 | × 1.05 |
| Guidance raised | × 1.03 |

Sentiment deliberately moves confidence only — never the fair value itself. Short-term
news is too noisy to shift a 10-year DCF.

### NLP sentiment layer

Three backends, auto-selected at import based on installed packages:

| Backend | Package | Quality | Requires |
|---|---|---|---|
| **FinBERT** | `transformers + torch` | ★★★★★ | GPU optional |
| **VADER** | `vaderSentiment` | ★★★☆☆ | CPU only |
| **Keyword** | built-in | ★★☆☆☆ | Nothing extra |

The keyword backend uses the **Loughran-McDonald** financial lexicon — domain-specific
word lists that correctly score terms like *beat*, *raised*, *miss*, *warning*.

Beyond positive/negative polarity the layer also detects:
- **Uncertainty score** (0–1) — hedging language, forward ambiguity
- **Guidance revision** — *raised / lowered / reiterated* from earnings text
- **Analyst consensus** — scraped from FinViz (buy / hold / sell)

## Data sources

| Layer | Source | Key required |
|---|---|---|
| Financials, prices, news | Yahoo Finance (`yfinance`) | No |
| News headlines, analyst rating | FinViz (scraper) | No |
| News articles + summaries | Alpaca News API | Yes — free |
| VIX | CBOE public CSV | No |
| Macro rates, CPI, GDP | FRED St. Louis Fed (`fredapi`) | Yes — free |
| Detailed financials, peers | Financial Modeling Prep (FMP) | Yes — free (250 req/day) |

The app runs without any API keys: yfinance covers prices and financials, macro falls
back to hard-coded mid-cycle defaults with a warning, and the keyword NLP backend
requires no external service.

## Installation

```bash
git clone <repo>
cd FairPrice
pip install -r requirements.txt

# Optional: FinBERT (best NLP quality — needs ~500 MB download)
pip install transformers torch

cp .env.example .env
# fill in keys as needed (see Configuration below)
```

## Usage

```bash
# Analyse a single ticker (AAPL is the default)
python main.py AAPL

# Any ticker, with debug logging
python main.py NVDA --log-level DEBUG

# Bypass disk cache (forces fresh API calls)
python main.py MSFT --no-cache

# Compare two or more tickers side by side
python main.py --compare AAPL MSFT GOOG
```

Comparison output:

```
==========================================================
  FairPrice  |  COMPARE: AAPL  ·  MSFT  ·  GOOG
==========================================================

  Metric                        AAPL          MSFT          GOOG
  ──────────────────    ────────────  ────────────  ────────────
  Company                 Apple Inc. Microsoft Cor Alphabet Inc.
  Sector                  Technology    Technology Communication
  Price                      $312.51       $426.99       $386.12
  Fair value (base)          $106.81       $158.41       $147.94
    range low                 $96.10       $125.26       $114.88
    range high               $115.22       $191.11       $187.38
  Margin of safety             -193%         -170%         -161%
  Confidence                    0.62          0.48          0.45
  Sentiment                  +0.13 ~       +0.15 ~       +0.12 ~
  Analyst                          –             –             –
  Verdict                     ▼ OVER        ~ FAIR        ~ FAIR
```

A ticker that fails to fetch is reported inline and skipped, so one bad symbol
never aborts the whole comparison.

Example output:

```
==========================================================
  FairPrice  |  AAPL
==========================================================

  Apple Inc.
  Technology  /  Consumer Electronics  /  United States
  NMS  |  USD

──────────────────────────────────────  VALUATION

  Model                   Value
  ────────────────────    ──────────
  DCF (base)              $   96.25
  ML Forecast             $  118.66

  Current price           $  308.33
  Fair value (low)        $   94.53
  Fair value (base)       $  104.87
  Fair value (high)       $  113.06

  Margin of safety         −194.0 %
  Confidence score            0.60  (0–1)

  ▼  POTENTIALLY OVERVALUED   (trading > 20% above fair value)

──────────────────────────────────────  SENTIMENT  [keyword]

  Score   +0.042  [          │          ] ~ NEUTRAL
  Positive  38%  |  Negative  33%  |  Neutral  28%
  Uncertainty   : 0.18
  Analyst (FinViz): BUY
  Articles: 8  |  Alpaca: 12  |  Sources: yfinance, finviz, alpaca

  Recent headlines:
    › Apple reports record services revenue in latest quarter
    › Analysts debate whether AI features justify current premium
```

## Notebooks

Four Jupyter notebooks walk through the full pipeline, from raw data to actionable output:

| Notebook | What it covers |
|---|---|
| `01_data_exploration.ipynb` | Inspect raw financials, market data, and macro inputs for any ticker |
| `02_single_stock_valuation.ipynb` | DCF waterfall, FCF projection, sensitivity heatmap, model comparison, verdict, **sentiment overlay** |
| `03_sector_screener.ipynb` | Screen 30+ tickers in parallel; opportunity map; fundamentals heatmap; buy zone / avoid zone; **sentiment-adjusted composite signal** |
| `04_sentiment_analysis.ipynb` | NLP deep-dive: backend comparison, uncertainty & guidance detection, live single-stock analysis, universe sweep, valuation × sentiment quadrant, high-conviction watchlist |

```bash
cd notebooks
jupyter notebook
```

## Project layout

```
FairPrice/
├── fairprice/
│   ├── config.py           # pydantic-settings; reads .env
│   ├── schemas.py          # shared dataclasses (ValuationResult, SentimentData, …)
│   ├── valuation.py        # ensemble engine + sentiment confidence adjustment
│   ├── data/
│   │   ├── base.py         # RateLimiter + CachedClient (diskcache)
│   │   ├── financials.py   # FinancialsClient  — yfinance + FMP
│   │   ├── market.py       # MarketClient      — prices, beta, VIX
│   │   └── macro.py        # MacroClient       — FRED, fallback defaults
│   ├── models/
│   │   ├── dcf.py          # 3-stage DCF + scenario/sensitivity helpers
│   │   ├── relative.py     # peer multiples (IQR outlier filter)
│   │   └── ml_forecast.py  # OLS log-revenue trend + analyst blend
│   └── nlp/
│       ├── __init__.py     # SentimentClient — 1-hour cached entry point
│       ├── lexicon.py      # Loughran-McDonald POSITIVE / NEGATIVE / UNCERTAINTY sets
│       ├── sentiment.py    # FinBERT → VADER → keyword scorer; uncertainty & guidance
│       └── sources.py      # fetch_yfinance_news / fetch_finviz / fetch_alpaca_news
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_single_stock_valuation.ipynb
│   ├── 03_sector_screener.ipynb
│   └── 04_sentiment_analysis.ipynb
├── tests/
│   ├── test_models.py      # 20 unit tests — no network needed
│   └── test_nlp.py         # 27 unit tests — no network needed
├── main.py
├── requirements.txt
└── .env.example
```

## Configuration

Copy `.env.example` to `.env` and fill in whichever keys you have:

```
# Macro data — https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=...

# Detailed financials + peer lists — https://financialmodelingprep.com/developer
FMP_API_KEY=...

# Optional: real-time prices — https://polygon.io
POLYGON_API_KEY=...

# Alpaca News API (same credentials as any Alpaca brokerage account)
# https://alpaca.markets → Paper Trading → API Keys
ALPACA_API_KEY=...
ALPACA_SECRET=...
```

Optional cache overrides:

```
CACHE_TTL_PRICES=300       # seconds; default 5 min
CACHE_TTL_FINANCIALS=86400 # default 1 day
CACHE_TTL_MACRO=3600       # default 1 hour
```

API responses are cached to `.cache/` so repeated runs are fast and stay within free
tier limits. Pass `--no-cache` to force fresh calls.

## Tests

```bash
# All unit tests — no network, no API keys needed (~47 tests)
pytest tests/test_models.py tests/test_nlp.py -v

# Including live API integration tests
pytest tests/ -v -m network
```

## Limitations and caveats

- **Not suitable for banks or insurers** — DCF assumes operating-business FCF; financial
  companies require dividend-discount or residual-income models.
- **Negative FCF** — DCF is skipped; ensemble falls back to relative + ML, lowering
  confidence. Companies burning cash get a wider uncertainty band.
- **Peer quality** — FMP API key required for peer lists. Without it, relative valuation
  is skipped entirely.
- **Analyst consensus** — yfinance `earningsGrowth` / `revenueGrowth` are often stale or
  absent for smaller companies; the ML model then relies on historical trend only.
- **Sentiment at scale** — the keyword backend is always available; VADER and FinBERT give
  meaningfully better results on ambiguous or ironic financial language.
- **All estimates are point-in-time** — re-run after each earnings release.
- **Not financial advice.**

## Planned

- Sector-aware equity risk premium (ERP) adjustment
- Earnings-date aware cache invalidation
