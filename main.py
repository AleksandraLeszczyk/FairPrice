"""
FairPrice — demo entry point.

Usage:
    python main.py AAPL
    python main.py MSFT --no-cache
    python main.py NVDA --log-level DEBUG
    python main.py --compare AAPL MSFT GOOG
"""
from __future__ import annotations

import argparse
import logging
import sys

_SEP = "=" * 58
_LINE = "─" * 58


def main() -> None:
    parser = argparse.ArgumentParser(description="FairPrice stock valuation")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--compare", nargs="+", metavar="TICKER",
                        help="Compare two or more tickers side by side")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    from fairprice.data import FinancialsClient, MacroClient, MarketClient
    from fairprice.nlp import SentimentClient

    fin = FinancialsClient()
    mkt = MarketClient()
    mac = MacroClient()
    nlp = SentimentClient()

    if args.compare:
        tickers = [t.upper() for t in args.compare]
        if len(tickers) < 2:
            parser.error("--compare needs at least two tickers")
        _compare(tickers, fin, mkt, mac, nlp)
        return

    ticker = args.ticker.upper()
    _analyze_single(ticker, fin, mkt, mac, nlp)


def _analyze_single(ticker, fin, mkt, mac, nlp) -> None:
    import fairprice.valuation as valuation_engine

    print(f"\n{_SEP}")
    print(f"  FairPrice  |  {ticker}")
    print(_SEP)

    # ── Profile ──────────────────────────────────────────────────────────
    profile = fin.get_profile(ticker)
    print(f"\n  {profile.name}")
    print(f"  {profile.sector}  /  {profile.industry}  /  {profile.country}")
    print(f"  {profile.exchange}  |  {profile.currency}")

    # ── Financials ────────────────────────────────────────────────────────
    print(f"\n{_LINE}  FINANCIALS")
    stmts = fin.get_statements(ticker)
    print(f"  Source: {stmts.source}")

    _show_df(stmts.income_statement,
             ["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
             "Income Statement (annual, $B)", scale=1e9)

    _show_df(stmts.cash_flow,
             ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
             "Cash Flow (annual, $B)", scale=1e9)

    if stmts.ttm:
        print("\n  TTM key metrics ($B):")
        for k, v in sorted(stmts.ttm.items()):
            print(f"    {k:<40} {v/1e9:>8.2f}")

    peers = fin.get_peers(ticker)
    if peers:
        print(f"\n  Peers: {', '.join(peers)}")

    # ── Market data ───────────────────────────────────────────────────────
    print(f"\n{_LINE}  MARKET DATA")
    market = mkt.get_market_data(ticker)
    print(f"  Price              ${market.current_price:>10.2f}")
    print(f"  Market cap         ${market.market_cap/1e9:>9.1f} B")
    print(f"  Shares outstanding  {market.shares_outstanding/1e6:>9.0f} M")
    print(f"  Beta (1Y vs SPY)    {market.beta_1y:>9.3f}")
    print(f"  Beta (reported)     {market.beta_reported:>9.3f}")
    print(f"  Dividend yield      {market.dividend_yield*100:>9.2f} %")
    print(f"  10Y Treasury        {market.treasury_10y*100:>9.2f} %")
    print(f"  VIX                 {market.vix:>9.1f}")

    # ── Macro ─────────────────────────────────────────────────────────────
    print(f"\n{_LINE}  MACRO")
    macro = mac.get_macro_data()
    # Populate VIX into macro (market is the authoritative source)
    macro.vix = market.vix
    print(f"  Risk-free rate      {macro.risk_free_rate*100:>7.2f} %  (10Y Treasury)")
    print(f"  Equity risk premium {macro.equity_risk_premium*100:>7.2f} %  (historical avg)")
    print(f"  Inflation (CPI YoY) {macro.inflation_rate*100:>7.2f} %")
    print(f"  Real GDP growth     {macro.real_gdp_growth*100:>7.2f} %")
    print(f"  Fed funds rate      {macro.fed_funds_rate*100:>7.2f} %")
    print(f"  Unemployment        {macro.unemployment_rate*100:>7.2f} %")

    # ── Valuation ─────────────────────────────────────────────────────────
    print(f"\n{_LINE}  VALUATION")
    sent = nlp.get_sentiment(ticker)
    result = valuation_engine.estimate(stmts, market, macro, peers, sentiment=sent)

    print(f"\n  {'Model':<20} {'Value':>10}")
    print(f"  {'─'*20}  {'─'*10}")
    if result.dcf_value:
        print(f"  {'DCF (base)':<20} ${result.dcf_value:>9.2f}")
    if result.relative_value:
        print(f"  {'Relative':<20} ${result.relative_value:>9.2f}")
    if result.ml_value:
        print(f"  {'ML Forecast':<20} ${result.ml_value:>9.2f}")

    print()
    print(f"  Current price       ${result.current_price:>9.2f}")
    print(f"  Fair value (low)    ${result.fair_value_low:>9.2f}")
    print(f"  Fair value (base)   ${result.fair_value_base:>9.2f}")
    print(f"  Fair value (high)   ${result.fair_value_high:>9.2f}")

    mos_pct = result.margin_of_safety * 100
    sign = "+" if mos_pct >= 0 else ""
    print(f"\n  Margin of safety    {sign}{mos_pct:>8.1f} %")
    print(f"  Confidence score     {result.confidence_score:>8.2f}  (0–1)")

    if result.notes:
        print(f"\n  Notes:")
        for note in result.notes:
            print(f"    • {note}")

    _verdict(result.margin_of_safety, result.confidence_score)

    # ── Sentiment ─────────────────────────────────────────────────────────
    if sent and (sent.n_articles + sent.n_reddit_posts) > 0:
        print(f"\n{_LINE}  SENTIMENT  [{sent.backend}]")
        score_bar = _sentiment_bar(sent.sentiment_score)
        print(f"  Score   {sent.sentiment_score:+.3f}  {score_bar}")
        print(f"  Positive {sent.positive_ratio*100:4.0f}%  |  "
              f"Negative {sent.negative_ratio*100:4.0f}%  |  "
              f"Neutral {sent.neutral_ratio*100:4.0f}%")
        print(f"  Uncertainty   : {sent.uncertainty_score:.2f}")
        if sent.analyst_consensus:
            print(f"  Analyst (FinViz): {sent.analyst_consensus.upper()}")
        if sent.guidance_revision:
            print(f"  Guidance revision: {sent.guidance_revision.upper()}")
        print(f"  Articles: {sent.n_articles}  |  Alpaca: {sent.n_alpaca_articles}"
              f"  |  Sources: {', '.join(sent.sources) or 'none'}")
        if sent.top_headlines:
            print(f"\n  Recent headlines:")
            for h in sent.top_headlines:
                print(f"    › {h[:90]}")

    print(f"\n{_SEP}\n")


# ── Compare mode ───────────────────────────────────────────────────────────

def _compute(ticker, fin, mkt, mac, nlp):
    """Fetch data and run the ensemble for one ticker. Returns a row dict."""
    import fairprice.valuation as valuation_engine

    profile = fin.get_profile(ticker)
    stmts = fin.get_statements(ticker)
    market = mkt.get_market_data(ticker)
    macro = mac.get_macro_data()
    macro.vix = market.vix
    peers = fin.get_peers(ticker)
    sent = nlp.get_sentiment(ticker)
    result = valuation_engine.estimate(stmts, market, macro, peers, sentiment=sent)
    return {"profile": profile, "result": result, "sentiment": sent}


def _compare(tickers, fin, mkt, mac, nlp) -> None:
    print(f"\n{_SEP}")
    print(f"  FairPrice  |  COMPARE: {'  ·  '.join(tickers)}")
    print(_SEP)

    rows: dict[str, dict] = {}
    for ticker in tickers:
        try:
            rows[ticker] = _compute(ticker, fin, mkt, mac, nlp)
        except Exception as exc:  # noqa: BLE001 — one bad ticker shouldn't abort the run
            print(f"\n  ⚠  {ticker}: failed to analyse ({exc})")

    ok = [t for t in tickers if t in rows]
    if not ok:
        print(f"\n{_SEP}\n")
        return

    col_w = 14
    label_w = 20

    def line(label: str, cells: list[str]) -> None:
        body = "".join(f"{c:>{col_w}}" for c in cells)
        print(f"  {label:<{label_w}}{body}")

    print()
    line("Metric", ok)
    line("─" * (label_w - 2), ["─" * (col_w - 2)] * len(ok))

    def col(fn) -> list[str]:
        return [fn(rows[t]) for t in ok]

    line("Company", col(lambda r: r["profile"].name[:col_w - 1]))
    line("Sector", col(lambda r: r["profile"].sector[:col_w - 1]))
    line("Price", col(lambda r: f"${r['result'].current_price:,.2f}"))
    line("Fair value (base)", col(lambda r: f"${r['result'].fair_value_base:,.2f}"))
    line("  range low", col(lambda r: f"${r['result'].fair_value_low:,.2f}"))
    line("  range high", col(lambda r: f"${r['result'].fair_value_high:,.2f}"))
    line("Margin of safety", col(_fmt_mos))
    line("Confidence", col(lambda r: f"{r['result'].confidence_score:.2f}"))
    line("Sentiment", col(_fmt_sentiment))
    line("Analyst", col(lambda r: (r["sentiment"].analyst_consensus or "–").upper()
                                   if r["sentiment"] else "–"))
    line("Verdict", col(lambda r: _verdict_tag(r["result"].margin_of_safety,
                                               r["result"].confidence_score)))

    print(f"\n{_SEP}\n")


def _fmt_mos(r) -> str:
    mos = r["result"].margin_of_safety * 100
    sign = "+" if mos >= 0 else ""
    return f"{sign}{mos:.0f}%"


def _fmt_sentiment(r) -> str:
    sent = r["sentiment"]
    if not sent or (sent.n_articles + sent.n_alpaca_articles) == 0:
        return "n/a"
    score = sent.sentiment_score
    tag = "▼" if score < -0.3 else "▲" if score > 0.3 else "~"
    return f"{score:+.2f} {tag}"


def _verdict_tag(mos: float, conf: float) -> str:
    if conf < 0.30:
        return "LOW CONF"
    if mos > 0.20 and conf >= 0.50:
        return "★ UNDER"
    if mos < -0.20 and conf >= 0.50:
        return "▼ OVER"
    return "~ FAIR"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _show_df(df, cols: list[str], label: str, scale: float = 1.0) -> None:
    if df is None or df.empty:
        return
    available = [c for c in cols if c in df.columns]
    if not available:
        return
    subset = df[available].tail(5) / scale
    print(f"\n  {label}:")
    print(subset.to_string(float_format=lambda x: f"{x:8.2f}").replace("\n", "\n  "))


def _sentiment_bar(score: float, width: int = 20) -> str:
    """ASCII bar: negative ◀ │ ▶ positive, centred at zero."""
    mid = width // 2
    pos = int((score + 1) / 2 * width)
    bar = [" "] * width
    if 0 <= pos < width:
        bar[pos] = "█"
    bar[mid] = "│" if bar[mid] == " " else "█"
    label = "▼ BEARISH" if score < -0.3 else "▲ BULLISH" if score > 0.3 else "~ NEUTRAL"
    return f"[{''.join(bar)}] {label}"


def _verdict(mos: float, conf: float) -> None:
    print()
    if conf < 0.30:
        label = "⚠  LOW CONFIDENCE — data too sparse for a reliable estimate"
    elif mos > 0.20 and conf >= 0.50:
        label = "★  POTENTIALLY UNDERVALUED  (MoS > 20%, confidence adequate)"
    elif mos < -0.20 and conf >= 0.50:
        label = "▼  POTENTIALLY OVERVALUED   (trading > 20% above fair value)"
    else:
        label = "~  FAIRLY VALUED  (within ±20% of estimated fair value)"
    print(f"  {label}")


if __name__ == "__main__":
    main()
