"""
Unit tests for the NLP / sentiment layer.

All tests are offline — no network calls.
"""
import pytest

from fairprice.nlp.lexicon import (
    NEGATIVE,
    POSITIVE,
    UNCERTAINTY,
    GUIDANCE_LOWERED,
    GUIDANCE_RAISED,
)
from fairprice.nlp.sentiment import (
    detect_guidance_revision,
    detect_uncertainty,
    score_texts,
    active_backend,
    _keyword,
    _tokenise,
)
from fairprice.schemas import SentimentData


# ── Lexicon sanity ────────────────────────────────────────────────────────────

class TestLexicon:
    def test_no_overlap_positive_negative(self):
        overlap = POSITIVE & NEGATIVE
        assert not overlap, f"Overlap between POSITIVE and NEGATIVE: {overlap}"

    def test_uncertainty_has_entries(self):
        assert len(UNCERTAINTY) >= 10

    def test_known_positive_words(self):
        for word in ("beat", "record", "growth", "upgrade"):
            assert word in POSITIVE

    def test_known_negative_words(self):
        for word in ("miss", "loss", "downgrade", "warning"):
            assert word in NEGATIVE


# ── Keyword scorer ────────────────────────────────────────────────────────────

class TestKeywordScorer:
    def test_positive_headline(self):
        results = _keyword(["Apple beat earnings estimates and raised guidance"])
        assert results[0]["positive"] > results[0]["negative"]

    def test_negative_headline(self):
        results = _keyword(["Company missed expectations and issued a profit warning"])
        assert results[0]["negative"] > results[0]["positive"]

    def test_neutral_headline(self):
        results = _keyword(["Company reports quarterly results on Thursday"])
        # No strong signal → neutral dominates
        assert results[0]["neutral"] >= 0.5

    def test_empty_input(self):
        assert score_texts([]) == []

    def test_output_shape(self):
        texts = ["Good earnings", "Bad quarter"]
        results = score_texts(texts)
        assert len(results) == 2
        for r in results:
            assert set(r.keys()) == {"positive", "negative", "neutral"}
            assert all(0.0 <= v <= 1.0 for v in r.values())

    def test_tokenise(self):
        # re.findall(r"[a-z]+") on "apple's q3 2024!" → ["apple", "s", "q"]
        assert _tokenise("Apple's Q3 2024!") == ["apple", "s", "q"]
        tokens = _tokenise("Hello, World! 123")
        assert all(t.islower() for t in tokens)
        assert all(t.isalpha() for t in tokens)


# ── Uncertainty detection ─────────────────────────────────────────────────────

class TestUncertainty:
    def test_high_uncertainty_text(self):
        text = "The outlook remains uncertain amid volatile market conditions and challenging headwinds"
        score = detect_uncertainty(text)
        assert score > 0.2

    def test_low_uncertainty_text(self):
        text = "Apple reported record revenue of $120 billion this quarter"
        score = detect_uncertainty(text)
        assert score < 0.2

    def test_score_bounded(self):
        for text in ["", "hello", "uncertain volatile risky concerning"]:
            score = detect_uncertainty(text)
            assert 0.0 <= score <= 1.0


# ── Guidance revision ─────────────────────────────────────────────────────────

class TestGuidanceRevision:
    def test_guidance_raised(self):
        texts = ["Management raised guidance for the full year above consensus"]
        assert detect_guidance_revision(texts) == "raised"

    def test_guidance_lowered(self):
        texts = ["Company issued a profit warning and lowered guidance"]
        assert detect_guidance_revision(texts) == "lowered"

    def test_guidance_reiterated(self):
        texts = ["Management reaffirmed guidance in line with expectations"]
        assert detect_guidance_revision(texts) == "reiterated"

    def test_lowered_beats_reiterated(self):
        # When both signals appear, lowered should win (more conservative)
        texts = ["maintained guidance but later lowered guidance significantly"]
        assert detect_guidance_revision(texts) == "lowered"

    def test_no_guidance_signal(self):
        texts = ["Quarterly results next week", "CEO speaks at conference"]
        assert detect_guidance_revision(texts) is None

    def test_empty_list(self):
        assert detect_guidance_revision([]) is None


# ── Backend ───────────────────────────────────────────────────────────────────

def test_active_backend_is_valid():
    backend = active_backend()
    assert backend in ("finbert", "vader", "keyword")


# ── SentimentData schema ──────────────────────────────────────────────────────

def test_sentiment_data_fields():
    sd = SentimentData(
        ticker="TEST",
        sentiment_score=0.25,
        positive_ratio=0.4,
        negative_ratio=0.15,
        neutral_ratio=0.45,
        uncertainty_score=0.1,
        guidance_revision="raised",
        analyst_consensus="buy",
        n_articles=10,
        n_alpaca_articles=8,
        backend="keyword",
        sources=["yfinance", "finviz", "alpaca"],
        top_headlines=["Headline 1", "Headline 2"],
    )
    assert sd.ticker == "TEST"
    assert sd.sentiment_score == 0.25
    assert sd.guidance_revision == "raised"
    assert sd.n_alpaca_articles == 8
    assert len(sd.top_headlines) == 2


# ── Valuation integration ─────────────────────────────────────────────────────

class TestSentimentAdjustment:
    """Test that sentiment adjusts confidence without breaking valuation."""

    def _make_sentiment(self, score: float, uncertainty: float = 0.1,
                         guidance: str = None) -> SentimentData:
        return SentimentData(
            ticker="TEST", sentiment_score=score,
            positive_ratio=max(0, score), negative_ratio=max(0, -score),
            neutral_ratio=1 - abs(score), uncertainty_score=uncertainty,
            guidance_revision=guidance, analyst_consensus=None,
            n_articles=10, n_alpaca_articles=8, backend="keyword",
        )

    def test_bearish_uncertain_reduces_confidence(self):
        from fairprice.valuation import _sentiment_adjust
        sent = self._make_sentiment(-0.5, uncertainty=0.7)
        adj = _sentiment_adjust(0.8, sent, [])
        assert adj < 0.8

    def test_bullish_boosts_confidence(self):
        from fairprice.valuation import _sentiment_adjust
        sent = self._make_sentiment(0.5)
        adj = _sentiment_adjust(0.7, sent, [])
        assert adj > 0.7

    def test_neutral_no_change(self):
        from fairprice.valuation import _sentiment_adjust
        sent = self._make_sentiment(0.0, uncertainty=0.1)
        adj = _sentiment_adjust(0.7, sent, [])
        assert adj == pytest.approx(0.7, abs=0.01)

    def test_confidence_never_exceeds_1(self):
        from fairprice.valuation import _sentiment_adjust
        sent = self._make_sentiment(0.9, guidance="raised")
        adj = _sentiment_adjust(0.99, sent, [])
        assert adj <= 1.0

    def test_none_sentiment_passthrough(self):
        from fairprice.valuation import _sentiment_adjust
        adj = _sentiment_adjust(0.65, None, [])
        assert adj == 0.65

    def test_notes_populated_on_adjustment(self):
        from fairprice.valuation import _sentiment_adjust
        notes: list = []
        sent = self._make_sentiment(-0.4, uncertainty=0.6)
        _sentiment_adjust(0.8, sent, notes)
        assert len(notes) >= 1
        assert any("confidence" in n.lower() for n in notes)
