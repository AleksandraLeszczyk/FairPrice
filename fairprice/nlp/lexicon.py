"""
Financial sentiment lexicon — compact subset of Loughran-McDonald (2011).

Used as the always-available fallback when neither FinBERT nor VADER is installed.
Also used for uncertainty and guidance-revision detection in all backends.
"""

POSITIVE: frozenset[str] = frozenset({
    "beat", "beats", "exceeded", "exceeds", "surpass", "surpassed", "outperform",
    "outperformed", "record", "records", "strong", "stronger", "strongest",
    "robust", "growth", "grew", "expand", "expanded", "expansion", "raised",
    "raise", "upgrade", "upgraded", "bullish", "momentum", "surge", "surged",
    "accelerate", "accelerated", "gain", "gained", "positive", "profitable",
    "profitability", "breakthrough", "innovative", "demand", "upside",
    "recovery", "recover", "rebound", "resilient", "optimistic", "confidence",
    "opportunity", "opportunities", "leader", "leading", "dominance",
    "margin", "margins", "efficiency", "dividend", "buyback", "repurchase",
    "approval", "approved", "launch", "launched", "partnership", "synergy",
    "ahead", "above", "better", "best", "solid", "healthy", "impressive",
})

NEGATIVE: frozenset[str] = frozenset({
    "miss", "missed", "misses", "disappoint", "disappointed", "disappointing",
    "decline", "declined", "declines", "loss", "losses", "weak", "weaker",
    "weakest", "concern", "concerns", "downgrade", "downgraded", "bearish",
    "warning", "warn", "warned", "risk", "risks", "risky", "threat", "threats",
    "shortfall", "cut", "cuts", "cutting", "reduce", "reduced", "reduction",
    "layoff", "layoffs", "restructure", "restructuring", "investigation",
    "lawsuit", "litigation", "penalty", "fine", "breach", "breach",
    "disappointing", "underperform", "underperformed", "below", "pressure",
    "pressures", "headwind", "headwinds", "slowing", "slowed", "slow",
    "deteriorate", "deteriorated", "deterioration", "volatile", "volatility",
    "uncertainty", "challenging", "challenges", "difficult", "difficulties",
    "default", "bankruptcy", "insolvency", "recall", "scandal", "fraud",
    "probe", "subpoena", "impairment", "writedown", "write-off", "goodwill",
    "negative", "adverse", "unfavorable", "unfavourable", "worse", "worst",
})

UNCERTAINTY: frozenset[str] = frozenset({
    "uncertain", "uncertainty", "unpredictable", "unclear", "unknown",
    "cautious", "caution", "careful", "conservative", "prudent",
    "challenging", "challenges", "difficult", "difficulties",
    "volatile", "volatility", "turbulent", "turbulence",
    "risk", "risks", "concern", "concerns", "worry", "worried",
    "headwind", "headwinds", "downside", "pressure", "pressures",
    "dependent", "depends", "depending", "contingent", "subject to",
    "may", "might", "could", "possible", "possibly", "perhaps",
    "expect", "expected", "expectation", "anticipate", "anticipated",
    "outlook", "guidance", "forecast", "estimate", "projected",
    "monitoring", "watchful", "reassess", "reassessing",
})

# Phrase-level guidance signals (lowercase, partial match)
GUIDANCE_RAISED: list[str] = [
    "raised guidance", "raises guidance", "raised its guidance",
    "increased guidance", "raised outlook", "increased outlook",
    "raised its outlook", "raised forecast", "raised its forecast",
    "above guidance", "above expectations", "beat expectations",
    "beat estimates", "exceeded expectations", "exceeded estimates",
    "above consensus", "upside guidance", "guidance raise",
]

GUIDANCE_LOWERED: list[str] = [
    "lowered guidance", "lowers guidance", "lowered its guidance",
    "cut guidance", "cuts guidance", "reduced guidance",
    "lowered outlook", "reduced outlook", "lowered forecast",
    "below guidance", "below expectations", "missed expectations",
    "missed estimates", "below consensus", "profit warning",
    "issued a warning", "cautious guidance", "guidance cut",
    "negative guidance", "guidance below",
]

GUIDANCE_REITERATED: list[str] = [
    "maintained guidance", "reaffirmed guidance", "reiterated guidance",
    "maintained its guidance", "reaffirmed its outlook",
    "in line with expectations", "in line with estimates",
    "met expectations", "met estimates", "on track",
]
