#!/usr/bin/env python3
"""Shared scoring engine: weighted, explainable, and honest about missing data.

Every horizon scores 0-100, but the horizons do not share weights, criteria or
even which metrics are admissible. What they share is this machinery, and three
properties it enforces.

1. NORMALISATION FOR MISSING DATA
    A criterion whose input is unavailable is dropped from BOTH the earned and
    the available total, rather than scored as zero. Scoring it zero would
    conflate "this company is bad" with "we could not measure it", and would
    systematically push recently-listed companies and lenders -- which
    legitimately lack whole categories of metric -- to the bottom of every
    screen. The score is therefore earned/available, and `coverage` reports how
    much of the intended framework actually applied.

    The corollary matters: a 90/100 built from 30% of the criteria is not
    comparable to a 90/100 built from all of them, so coverage travels with the
    score everywhere and low coverage is surfaced as a caveat rather than
    quietly averaged away.

2. EVERY POINT IS EXPLAINED
    Each criterion records why it earned or failed to earn its weight. The
    output is an audit trail, not a number -- which is the entire basis for
    trusting the long-horizon engine, where no backtest can honestly validate
    the weights (see README). A ranking you can read is worth more than a
    ranking you can only sort.

3. PENALTIES ARE SEPARATE FROM CRITERIA
    Governance red flags, event risk and overextension subtract from the final
    score rather than participating in the weighted average. A severely pledged
    promoter should not be offset by a good margin trend -- it is a veto-shaped
    risk, and averaging it away is exactly how screens end up recommending
    companies that later blow up.

BANDS, NOT BOOLEANS, WHERE THE UNDERLYING RELATIONSHIP IS GRADED
    `add_banded` exists because most of these relationships are not step
    functions. ROCE of 24% is not meaningfully worse than 26%, but both differ
    from 8%. Bands give partial credit at the boundaries so a screen does not
    hinge on an arbitrary cut-off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A score built from less than this share of its intended criteria is reported
# with an explicit low-confidence caveat: the number is still the best estimate
# available, but it is not comparable to a fully-measured one.
MIN_COVERAGE_FOR_CONFIDENCE = 0.60


def _missing(value) -> bool:
    """True when a metric is absent. Treats None, NaN and NaT alike."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class Score:
    """Accumulates weighted criteria, reasons, warnings and penalties."""

    def __init__(self, ticker: str, horizon: str):
        self.ticker = ticker
        self.horizon = horizon
        self.earned = 0.0
        self.available = 0.0
        self.reasons: list[str] = []
        self.misses: list[str] = []
        self.warnings: list[str] = []
        self.penalties: list[tuple[str, float]] = []
        self.skipped: list[str] = []
        self.facts: dict = {}

    # ---- criteria --------------------------------------------------------- #
    def add(self, condition: bool, weight: float, reason: str,
            miss: str | None = None, available: bool = True) -> None:
        """Score one criterion.

        `available=False` removes it from the denominator entirely -- use it
        when the input metric is missing, so the company is not punished for
        data we do not have.
        """
        if not available:
            self.skipped.append(reason)
            return
        self.available += weight
        if condition:
            self.earned += weight
            self.reasons.append(reason)
        elif miss:
            self.misses.append(miss)

    def add_metric(self, value, weight: float, test, reason: str,
                   miss: str | None = None) -> None:
        """Score a criterion whose availability follows from its metric."""
        if _missing(value):
            self.skipped.append(reason)
            return
        self.add(bool(test(value)), weight, reason, miss=miss)

    def add_banded(self, value, weight: float, bands, label: str) -> None:
        """Award graded credit from (threshold, fraction, description) bands.

        Bands are evaluated in order and the first match wins, so they must be
        listed best-first. `fraction` is the share of `weight` earned.
        """
        if _missing(value):
            self.skipped.append(label)
            return
        self.available += weight
        for threshold, fraction, description in bands:
            if threshold(value):
                self.earned += weight * fraction
                if fraction > 0:
                    self.reasons.append(description)
                else:
                    self.misses.append(description)
                return
        self.misses.append(f"{label}: no band matched ({value})")

    # ---- penalties --------------------------------------------------------- #
    def penalise(self, points: float, reason: str) -> None:
        """Subtract from the final score, outside the weighted average.

        Used for risks that must not be offset by unrelated strengths.
        """
        if points > 0:
            self.penalties.append((reason, float(points)))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fact(self, **kwargs) -> None:
        """Attach context values for the report (not scored)."""
        self.facts.update(kwargs)

    # ---- results ----------------------------------------------------------- #
    @property
    def coverage(self) -> float:
        """Share of the intended framework that could actually be applied.

        Measured by criterion count rather than weight, so a score is judged on
        how much of its reasoning was possible, independent of how heavily any
        one criterion happens to be weighted.
        """
        applied = len(self.reasons) + len(self.misses)
        attempted = applied + len(self.skipped)
        return applied / attempted if attempted else 0.0

    @property
    def raw_score(self) -> float:
        """Weighted score before penalties, normalised to 0-100."""
        if self.available <= 0:
            return float("nan")
        return self.earned / self.available * 100.0

    @property
    def penalty_total(self) -> float:
        return sum(points for _, points in self.penalties)

    @property
    def score(self) -> float:
        """Final 0-100 score, penalties applied, floored at zero."""
        base = self.raw_score
        if np.isnan(base):
            return float("nan")
        return max(0.0, base - self.penalty_total)

    @property
    def confident(self) -> bool:
        return self.coverage >= MIN_COVERAGE_FOR_CONFIDENCE

    def to_dict(self) -> dict:
        caveats = list(self.warnings)
        if not self.confident:
            caveats.append(
                f"Low data coverage ({self.coverage:.0%} of criteria applied) -- "
                "this score is not directly comparable to a fully-measured one")
        return {
            "ticker": self.ticker,
            "horizon": self.horizon,
            "score": round(self.score, 1) if not np.isnan(self.score) else np.nan,
            "raw_score": round(self.raw_score, 1) if not np.isnan(self.raw_score) else np.nan,
            "penalty": round(self.penalty_total, 1),
            "coverage": round(self.coverage, 3),
            "confident": self.confident,
            "reasons": list(self.reasons),
            "misses": list(self.misses),
            "warnings": caveats,
            "penalties": [f"{reason} (-{points:g})" for reason, points in self.penalties],
            "skipped": list(self.skipped),
            **self.facts,
        }


VALUATION_INPUTS = ("pe_own_pctile", "pe_sector_z", "fcf_yield", "peg")


def valuation_unmeasurable(valuation: dict) -> bool:
    """True when no valuation signal at all is available for this company.

    Either the multiples were suppressed (currency mismatch) or every valuation
    input is missing -- which happens for recently-listed names with too little
    price history for an own-history percentile and too few sector peers for a
    z-score.

    Callers award NEUTRAL credit in this case rather than skipping the block.
    Skipping removes it from the denominator, which is correct for one missing
    input among many but wrong when a whole fifth of the score vanishes: the
    company is then silently rescored on its strengths alone. Observed live on
    INFY, which topped the long-term screen at 94/100 precisely because its
    multiples are currency-suppressed, and again in the medium engine where a
    suppressed stock outranked one whose cheapness was demonstrated.
    Normalisation had turned "we cannot price it" into "it is priced well".
    """
    if valuation.get("currency_suppressed"):
        return True
    return all(_missing(valuation.get(key)) for key in VALUATION_INPUTS)


def conviction_tier(score: float, confident: bool = True) -> str:
    """Map a score to a plain-language tier.

    Deliberately coarse. Ranking to one decimal place implies a precision these
    inputs do not support, and a five-way split communicates the same ordering
    without inviting false confidence in the difference between 71 and 73.
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "unrated"
    if not confident:
        return "insufficient data"
    if score >= 80:
        return "strong"
    if score >= 65:
        return "good"
    if score >= 50:
        return "moderate"
    if score >= 35:
        return "weak"
    return "avoid"


def rank_scores(scores: list[dict], top: int | None = None,
                require_confident: bool = False) -> pd.DataFrame:
    """Turn scored dicts into a ranked frame."""
    if not scores:
        return pd.DataFrame()
    frame = pd.DataFrame(scores)
    if require_confident and "confident" in frame.columns:
        frame = frame[frame["confident"]]
    frame = (frame.sort_values("score", ascending=False, na_position="last")
                  .reset_index(drop=True))
    frame.insert(0, "rank", frame.index + 1)
    if "score" in frame.columns and "confident" in frame.columns:
        frame["tier"] = [conviction_tier(s, c)
                         for s, c in zip(frame["score"], frame["confident"])]
    return frame.head(top) if top else frame
