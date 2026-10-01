#!/usr/bin/env python3
"""Short-term scoring engine: weeks to months.

WHAT DRIVES RETURN AT THIS HORIZON
    Over weeks to months, return is dominated by trend structure, near-term price
    momentum, participation and execution. The engine is therefore technical by
    design: it scores whether price is stacked above key moving averages, whether
    those averages are rising, whether weekly trend agrees, and whether trend
    strength is expanding rather than fading.

WHAT IS DELIBERATELY EXCLUDED, AND WHY
    Fundamentals do not forecast the next few weeks with enough precision to be
    a primary signal. They act solely as a junk filter -- a loss-making or
    structurally weak company may still bounce, but it does not deserve the same
    risk budget as a clean trend. Valuation is also excluded: a stock can stay
    expensive for months while momentum persists, and cheap stocks can keep
    falling.

    At this horizon risk management is the actual edge, not signal selection.
    The score says whether a setup is attractive; the attached risk plan says
    where the trade is wrong, how much is at risk per share, and what 2R/3R
    outcomes look like before capital is committed.

WEIGHTS
    Trend & structure   45   MA stack, slopes, weekly trend, structure, ADX
    Price momentum      25   20-day, 60-day and 3-month returns
    Confirmation        25   healthy RSI, MACD, volume confirmation
    Relative strength    5   market-relative trend when available

    Trend gets nearly half the score because short-term trades fail fastest when
    they fight structure. RSI is rewarded only in a healthy band: 45-65 shows
    participation without exhaustion, while very high RSI is a penalty rather
    than a reason to chase.

PENALTIES SIT OUTSIDE THE AVERAGE
    Illiquidity, blow-off volume, extreme overbought readings and trading below
    the 200-day average subtract directly. A signal that cannot be traded, or is
    already exhausted, is worse than a merely low-scoring setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.scoring.base import Score, _missing

HORIZON = "short"

W_TREND_STRUCTURE = 45.0
W_PRICE_MOMENTUM = 25.0
W_CONFIRMATION = 25.0
W_RELATIVE_STRENGTH = 5.0


def risk_plan(technical: dict, atr_multiple: float = 2.0) -> dict:
    """Concrete ATR-based trade plan for a short-term setup."""
    technical = technical or {}
    entry = technical.get("latest_close")
    atr = technical.get("atr")
    suggested_hold = "4-10 weeks"
    if _missing(entry) or _missing(atr):
        return {"stop_loss": np.nan, "risk_per_share": np.nan,
                "target_2r": np.nan, "target_3r": np.nan,
                "suggested_hold": suggested_hold}

    entry = float(entry)
    risk_per_share = float(atr_multiple) * float(atr)
    stop_loss = entry - risk_per_share
    return {"stop_loss": stop_loss,
            "risk_per_share": risk_per_share,
            "target_2r": entry + 2.0 * risk_per_share,
            "target_3r": entry + 3.0 * risk_per_share,
            "suggested_hold": suggested_hold}


def score_short_term(fundamentals: dict | None = None, valuation: dict | None = None,
                     governance: dict | None = None,
                     technical: dict | None = None) -> Score:
    """Score one company for a weeks-to-months hold."""
    if technical is None and fundamentals and "latest_close" in fundamentals:
        technical = fundamentals
        fundamentals = {}
    fundamentals = fundamentals or {}
    valuation = valuation or {}
    governance = governance or {}
    technical = technical or {}

    ticker = fundamentals.get("ticker") or technical.get("symbol") or "?"
    score = Score(ticker, HORIZON)
    score.fact(entry=technical.get("latest_close"), **risk_plan(technical))

    # ====================== TREND & STRUCTURE (45) ========================== #
    score.add_banded(
        technical.get("close_vs_sma20_pct"), W_TREND_STRUCTURE * 0.14,
        [(lambda v: 0 <= v <= 8, 1.0, "Price is above SMA20 without being stretched"),
         (lambda v: 8 < v <= 15, 0.65, "Price is above SMA20 but extended"),
         (lambda v: -2 <= v < 0, 0.35, "Price is near SMA20 support"),
         (lambda v: True, 0.0, "Price is below SMA20 or vertically stretched")],
        "Price vs SMA20")

    score.add_banded(
        technical.get("close_vs_sma50_pct"), W_TREND_STRUCTURE * 0.14,
        [(lambda v: v >= 5, 1.0, "Price is above SMA50"),
         (lambda v: v >= 0, 0.75, "Price is modestly above SMA50"),
         (lambda v: v >= -3, 0.25, "Price is testing SMA50"),
         (lambda v: True, 0.0, "Price is below SMA50")],
        "Price vs SMA50")

    score.add_banded(
        technical.get("close_vs_sma200_pct"), W_TREND_STRUCTURE * 0.12,
        [(lambda v: v >= 5, 1.0, "Price is above SMA200 -- primary trend supports the trade"),
         (lambda v: v >= 0, 0.7, "Price is just above SMA200"),
         (lambda v: v >= -5, 0.2, "Price is near SMA200 but primary trend is fragile"),
         (lambda v: True, 0.0, "Price is below SMA200")],
        "Price vs SMA200")

    score.add_banded(
        technical.get("sma20_slope_10d"), W_TREND_STRUCTURE * 0.10,
        [(lambda v: v >= 2.0, 1.0, "SMA20 is rising quickly"),
         (lambda v: v >= 0.5, 0.7, "SMA20 is rising"),
         (lambda v: v >= -0.5, 0.25, "SMA20 is flat"),
         (lambda v: True, 0.0, "SMA20 is falling")],
        "SMA20 slope")

    score.add_banded(
        technical.get("sma50_slope_10d"), W_TREND_STRUCTURE * 0.10,
        [(lambda v: v >= 1.2, 1.0, "SMA50 is rising"),
         (lambda v: v >= 0, 0.65, "SMA50 slope is positive"),
         (lambda v: v >= -0.5, 0.25, "SMA50 is flat"),
         (lambda v: True, 0.0, "SMA50 is falling")],
        "SMA50 slope")

    score.add_banded(
        technical.get("weekly_close_vs_ema20_pct"), W_TREND_STRUCTURE * 0.14,
        [(lambda v: v >= 4, 1.0, "Weekly EMA trend confirms the daily setup"),
         (lambda v: v >= 0, 0.7, "Weekly EMA trend is positive"),
         (lambda v: v >= -3, 0.25, "Weekly EMA trend is neutral"),
         (lambda v: True, 0.0, "Weekly EMA trend is negative")],
        "Weekly EMA trend")

    score.add_banded(
        technical.get("structure_hh_hl_60d"), W_TREND_STRUCTURE * 0.12,
        [(lambda v: v >= 1, 1.0, "60-day structure is higher-highs and higher-lows"),
         (lambda v: True, 0.0, "60-day structure is not higher-highs and higher-lows")],
        "60-day structure")

    score.add_banded(
        technical.get("adx"), W_TREND_STRUCTURE * 0.12,
        [(lambda v: v >= 30, 1.0, "ADX shows strong trend strength"),
         (lambda v: v >= 22, 0.75, "ADX shows tradable trend strength"),
         (lambda v: v >= 16, 0.35, "ADX trend strength is emerging"),
         (lambda v: True, 0.0, "ADX shows little trend strength")],
        "ADX level")

    score.add_banded(
        technical.get("adx_trajectory_10d"), W_TREND_STRUCTURE * 0.12,
        [(lambda v: v >= 5, 1.0, "ADX is rising -- trend strength is expanding"),
         (lambda v: v >= 1, 0.7, "ADX trajectory is positive"),
         (lambda v: v >= -2, 0.25, "ADX trajectory is flat"),
         (lambda v: True, 0.0, "ADX is falling -- trend strength is fading")],
        "ADX trajectory")

    # ====================== PRICE MOMENTUM (25) ============================= #
    score.add_banded(
        technical.get("roc_20d"), W_PRICE_MOMENTUM * 0.35,
        [(lambda v: 4 <= v <= 15, 1.0, "20-day momentum is strong but not vertical"),
         (lambda v: 0 <= v < 4, 0.55, "20-day momentum is positive"),
         (lambda v: 15 < v <= 25, 0.45, "20-day momentum is extended"),
         (lambda v: True, 0.0, "20-day momentum is weak or exhausted")],
        "20-day momentum")

    score.add_banded(
        technical.get("roc_60d"), W_PRICE_MOMENTUM * 0.35,
        [(lambda v: 8 <= v <= 30, 1.0, "60-day momentum supports the setup"),
         (lambda v: 0 <= v < 8, 0.55, "60-day momentum is positive"),
         (lambda v: 30 < v <= 45, 0.45, "60-day momentum is extended"),
         (lambda v: True, 0.0, "60-day momentum is weak or exhausted")],
        "60-day momentum")

    score.add_banded(
        technical.get("ret_3m"), W_PRICE_MOMENTUM * 0.30,
        [(lambda v: 10 <= v <= 35, 1.0, "Three-month return confirms momentum"),
         (lambda v: 0 <= v < 10, 0.55, "Three-month return is positive"),
         (lambda v: 35 < v <= 55, 0.35, "Three-month return is stretched"),
         (lambda v: True, 0.0, "Three-month return is weak or exhausted")],
        "Three-month return")

    # ====================== CONFIRMATION (25) =============================== #
    score.add_banded(
        technical.get("rsi"), W_CONFIRMATION * 0.35,
        [(lambda v: 45 <= v <= 65, 1.0, "RSI is in the healthy participation band"),
         (lambda v: 40 <= v < 45, 0.45, "RSI is improving from a low base"),
         (lambda v: 65 < v <= 70, 0.45, "RSI is firm but approaching overbought"),
         (lambda v: 70 < v <= 75, 0.1, "RSI is overbought"),
         (lambda v: True, 0.0, "RSI is weak or extremely overbought")],
        "RSI health")

    score.add_banded(
        technical.get("macd_vs_signal"), W_CONFIRMATION * 0.30,
        [(lambda v: v > 0, 1.0, "MACD is above signal"),
         (lambda v: v >= -0.2, 0.35, "MACD is near a bullish cross"),
         (lambda v: True, 0.0, "MACD is below signal")],
        "MACD confirmation")

    score.add_banded(
        technical.get("volume_vs_20d_avg"), W_CONFIRMATION * 0.35,
        [(lambda v: 1.2 <= v <= 2.5, 1.0, "Volume confirms the move"),
         (lambda v: 0.8 <= v < 1.2, 0.45, "Volume is normal"),
         (lambda v: 2.5 < v <= 3.0, 0.35, "Volume is high but not yet a blow-off"),
         (lambda v: True, 0.0, "Volume does not confirm, or is a blow-off")],
        "Volume confirmation")

    # ====================== RELATIVE STRENGTH (5) =========================== #
    rel = (technical.get("relative_strength_3m")
           if "relative_strength_3m" in technical else technical.get("rs_3m"))
    score.add_banded(
        rel, W_RELATIVE_STRENGTH,
        [(lambda v: v >= 8, 1.0, "Outperforming the broader market"),
         (lambda v: v >= 0, 0.65, "Holding up better than the broader market"),
         (lambda v: v >= -5, 0.25, "Relative strength is neutral"),
         (lambda v: True, 0.0, "Underperforming the broader market")],
        "Relative strength")

    # ====================== PENALTIES ======================================= #
    _apply_penalties(score, fundamentals, technical)

    return score


def _apply_penalties(score: Score, fundamentals: dict, technical: dict) -> None:
    """Risks that matter more than a short-term setup score."""
    turnover = technical.get("avg_turnover_20d")
    if not _missing(turnover):
        if turnover < 5_000_000:
            score.penalise(10, "Illiquid: average traded value below 50 lakh")
        elif turnover < 20_000_000:
            score.penalise(4, "Thin liquidity: average traded value below 2 crore")

    rsi = technical.get("rsi")
    if not _missing(rsi) and rsi > 75:
        score.penalise(8, "Extremely overbought RSI above 75")

    volume = technical.get("volume_vs_20d_avg")
    if not _missing(volume) and volume > 3:
        score.penalise(7, "Blow-off volume above 3x 20-day average -- possible exhaustion")

    vs_sma200 = technical.get("close_vs_sma200_pct")
    if not _missing(vs_sma200) and vs_sma200 < 0:
        score.penalise(8, "Price below SMA200 -- fighting the primary trend")

    if fundamentals.get("is_lender"):
        score.warn("Lender: short-term score is technical; asset quality still needs a separate check.")
    if fundamentals.get("profitable_years") == 0 or fundamentals.get("net_income_cagr_3y") == -100.0:
        score.penalise(5, "Fundamental junk filter: loss-making profile")
