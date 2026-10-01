#!/usr/bin/env python3
"""Medium-term scoring engine: 1-2 year holds.

WHAT DRIVES RETURN AT THIS HORIZON
    Over 1-2 years, returns are driven by earnings revisions, post-earnings-
    announcement drift, price momentum and multiple re-rating. This is the
    horizon where fundamentals and the tape meet: the market is still reacting
    to new information, but the holding period is long enough for a genuine
    change in the earnings path to matter.

    The sharpest earnings signal is acceleration, not the growth level. Post-
    earnings-announcement drift responds to a CHANGE in the growth rate: a
    company moving from +8% to +25% YoY is usually repriced more forcefully than
    one that stays at +25%. Quarterly revenue and EPS growth are therefore
    scored, but acceleration is given the larger share.

WHAT IS DELIBERATELY EXCLUDED, AND WHY
    No long-horizon compounding metrics such as 5-year return, dividend yield or
    FCF durability dominate here. They matter for a 2-5 year owner, but over the
    next 12-24 months the market pays for revisions, momentum and room to re-
    rate. No intraday or very short-term chart triggers either: those belong in
    the short-term engine, where execution and risk management are the edge.

WEIGHTS
    Earnings momentum       30   quarterly YoY growth and acceleration
    Price momentum          25   12-1 momentum first, then 6-month return
    Re-rating room          20   P/E vs own history, sector and PEG
    Quality floor           15   ROCE/ROE and leverage
    Trend confirmation      10   SMA200, weekly trend and ADX

    Price momentum gets the second-largest weight because 12-1 momentum is the
    strongest documented factor at this horizon (Jegadeesh-Titman). The most
    recent month is skipped deliberately: short-term reversal contaminates that
    month, so 12-1 captures persistent trend without chasing a noisy last leg.
    Quality is a floor, not the thesis -- it stops momentum from dragging in
    leveraged junk that rallies hard and then collapses.

PENALTIES SIT OUTSIDE THE AVERAGE
    Severe pledging, promoter sell-downs, vertical overextension and current
    losses subtract directly. These are path risks: they can break a medium-
    term trade even when the weighted criteria still look attractive.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.scoring.base import Score, _missing, valuation_unmeasurable as _valuation_unmeasurable

HORIZON = "medium"

W_EARNINGS_MOMENTUM = 30.0
W_PRICE_MOMENTUM = 25.0
W_VALUATION = 20.0
W_QUALITY = 15.0
W_TREND = 10.0


def score_medium_term(fundamentals: dict, valuation: dict | None = None,
                      governance: dict | None = None,
                      technical: dict | None = None) -> Score:
    """Score one company for a 1-2 year hold."""
    fundamentals = fundamentals or {}
    valuation = valuation or {}
    governance = governance or {}
    technical = technical or {}

    ticker = fundamentals.get("ticker") or technical.get("symbol") or "?"
    score = Score(ticker, HORIZON)
    is_lender = bool(fundamentals.get("is_lender"))

    score.fact(is_lender=is_lender,
               latest_quarter=fundamentals.get("latest_quarter"),
               sector=valuation.get("sector"))

    # ====================== EARNINGS MOMENTUM (30) ========================== #
    # Acceleration carries the larger share because PEAD is a reaction to a
    # change in expectations, not merely to a high reported growth rate.
    score.add_banded(
        fundamentals.get("q_revenue_acceleration"), W_EARNINGS_MOMENTUM * 0.30,
        [(lambda v: v >= 10, 1.0, "Revenue growth accelerated by 10pp+ YoY"),
         (lambda v: v >= 4, 0.75, "Revenue growth acceleration is positive"),
         (lambda v: v >= -2, 0.35, "Revenue growth broadly stable"),
         (lambda v: True, 0.0, "Revenue growth is decelerating")],
        "Revenue acceleration")

    score.add_banded(
        fundamentals.get("q_eps_acceleration"), W_EARNINGS_MOMENTUM * 0.35,
        [(lambda v: v >= 12, 1.0, "EPS growth accelerated sharply"),
         (lambda v: v >= 5, 0.75, "EPS growth acceleration is positive"),
         (lambda v: v >= -3, 0.35, "EPS growth broadly stable"),
         (lambda v: True, 0.0, "EPS growth is decelerating")],
        "EPS acceleration")

    score.add_banded(
        fundamentals.get("q_revenue_yoy"), W_EARNINGS_MOMENTUM * 0.15,
        [(lambda v: v >= 25, 1.0, "Quarterly revenue growth above 25% YoY"),
         (lambda v: v >= 15, 0.75, "Quarterly revenue growth 15-25% YoY"),
         (lambda v: v >= 5, 0.4, "Quarterly revenue growth is positive"),
         (lambda v: True, 0.0, "Quarterly revenue is flat or shrinking")],
        "Quarterly revenue growth")

    score.add_banded(
        fundamentals.get("q_eps_yoy"), W_EARNINGS_MOMENTUM * 0.20,
        [(lambda v: v >= 30, 1.0, "Quarterly EPS growth above 30% YoY"),
         (lambda v: v >= 15, 0.75, "Quarterly EPS growth 15-30% YoY"),
         (lambda v: v >= 0, 0.35, "Quarterly EPS growth is positive but modest"),
         (lambda v: True, 0.0, "Quarterly EPS is shrinking")],
        "Quarterly EPS growth")

    # ====================== PRICE MOMENTUM (25) ============================= #
    score.add_banded(
        technical.get("mom_12_1"), W_PRICE_MOMENTUM * 0.70,
        [(lambda v: v >= 35, 1.0, "Strong 12-1 momentum -- persistent medium-term trend"),
         (lambda v: v >= 18, 0.8, "Positive 12-1 momentum"),
         (lambda v: v >= 5, 0.45, "Mild 12-1 momentum"),
         (lambda v: v >= -5, 0.15, "Flat 12-1 momentum"),
         (lambda v: True, 0.0, "Negative 12-1 momentum")],
        "12-1 momentum")

    score.add_banded(
        technical.get("ret_6m"), W_PRICE_MOMENTUM * 0.30,
        [(lambda v: v >= 25, 1.0, "Six-month return above 25%"),
         (lambda v: v >= 12, 0.75, "Six-month return 12-25%"),
         (lambda v: v >= 0, 0.35, "Six-month return is positive"),
         (lambda v: True, 0.0, "Negative six-month return")],
        "Six-month return")

    # ====================== VALUATION / RE-RATING ROOM (20) ================= #
    # Unmeasurable valuation is scored NEUTRAL rather than skipped, for the same
    # reason as in long_term.py: skipping removes the block from the denominator
    # and quietly rescores the company on its strengths alone. Observed live,
    # a currency-suppressed stock scored 97.4 here against 96.4 for one whose
    # cheapness was actually demonstrated -- the data gap was outranking the
    # evidence. Neutral credit says "valuation unknown, assume average".
    if _valuation_unmeasurable(valuation):
        score.available += W_VALUATION
        score.earned += W_VALUATION * 0.5
        reason = ("Valuation could not be measured -- scored neutral rather "
                  "than skipped, so a data gap cannot inflate the rank")
        score.misses.append(reason)
        score.warn(reason)
    else:
        score.add_banded(
            valuation.get("pe_own_pctile"), W_VALUATION * 0.45,
            [(lambda v: v >= 75, 1.0, "P/E has substantial room to re-rate vs own history"),
             (lambda v: v >= 55, 0.75, "P/E below its own median"),
             (lambda v: v >= 35, 0.4, "P/E around its own history"),
             (lambda v: v >= 15, 0.15, "P/E is high versus own history"),
             (lambda v: True, 0.0, "P/E leaves little own-history re-rating room")],
            "P/E re-rating room")

        score.add_banded(
            valuation.get("pe_sector_z"), W_VALUATION * 0.30,
            [(lambda v: v >= 1.0, 1.0, "Materially cheaper than sector peers"),
             (lambda v: v >= 0.2, 0.7, "Cheaper than sector peers"),
             (lambda v: v >= -0.5, 0.4, "Valuation in line with sector peers"),
             (lambda v: True, 0.0, "Expensive versus sector peers")],
            "Sector re-rating room")

        score.add_banded(
            valuation.get("peg"), W_VALUATION * 0.25,
            [(lambda v: v <= 0.8, 1.0, "PEG below 0.8 -- growth is underpriced"),
             (lambda v: v <= 1.5, 0.7, "PEG 0.8-1.5 -- growth reasonably priced"),
             (lambda v: v <= 2.5, 0.3, "PEG 1.5-2.5 -- some growth priced in"),
             (lambda v: True, 0.0, "PEG above 2.5 -- little re-rating room")],
            "PEG")

    # ====================== QUALITY FLOOR (15) ============================== #
    if is_lender:
        score.add_banded(
            fundamentals.get("roe_mean_5y"), W_QUALITY * 0.65,
            [(lambda v: v >= 18, 1.0, "ROE above 18% (5y avg) -- strong lender quality floor"),
             (lambda v: v >= 14, 0.75, "ROE 14-18% (5y avg)"),
             (lambda v: v >= 10, 0.45, "ROE 10-14% (5y avg)"),
             (lambda v: True, 0.0, "ROE below 10% -- weak lender quality floor")],
            "ROE (lender)")
        score.skipped.append("Leverage (lender)")
    else:
        score.add_banded(
            fundamentals.get("roce_mean_5y"), W_QUALITY * 0.65,
            [(lambda v: v >= 24, 1.0, "ROCE above 24% (5y avg) -- quality floor is strong"),
             (lambda v: v >= 18, 0.8, "ROCE 18-24% (5y avg)"),
             (lambda v: v >= 12, 0.45, "ROCE 12-18% (5y avg)"),
             (lambda v: True, 0.0, "ROCE below 12% -- weak quality floor")],
            "ROCE floor")

        score.add_banded(
            fundamentals.get("debt_to_equity"), W_QUALITY * 0.35,
            [(lambda v: v <= 0.25, 1.0, "Low leverage supports the momentum setup"),
             (lambda v: v <= 0.75, 0.7, "Manageable leverage"),
             (lambda v: v <= 1.5, 0.3, "Elevated leverage"),
             (lambda v: True, 0.0, "High leverage -- momentum can unwind violently")],
            "Leverage")

    # ====================== TREND CONFIRMATION (10) ========================= #
    score.add_banded(
        technical.get("close_vs_sma200_pct"), W_TREND * 0.40,
        [(lambda v: v >= 8, 1.0, "Price is comfortably above the 200-day average"),
         (lambda v: v >= 0, 0.7, "Price is above the 200-day average"),
         (lambda v: v >= -5, 0.25, "Price is testing the 200-day average"),
         (lambda v: True, 0.0, "Price is below the 200-day average")],
        "Primary trend")

    score.add_banded(
        technical.get("weekly_close_vs_ema20_pct"), W_TREND * 0.30,
        [(lambda v: v >= 5, 1.0, "Weekly trend is firmly positive"),
         (lambda v: v >= 0, 0.65, "Weekly trend is positive"),
         (lambda v: v >= -3, 0.25, "Weekly trend is flattening"),
         (lambda v: True, 0.0, "Weekly trend is negative")],
        "Weekly trend")

    score.add_banded(
        technical.get("adx"), W_TREND * 0.30,
        [(lambda v: v >= 28, 1.0, "ADX confirms a strong trend"),
         (lambda v: v >= 20, 0.7, "ADX confirms a tradable trend"),
         (lambda v: v >= 15, 0.3, "ADX shows a weak trend"),
         (lambda v: True, 0.0, "ADX shows little trend strength")],
        "ADX trend strength")

    # ====================== PENALTIES ======================================= #
    _apply_penalties(score, fundamentals, valuation, governance, technical)

    return score


def _fmt_pct(value) -> str:
    return "unknown" if _missing(value) else f"{float(value):.0f}%"


def _apply_penalties(score: Score, fundamentals: dict, valuation: dict,
                     governance: dict, technical: dict) -> None:
    """Risks that can break a 1-2 year thesis before the average catches up."""
    band = governance.get("pledge_band")
    pledge = governance.get("promoter_pledge_pct")
    if band == "severe":
        score.penalise(12, f"Severe promoter pledging ({_fmt_pct(pledge)} of stake)")
    elif band == "elevated":
        score.penalise(7, f"Elevated promoter pledging ({_fmt_pct(pledge)} of stake)")

    trend = governance.get("trend_band")
    change = governance.get("promoter_change_4q_pp")
    if trend == "selling_heavily":
        score.penalise(8, f"Promoters sold down {abs(float(change)):.1f}pp over four quarters")
    elif trend == "selling":
        score.penalise(3, f"Promoters reduced stake {abs(float(change)):.1f}pp over four quarters")

    dist_high = technical.get("dist_from_52w_high")
    rsi = technical.get("rsi")
    if not _missing(dist_high) and not _missing(rsi) and dist_high >= -2 and rsi > 70:
        score.penalise(6, "Overextended at the 52-week high with RSI above 70")

    if valuation.get("is_loss_making"):
        score.penalise(8, "Currently loss-making")

    if valuation.get("currency_suppressed"):
        score.warn(valuation.get("valuation_caveat")
                   or "Valuation multiples suppressed (currency mismatch)")

    if fundamentals.get("is_lender"):
        score.warn("Lender: ROE substitutes for ROCE; asset quality is not captured here.")
