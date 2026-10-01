#!/usr/bin/env python3
"""Long-term scoring engine: 2-5 year holds.

WHAT DRIVES RETURN AT THIS HORIZON
    Over 2-5 years, total return decomposes into earnings growth, dividends,
    and the change in the multiple. Nothing else survives the time frame. That
    single fact determines everything below: the engine scores the durability
    of the earnings stream and the price paid for it, and ignores the things
    that dominate a weekly chart.

WHAT IS DELIBERATELY EXCLUDED, AND WHY
    No RSI, MACD, breakout proximity or volume signals. These have no
    demonstrated predictive content over multi-year horizons, and including
    them would import short-horizon noise into a decision measured in years.

    No 12-month price momentum either -- which is a more interesting omission,
    because momentum is genuinely powerful at 6-18 months. At 3-5 years the
    effect historically REVERSES (long-horizon reversal): yesterday's winners
    tend to underperform. Carrying momentum across from the medium-term engine
    would therefore be actively harmful, not merely useless.

WEIGHTS
    Quality      35   ROCE level, consistency, margin trend, earnings quality
    Growth       25   revenue/EPS CAGR and its stability
    Health       20   leverage, coverage, deleveraging, dilution
    Valuation    20   entry multiple vs own history and sector

    Quality carries the most weight because it is the most persistent: a
    business earning 25% on capital tends to keep doing so, while a single
    year's growth rate does not repeat reliably. Valuation is weighted lowest
    of the four but is never ignored -- paying too much caps the outcome even
    when the business is excellent.

PENALTIES SIT OUTSIDE THE AVERAGE
    Promoter pledging, sell-downs, accounting-quality failures and persistent
    cash-conversion problems subtract directly. These are the risks that end
    holdings permanently, and they must not be offset by a good margin trend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.scoring.base import Score, _missing, valuation_unmeasurable

HORIZON = "long"

W_QUALITY = 35.0
W_GROWTH = 25.0
W_HEALTH = 20.0
W_VALUATION = 20.0


def _valuation_unmeasurable(valuation: dict) -> bool:
    """Kept as a module-level alias so the long-term engine reads standalone.

    The implementation lives in base.py because the medium-term engine needs
    exactly the same rule -- both have a 20-point valuation block that must not
    be silently dropped from the denominator.
    """
    return valuation_unmeasurable(valuation)


def score_long_term(fundamentals: dict, valuation: dict | None = None,
                    governance: dict | None = None,
                    technical: dict | None = None) -> Score:
    """Score one company for a 2-5 year hold."""
    valuation = valuation or {}
    governance = governance or {}
    technical = technical or {}

    ticker = fundamentals.get("ticker", "?")
    score = Score(ticker, HORIZON)
    is_lender = bool(fundamentals.get("is_lender"))

    score.fact(is_lender=is_lender,
               years_of_data=fundamentals.get("years_of_data"),
               sector=valuation.get("sector"))

    # ====================== QUALITY (35) ==================================== #
    # Capital efficiency is the single most persistent business characteristic,
    # so it carries the largest share.
    if is_lender:
        # ROCE is meaningless where interest is revenue and leverage is the
        # model. ROE carries the weight instead; the dedicated lender engine in
        # financials.py adds asset-quality detail this generic path cannot.
        score.add_banded(
            fundamentals.get("roe_mean_5y"), W_QUALITY * 0.5,
            [(lambda v: v >= 18, 1.0, "ROE above 18% (5y avg) -- strong for a lender"),
             (lambda v: v >= 14, 0.75, "ROE 14-18% (5y avg)"),
             (lambda v: v >= 10, 0.45, "ROE 10-14% (5y avg)"),
             (lambda v: v >= 0, 0.1, "ROE below 10% -- weak returns on equity"),
             (lambda v: True, 0.0, "Negative ROE")],
            "ROE (lender)")
    else:
        score.add_banded(
            fundamentals.get("roce_mean_5y"), W_QUALITY * 0.40,
            [(lambda v: v >= 25, 1.0, "ROCE above 25% (5y avg) -- excellent capital efficiency"),
             (lambda v: v >= 18, 0.8, "ROCE 18-25% (5y avg) -- strong"),
             (lambda v: v >= 13, 0.55, "ROCE 13-18% (5y avg) -- adequate"),
             (lambda v: v >= 9, 0.25, "ROCE 9-13% (5y avg) -- modest"),
             (lambda v: True, 0.0, "ROCE below 9% -- earns little on capital employed")],
            "ROCE level")

        # Consistency is scored separately from level because a compounder is
        # defined by repeating a high return, not achieving one once.
        score.add_banded(
            fundamentals.get("roce_std_5y"), W_QUALITY * 0.20,
            [(lambda v: v <= 3, 1.0, "Very stable ROCE (5y std <3pp)"),
             (lambda v: v <= 6, 0.7, "Stable ROCE (5y std <6pp)"),
             (lambda v: v <= 12, 0.35, "Variable ROCE (5y std <12pp)"),
             (lambda v: True, 0.0, "Erratic ROCE -- returns on capital are not repeatable")],
            "ROCE consistency")

    score.add_banded(
        fundamentals.get("net_margin_trend"), W_QUALITY * 0.15,
        [(lambda v: v >= 0.5, 1.0, "Net margin expanding"),
         (lambda v: v >= -0.1, 0.6, "Net margin stable"),
         (lambda v: v >= -1.0, 0.2, "Net margin drifting down"),
         (lambda v: True, 0.0, "Net margin contracting materially")],
        "Margin trend")

    # Cash conversion: profit is an opinion, cash is a fact.
    score.add_banded(
        fundamentals.get("cfo_to_net_income_mean"), W_QUALITY * 0.25,
        [(lambda v: v >= 1.1, 1.0, "Operating cash flow exceeds reported profit"),
         (lambda v: v >= 0.85, 0.75, "Profit converts well to cash"),
         (lambda v: v >= 0.6, 0.3, "Weak cash conversion"),
         (lambda v: True, 0.0, "Profit is not converting to cash -- accruals warning")],
        "Earnings quality")

    # ====================== GROWTH (25) ===================================== #
    score.add_banded(
        fundamentals.get("revenue_cagr_5y"), W_GROWTH * 0.35,
        [(lambda v: v >= 20, 1.0, "Revenue CAGR above 20% (5y)"),
         (lambda v: v >= 12, 0.8, "Revenue CAGR 12-20% (5y)"),
         (lambda v: v >= 7, 0.55, "Revenue CAGR 7-12% (5y)"),
         (lambda v: v >= 3, 0.25, "Revenue CAGR 3-7% (5y) -- barely ahead of inflation"),
         (lambda v: True, 0.0, "Revenue stagnant or shrinking over 5y")],
        "Revenue growth 5y")

    score.add_banded(
        fundamentals.get("eps_cagr_5y"), W_GROWTH * 0.40,
        [(lambda v: v >= 20, 1.0, "EPS CAGR above 20% (5y)"),
         (lambda v: v >= 12, 0.8, "EPS CAGR 12-20% (5y)"),
         (lambda v: v >= 7, 0.5, "EPS CAGR 7-12% (5y)"),
         (lambda v: v >= 0, 0.2, "EPS barely growing over 5y"),
         (lambda v: True, 0.0, "EPS declining over 5y")],
        "EPS growth 5y")

    # Consistency of profitability. A company profitable in every year of the
    # window is a different risk from one that dips into loss.
    years = fundamentals.get("years_of_data") or 0
    profitable = fundamentals.get("profitable_years")
    if not _missing(profitable) and years >= 3:
        score.add(profitable == years, W_GROWTH * 0.25,
                  f"Profitable in all {years} reported years",
                  miss=f"Loss-making in {years - int(profitable)} of {years} years")
    else:
        score.skipped.append("Profit consistency")

    # ====================== FINANCIAL HEALTH (20) =========================== #
    if is_lender:
        # Debt/equity and interest coverage do not apply to a lender: borrowing
        # IS the raw material. Only dilution is scored generically here.
        score.add_banded(
            fundamentals.get("share_count_cagr"), W_HEALTH * 0.5,
            [(lambda v: v <= 0.5, 1.0, "No meaningful dilution"),
             (lambda v: v <= 3, 0.6, "Mild dilution"),
             (lambda v: v <= 8, 0.2, "Notable dilution"),
             (lambda v: True, 0.0, "Heavy dilution -- per-share value diluted")],
            "Dilution")
    else:
        score.add_banded(
            fundamentals.get("debt_to_equity"), W_HEALTH * 0.35,
            [(lambda v: v <= 0.1, 1.0, "Effectively debt-free"),
             (lambda v: v <= 0.5, 0.8, "Low leverage (D/E below 0.5)"),
             (lambda v: v <= 1.0, 0.5, "Moderate leverage (D/E 0.5-1.0)"),
             (lambda v: v <= 2.0, 0.2, "High leverage (D/E 1-2)"),
             (lambda v: True, 0.0, "Very high leverage (D/E above 2)")],
            "Leverage")

        score.add_banded(
            fundamentals.get("interest_coverage"), W_HEALTH * 0.25,
            [(lambda v: v >= 10, 1.0, "Interest comfortably covered (>10x)"),
             (lambda v: v >= 5, 0.75, "Interest well covered (5-10x)"),
             (lambda v: v >= 2.5, 0.4, "Interest adequately covered (2.5-5x)"),
             (lambda v: True, 0.0, "Thin interest cover -- vulnerable to a rate or earnings shock")],
            "Interest coverage")

        # Deleveraging is one of the most reliable re-rating catalysts.
        score.add_banded(
            fundamentals.get("debt_to_equity_trend"), W_HEALTH * 0.20,
            [(lambda v: v <= -0.05, 1.0, "Deleveraging over time"),
             (lambda v: v <= 0.05, 0.6, "Leverage stable"),
             (lambda v: True, 0.0, "Leverage rising")],
            "Debt trend")

        score.add_banded(
            fundamentals.get("share_count_cagr"), W_HEALTH * 0.20,
            [(lambda v: v <= 0.5, 1.0, "No meaningful dilution"),
             (lambda v: v <= 3, 0.6, "Mild dilution"),
             (lambda v: v <= 8, 0.2, "Notable dilution"),
             (lambda v: True, 0.0, "Heavy dilution -- per-share value diluted")],
            "Dilution")

    # ====================== VALUATION (20) ================================== #
    # Own-history percentile first: it answers "is this cheap for itself", which
    # is what re-rating potential depends on, and it is comparable across
    # sectors in a way an absolute multiple never is.
    #
    # WHY AN UNMEASURABLE VALUATION IS SCORED NEUTRAL RATHER THAN SKIPPED
    #   Skipping a criterion removes it from the denominator, which is right
    #   when one input among many is missing. But valuation is a fifth of this
    #   score, and when the WHOLE block is unavailable the normalisation quietly
    #   rescores the company on its strengths alone. Observed live: INFY topped
    #   the long-term screen at 94/100 precisely because its multiples are
    #   currency-suppressed, so the one dimension that might have held it back
    #   never applied. "We cannot price it" had become "it is priced well".
    #
    #   Awarding neutral credit says the honest thing instead -- valuation is
    #   unknown, so assume it is average -- and keeps the company rankable
    #   without letting a data gap win it the screen.
    if _valuation_unmeasurable(valuation):
        score.available += W_VALUATION
        score.earned += W_VALUATION * 0.5
        reason = ("Valuation could not be measured -- scored neutral rather "
                  "than skipped, so a data gap cannot inflate the rank")
        score.misses.append(reason)
        score.warn(reason)
    else:
        score.add_banded(
            valuation.get("pe_own_pctile"), W_VALUATION * 0.40,
            [(lambda v: v >= 75, 1.0, "P/E in the cheapest quartile of its own 5y range"),
             (lambda v: v >= 55, 0.75, "P/E below its own 5y median"),
             (lambda v: v >= 35, 0.4, "P/E around its own 5y median"),
             (lambda v: v >= 15, 0.15, "P/E in the expensive part of its own range"),
             (lambda v: True, 0.0, "P/E near the top of its own 5y range -- little re-rating room")],
            "Valuation vs own history")

        score.add_banded(
            valuation.get("pe_sector_z"), W_VALUATION * 0.25,
            [(lambda v: v >= 1.0, 1.0, "Materially cheaper than sector peers"),
             (lambda v: v >= 0.2, 0.7, "Cheaper than sector peers"),
             (lambda v: v >= -0.5, 0.4, "In line with sector peers"),
             (lambda v: True, 0.0, "More expensive than sector peers")],
            "Valuation vs sector")

        # FCF yield is the cash-based valuation cross-check: it cannot be
        # produced by accounting choices the way an earnings multiple can.
        score.add_banded(
            valuation.get("fcf_yield"), W_VALUATION * 0.20,
            [(lambda v: v >= 6, 1.0, "FCF yield above 6%"),
             (lambda v: v >= 3.5, 0.7, "FCF yield 3.5-6%"),
             (lambda v: v >= 1.5, 0.35, "FCF yield 1.5-3.5%"),
             (lambda v: True, 0.0, "Minimal or negative free cash flow yield")],
            "FCF yield")

        score.add_banded(
            valuation.get("peg"), W_VALUATION * 0.15,
            [(lambda v: v <= 1.0, 1.0, "PEG below 1 -- growth cheaply priced"),
             (lambda v: v <= 1.8, 0.65, "PEG 1-1.8 -- growth reasonably priced"),
             (lambda v: v <= 3.0, 0.25, "PEG 1.8-3"),
             (lambda v: True, 0.0, "PEG above 3 -- paying a lot for the growth")],
            "PEG")

    # ====================== PENALTIES ======================================= #
    _apply_penalties(score, fundamentals, valuation, governance)

    return score


def _apply_penalties(score: Score, fundamentals: dict, valuation: dict,
                     governance: dict) -> None:
    """Risks that must not be averaged away by unrelated strengths."""
    # Governance. Pledging is reflexive risk: a price fall forces lender selling
    # which drives the price down further, so it is scored as a cliff.
    band = governance.get("pledge_band")
    pledge = governance.get("promoter_pledge_pct")
    if band == "severe":
        score.penalise(15, f"Severe promoter pledging ({pledge:.0f}% of stake)")
    elif band == "elevated":
        score.penalise(8, f"Elevated promoter pledging ({pledge:.0f}% of stake)")
    elif band == "moderate":
        score.penalise(3, f"Some promoter pledging ({pledge:.0f}% of stake)")

    trend = governance.get("trend_band")
    change = governance.get("promoter_change_4q_pp")
    if trend == "selling_heavily":
        score.penalise(10, f"Promoters sold down {abs(change):.1f}pp over four quarters")
    elif trend == "selling":
        score.penalise(4, f"Promoters reduced stake {abs(change):.1f}pp over four quarters")

    # Accounting quality. Sustained cash-conversion failure is the single most
    # common precursor to a long-term holding going wrong.
    cfo = fundamentals.get("cfo_to_net_income_mean")
    if not _missing(cfo) and cfo < 0.5:
        score.penalise(12, f"Operating cash flow only {cfo:.0%} of reported profit "
                           "over the period -- earnings are not cash-backed")

    # A company that has never generated free cash flow is funding itself from
    # external capital, which is a different proposition over a 2-5 year hold.
    fcf_years = fundamentals.get("positive_fcf_years")
    years = fundamentals.get("years_of_data") or 0
    if not _missing(fcf_years) and years >= 4 and fcf_years == 0:
        score.penalise(10, "No positive free cash flow in any reported year")

    if fundamentals.get("is_lender"):
        score.warn("Lender: scored without ROCE/leverage criteria. "
                   "Asset quality (GNPA, provisioning, capital adequacy) is not "
                   "captured here and must be checked separately.")

    if valuation.get("currency_suppressed"):
        score.warn(valuation.get("valuation_caveat")
                   or "Valuation multiples suppressed (currency mismatch)")

    if valuation.get("is_loss_making"):
        score.penalise(8, "Currently loss-making")

    years = fundamentals.get("years_of_data") or 0
    if years and years < 4:
        score.warn(f"Only {years} years of financial history -- a 2-5 year "
                   "judgement is being made on a short record")
