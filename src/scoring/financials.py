#!/usr/bin/env python3
"""Scoring override for lenders: banks, NBFCs and financiers.

WHY A SEPARATE MODEL IS NOT OPTIONAL
    A lender's economics invert the assumptions every other engine rests on.
    Interest expense is not a financing cost, it is the cost of goods sold.
    Leverage is not a risk to be minimised, it is the business model -- a bank
    running at 8x assets-to-equity is normal, and one at 1x has no business.
    Deposits are liabilities that the franchise's value comes from attracting.

    So the generic metrics do not merely lose accuracy, they invert:

        ROCE = EBIT / (debt + equity)   EBIT is not reported at all; "capital
                                        employed" would count deposits as
                                        invested capital
        Debt/Equity                     high is healthy, and the generic engine
                                        penalises it
        EV/EBITDA                       enterprise value is meaningless when
                                        debt is inventory
        Interest coverage               interest is revenue, not a burden

    Confirmed in the data: of 101 Financial Services names in the Nifty 500,
    EBIT is present for 55 and operating income for 40, while net interest
    income is present for all 101. The generic model would score most of them
    on whichever metrics happened to survive, which is worse than not scoring
    them at all.

WHAT THIS MODEL CAN AND CANNOT SEE -- STATED PLAINLY
    Available for all 101 lenders: net interest income, total assets, equity,
    tangible book value, net income, EPS, and price-to-book.

    NOT available from this data source, at all: gross and net NPA, provision
    coverage, capital adequacy (CRAR/Tier-1), CASA ratio, slippage and credit
    cost. These are the metrics that actually distinguish a good lender from a
    dangerous one -- asset quality is the whole game, and a bank can post
    excellent ROA right up until the credit cycle turns.

    The model therefore scores profitability, capitalisation and valuation, and
    every output carries an explicit caveat that asset quality is unverified.
    That is a real limitation, not a formality: this engine can tell you a
    lender is profitably run and reasonably priced. It cannot tell you whether
    the loan book is sound, and it says so rather than implying otherwise.

WEIGHTS
    Profitability   35   ROA and ROE -- ROA first, since it is the measure that
                         cannot be flattered by leverage
    Capitalisation  25   equity/assets, a solvency buffer standing in for the
                         capital-adequacy data this source does not carry
    Growth          20   asset and earnings growth, with very fast loan growth
                         treated as a WARNING rather than a strength
    Valuation       20   price-to-book, the standard lender multiple
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.scoring.base import Score, _missing, valuation_unmeasurable

HORIZON = "long"

W_PROFITABILITY = 35.0
W_CAPITALISATION = 25.0
W_GROWTH = 20.0
W_VALUATION = 20.0

# Asset growth beyond this is scored as a risk rather than a success. Rapid
# loan growth is the most reliable leading indicator of future bad loans: books
# that double in two years are underwriting borrowers the incumbents declined,
# and the defaults surface two to three years later -- comfortably inside the
# long-horizon holding period this engine is used for.
ASSET_GROWTH_CAUTION = 30.0

# --------------------------------------------------------------------------- #
# Sub-classification of NSE's "Financial Services" macro-sector.
#
# The NSE bucket holds 101 Nifty 500 names and bundles at least four distinct
# business models. Applying the lender model to all of them reproduces, one
# level down, exactly the error this module exists to correct -- and it was
# visible in the output: life insurers (HDFCLIFE, ICICIPRULI, CANHLIFE) fell to
# the bottom at 0-19/100, each carrying a 12-point "thin capital buffer"
# penalty, because an insurer's balance sheet is mostly policyholder funds so
# equity/assets is structurally tiny. Meanwhile fee-based brokers and wealth
# managers (ANANDRATHI at 29.8% "ROA", GROWW at 11.2%) ranked near the top of a
# lending model that does not describe them at all.
#
# The split is made on yfinance's sub-industry, and EBIT availability confirms
# it empirically:
#
#   Banks - Regional      26 names,  0 with EBIT   -> genuine lenders
#   Credit Services       25 names, 12 with EBIT   -> NBFCs, lenders
#   Mortgage Finance       9 names,  5 with EBIT   -> HFCs, lenders
#   Asset Management      11 names,  9 with EBIT   -> fee business, generic
#   Capital Markets        8 names,  7 with EBIT   -> fee business, generic
#   Exchanges/Data         3 names,  3 with EBIT   -> fee business, generic
#   Insurance - *         13 names, 13 with EBIT   -> neither model fits
# --------------------------------------------------------------------------- #
LENDER_INDUSTRIES = {
    "Banks - Regional", "Banks - Diversified", "Banks",
    "Credit Services", "Mortgage Finance",
}

INSURER_INDUSTRIES = {
    "Insurance - Life", "Insurance - Diversified", "Insurance - Reinsurance",
    "Insurance - Property & Casualty", "Insurance Brokers",
    "Insurance - Specialty",
}

MODEL_LENDER = "lender"
MODEL_INSURER = "insurer"
MODEL_GENERIC = "generic"


def classify_financial(industry: str | None, nse_is_financial: bool = True) -> str:
    """Choose the scoring model for a company in NSE's Financial Services bucket.

    Returns MODEL_LENDER, MODEL_INSURER or MODEL_GENERIC. Anything that is not
    a lender or an insurer -- asset managers, brokers, exchanges, and the four
    fintech names NSE files here -- is an ordinary fee-earning business that
    reports EBIT and belongs in the generic engine.
    """
    if not nse_is_financial:
        return MODEL_GENERIC
    name = (industry or "").strip()
    if name in LENDER_INDUSTRIES:
        return MODEL_LENDER
    if name in INSURER_INDUSTRIES:
        return MODEL_INSURER
    if not name:
        # Unknown sub-industry inside the financial bucket. Defaulting to the
        # lender model risks the insurer failure above, so fall back to generic
        # and let the missing-data normalisation handle whatever is absent.
        return MODEL_GENERIC
    return MODEL_GENERIC


INSURER_CAVEAT = (
    "Insurer: scored on the generic model because neither the lending nor the "
    "industrial framework describes insurance economics. Embedded value, VNB "
    "margin, solvency ratio and persistency -- the measures that actually value "
    "an insurer -- are unavailable from this data source. Treat this score as a "
    "rough quality and valuation read only."
)


def model_map(info, financial_tickers) -> dict:
    """Map every ticker to its scoring model: lender, insurer or generic.

    Must be called BEFORE fundamental metrics are computed, because the lender
    flag decides whether EBIT-derived metrics are nulled. Marking all 101 NSE
    financials as lenders would strip ROCE and margins from asset managers and
    insurers that legitimately report them.
    """
    financial_tickers = set(financial_tickers or ())
    industries = {}
    if info is not None and len(info):
        industries = dict(zip(info["ticker"],
                              info.get("industry", [None] * len(info))))
    return {ticker: classify_financial(industries.get(ticker),
                                       ticker in financial_tickers)
            for ticker in industries}


def lender_tickers(info, financial_tickers) -> set:
    """Tickers that should be scored as lenders (and have EBIT metrics nulled)."""
    return {ticker for ticker, model in model_map(info, financial_tickers).items()
            if model == MODEL_LENDER}


def _derive_lender_metrics(fundamentals: dict, info: dict | None = None) -> dict:
    """Lender-specific ratios derived from the line items that do exist."""
    info = info or {}
    out = {}

    # ROA is the primary lender profitability measure precisely because it
    # cannot be inflated by gearing: two banks with identical ROA and different
    # leverage report very different ROE, and the difference is risk, not skill.
    out["roa"] = fundamentals.get("roa_latest")
    out["roe"] = fundamentals.get("roe_latest")
    out["roe_mean_5y"] = fundamentals.get("roe_mean_5y")

    # Equity/assets: the solvency buffer. A crude stand-in for capital adequacy,
    # which this data source does not carry. Indian banks typically run 6-12%;
    # NBFCs are usually higher because they fund with borrowings not deposits.
    out["equity_to_assets"] = fundamentals.get("equity_to_assets")

    # Net interest margin proxy. True NIM divides by average EARNING assets;
    # total assets is the closest available denominator, so this reads slightly
    # low and is used for ranking rather than as a reported figure.
    out["nim_proxy"] = fundamentals.get("nim_proxy")

    # The loan book is the business, so balance-sheet growth is the growth
    # measure -- revenue CAGR is a poor substitute for a lender.
    growth = fundamentals.get("asset_cagr_3y")
    if _missing(growth):
        growth = fundamentals.get("revenue_cagr_3y")
    out["asset_growth_3y"] = growth

    out["eps_cagr_3y"] = fundamentals.get("eps_cagr_3y")
    out["price_to_book"] = info.get("priceToBook")
    return out


def score_any(fundamentals: dict, valuation: dict | None = None,
              governance: dict | None = None, info: dict | None = None,
              technical: dict | None = None,
              nse_is_financial: bool = False) -> Score:
    """Route a company to the right long-horizon model and score it.

    This is the entry point callers should use, so the lender/insurer/generic
    decision lives in one place rather than being re-derived at each call site.
    """
    from src.scoring.long_term import score_long_term

    info = info or {}
    model = classify_financial(info.get("industry"), nse_is_financial)

    if model == MODEL_LENDER:
        return score_financial(fundamentals, valuation, governance, info, technical)

    # Insurers and fee-based financials are ordinary P&L businesses: they report
    # EBIT, so the generic engine applies. The lender flag is cleared first,
    # otherwise the generic engine would null the very metrics they do have.
    generic_fundamentals = dict(fundamentals)
    generic_fundamentals["is_lender"] = False
    score = score_long_term(generic_fundamentals, valuation, governance, technical)
    if model == MODEL_INSURER:
        score.warn(INSURER_CAVEAT)
    score.fact(model=model)
    return score


def score_financial(fundamentals: dict, valuation: dict | None = None,
                    governance: dict | None = None, info: dict | None = None,
                    technical: dict | None = None) -> Score:
    """Score a lender for a long-horizon hold."""
    valuation = valuation or {}
    governance = governance or {}
    info = info or {}

    ticker = fundamentals.get("ticker", "?")
    score = Score(ticker, HORIZON)
    derived = _derive_lender_metrics(fundamentals, info)
    score.fact(is_lender=True, model="financials", **derived)

    # ====================== PROFITABILITY (35) ============================== #
    # ROA leads because it is leverage-independent.
    score.add_banded(
        derived["roa"], W_PROFITABILITY * 0.55,
        [(lambda v: v >= 2.0, 1.0, "ROA above 2% -- excellent for a lender"),
         (lambda v: v >= 1.3, 0.8, "ROA 1.3-2% -- strong"),
         (lambda v: v >= 0.8, 0.5, "ROA 0.8-1.3% -- adequate"),
         (lambda v: v >= 0.3, 0.2, "ROA 0.3-0.8% -- thin"),
         (lambda v: True, 0.0, "ROA below 0.3% -- barely profitable on assets")],
        "Return on assets")

    score.add_banded(
        derived["roe_mean_5y"], W_PROFITABILITY * 0.45,
        [(lambda v: v >= 18, 1.0, "ROE above 18% (5y avg)"),
         (lambda v: v >= 14, 0.8, "ROE 14-18% (5y avg)"),
         (lambda v: v >= 10, 0.5, "ROE 10-14% (5y avg)"),
         (lambda v: v >= 5, 0.2, "ROE 5-10% (5y avg) -- weak"),
         (lambda v: True, 0.0, "ROE below 5% -- poor returns on shareholder capital")],
        "Return on equity")

    # ====================== CAPITALISATION (25) ============================= #
    score.add_banded(
        derived["equity_to_assets"], W_CAPITALISATION * 0.70,
        [(lambda v: v >= 15, 1.0, "Equity above 15% of assets -- well capitalised"),
         (lambda v: v >= 10, 0.8, "Equity 10-15% of assets -- comfortable"),
         (lambda v: v >= 7, 0.55, "Equity 7-10% of assets -- adequate"),
         (lambda v: v >= 5, 0.25, "Equity 5-7% of assets -- thin buffer"),
         (lambda v: True, 0.0, "Equity below 5% of assets -- little loss-absorbing buffer")],
        "Capitalisation")

    score.add_banded(
        derived["nim_proxy"], W_CAPITALISATION * 0.30,
        [(lambda v: v >= 4.0, 1.0, "Wide net interest margin"),
         (lambda v: v >= 2.5, 0.75, "Healthy net interest margin"),
         (lambda v: v >= 1.5, 0.4, "Modest net interest margin"),
         (lambda v: True, 0.0, "Thin net interest margin -- little spread on lending")],
        "Net interest margin")

    # ====================== GROWTH (20) ===================================== #
    # Deliberately not "more is better". Fast loan growth is how a lender buys
    # future credit losses, so the top band is moderate growth and the fastest
    # growers are capped and warned about instead.
    growth = derived["asset_growth_3y"]
    if _missing(growth):
        score.skipped.append("Asset growth")
    else:
        score.available += W_GROWTH * 0.5
        if growth > ASSET_GROWTH_CAUTION:
            score.earned += W_GROWTH * 0.5 * 0.45
            score.misses.append(
                f"Assets growing {growth:.0f}% a year -- very fast loan growth "
                "often precedes a rise in bad loans, so this is capped rather "
                "than rewarded")
            score.warn(f"Rapid balance-sheet growth ({growth:.0f}% CAGR) without "
                       "visibility on underwriting standards")
        elif growth >= 15:
            score.earned += W_GROWTH * 0.5
            score.reasons.append(f"Assets growing {growth:.0f}% a year -- strong "
                                 "but not reckless")
        elif growth >= 8:
            score.earned += W_GROWTH * 0.5 * 0.75
            score.reasons.append(f"Assets growing {growth:.0f}% a year")
        elif growth >= 0:
            score.earned += W_GROWTH * 0.5 * 0.3
            score.misses.append(f"Assets growing only {growth:.0f}% a year")
        else:
            score.misses.append("Balance sheet shrinking")

    score.add_banded(
        derived["eps_cagr_3y"], W_GROWTH * 0.5,
        [(lambda v: v >= 18, 1.0, "EPS CAGR above 18% (3y)"),
         (lambda v: v >= 10, 0.75, "EPS CAGR 10-18% (3y)"),
         (lambda v: v >= 3, 0.4, "EPS CAGR 3-10% (3y)"),
         (lambda v: v >= 0, 0.15, "EPS broadly flat"),
         (lambda v: True, 0.0, "EPS declining")],
        "Earnings growth")

    # ====================== VALUATION (20) ================================== #
    # Price-to-book is the lender multiple: earnings are provision-dependent and
    # therefore manipulable in the short run, while book value is the thing
    # being bought. P/E is deliberately not used here.
    #
    # Measurability must be judged on P/B specifically, not on the generic
    # valuation inputs. base.valuation_unmeasurable() tests pe_own_pctile,
    # pe_sector_z, fcf_yield and peg -- none of which this model consults -- so
    # deferring to it marked every lender unmeasurable and silently dropped P/B
    # from the score. A bank at 0.9x book and one at 6.0x book both scored 75.1.
    book_multiple = (valuation.get("pb") if not _missing(valuation.get("pb"))
                     else derived["price_to_book"])
    if valuation.get("currency_suppressed") or _missing(book_multiple):
        score.available += W_VALUATION
        score.earned += W_VALUATION * 0.5
        reason = ("Valuation could not be measured -- scored neutral rather "
                  "than skipped, so a data gap cannot inflate the rank")
        score.misses.append(reason)
        score.warn(reason)
    else:
        score.add_banded(
            book_multiple, W_VALUATION * 0.65,
            [(lambda v: v <= 1.0, 1.0, "Trading at or below book value"),
             (lambda v: v <= 1.8, 0.8, "P/B 1-1.8 -- reasonable for a lender"),
             (lambda v: v <= 3.0, 0.5, "P/B 1.8-3"),
             (lambda v: v <= 4.5, 0.2, "P/B 3-4.5 -- richly valued"),
             (lambda v: True, 0.0, "P/B above 4.5 -- priced for sustained excellence")],
            "Price to book")

        score.add_banded(
            valuation.get("pb_own_pctile"), W_VALUATION * 0.35,
            [(lambda v: v >= 70, 1.0, "P/B cheap versus its own history"),
             (lambda v: v >= 45, 0.65, "P/B around its own history"),
             (lambda v: v >= 20, 0.3, "P/B expensive versus its own history"),
             (lambda v: True, 0.0, "P/B near the top of its own range")],
            "P/B vs own history")

    _apply_lender_penalties(score, fundamentals, valuation, governance, derived)
    return score


def _apply_lender_penalties(score: Score, fundamentals: dict, valuation: dict,
                            governance: dict, derived: dict) -> None:
    # The caveat that matters most, attached to every lender without exception.
    score.warn("Asset quality is NOT assessed: gross/net NPA, provision "
               "coverage, capital adequacy and credit cost are unavailable from "
               "this data source. A lender can report excellent ROA until the "
               "credit cycle turns -- verify the loan book independently before "
               "acting on this score.")

    band = governance.get("pledge_band")
    pledge = governance.get("promoter_pledge_pct")
    if band == "severe":
        score.penalise(15, f"Severe promoter pledging ({pledge:.0f}% of stake)")
    elif band == "elevated":
        score.penalise(8, f"Elevated promoter pledging ({pledge:.0f}% of stake)")

    if governance.get("trend_band") == "selling_heavily":
        change = governance.get("promoter_change_4q_pp")
        score.penalise(10, f"Promoters sold down {abs(change):.1f}pp over four quarters")

    # Thin capitalisation is an existential risk for a lender rather than a
    # scoring nuance: the buffer is what absorbs loan losses.
    equity_ratio = derived.get("equity_to_assets")
    if not _missing(equity_ratio) and equity_ratio < 5:
        score.penalise(12, f"Equity is only {equity_ratio:.1f}% of assets -- "
                           "a thin buffer against credit losses")

    if valuation.get("is_loss_making"):
        score.penalise(10, "Currently loss-making")

    if valuation.get("currency_suppressed"):
        score.warn(valuation.get("valuation_caveat")
                   or "Valuation multiples suppressed (currency mismatch)")
