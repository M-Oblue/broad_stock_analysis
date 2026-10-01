#!/usr/bin/env python3
"""Fundamental metrics derived from raw statement line items.

Everything here is computed from statements only -- statement divided by
statement. That is deliberate and it is what makes these metrics immune to the
currency mismatch that corrupts price-based ratios: if revenue, EBIT and equity
are all reported in USD, their ratios are identical to the INR ones. So ROCE,
margins, CAGRs, coverage and D/E stay valid for INFY and HCLTECH even while
their EV/EBITDA is unusable. Valuation (price over statement) is handled
separately in valuation.py, where the mismatch does matter.

WHAT THESE MEASURE, AND WHY THESE
    Quality      ROCE and its 5-year consistency. A single good year is noise;
                 what distinguishes a compounder is earning a high return on
                 capital *repeatedly*. Hence roce_mean AND roce_std -- the
                 standard deviation is as informative as the level.
    Earnings     CFO/Net Income. Profit is an opinion, cash is a fact. A company
    quality      reporting rising profit while operating cash flow lags is the
                 classic accruals warning, and it is invisible to P/E.
    Growth       Revenue and EPS CAGR over 3y and 5y, plus quarterly YoY and
                 growth ACCELERATION -- the medium-horizon engine needs the
                 second derivative, since post-earnings drift responds to
                 change in the growth rate, not its level.
    Solvency     Interest coverage, net debt/EBITDA, D/E, and the debt TREND.
                 Deleveraging is one of the most reliable re-rating catalysts.
    Dilution     Share count trend. Growth funded by issuing stock is not the
                 same as growth funded by cash flow, and per-share value is
                 what an investor actually owns.

LENDERS ARE HANDLED BY OMISSION, NOT BY GUESSWORK
    Banks and NBFCs have no EBIT, no Operating Income and no current
    assets/liabilities. Rather than substitute a proxy and emit a confident
    wrong number, every EBIT-derived metric returns NaN for them and
    `is_lender` is set so the scoring layer routes to the financials model.
    ROE, ROA, growth, accruals and dilution remain meaningful and are computed
    normally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Minimum annual periods required before a trend/consistency figure is
# meaningful. Two points define a line but say nothing about stability.
MIN_YEARS_FOR_TREND = 3

# A quarter's YoY comparison needs the same quarter one year earlier, i.e. 5
# quarterly observations to also compute acceleration.
QUARTERS_PER_YEAR = 4

# How far a candidate quarter may sit from exactly one year before the
# reference and still count as "the same quarter last year". Indian quarter-ends
# land on fixed dates (Mar/Jun/Sep/Dec 31) so the true match is within a few
# days; 45 days accepts reporting-date drift while still rejecting an adjacent
# quarter, which is 91 days away.
YEAR_AGO_TOLERANCE_DAYS = 45


def _safe_div(numerator, denominator, *, positive_denominator=True):
    """Divide, returning NaN rather than inf/garbage on a bad denominator.

    `positive_denominator` guards ratios that are meaningless when the
    denominator is negative -- a margin computed on negative revenue, or ROE on
    negative equity, produces a *positive-looking* number that would rank well
    in a screen. Silent sign flips are the dangerous case.
    """
    if numerator is None or denominator is None:
        return np.nan
    try:
        n, d = float(numerator), float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(n) or np.isnan(d) or d == 0:
        return np.nan
    if positive_denominator and d < 0:
        return np.nan
    return n / d


def _cagr(first, last, years):
    """Compound annual growth rate.

    Undefined when the starting value is non-positive: a company going from a
    loss to a profit has no meaningful growth *rate*, and forcing one produces
    either a complex number or a nonsense percentage. Returns NaN so the
    scoring layer treats it as missing rather than as spectacular growth.
    """
    if first is None or last is None or years <= 0:
        return np.nan
    try:
        f, l = float(first), float(last)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(f) or np.isnan(l) or f <= 0:
        return np.nan
    if l <= 0:
        return -100.0  # collapsed to a loss: a real, reportable outcome
    return ((l / f) ** (1.0 / years) - 1.0) * 100.0


def _slope_per_year(series: pd.Series):
    """Least-squares slope of a yearly series, in units per year.

    Used for margin and debt trends. A regression over all available years is
    more robust than first-vs-last, which a single outlier year can invert.
    """
    clean = series.dropna()
    if len(clean) < MIN_YEARS_FOR_TREND:
        return np.nan
    x = np.arange(len(clean), dtype=float)
    y = clean.to_numpy(dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _annual_frame(statements: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Wide annual statement frame for one ticker, oldest period first."""
    rows = statements[(statements["ticker"] == ticker)
                      & (statements["period_type"] == "annual")]
    if rows.empty:
        return pd.DataFrame()
    wide = rows.pivot_table(index="period_end", columns="item", values="value",
                            aggfunc="first")
    return wide.sort_index()


def _quarterly_frame(statements: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rows = statements[(statements["ticker"] == ticker)
                      & (statements["period_type"] == "quarterly")
                      & (statements["statement"] == "income")]
    if rows.empty:
        return pd.DataFrame()
    wide = rows.pivot_table(index="period_end", columns="item", values="value",
                            aggfunc="first")
    return wide.sort_index()


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    """Column as a float Series, or an all-NaN Series when absent.

    Missing columns are normal, not exceptional: lenders have no `ebit` at all.
    Returning an aligned NaN series lets every downstream computation proceed
    uniformly and produce NaN, instead of each call site testing for presence.
    """
    if frame.empty or name not in frame.columns:
        return pd.Series(dtype=float, index=frame.index if not frame.empty else None)
    return pd.to_numeric(frame[name], errors="coerce")


def _last(series: pd.Series):
    clean = series.dropna() if series is not None else pd.Series(dtype=float)
    return float(clean.iloc[-1]) if len(clean) else np.nan


def compute_roce_series(annual: pd.DataFrame) -> pd.Series:
    """ROCE per year = EBIT / (total debt + shareholders' equity).

    Capital employed is debt + equity rather than assets - current liabilities:
    both are standard, but debt+equity is the convention Indian screeners use
    and it is available for far more of this universe (current liabilities are
    absent for every lender and some others).
    """
    ebit = _col(annual, "ebit")
    debt = _col(annual, "total_debt")
    equity = _col(annual, "stockholders_equity")
    # A lender has no EBIT column at all, so `_col` hands back an all-NaN
    # series rather than an empty one. Return empty explicitly so callers can
    # distinguish "no capital-efficiency data exists" from "computed to NaN".
    if ebit.dropna().empty or equity.dropna().empty:
        return pd.Series(dtype=float)
    capital = debt.fillna(0) + equity
    roce = pd.Series(
        [_safe_div(e, c) for e, c in zip(ebit, capital)], index=annual.index)
    return roce * 100.0


def compute_for_ticker(statements: pd.DataFrame, ticker: str,
                       is_lender: bool = False) -> dict:
    """All fundamental metrics for one ticker."""
    annual = _annual_frame(statements, ticker)
    quarterly = _quarterly_frame(statements, ticker)

    out: dict = {"ticker": ticker, "is_lender": bool(is_lender),
                 "annual_periods": int(len(annual)),
                 "quarterly_periods": int(len(quarterly))}

    if annual.empty:
        out["has_fundamentals"] = False
        return out
    out["has_fundamentals"] = True

    revenue = _col(annual, "total_revenue")
    ebit = _col(annual, "ebit")
    ebitda = _col(annual, "ebitda")
    net_income = _col(annual, "net_income")
    gross_profit = _col(annual, "gross_profit")
    equity = _col(annual, "stockholders_equity")
    assets = _col(annual, "total_assets")
    debt = _col(annual, "total_debt")
    cash = _col(annual, "cash")
    interest = _col(annual, "interest_expense")
    cfo = _col(annual, "operating_cash_flow")
    fcf = _col(annual, "free_cash_flow")
    shares = _col(annual, "shares_outstanding")
    eps = _col(annual, "diluted_eps")
    if eps.dropna().empty:
        eps = _col(annual, "basic_eps")
    current_assets = _col(annual, "current_assets")
    current_liabilities = _col(annual, "current_liabilities")

    # ---- profitability ---------------------------------------------------- #
    roce = compute_roce_series(annual)
    roce_clean = roce.dropna()
    out["roce_latest"] = _last(roce)
    out["roce_mean_5y"] = float(roce_clean.mean()) if len(roce_clean) else np.nan
    # Consistency is the point: a compounder earns high returns repeatedly.
    out["roce_std_5y"] = (float(roce_clean.std(ddof=0))
                          if len(roce_clean) >= MIN_YEARS_FOR_TREND else np.nan)
    out["roce_min_5y"] = float(roce_clean.min()) if len(roce_clean) else np.nan
    out["roce_trend"] = _slope_per_year(roce)

    out["roe_latest"] = _safe_div(_last(net_income), _last(equity)) * 100 \
        if not np.isnan(_safe_div(_last(net_income), _last(equity))) else np.nan
    roe_series = pd.Series([_safe_div(n, e) for n, e in zip(net_income, equity)],
                           index=annual.index) * 100
    out["roe_mean_5y"] = float(roe_series.dropna().mean()) if roe_series.notna().any() else np.nan
    out["roa_latest"] = _safe_div(_last(net_income), _last(assets)) * 100 \
        if not np.isnan(_safe_div(_last(net_income), _last(assets))) else np.nan

    # ---- margins and their direction -------------------------------------- #
    net_margin = pd.Series([_safe_div(n, r) for n, r in zip(net_income, revenue)],
                           index=annual.index) * 100
    op_margin = pd.Series([_safe_div(e, r) for e, r in zip(ebit, revenue)],
                          index=annual.index) * 100
    gross_margin = pd.Series([_safe_div(g, r) for g, r in zip(gross_profit, revenue)],
                             index=annual.index) * 100
    ebitda_margin = pd.Series([_safe_div(e, r) for e, r in zip(ebitda, revenue)],
                              index=annual.index) * 100

    out["net_margin"] = _last(net_margin)
    out["operating_margin"] = _last(op_margin)
    out["gross_margin"] = _last(gross_margin)
    out["ebitda_margin"] = _last(ebitda_margin)
    out["net_margin_trend"] = _slope_per_year(net_margin)
    out["operating_margin_trend"] = _slope_per_year(op_margin)
    out["gross_margin_trend"] = _slope_per_year(gross_margin)

    # ---- earnings quality -------------------------------------------------- #
    # CFO/NI below 1 sustained means reported profit is not arriving as cash.
    out["cfo_to_net_income"] = _safe_div(_last(cfo), _last(net_income))
    cfo_ni = pd.Series([_safe_div(c, n) for c, n in zip(cfo, net_income)],
                       index=annual.index)
    out["cfo_to_net_income_mean"] = (float(cfo_ni.dropna().mean())
                                     if cfo_ni.notna().any() else np.nan)
    out["fcf_margin"] = _safe_div(_last(fcf), _last(revenue)) * 100 \
        if not np.isnan(_safe_div(_last(fcf), _last(revenue))) else np.nan
    out["fcf_to_net_income"] = _safe_div(_last(fcf), _last(net_income))
    out["positive_fcf_years"] = int((fcf.dropna() > 0).sum())
    out["profitable_years"] = int((net_income.dropna() > 0).sum())
    out["years_of_data"] = int(len(annual))

    # ---- growth ------------------------------------------------------------ #
    for label, series in (("revenue", revenue), ("eps", eps),
                          ("net_income", net_income)):
        clean = series.dropna()
        for horizon in (3, 5):
            key = f"{label}_cagr_{horizon}y"
            if len(clean) > horizon:
                out[key] = _cagr(clean.iloc[-(horizon + 1)], clean.iloc[-1], horizon)
            elif len(clean) >= 2:
                # Use the full span available rather than reporting nothing,
                # but only for the shorter horizon so a 5y figure is never
                # silently a 2y figure wearing a 5y label.
                span = len(clean) - 1
                out[key] = (_cagr(clean.iloc[0], clean.iloc[-1], span)
                            if horizon == 3 else np.nan)
            else:
                out[key] = np.nan

    # ---- solvency ---------------------------------------------------------- #
    out["debt_to_equity"] = _safe_div(_last(debt), _last(equity))
    net_debt = _last(debt) - (_last(cash) if not np.isnan(_last(cash)) else 0.0)
    out["net_debt"] = net_debt
    out["net_debt_to_ebitda"] = _safe_div(net_debt, _last(ebitda))
    # Interest expense is reported positive; coverage is meaningless at zero
    # interest (debt-free), which is a good thing, so it is reported as NaN and
    # the scoring layer credits zero-debt separately.
    out["interest_coverage"] = _safe_div(_last(ebit), abs(_last(interest))) \
        if _last(interest) not in (0,) else np.nan
    out["current_ratio"] = _safe_div(_last(current_assets), _last(current_liabilities))

    de_series = pd.Series([_safe_div(d, e) for d, e in zip(debt, equity)],
                          index=annual.index)
    # Negative slope = deleveraging = a classic re-rating catalyst.
    out["debt_to_equity_trend"] = _slope_per_year(de_series)

    # ---- dilution ---------------------------------------------------------- #
    shares_clean = shares.dropna()
    if len(shares_clean) >= 2:
        years = len(shares_clean) - 1
        out["share_count_cagr"] = _cagr(shares_clean.iloc[0],
                                        shares_clean.iloc[-1], years)
    else:
        out["share_count_cagr"] = np.nan

    # ---- quarterly earnings momentum (medium horizon) ---------------------- #
    out.update(_quarterly_momentum(quarterly))

    # ---- lender handling ---------------------------------------------------- #
    out["net_interest_income"] = _last(_col(annual, "net_interest_income"))
    # Raw latest balances, kept so the lender model can derive its own ratios
    # (equity/assets as a capital-buffer proxy, NII/assets as a margin proxy)
    # without re-reading the statement frame.
    out["total_assets_latest"] = _last(assets)
    out["equity_latest"] = _last(equity)
    out["equity_to_assets"] = (
        _safe_div(_last(equity), _last(assets)) * 100
        if not np.isnan(_safe_div(_last(equity), _last(assets))) else np.nan)
    out["nim_proxy"] = (
        _safe_div(out["net_interest_income"], _last(assets)) * 100
        if not np.isnan(_safe_div(out["net_interest_income"], _last(assets)))
        else np.nan)
    # Asset growth is the lender's equivalent of revenue growth: the balance
    # sheet IS the business, so loan-book expansion is the growth measure.
    assets_clean = assets.dropna()
    out["asset_cagr_3y"] = (_cagr(assets_clean.iloc[-4], assets_clean.iloc[-1], 3)
                            if len(assets_clean) > 3 else np.nan)

    if is_lender or np.isnan(out["roce_latest"]):
        # EBIT-derived figures are absent or meaningless for lenders. Null them
        # explicitly rather than leaving a partially-populated row that looks
        # comparable to a non-financial.
        for key in ("roce_latest", "roce_mean_5y", "roce_std_5y", "roce_min_5y",
                    "roce_trend", "operating_margin", "operating_margin_trend",
                    "interest_coverage", "net_debt_to_ebitda", "ebitda_margin"):
            if is_lender:
                out[key] = np.nan
        out["is_lender"] = bool(is_lender)

    return out


def _quarterly_momentum(quarterly: pd.DataFrame) -> dict:
    """Latest-quarter YoY growth and its acceleration.

    Acceleration is the medium-horizon signal that matters: post-earnings drift
    responds to a CHANGE in the growth rate, so a company going from +10% to
    +25% YoY is a different proposition from one steady at +25%.

    YEAR-AGO MATCHING IS BY DATE, NOT BY POSITION
        yfinance's quarterly series has holes -- RELIANCE is missing 2025-09-30
        entirely, and INFY reports revenue for only 5 of its 7 quarters. Taking
        `iloc[-5]` as "the same quarter last year" therefore silently compares
        Jun-2026 against Mar-2025, five quarters apart, and reports the result
        as a YoY growth rate. It produced a plausible-looking 18.4% for
        RELIANCE that was simply measuring the wrong interval.

        So the comparison quarter is located by DATE: the observation closest to
        exactly one year before the reference, accepted only within a tolerance.
        If the true year-ago quarter is missing, the answer is NaN -- a missing
        metric is recoverable, a wrong one is not.
    """
    out = {
        "q_revenue_yoy": np.nan, "q_eps_yoy": np.nan,
        "q_revenue_yoy_prev": np.nan, "q_eps_yoy_prev": np.nan,
        "q_revenue_acceleration": np.nan, "q_eps_acceleration": np.nan,
        "latest_quarter": None,
    }
    if quarterly.empty:
        return out

    out["latest_quarter"] = str(quarterly.index[-1])
    revenue = _col(quarterly, "total_revenue").dropna()
    eps = _col(quarterly, "diluted_eps").dropna()
    if eps.empty:
        eps = _col(quarterly, "basic_eps").dropna()

    def yoy(series, offset=0):
        """Growth vs the same quarter a year earlier, `offset` quarters back."""
        if len(series) < 2 + offset:
            return np.nan
        dated = series.copy()
        dated.index = pd.to_datetime(dated.index, errors="coerce")
        dated = dated[dated.index.notna()].sort_index()
        if len(dated) < 2 + offset:
            return np.nan

        position = len(dated) - 1 - offset
        if position < 0:
            return np.nan
        reference_date = dated.index[position]
        current = dated.iloc[position]

        target = reference_date - pd.DateOffset(years=1)
        candidates = dated.iloc[:position]
        if candidates.empty:
            return np.nan
        offsets = np.abs((candidates.index - target).days.to_numpy())
        best_position = int(np.argmin(offsets))
        if offsets[best_position] > YEAR_AGO_TOLERANCE_DAYS:
            return np.nan  # the true year-ago quarter simply isn't present

        year_ago = candidates.iloc[best_position]
        if pd.isna(year_ago) or pd.isna(current) or year_ago <= 0:
            return np.nan  # growth off a zero/negative base is uninterpretable
        return (float(current) / float(year_ago) - 1.0) * 100.0

    out["q_revenue_yoy"] = yoy(revenue)
    out["q_revenue_yoy_prev"] = yoy(revenue, offset=1)
    out["q_eps_yoy"] = yoy(eps)
    out["q_eps_yoy_prev"] = yoy(eps, offset=1)

    for metric in ("revenue", "eps"):
        now, prev = out[f"q_{metric}_yoy"], out[f"q_{metric}_yoy_prev"]
        if not (np.isnan(now) or np.isnan(prev)):
            out[f"q_{metric}_acceleration"] = now - prev

    return out


def compute_all(statements: pd.DataFrame,
                lender_symbols: set[str] | None = None,
                verbose: bool = False) -> pd.DataFrame:
    """Fundamental metrics for every ticker present in `statements`."""
    lender_symbols = lender_symbols or set()
    tickers = sorted(statements["ticker"].dropna().unique())
    records = []
    for i, ticker in enumerate(tickers, 1):
        records.append(compute_for_ticker(
            statements, ticker, is_lender=ticker in lender_symbols))
        if verbose and i % 100 == 0:
            print(f"  {i}/{len(tickers)}")
    return pd.DataFrame(records)
