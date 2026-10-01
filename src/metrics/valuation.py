#!/usr/bin/env python3
"""Valuation metrics, measured against a stock's own history and its sector.

ABSOLUTE MULTIPLES ARE NOT COMPARABLE, SO NOTHING HERE RANKS ON THEM
    A P/E of 45 is expensive for a steel mill and cheap for a consumer-staples
    compounder. Screening the Nifty 500 on raw P/E therefore does not find
    cheap companies -- it finds cheap *sectors*, and ranks the index roughly by
    industry every single run. Worse, at the long horizon it systematically
    buys structurally-low-multiple businesses (commodities, PSUs, capital-
    intensive cyclicals) and calls them value.

    So every multiple here is additionally expressed two ways:
      own-history percentile  -- where today's multiple sits within the same
                                 stock's own 5-year range. This is the one that
                                 answers "is this stock cheap *for itself*",
                                 which is what re-rating potential depends on.
      sector z-score          -- versus the median of its NSE industry, which
                                 answers "is this cheap versus its peers".

    The own-history measure needs a price series (multiples are recomputed
    back through time using trailing fundamentals), so it degrades to NaN for
    the 42 recently-listed names with under ~500 bars rather than being
    computed off a stub of history.

THIS IS THE MODULE THE CURRENCY MISMATCH ACTUALLY BREAKS
    Every ratio here divides a price-derived numerator (market cap, enterprise
    value, share price) by a statement-derived denominator (earnings, book
    value, revenue, free cash flow). When a company reports statements in a
    different currency from its listing -- INFY and HCLTECH report in USD while
    priced in INR -- the ratio is wrong by the FX rate. yfinance's own
    enterpriseToEbitda reads 936 for INFY against a true value near 10.6.

    Two stocks silently carrying a valuation metric inflated ~88x would sit at
    the extreme of any screen sorted on it, so `suppressed` from
    src/data/validate.py is applied here: affected ratios become NaN and the
    reason is recorded in `valuation_caveat`. Nothing is silently emitted.

NEGATIVE EARNINGS ARE NOT CHEAP
    A loss-making company has a negative P/E. Sorted ascending, it ranks as the
    cheapest stock in the index. Negative multiples are therefore mapped to NaN
    with an explicit `is_loss_making` flag rather than being allowed to win a
    value screen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Trading days of price history required before an own-history percentile is
# reported. Below this the percentile describes a stub of history and would
# read as a confident statement about a range that was never observed.
MIN_BARS_FOR_HISTORY = 400

# Minimum peers in an industry before a sector-relative z-score means anything.
MIN_PEERS_FOR_SECTOR = 4

# Multiples for which "lower is cheaper" holds. Used to orient percentiles so
# that a high score consistently means good value across every metric.
LOWER_IS_CHEAPER = {"pe", "pb", "ps", "ev_ebitda", "ev_sales", "p_fcf"}


def _positive(value):
    """Return value if it is a usable positive number, else NaN.

    Valuation multiples built on a negative denominator (loss, negative book
    value) are not merely imprecise -- they invert, and a naive ascending sort
    promotes them to the top of a value screen.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(v) or v <= 0:
        return np.nan
    return v


def _ratio(numerator, denominator):
    n, d = _positive(numerator), _positive(denominator)
    if np.isnan(n) or np.isnan(d):
        return np.nan
    return n / d


def _ttm(statements: pd.DataFrame, ticker: str, item: str,
         quarters: int = 4) -> float:
    """Trailing-twelve-month sum of a quarterly income item.

    Preferred over the last annual figure because an annual number can be up to
    fifteen months stale, which at the medium horizon is most of the holding
    period. Falls back to the latest annual value when too few quarters exist.
    """
    rows = statements[(statements["ticker"] == ticker)
                      & (statements["period_type"] == "quarterly")
                      & (statements["item"] == item)]
    if not rows.empty:
        series = (rows.assign(period_end=pd.to_datetime(rows["period_end"]))
                      .sort_values("period_end")["value"].dropna())
        if len(series) >= quarters:
            return float(series.iloc[-quarters:].sum())

    annual = statements[(statements["ticker"] == ticker)
                        & (statements["period_type"] == "annual")
                        & (statements["item"] == item)]
    if annual.empty:
        return np.nan
    series = (annual.assign(period_end=pd.to_datetime(annual["period_end"]))
                    .sort_values("period_end")["value"].dropna())
    return float(series.iloc[-1]) if len(series) else np.nan


def _latest_annual(statements: pd.DataFrame, ticker: str, item: str) -> float:
    rows = statements[(statements["ticker"] == ticker)
                      & (statements["period_type"] == "annual")
                      & (statements["item"] == item)]
    if rows.empty:
        return np.nan
    series = (rows.assign(period_end=pd.to_datetime(rows["period_end"]))
                  .sort_values("period_end")["value"].dropna())
    return float(series.iloc[-1]) if len(series) else np.nan


def compute_for_ticker(ticker: str, statements: pd.DataFrame,
                       info_row: pd.Series | dict,
                       suppressed: set[str] | None = None) -> dict:
    """Current valuation multiples for one ticker, computed from statements.

    Multiples are recomputed here rather than read from `.info` because
    yfinance's own ratios are inconsistently correct -- for INFY it reports a
    sane priceToBook (4.62) alongside an enterpriseToEbitda of 936. Deriving
    them from market cap and statement lines gives one consistent basis, and
    makes the currency suppression meaningful.
    """
    suppressed = suppressed or set()
    info = dict(info_row) if not isinstance(info_row, dict) else info_row

    out: dict = {"ticker": ticker, "valuation_caveat": None,
                 "is_loss_making": False, "currency_suppressed": False}

    market_cap = _positive(info.get("marketCap"))
    enterprise_value = _positive(info.get("enterpriseValue"))

    net_income = _ttm(statements, ticker, "net_income")
    revenue = _ttm(statements, ticker, "total_revenue")
    ebitda = _ttm(statements, ticker, "ebitda")
    equity = _latest_annual(statements, ticker, "stockholders_equity")
    fcf = _latest_annual(statements, ticker, "free_cash_flow")

    out["is_loss_making"] = bool(not np.isnan(net_income) and net_income <= 0)

    out["pe"] = _ratio(market_cap, net_income)
    out["pb"] = _ratio(market_cap, equity)
    out["ps"] = _ratio(market_cap, revenue)
    out["ev_ebitda"] = _ratio(enterprise_value, ebitda)
    out["ev_sales"] = _ratio(enterprise_value, revenue)
    out["p_fcf"] = _ratio(market_cap, fcf)

    # Yields are the reciprocal view; kept explicitly because "higher is
    # better" reads more naturally in a score than "lower is better".
    out["earnings_yield"] = (100.0 / out["pe"]) if not np.isnan(out["pe"]) else np.nan
    out["fcf_yield"] = (100.0 / out["p_fcf"]) if not np.isnan(out["p_fcf"]) else np.nan

    dividend_yield = info.get("dividendYield")
    out["dividend_yield"] = (float(dividend_yield)
                             if dividend_yield is not None
                             and not pd.isna(dividend_yield) else np.nan)
    payout = info.get("payoutRatio")
    out["payout_ratio"] = (float(payout) * 100
                           if payout is not None and not pd.isna(payout) else np.nan)

    # PEG: valuation relative to growth. Only meaningful with positive growth --
    # a negative denominator flips the sign and makes a shrinking company look
    # like the cheapest growth stock in the index.
    growth = info.get("earningsGrowth")
    try:
        growth_pct = float(growth) * 100 if growth is not None else np.nan
    except (TypeError, ValueError):
        growth_pct = np.nan
    out["earnings_growth_pct"] = growth_pct
    out["peg"] = (out["pe"] / growth_pct
                  if not np.isnan(out["pe"]) and growth_pct is not None
                  and not np.isnan(growth_pct) and growth_pct > 0 else np.nan)

    # ---- currency suppression ---------------------------------------------- #
    # Applied last so it overrides everything computed above.
    #
    # NaN is truthy in Python, so a plain `if currency and financial_currency`
    # accepts a MISSING financialCurrency and then compares "INR" against the
    # string "nan", declaring a mismatch that does not exist. JSWDULUX.NS has no
    # financialCurrency and was suppressed on exactly this bug. Absent data must
    # mean "cannot determine", not "different".
    currency = info.get("currency")
    financial_currency = info.get("financialCurrency")
    have_both = not pd.isna(currency) and not pd.isna(financial_currency)
    mismatch = have_both and str(currency) != str(financial_currency)

    if mismatch or suppressed:
        out["currency_suppressed"] = True
        for field in ("pe", "pb", "ps", "ev_ebitda", "ev_sales", "p_fcf",
                      "earnings_yield", "fcf_yield", "peg"):
            out[field] = np.nan
        out["valuation_caveat"] = (
            f"statements in {financial_currency} but priced in {currency}; "
            "price/statement ratios suppressed (statement-only metrics such as "
            "ROCE, margins and CAGR remain valid)")
    elif not have_both:
        # Reported, not suppressed: NSE-listed companies report in INR in all
        # but a couple of cases, so withholding every multiple here would lose
        # far more information than the residual risk justifies.
        out["valuation_caveat"] = ("reporting currency unconfirmed; multiples "
                                   "assume statements and price share a currency")

    if out["is_loss_making"] and not out["currency_suppressed"]:
        out["valuation_caveat"] = ("loss-making: earnings multiples are NaN "
                                   "rather than negative, so it cannot rank as cheap")

    return out


def historical_multiple_series(prices: pd.DataFrame, ticker: str,
                               statements: pd.DataFrame,
                               metric: str = "pe") -> pd.Series:
    """Approximate a stock's own multiple through time.

    Method: hold the fundamental denominator fixed at its most recent trailing
    value and vary price. This deliberately measures *price* re-rating rather
    than reconstructing a true point-in-time multiple -- yfinance carries no
    point-in-time fundamentals, so an honest approximation is the most that is
    available. It answers "is the market paying more or less for this business
    than it usually does", which is the question the percentile is used for.

    The limitation is real and worth stating: for a company whose earnings
    changed sharply over the window, the early part of the series reflects
    today's earnings against yesterday's price.
    """
    symbol_prices = prices[prices["symbol"] == ticker].sort_values("date")
    if symbol_prices.empty or len(symbol_prices) < MIN_BARS_FOR_HISTORY:
        return pd.Series(dtype=float)

    denominators = {
        "pe": _ttm(statements, ticker, "net_income"),
        "pb": _latest_annual(statements, ticker, "stockholders_equity"),
        "ps": _ttm(statements, ticker, "total_revenue"),
    }
    denominator = _positive(denominators.get(metric))
    shares = _positive(_latest_annual(statements, ticker, "shares_outstanding"))
    if np.isnan(denominator) or np.isnan(shares):
        return pd.Series(dtype=float)

    per_share = denominator / shares
    if per_share <= 0:
        return pd.Series(dtype=float)

    series = pd.Series(symbol_prices["close"].to_numpy() / per_share,
                       index=pd.to_datetime(symbol_prices["date"]))
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def own_history_percentile(current: float, history: pd.Series,
                           lower_is_cheaper: bool = True) -> float:
    """Where `current` sits in its own history, oriented so higher = cheaper.

    Returns 0-100. For a multiple where lower is cheaper, a result of 90 means
    the stock is cheaper than 90% of its own history.
    """
    if history is None or len(history) < MIN_BARS_FOR_HISTORY or np.isnan(current):
        return np.nan
    rank = float((history < current).sum()) / len(history) * 100.0
    return 100.0 - rank if lower_is_cheaper else rank


def add_sector_relative(valuations: pd.DataFrame,
                        sector_map: pd.Series | dict,
                        metrics=("pe", "pb", "ev_ebitda", "ps")) -> pd.DataFrame:
    """Add per-industry z-scores, oriented so higher = cheaper.

    Uses median and median-absolute-deviation rather than mean/stdev: valuation
    distributions within an industry are right-skewed and a single 200x P/E
    would drag the mean far enough to make genuinely expensive peers look
    average.
    """
    out = valuations.copy()
    if isinstance(sector_map, dict):
        sector_map = pd.Series(sector_map)
    out["sector"] = out["ticker"].map(sector_map)

    for metric in metrics:
        column = f"{metric}_sector_z"
        out[column] = np.nan
        if metric not in out.columns:
            continue
        for sector, group in out.groupby("sector"):
            values = group[metric].dropna()
            if len(values) < MIN_PEERS_FOR_SECTOR:
                continue
            median = values.median()
            mad = (values - median).abs().median()
            if mad == 0 or np.isnan(mad):
                continue
            # 1.4826 scales MAD to be comparable with a standard deviation
            # under normality, keeping the z-score interpretable.
            z = (group[metric] - median) / (1.4826 * mad)
            out.loc[group.index, column] = -z if metric in LOWER_IS_CHEAPER else z

    return out


def compute_all(statements: pd.DataFrame, info: pd.DataFrame,
                prices: pd.DataFrame | None = None,
                sector_map=None, suppressed: dict | None = None,
                verbose: bool = False) -> pd.DataFrame:
    """Valuation metrics for every ticker in `info`."""
    suppressed = suppressed or {}
    info_indexed = info.set_index("ticker", drop=False)
    tickers = sorted(info_indexed.index.unique())

    records = []
    for i, ticker in enumerate(tickers, 1):
        row = info_indexed.loc[ticker]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        record = compute_for_ticker(ticker, statements, row,
                                    suppressed=suppressed.get(ticker))

        if prices is not None and not record["currency_suppressed"]:
            for metric in ("pe", "pb", "ps"):
                history = historical_multiple_series(prices, ticker,
                                                     statements, metric)
                record[f"{metric}_own_pctile"] = own_history_percentile(
                    record.get(metric, np.nan), history,
                    lower_is_cheaper=metric in LOWER_IS_CHEAPER)
        records.append(record)

        if verbose and i % 100 == 0:
            print(f"  {i}/{len(tickers)}")

    frame = pd.DataFrame(records)
    if sector_map is not None:
        frame = add_sector_relative(frame, sector_map)
    return frame
