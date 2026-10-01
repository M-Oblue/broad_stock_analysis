#!/usr/bin/env python3
"""Data-quality gates run between acquisition and metric computation.

Every check here exists because the underlying feed was observed producing
wrong-but-plausible values. Wrong-and-obvious is harmless -- it crashes. This
module targets the failures that look like data.

THE CURRENCY MISMATCH (primary guard)
    Some Indian companies report financial statements in USD while their shares
    are priced in INR. yfinance exposes this as `financialCurrency` != `currency`
    but does not reconcile the two, so any ratio dividing a price-derived
    numerator by a statement-derived denominator is wrong by the FX rate:

        INFY: EV 4,195,311,943,680 (INR) / EBITDA 4,481,999,872 (USD) = 936.0
        true value ~ 936 / 88 = 10.6

    Measured across the Nifty 500 this affects only INFY and HCLTECH -- and a
    random 60-ticker sample found ZERO, which is precisely why it needs a
    systematic check rather than eyeballing. Two stocks silently carrying a
    valuation metric off by ~88x would rank at the extreme of any screen that
    sorts on it.

    Crucially the damage is selective, so the fix must be too:

        statement / statement  ->  SAFE   (ROCE, margins, CAGRs, D/E, coverage)
                                           currency cancels in the ratio
        price / statement      ->  BROKEN (EV/EBITDA, P/FCF, FCF yield, P/S)

    The response is to FX-convert where a rate is available and suppress with an
    explicit flag otherwise -- never to silently emit the number.

OTHER GATES
    - Staleness: statements older than ~15 months mean yfinance is missing
      recent quarters, not that the company stopped reporting (Indian companies
      report quarterly by law).
    - Coverage: tickers with no statements at all, so a run reports true
      universe coverage rather than silently screening fewer names.
    - Sanity: negative revenue/assets, impossible margins, zero equity.
    - Internal consistency: yfinance's own ratios vs ones recomputed from
      statements, which is how the currency bug is detectable without an FX feed.

USAGE
    python -m src.data.validate              # full report
    python -m src.data.validate --strict     # non-zero exit on any error
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.io_console import enable_utf8

# Beyond this, yfinance is missing quarters rather than the company being quiet.
MAX_STATEMENT_AGE_DAYS = 460  # ~15 months

# Margins outside this band indicate a units or sign error, not a real business.
MARGIN_SANITY = (-5.0, 5.0)  # i.e. -500% .. +500%

# Ratios that divide a price-derived numerator by a statement-derived
# denominator. These are the ones a currency mismatch corrupts.
PRICE_OVER_STATEMENT_FIELDS = [
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "enterpriseToRevenue",
]


class ValidationReport:
    """Collected findings, split by severity.

    ERROR  -- the value is wrong; do not score on it.
    WARN   -- suspicious or degraded; usable with care.
    INFO   -- coverage//context, no action needed.
    """

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        # ticker -> set of field names that must not be used downstream
        self.suppressed: dict[str, set[str]] = {}

    def error(self, msg): self.errors.append(msg)
    def warn(self, msg): self.warnings.append(msg)
    def note(self, msg): self.info.append(msg)

    def suppress(self, ticker: str, fields) -> None:
        self.suppressed.setdefault(ticker, set()).update(fields)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = []
        for tag, items in (("FAIL", self.errors), ("WARN", self.warnings),
                           ("INFO", self.info)):
            for item in items:
                lines.append(f"[{tag}] {item}")
        if not lines:
            lines.append("[OK] No data-quality issues found.")
        return "\n".join(lines)


def check_currency_mismatch(info: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """Flag tickers whose statement currency differs from their price currency.

    Returns the offending rows. Suppresses every price/statement ratio for them
    so no downstream screen can rank on a number that is off by an FX rate.
    """
    if not {"currency", "financialCurrency"}.issubset(info.columns):
        report.warn("info cache lacks currency/financialCurrency -- cannot run "
                    "the currency mismatch guard")
        return pd.DataFrame()

    both = info.dropna(subset=["currency", "financialCurrency"])
    bad = both[both["currency"].astype(str) != both["financialCurrency"].astype(str)]

    if bad.empty:
        report.note(f"Currency guard: all {len(both)} tickers report statements "
                    "in their trading currency")
        return bad

    for _, row in bad.iterrows():
        ticker = row["ticker"]
        present = [f for f in PRICE_OVER_STATEMENT_FIELDS
                   if f in info.columns and pd.notna(row.get(f))]
        report.suppress(ticker, present)
        detail = ""
        if pd.notna(row.get("enterpriseToEbitda")):
            detail = f" (e.g. enterpriseToEbitda={float(row['enterpriseToEbitda']):.1f})"
        report.error(
            f"{ticker}: statements in {row['financialCurrency']} but priced in "
            f"{row['currency']} -- every price/statement ratio is wrong by the FX "
            f"rate{detail}. Suppressed: {', '.join(present) or 'none present'}. "
            "Statement-only metrics (ROCE, margins, CAGR, D/E) remain valid."
        )
    return bad


def check_statement_coverage(statements: pd.DataFrame, info: pd.DataFrame,
                             expected: list[str] | None,
                             report: ValidationReport) -> None:
    """Report tickers with no statements, so coverage is explicit."""
    have = set(statements["ticker"].unique()) if not statements.empty else set()
    universe = set(expected) if expected else set(info["ticker"].unique())
    missing = sorted(universe - have)

    coverage = 100 * len(have & universe) / max(len(universe), 1)
    report.note(f"Statement coverage: {len(have & universe)}/{len(universe)} "
                f"tickers ({coverage:.1f}%)")

    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        report.warn(f"{len(missing)} ticker(s) have no financial statements and "
                    f"cannot be scored on fundamentals: {shown}")


def check_staleness(statements: pd.DataFrame, report: ValidationReport) -> None:
    """Indian companies report quarterly; a long gap means missing data."""
    if statements.empty:
        return
    annual = statements[statements["period_type"] == "annual"]
    if annual.empty:
        return
    latest = (annual.assign(period_end=pd.to_datetime(annual["period_end"]))
                    .groupby("ticker")["period_end"].max())
    age = (pd.Timestamp(date.today()) - latest).dt.days
    stale = age[age > MAX_STATEMENT_AGE_DAYS].sort_values(ascending=False)
    if len(stale):
        shown = ", ".join(f"{t} ({d}d)" for t, d in list(stale.items())[:8])
        report.warn(f"{len(stale)} ticker(s) have stale annual statements "
                    f"(>{MAX_STATEMENT_AGE_DAYS}d): {shown}"
                    + (" ..." if len(stale) > 8 else ""))


def check_value_sanity(statements: pd.DataFrame, report: ValidationReport) -> None:
    """Catch impossible statement values (units/sign errors)."""
    if statements.empty:
        return
    wide = (statements[statements["period_type"] == "annual"]
            .pivot_table(index=["ticker", "period_end"], columns="item",
                         values="value", aggfunc="first")
            .reset_index())
    if wide.empty:
        return

    if "total_revenue" in wide:
        bad = wide[wide["total_revenue"] < 0]["ticker"].unique()
        if len(bad):
            report.error(f"{len(bad)} ticker(s) report negative revenue: "
                         f"{', '.join(sorted(bad)[:8])}")

    if "total_assets" in wide:
        bad = wide[wide["total_assets"] <= 0]["ticker"].unique()
        if len(bad):
            report.error(f"{len(bad)} ticker(s) report non-positive total assets: "
                         f"{', '.join(sorted(bad)[:8])}")

    if {"net_income", "total_revenue"}.issubset(wide.columns):
        calc = wide.dropna(subset=["net_income", "total_revenue"])
        calc = calc[calc["total_revenue"] > 0]
        margin = calc["net_income"] / calc["total_revenue"]
        outliers = calc[(margin < MARGIN_SANITY[0]) | (margin > MARGIN_SANITY[1])]
        if len(outliers):
            names = ", ".join(sorted(outliers["ticker"].unique())[:8])
            report.warn(f"{outliers['ticker'].nunique()} ticker(s) show net "
                        f"margins outside {MARGIN_SANITY} -- likely a one-off or "
                        f"a units issue: {names}")

    if "stockholders_equity" in wide:
        neg = wide[wide["stockholders_equity"] < 0]["ticker"].unique()
        if len(neg):
            report.warn(f"{len(neg)} ticker(s) have negative shareholders' equity "
                        f"-- ROE/ROCE are meaningless for these: "
                        f"{', '.join(sorted(neg)[:8])}")


def check_info_sanity(info: pd.DataFrame, report: ValidationReport) -> None:
    """Sanity-check the wide info snapshot."""
    if info.empty:
        return

    if "status" in info.columns:
        for status, n in info["status"].value_counts().items():
            if status == "ok":
                continue
            report.warn(f"{n} ticker(s) with fetch status '{status}'")

    if "sector" in info.columns:
        unknown = info[info["sector"].isna() | info["sector"].eq("Unknown")]
        if len(unknown):
            report.note(f"{len(unknown)} ticker(s) have no yfinance sector -- "
                        "NSE industry from the universe file is used instead")

    if "marketCap" in info.columns:
        bad = info[info["marketCap"].notna() & (info["marketCap"] <= 0)]
        if len(bad):
            report.error(f"{len(bad)} ticker(s) report non-positive market cap: "
                         f"{', '.join(sorted(bad['ticker'])[:8])}")

    # A negative trailing P/E is not an error (loss-making company), but it must
    # not be ranked as "cheap". Flag so the valuation layer handles the sign.
    if "trailingPE" in info.columns:
        neg = info[info["trailingPE"].notna() & (info["trailingPE"] < 0)]
        if len(neg):
            report.note(f"{len(neg)} ticker(s) have negative trailing P/E "
                        "(loss-making) -- valuation scoring must not read these "
                        "as cheap")


def check_price_coverage(prices: pd.DataFrame, min_bars: int,
                         report: ValidationReport) -> None:
    """Flag symbols with too little history for long-horizon metrics."""
    if prices is None or prices.empty:
        report.warn("No price data available to validate")
        return
    bars = prices.groupby("symbol").size()
    thin = bars[bars < min_bars].sort_values()
    report.note(f"Price history: {len(bars)} symbols, median "
                f"{int(bars.median())} bars")
    if len(thin):
        shown = ", ".join(f"{s} ({n})" for s, n in list(thin.items())[:8])
        report.warn(f"{len(thin)} symbol(s) have <{min_bars} bars -- recently "
                    f"listed, so long-horizon metrics (valuation percentiles, "
                    f"12-1 momentum) will be unavailable: {shown}"
                    + (" ..." if len(thin) > 8 else ""))


def validate_all(statements=None, info=None, prices=None,
                 expected_tickers=None, min_bars: int = 500) -> ValidationReport:
    """Run every gate and return a single report."""
    report = ValidationReport()

    if info is not None and not info.empty:
        check_currency_mismatch(info, report)
        check_info_sanity(info, report)
    if statements is not None and not statements.empty:
        check_statement_coverage(statements, info if info is not None
                                 else pd.DataFrame(), expected_tickers, report)
        check_staleness(statements, report)
        check_value_sanity(statements, report)
    if prices is not None:
        check_price_coverage(prices, min_bars, report)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any ERROR is found")
    parser.add_argument("--min-bars", type=int, default=500,
                        help="price bars required for long-horizon metrics")
    args = parser.parse_args()

    enable_utf8()

    from src.data.fetch_fundamentals import _load_info_cache, _load_statements_cache
    from src.data.fetch_prices import _load_cache as _load_price_cache
    from src.data.universe import load_universe, tickers as universe_tickers

    statements = _load_statements_cache()
    info = _load_info_cache()
    prices = _load_price_cache()

    try:
        expected = universe_tickers(load_universe(refresh=False, verbose=False))
    except Exception:
        expected = None

    report = validate_all(statements=statements, info=info, prices=prices,
                          expected_tickers=expected, min_bars=args.min_bars)

    print("=" * 78)
    print("DATA VALIDATION")
    print("=" * 78)
    print(report.render())

    if report.suppressed:
        print("\n" + "-" * 78)
        print("SUPPRESSED FIELDS (must not be scored)")
        print("-" * 78)
        for ticker, fields in sorted(report.suppressed.items()):
            print(f"  {ticker:<16s} {', '.join(sorted(fields))}")

    print(f"\nSummary: {len(report.errors)} error(s), "
          f"{len(report.warnings)} warning(s)")

    return 1 if (args.strict and not report.ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
