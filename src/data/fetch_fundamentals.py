#!/usr/bin/env python3
"""Fundamentals acquisition: financial statements + quote-level info.

Produces two caches, deliberately separate because they have different shapes
and different refresh economics:

    data/cache/statements.csv   long format, one row per ticker/statement/label/
                                period -- 5 annual years + ~8 quarters
    data/cache/info.csv         wide, one row per ticker -- the `.info` snapshot

WHY RAW STATEMENT LINES, NOT PRE-COMPUTED RATIOS
    Storing the underlying line items (revenue, EBIT, equity, debt, FCF ...)
    rather than finished ratios means adding a new metric later is a pure
    recompute over local data. Caching ROCE instead would force a full refetch
    of 500 tickers every time the metric set changes -- roughly 25 minutes of
    network for something that is arithmetic on data already held.

WHY BOTH ANNUAL AND QUARTERLY
    They serve different horizons. Annual statements drive long-term quality
    (5y ROCE consistency, margin trend, CAGRs). Quarterly income drives the
    medium-term engine's earnings momentum and acceleration -- the whole point
    of post-earnings-announcement drift is that it fires on the latest quarter,
    which an annual series cannot see.

TWO REALITIES THIS HANDLES
    1. Lenders genuinely lack rows. Banks and NBFCs have no EBIT, no Operating
       Income and no Current Assets/Liabilities -- verified on HDFCBANK. That is
       correct reporting, not a fetch failure, so a missing label is recorded as
       absent rather than retried forever.
    2. Some tickers return completely empty statements (verified on
       TATAMOTORS.NS). That IS a failure and is retried, then recorded so the
       run reports real coverage instead of silently screening fewer stocks.

CURRENCY
    `currency` and `financialCurrency` are always captured, because a mismatch
    silently corrupts every price-over-statement ratio. INFY and HCLTECH report
    in USD while priced in INR, which is why yfinance's own enterpriseToEbitda
    reads 936 and 1168 instead of ~10.6. src/data/validate.py consumes these.

USAGE
    python -m src.data.fetch_fundamentals                 # refresh stale only
    python -m src.data.fetch_fundamentals --force         # ignore TTL
    python -m src.data.fetch_fundamentals --symbols INFY.NS HDFCBANK.NS
    python -m src.data.fetch_fundamentals --ttl 14        # custom staleness
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.io_console import enable_utf8
from src.common.run_paths import cache_file, ensure_parent
from src.data.universe import load_universe, tickers as universe_tickers

STATEMENTS_CACHE = "statements.csv"
INFO_CACHE = "info.csv"

# Bump when the captured label/field set changes. Rows written under an older
# version are refetched rather than reused.
#
# Version gating is checked EXPLICITLY rather than by testing for key presence,
# because the cache is rewritten as one DataFrame: pandas unions columns across
# all records, so the moment a single ticker is refetched with a new field,
# every older row silently gains that column as NaN. Such rows would look
# complete while serving nothing.
SCHEMA_VERSION = 1

DEFAULT_TTL_DAYS = 30  # statements only change quarterly; 30d is generous

# --------------------------------------------------------------------------- #
# Curated line items. yfinance exposes 50-90 rows per statement; capturing only
# what the metrics layer consumes keeps the cache ~90k rows instead of ~500k.
#
# Each entry is (canonical_name, [candidate labels in priority order]). Labels
# vary by company and yfinance version, so aliases avoid silently losing a
# metric for a subset of the universe.
# --------------------------------------------------------------------------- #
INCOME_LABELS = [
    ("total_revenue", ["Total Revenue", "Operating Revenue"]),
    ("gross_profit", ["Gross Profit"]),
    ("operating_income", ["Operating Income", "Total Operating Income As Reported"]),
    ("ebit", ["EBIT"]),
    ("ebitda", ["EBITDA", "Normalized EBITDA"]),
    ("net_income", ["Net Income", "Net Income Common Stockholders",
                    "Net Income From Continuing Operation Net Minority Interest"]),
    ("basic_eps", ["Basic EPS"]),
    ("diluted_eps", ["Diluted EPS"]),
    ("interest_expense", ["Interest Expense", "Interest Expense Non Operating"]),
    ("pretax_income", ["Pretax Income"]),
    ("tax_provision", ["Tax Provision"]),
    ("diluted_shares", ["Diluted Average Shares"]),
    # Lender-specific: net interest income is the closest thing a bank has to
    # an operating-profit line, and is absent for non-financials.
    ("net_interest_income", ["Net Interest Income"]),
]

BALANCE_LABELS = [
    ("total_assets", ["Total Assets"]),
    ("stockholders_equity", ["Stockholders Equity", "Common Stock Equity",
                             "Total Equity Gross Minority Interest"]),
    ("total_debt", ["Total Debt"]),
    ("current_assets", ["Current Assets"]),
    ("current_liabilities", ["Current Liabilities"]),
    ("cash", ["Cash And Cash Equivalents",
              "Cash Cash Equivalents And Short Term Investments"]),
    ("invested_capital", ["Invested Capital"]),
    ("retained_earnings", ["Retained Earnings"]),
    ("shares_outstanding", ["Ordinary Shares Number", "Share Issued"]),
    ("tangible_book_value", ["Tangible Book Value"]),
]

CASHFLOW_LABELS = [
    ("operating_cash_flow", ["Operating Cash Flow",
                             "Cash Flow From Continuing Operating Activities"]),
    ("free_cash_flow", ["Free Cash Flow"]),
    ("capital_expenditure", ["Capital Expenditure"]),
    ("dividends_paid", ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
]

# Wide `.info` fields. Valuation multiples are captured but treated as
# untrusted: the metrics layer recomputes them from statements where possible,
# because yfinance's own ratios are inconsistently correct (priceToBook is fine
# for INFY while enterpriseToEbitda is off by the USD-INR rate).
INFO_FIELDS = [
    # identity / classification
    "sector", "industry", "longName", "quoteType",
    # currency -- the mismatch guard depends on these two
    "currency", "financialCurrency",
    # size
    "marketCap", "enterpriseValue", "sharesOutstanding", "floatShares",
    # valuation
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "enterpriseToRevenue", "bookValue",
    "trailingEps", "forwardEps",
    # profitability
    "returnOnEquity", "returnOnAssets", "profitMargins", "grossMargins",
    "operatingMargins", "ebitdaMargins",
    # growth
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    # balance sheet / cash
    "debtToEquity", "currentRatio", "quickRatio", "totalCash", "totalDebt",
    "freeCashflow", "operatingCashflow", "totalRevenue", "ebitda",
    # shareholder returns
    "dividendYield", "payoutRatio",
    # risk / ownership
    "beta", "heldPercentInsiders", "heldPercentInstitutions",
    # analyst context (used only as a sanity cross-check, never scored)
    "numberOfAnalystOpinions", "targetMeanPrice", "recommendationKey",
]

STATEMENT_COLUMNS = ["ticker", "statement", "period_type", "period_end",
                     "item", "value"]


def _extract(df: pd.DataFrame, labels, ticker: str, statement: str,
             period_type: str) -> list[dict]:
    """Pull curated labels out of one yfinance statement frame, long format."""
    if df is None or df.empty:
        return []
    rows = []
    for canonical, candidates in labels:
        label = next((c for c in candidates if c in df.index), None)
        if label is None:
            continue  # genuinely not reported (e.g. EBIT for a bank)
        series = df.loc[label]
        # A duplicated label yields a frame; take the first occurrence.
        if isinstance(series, pd.DataFrame):
            series = series.iloc[0]
        for period_end, value in series.items():
            if pd.isna(value):
                continue
            rows.append({
                "ticker": ticker,
                "statement": statement,
                "period_type": period_type,
                "period_end": pd.Timestamp(period_end).date().isoformat(),
                "item": canonical,
                "value": float(value),
            })
    return rows


def fetch_one(ticker: str, retries: int = 1) -> tuple[list[dict], dict, str]:
    """Fetch statements + info for one ticker.

    Returns (statement_rows, info_record, status) where status is one of
    'ok', 'no_statements', or 'failed'. The distinction matters: a bank missing
    EBIT is 'ok', a ticker returning nothing at all is not.
    """
    import yfinance as yf

    last_error = None
    for attempt in range(retries + 1):
        try:
            stock = yf.Ticker(ticker)
            rows = []
            rows += _extract(stock.income_stmt, INCOME_LABELS, ticker, "income", "annual")
            rows += _extract(stock.balance_sheet, BALANCE_LABELS, ticker, "balance", "annual")
            rows += _extract(stock.cashflow, CASHFLOW_LABELS, ticker, "cashflow", "annual")
            rows += _extract(stock.quarterly_income_stmt, INCOME_LABELS,
                             ticker, "income", "quarterly")
            rows += _extract(stock.quarterly_balance_sheet, BALANCE_LABELS,
                             ticker, "balance", "quarterly")

            try:
                info = stock.info or {}
            except Exception:
                info = {}

            record = {f: info.get(f) for f in INFO_FIELDS}
            record["ticker"] = ticker
            record["fetched_at"] = date.today().isoformat()
            record["schema_version"] = SCHEMA_VERSION

            if not rows:
                # Empty statements are a real failure mode (TATAMOTORS.NS), so
                # retry before believing it.
                if attempt < retries:
                    time.sleep(1.5)
                    continue
                record["status"] = "no_statements"
                return [], record, "no_statements"

            record["status"] = "ok"
            return rows, record, "ok"

        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    return [], {"ticker": ticker, "fetched_at": date.today().isoformat(),
                "schema_version": SCHEMA_VERSION, "status": "failed",
                "error": f"{type(last_error).__name__}: {last_error}"}, "failed"


def _load_info_cache() -> pd.DataFrame:
    path = cache_file(INFO_CACHE)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError):
        return pd.DataFrame()


def _load_statements_cache() -> pd.DataFrame:
    path = cache_file(STATEMENTS_CACHE)
    if not path.exists():
        return pd.DataFrame(columns=STATEMENT_COLUMNS)
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, OSError):
        return pd.DataFrame(columns=STATEMENT_COLUMNS)
    if not set(STATEMENT_COLUMNS).issubset(df.columns):
        return pd.DataFrame(columns=STATEMENT_COLUMNS)
    return df


def _is_fresh(row, ttl_days: int) -> bool:
    """A cached row is reusable only if it is current-schema AND within TTL."""
    if int(row.get("schema_version", -1) or -1) != SCHEMA_VERSION:
        return False
    stamp = row.get("fetched_at")
    if not isinstance(stamp, str):
        return False
    try:
        age = (date.today() - datetime.fromisoformat(stamp).date()).days
    except ValueError:
        return False
    if age > ttl_days:
        return False
    # A previous hard failure is worth retrying sooner than a good record.
    return row.get("status") == "ok"


def fetch_fundamentals(symbols: list[str], ttl_days: int = DEFAULT_TTL_DAYS,
                       force: bool = False, verbose: bool = True,
                       save_every: int = 50):
    """Fetch/refresh fundamentals, skipping fresh cache entries."""
    def _say(msg):
        if verbose:
            print(msg)

    info_cache = _load_info_cache()
    stmt_cache = _load_statements_cache()

    fresh = set()
    if not force and not info_cache.empty:
        fresh = {r["ticker"] for _, r in info_cache.iterrows()
                 if _is_fresh(r, ttl_days)}

    todo = [s for s in symbols if s not in fresh]
    _say(f"[INFO] {len(fresh)} cached & fresh, {len(todo)} to fetch "
         f"(ttl={ttl_days}d, schema v{SCHEMA_VERSION})")
    if not todo:
        return stmt_cache, info_cache

    # Drop stale rows for the tickers being refetched so old and new don't mix.
    if not stmt_cache.empty:
        stmt_cache = stmt_cache[~stmt_cache["ticker"].isin(todo)]
    if not info_cache.empty:
        info_cache = info_cache[~info_cache["ticker"].isin(todo)]

    new_rows, new_info = [], []
    counts = {"ok": 0, "no_statements": 0, "failed": 0}
    started = time.time()

    for i, ticker in enumerate(todo, 1):
        rows, record, status = fetch_one(ticker)
        counts[status] += 1
        new_rows.extend(rows)
        new_info.append(record)

        if verbose and (i % 25 == 0 or i == len(todo)):
            rate = i / max(time.time() - started, 1e-9)
            eta = (len(todo) - i) / rate if rate else 0
            _say(f"  {i}/{len(todo)}  ok={counts['ok']} "
                 f"no_stmt={counts['no_statements']} fail={counts['failed']}  "
                 f"eta {eta/60:.1f}m")

        # Periodic flush: a 25-minute run that dies at minute 24 should not
        # throw away everything it fetched.
        if i % save_every == 0:
            _save(pd.concat([stmt_cache, pd.DataFrame(new_rows)], ignore_index=True)
                  if new_rows else stmt_cache,
                  pd.concat([info_cache, pd.DataFrame(new_info)], ignore_index=True))

    statements = (pd.concat([stmt_cache, pd.DataFrame(new_rows)], ignore_index=True)
                  if new_rows else stmt_cache)
    info = (pd.concat([info_cache, pd.DataFrame(new_info)], ignore_index=True)
            if new_info else info_cache)

    _save(statements, info)
    _say(f"\n[OK] fetched {len(todo)} in {(time.time()-started)/60:.1f}m -- "
         f"ok={counts['ok']} no_statements={counts['no_statements']} "
         f"failed={counts['failed']}")
    return statements, info


def _save(statements: pd.DataFrame, info: pd.DataFrame) -> None:
    if statements is not None and not statements.empty:
        path = ensure_parent(cache_file(STATEMENTS_CACHE))
        (statements.drop_duplicates(
            subset=["ticker", "statement", "period_type", "period_end", "item"],
            keep="last")
         .sort_values(["ticker", "statement", "period_type", "period_end"])
         .to_csv(path, index=False))
    if info is not None and not info.empty:
        path = ensure_parent(cache_file(INFO_CACHE))
        (info.drop_duplicates(subset=["ticker"], keep="last")
             .sort_values("ticker").to_csv(path, index=False))


def load_statements() -> pd.DataFrame:
    df = _load_statements_cache()
    if df.empty:
        raise FileNotFoundError(
            f"No statements cache at {cache_file(STATEMENTS_CACHE)}. "
            "Run: python -m src.data.fetch_fundamentals")
    return df


def load_info() -> pd.DataFrame:
    df = _load_info_cache()
    if df.empty:
        raise FileNotFoundError(
            f"No info cache at {cache_file(INFO_CACHE)}. "
            "Run: python -m src.data.fetch_fundamentals")
    return df


def wide_statements(statements: pd.DataFrame, period_type: str = "annual",
                    statement: str | None = None) -> pd.DataFrame:
    """Pivot long statement rows to ticker x period_end x item."""
    df = statements[statements["period_type"] == period_type]
    if statement:
        df = df[df["statement"] == statement]
    if df.empty:
        return pd.DataFrame()
    return (df.pivot_table(index=["ticker", "period_end"], columns="item",
                           values="value", aggfunc="first")
              .reset_index())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", metavar="TICKER")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL_DAYS,
                        help=f"cache staleness in days (default {DEFAULT_TTL_DAYS})")
    parser.add_argument("--force", action="store_true", help="ignore the TTL")
    parser.add_argument("--limit", type=int, help="only the first N symbols")
    args = parser.parse_args()

    enable_utf8()

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = universe_tickers(load_universe(refresh=False, verbose=True))
    if args.limit:
        symbols = symbols[:args.limit]

    statements, info = fetch_fundamentals(symbols, ttl_days=args.ttl, force=args.force)

    print(f"\nStatements : {len(statements):,} rows, "
          f"{statements['ticker'].nunique()} tickers")
    print(f"Info       : {len(info):,} rows")
    if "status" in info.columns:
        for status, n in info["status"].value_counts().items():
            print(f"  status {status:<14s} {n}")
    if {"currency", "financialCurrency"}.issubset(info.columns):
        both = info.dropna(subset=["currency", "financialCurrency"])
        mismatch = both[both["currency"] != both["financialCurrency"]]
        if len(mismatch):
            print(f"\n[WARN] {len(mismatch)} ticker(s) report statements in a "
                  "different currency than their price -- price/statement "
                  "ratios for these are invalid until converted:")
            for _, r in mismatch.iterrows():
                print(f"    {r['ticker']:<16s} price={r['currency']} "
                      f"statements={r['financialCurrency']}")
    print(f"\nCached to: {cache_file(STATEMENTS_CACHE)}")
    print(f"           {cache_file(INFO_CACHE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
