#!/usr/bin/env python3
"""Bulk OHLCV price history for the universe, with incremental refresh.

Writes long-format daily bars to data/cache/prices.csv:
    symbol,date,open,high,low,close,volume

WHY 5 YEARS BY DEFAULT
    The long-horizon engine needs valuation percentiles measured against a
    stock's own multi-year history, and the medium-horizon engine needs 12-1
    momentum (a 12-month lookback, skipping the most recent month). Anything
    less than ~3y makes those metrics either impossible or statistically
    meaningless, so the default period is deliberately generous.

THE SPLIT-ADJUSTMENT TRAP (the reason this isn't a naive append)
    Prices are fetched with auto_adjust=True, so history is adjusted for splits
    and dividends. That adjustment is applied *retroactively*: when a stock
    splits 1:5, every historical bar before the split is divided by 5 at the
    source. A naive incremental fetch ("I have data to Friday, get me Monday
    onward") would append correctly-adjusted new bars onto stale, unadjusted
    old bars, silently producing a 5x price cliff in the middle of the series.
    Every momentum, moving-average and volatility figure computed across that
    cliff would be garbage -- and nothing would error.

    So each incremental run re-fetches a small OVERLAP window and compares it
    against what is already cached. If the overlapping closes disagree by more
    than a tolerance, that symbol's history is re-downloaded in full. This
    turns a silent data-corruption bug into an automatic self-heal.

USAGE
    python -m src.data.fetch_prices                    # incremental refresh
    python -m src.data.fetch_prices --full             # force full re-download
    python -m src.data.fetch_prices --period 10y       # longer history
    python -m src.data.fetch_prices --symbols INFY.NS RELIANCE.NS
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.io_console import enable_utf8
from src.common.run_paths import cache_file, ensure_parent
from src.data.universe import load_universe, tickers as universe_tickers

PRICES_CACHE = "prices.csv"
COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]

# yfinance batches fine at this size; larger batches raise the cost of one
# failed request, smaller ones multiply round-trips.
BATCH_SIZE = 60

# Trading days of overlap re-fetched on an incremental run, used for the
# adjustment check described above. Needs to be long enough to survive a long
# weekend or exchange holiday and still contain real bars.
OVERLAP_DAYS = 10

# Relative tolerance when comparing overlapping closes. Floating-point noise and
# minor vendor restatements are ~1e-6; a split or bonus issue moves prices by
# tens of percent, so 1% cleanly separates "same data" from "re-adjusted".
ADJUSTMENT_TOLERANCE = 0.01


def _flatten(raw: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    """Turn yfinance's wide (ticker, field) frame into long format."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    frames = []
    # A single-ticker download returns flat columns; multi-ticker returns a
    # 2-level MultiIndex grouped by ticker. Normalise both to the same path.
    if raw.columns.nlevels == 1:
        present = {batch[0]: raw} if len(batch) == 1 else {}
    else:
        present = {t: raw[t] for t in batch if t in raw.columns.get_level_values(0)}

    for ticker, sub in present.items():
        sub = sub.dropna(how="all")
        if sub.empty:
            continue
        frame = pd.DataFrame({
            "symbol": ticker,
            "date": pd.to_datetime(sub.index).tz_localize(None).normalize(),
            "open": sub.get("Open"),
            "high": sub.get("High"),
            "low": sub.get("Low"),
            "close": sub.get("Close"),
            "volume": sub.get("Volume"),
        })
        frames.append(frame.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["close"])[COLUMNS]


def _download(batch: list[str], period: str | None = None,
              start=None, retries: int = 2) -> pd.DataFrame:
    """Download one batch, retrying transient failures."""
    import yfinance as yf

    for attempt in range(retries + 1):
        try:
            kwargs = dict(auto_adjust=True, progress=False, group_by="ticker",
                          threads=True)
            if start is not None:
                raw = yf.download(batch, start=start, **kwargs)
            else:
                raw = yf.download(batch, period=period, **kwargs)
            return _flatten(raw, batch)
        except Exception as exc:
            if attempt == retries:
                print(f"  [FAIL] batch of {len(batch)} failed: "
                      f"{type(exc).__name__}: {exc}")
                return pd.DataFrame(columns=COLUMNS)
            time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame(columns=COLUMNS)


def _load_cache() -> pd.DataFrame:
    path = cache_file(PRICES_CACHE)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path, parse_dates=["date"])
    except (pd.errors.ParserError, OSError, ValueError):
        return pd.DataFrame(columns=COLUMNS)
    if df.empty or not set(COLUMNS).issubset(df.columns):
        return pd.DataFrame(columns=COLUMNS)
    return df[COLUMNS]


def _detect_readjusted(cached: pd.DataFrame, fresh: pd.DataFrame) -> list[str]:
    """Symbols whose overlapping closes changed -- i.e. a corporate action
    retroactively re-adjusted their history, so the cache is now stale."""
    if cached.empty or fresh.empty:
        return []
    merged = cached.merge(fresh, on=["symbol", "date"], suffixes=("_old", "_new"))
    if merged.empty:
        return []
    old, new = merged["close_old"].astype(float), merged["close_new"].astype(float)
    denom = old.abs().replace(0, np.nan)
    merged["drift"] = ((new - old).abs() / denom)
    worst = merged.groupby("symbol")["drift"].max()
    return sorted(worst[worst > ADJUSTMENT_TOLERANCE].index.tolist())


def _batched(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_prices(symbols: list[str], period: str = "5y", full: bool = False,
                 verbose: bool = True) -> pd.DataFrame:
    """Fetch or incrementally refresh OHLCV history for `symbols`."""
    def _say(msg):
        if verbose:
            print(msg)

    cached = pd.DataFrame(columns=COLUMNS) if full else _load_cache()
    known = set(cached["symbol"].unique()) if not cached.empty else set()
    new_symbols = [s for s in symbols if s not in known]
    existing = [s for s in symbols if s in known]

    collected = []

    # --- symbols with no history: full-period download -------------------- #
    if new_symbols:
        _say(f"[INFO] Full download for {len(new_symbols)} symbol(s), period={period}")
        for i, batch in enumerate(_batched(new_symbols, BATCH_SIZE), 1):
            got = _download(batch, period=period)
            collected.append(got)
            _say(f"  batch {i}: {len(batch)} symbols -> {len(got):,} rows")

    # --- symbols already cached: overlap refresh + adjustment check -------- #
    if existing:
        last = cached[cached["symbol"].isin(existing)].groupby("symbol")["date"].max()
        start = (last.min() - timedelta(days=OVERLAP_DAYS)).date()
        _say(f"[INFO] Incremental refresh for {len(existing)} symbol(s) from {start}")

        fresh_parts = []
        for i, batch in enumerate(_batched(existing, BATCH_SIZE), 1):
            got = _download(batch, start=start)
            fresh_parts.append(got)
            _say(f"  batch {i}: {len(batch)} symbols -> {len(got):,} rows")
        fresh = (pd.concat(fresh_parts, ignore_index=True)
                 if fresh_parts else pd.DataFrame(columns=COLUMNS))

        readjusted = _detect_readjusted(cached, fresh)
        if readjusted:
            _say(f"[WARN] {len(readjusted)} symbol(s) were re-adjusted upstream "
                 f"(split/bonus); re-downloading their full history: "
                 f"{', '.join(readjusted[:8])}"
                 + (" ..." if len(readjusted) > 8 else ""))
            # Drop the stale history entirely, then refetch the full period.
            cached = cached[~cached["symbol"].isin(readjusted)]
            fresh = fresh[~fresh["symbol"].isin(readjusted)]
            for batch in _batched(readjusted, BATCH_SIZE):
                collected.append(_download(batch, period=period))

        collected.append(fresh)

    frames = [f for f in ([cached] + collected) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    # Newest row wins for a given symbol/date: a refreshed bar supersedes the
    # cached one (late-corrected volume, provisional close, etc).
    out = (out.drop_duplicates(subset=["symbol", "date"], keep="last")
              .sort_values(["symbol", "date"])
              .reset_index(drop=True))
    return out[COLUMNS]


def save_prices(df: pd.DataFrame) -> Path:
    path = ensure_parent(cache_file(PRICES_CACHE))
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return path


def load_prices() -> pd.DataFrame:
    """Read the cached price history (long format, `date` parsed)."""
    df = _load_cache()
    if df.empty:
        raise FileNotFoundError(
            f"No price cache at {cache_file(PRICES_CACHE)}. "
            "Run: python -m src.data.fetch_prices"
        )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", default="5y",
                        help="history length for full downloads (default 5y)")
    parser.add_argument("--full", action="store_true",
                        help="ignore the cache and re-download everything")
    parser.add_argument("--symbols", nargs="+", metavar="TICKER",
                        help="specific yfinance tickers instead of the universe")
    parser.add_argument("--no-refresh-universe", action="store_true",
                        help="use the cached universe rather than refetching it")
    args = parser.parse_args()

    enable_utf8()

    if args.symbols:
        symbols = args.symbols
    else:
        uni = load_universe(refresh=not args.no_refresh_universe, verbose=True)
        symbols = universe_tickers(uni)

    started = time.time()
    df = fetch_prices(symbols, period=args.period, full=args.full)

    if df.empty:
        print("[FAIL] No price data retrieved.")
        return 1

    path = save_prices(df)
    per_symbol = df.groupby("symbol")["date"].agg(["min", "max", "count"])
    missing = sorted(set(symbols) - set(df["symbol"].unique()))

    print(f"\n[OK] {len(df):,} rows for {df['symbol'].nunique()} symbols "
          f"in {time.time() - started:.0f}s")
    print(f"     Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"     Bars/symbol: median {int(per_symbol['count'].median())}, "
          f"min {int(per_symbol['count'].min())}, max {int(per_symbol['count'].max())}")
    if missing:
        print(f"     [WARN] {len(missing)} symbol(s) returned no data: "
              f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
    print(f"     Saved to   : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
