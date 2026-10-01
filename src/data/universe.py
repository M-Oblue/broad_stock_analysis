#!/usr/bin/env python3
"""Nifty 500 investable universe: constituents, NSE industry, and metadata.

Source of truth is NSE's published constituent file:
    https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv

which returns Company Name, Industry, Symbol, Series and ISIN. Two things make
it worth fetching rather than relying on the static seed list:

1. **Index membership drifts.** Stocks enter and leave the Nifty 500 at every
   review. A stale list silently screens names that are no longer in the index
   and misses new entrants.
2. **NSE's own `Industry` classification is better than yfinance's `sector`**
   for Indian equities -- it is the classification the index itself is built
   on, it is never missing, and it doesn't return "Unknown" (which yfinance
   does for a handful of recently-listed names).

Fallback chain, so a network failure degrades instead of killing the run:
    live NSE fetch  ->  data/cache/universe.csv  ->  data/raw/nifty500_tickers.csv

The seed file is symbols only, so the final fallback yields a usable ticker
list with empty industry metadata rather than nothing at all.

USAGE
    python -m src.data.universe                 # refresh + summary
    python -m src.data.universe --no-refresh    # use cache/seed only
    python -m src.data.universe --max-age 7     # refetch only if cache older
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.io_console import enable_utf8
from src.common.run_paths import cache_file, ensure_parent, raw_file

NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
UNIVERSE_CACHE = "universe.csv"
SEED_TICKERS = "nifty500_tickers.csv"

# NSE rejects requests without a browser-like User-Agent. A Referer is also
# needed for the www.nseindia.com API endpoints; harmless here and keeps one
# consistent session policy across every NSE call in this repo.
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# NSE macro-industries whose members are predominantly *lenders* (banks, NBFCs,
# insurers). ROCE, EBIT, D/E and EV/EBITDA are meaningless for these -- interest
# IS their revenue and leverage IS their business model -- so they route to the
# financials scoring override instead. Confirmed empirically: EBIT is simply
# absent from yfinance's income statement for exactly these names.
#
# This is the coarse NSE-level gate. It over-captures: "Financial Services" also
# holds asset managers, exchanges and fintechs that are ordinary fee-income
# businesses. The finer split needs yfinance's sub-industry and lives in
# src/scoring/financials.py, which is why this flag is named _sector_ not
# is_lender.
FINANCIAL_SECTORS = {"Financial Services"}

# NSE inserts placeholder rows named "Dummy <Company> Ltd." with a DUMMY-prefixed
# symbol during corporate restructurings (demergers in particular), to hold the
# index slot while the new entity lists. They are not tradeable and have no price
# history -- yfinance returns a bare 404. Dropping them here keeps the 404 out of
# every downstream fetch rather than handling it in each one.
PLACEHOLDER_SYMBOL_PREFIX = "DUMMY"


def _fetch_nse_constituents(timeout: int = 25) -> pd.DataFrame:
    """Download and normalise the live Nifty 500 constituent list."""
    import requests  # imported lazily so --no-refresh works without the dep

    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    response = session.get(NIFTY500_URL, timeout=timeout)
    response.raise_for_status()

    raw = pd.read_csv(io.StringIO(response.text))
    expected = {"Company Name", "Industry", "Symbol", "Series", "ISIN Code"}
    missing = expected - set(raw.columns)
    if missing:
        # Shape changed upstream -- fail loudly rather than cache a bad file.
        raise ValueError(f"NSE constituent file missing columns: {sorted(missing)}")

    return _normalise(raw)


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({
        "symbol": raw["Symbol"].astype("string").str.strip(),
        "company_name": raw["Company Name"].astype("string").str.strip(),
        "nse_industry": raw["Industry"].astype("string").str.strip(),
        "series": raw["Series"].astype("string").str.strip(),
        "isin": raw["ISIN Code"].astype("string").str.strip(),
    })
    df = df.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
    df = df[~df["symbol"].str.startswith(PLACEHOLDER_SYMBOL_PREFIX, na=False)]
    df["yf_ticker"] = df["symbol"] + ".NS"
    df["is_financial_sector"] = df["nse_industry"].isin(FINANCIAL_SECTORS)
    # Series BE is the trade-to-trade segment: no intraday netting, wider
    # spreads, thinner books. Not excluded here (that is the liquidity filter's
    # job, on real turnover) but flagged so short-horizon sizing can see it.
    df["is_trade_to_trade"] = df["series"].eq("BE")
    df["fetched_at"] = date.today().isoformat()
    return df.sort_values("symbol").reset_index(drop=True)


def _load_cache(max_age_days: int | None = None) -> pd.DataFrame | None:
    path = cache_file(UNIVERSE_CACHE)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, OSError):
        return None
    if df.empty or "symbol" not in df.columns:
        return None
    if max_age_days is not None and "fetched_at" in df.columns:
        stamp = str(df["fetched_at"].iloc[0])
        try:
            age = (date.today() - datetime.fromisoformat(stamp).date()).days
        except ValueError:
            return df  # unparseable stamp: usable, just not age-checkable
        if age > max_age_days:
            return None
    return df


def _load_seed() -> pd.DataFrame | None:
    """Last-resort fallback: the tracked symbols-only seed list.

    Yields tickers with empty industry metadata -- enough to run price-only
    (short-horizon) analysis, not enough for the sector-aware fundamental
    models, which is why callers get an explicit `metadata_complete` flag.
    """
    path = raw_file(SEED_TICKERS)
    if not path.exists():
        return None
    seed = pd.read_csv(path)
    col = "symbol" if "symbol" in seed.columns else seed.columns[0]
    symbols = (seed[col].astype("string").str.strip()
               .str.replace(r"\.NS$", "", regex=True))
    df = pd.DataFrame({"symbol": symbols.dropna().drop_duplicates()})
    df["company_name"] = pd.NA
    df["nse_industry"] = pd.NA
    df["series"] = pd.NA
    df["isin"] = pd.NA
    df["yf_ticker"] = df["symbol"] + ".NS"
    df["is_financial_sector"] = False
    df["is_trade_to_trade"] = False
    df["fetched_at"] = pd.NA
    return df.sort_values("symbol").reset_index(drop=True)


def load_universe(refresh: bool = True, max_age_days: int | None = 7,
                  verbose: bool = True) -> pd.DataFrame:
    """Return the Nifty 500 universe, refreshing from NSE when possible.

    Never raises on network failure -- falls back to cache, then to the seed
    list. The returned frame carries a `source` column so downstream code (and
    the user) can tell live data from a stale fallback.
    """
    def _say(msg):
        if verbose:
            print(msg)

    if refresh:
        try:
            df = _fetch_nse_constituents()
            df["source"] = "nse_live"
            path = ensure_parent(cache_file(UNIVERSE_CACHE))
            df.to_csv(path, index=False)
            _say(f"[OK] Nifty 500 constituents fetched from NSE: {len(df)} symbols")
            return df
        except Exception as exc:  # network, parse, or upstream shape change
            _say(f"[WARN] NSE fetch failed ({type(exc).__name__}: {exc}); "
                 "falling back to cache")

    cached = _load_cache(max_age_days=None if not refresh else max_age_days)
    if cached is not None:
        cached["source"] = "cache"
        stamp = cached["fetched_at"].iloc[0] if "fetched_at" in cached else "unknown"
        _say(f"[INFO] Using cached universe ({len(cached)} symbols, fetched {stamp})")
        return cached

    seed = _load_seed()
    if seed is not None:
        seed["source"] = "seed"
        _say(f"[WARN] Using seed ticker list ({len(seed)} symbols) -- no industry "
             "metadata, so sector-aware fundamental scoring will be degraded")
        return seed

    raise FileNotFoundError(
        "No universe available: NSE fetch failed, no cache, and no seed list at "
        f"{raw_file(SEED_TICKERS)}"
    )


def metadata_complete(df: pd.DataFrame) -> bool:
    """True when industry classification is present (i.e. not the seed fallback)."""
    return "nse_industry" in df.columns and df["nse_industry"].notna().any()


def tickers(df: pd.DataFrame) -> list[str]:
    """yfinance-form tickers (SYMBOL.NS)."""
    return df["yf_ticker"].dropna().tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-refresh", action="store_true",
                        help="skip the NSE fetch; use cache or seed only")
    parser.add_argument("--max-age", type=int, default=7, metavar="DAYS",
                        help="refetch if the cache is older than this (default 7)")
    args = parser.parse_args()

    enable_utf8()
    df = load_universe(refresh=not args.no_refresh, max_age_days=args.max_age)

    print(f"\nUniverse: {len(df)} symbols  (source: {df['source'].iloc[0]})")
    if metadata_complete(df):
        fin = int(df["is_financial_sector"].sum())
        t2t = int(df["is_trade_to_trade"].sum())
        print(f"  Financial-sector names : {fin}  -> routed to the lender scoring model")
        print(f"  Trade-to-trade (BE)    : {t2t}")
        print("\n  NSE industry breakdown:")
        for industry, count in df["nse_industry"].value_counts().items():
            print(f"    {industry:<38s} {count:>4d}")
    else:
        print("  [WARN] No industry metadata (seed fallback) -- fundamental "
              "scoring will be degraded")
    print(f"\nCached to: {cache_file(UNIVERSE_CACHE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
