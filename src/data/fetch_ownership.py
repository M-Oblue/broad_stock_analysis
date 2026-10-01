#!/usr/bin/env python3
"""Promoter ownership and share pledging, scraped from NSE's public JSON API.

This is the one governance dataset yfinance does not carry, and it matters most
at the long horizon: a promoter quietly pledging their stake or selling down is
among the earliest observable warnings of trouble, and it is invisible to any
price- or statement-based metric.

Two endpoints, both public, no login, no cookie priming required. Send a browser
User-Agent and a Referer -- note the NSE *homepage* returns 403 while these
`corporate-*` endpoints return 200, so priming a session against the homepage
(the usual advice) actively fails here.

    /api/corporate-share-holdings-master   20-25 quarters of the official
                                           shareholding pattern
    /api/corporate-pledgedata              current encumbrance disclosure

WHICH FIELD MEANS WHAT -- two traps, both verified live
---------------------------------------------------------------------------
1. PLEDGING: use `percPromoterShares`, NOT `numSharesPledged`.
   `numSharesPledged` counts ALL encumbered shares including non-promoter and
   NBFC pledges, so it reads alarmingly high for companies whose promoters have
   pledged nothing:

       ITC        numSharesPledged   254,334,331   promoter encumbrance 0.00%
       HDFCBANK   numSharesPledged   374,653,401   promoter encumbrance 0.00%
       YESBANK    numSharesPledged 4,281,424,941   promoter encumbrance 0.00%

   Scoring on it would red-flag exactly the highest-quality names in the index.
   `percPromoterShares` gives the real signal (INFY 0.00, ADANIENT 0.79,
   ZEEL 5.38 -- which matches the public record).

2. HOLDING %: use the shareholding-pattern endpoint, NOT the pledge endpoint's
   `percPromoterHolding`. The two disagree materially and the SHP filing is the
   authoritative one:

       ticker      pledge API    SHP (official)
       HDFCBANK        13.32          0.00     <- SHP is right
       INFY            20.76         13.82     <- SHP is right
       TCS             71.77         71.77     <- agree

   HDFC Bank genuinely has NO promoter following the HDFC Ltd merger; it is
   professionally managed. Taking the pledge endpoint's figure would invent a
   13% promoter stake that does not exist.

ZERO PROMOTER HOLDING IS NOT A RED FLAG
    Widely-held, professionally-managed companies (HDFCBANK, ITC, and most
    large private banks) have no identified promoter. That is a governance
    *structure*, not a governance *failure* -- often a positive. Such tickers
    are marked `has_promoter=False` so scoring skips promoter-based checks
    rather than penalising them for a null.

USAGE
    python -m src.data.fetch_ownership                  # whole universe (~4 min)
    python -m src.data.fetch_ownership --symbols INFY HDFCBANK
    python -m src.data.fetch_ownership --force          # ignore TTL
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.io_console import enable_utf8
from src.common.run_paths import cache_file, ensure_parent
from src.data.universe import load_universe

OWNERSHIP_CACHE = "ownership.csv"
HOLDING_HISTORY_CACHE = "promoter_holding_history.csv"

SCHEMA_VERSION = 1
DEFAULT_TTL_DAYS = 30  # shareholding patterns are filed quarterly

PLEDGE_URL = "https://www.nseindia.com/api/corporate-pledgedata?index=equities&symbol={sym}"
SHP_URL = ("https://www.nseindia.com/api/corporate-share-holdings-master"
           "?index=equities&symbol={sym}")

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Politeness delay between symbols. Measured at 0.4s: 12 symbols in 5.2s with
# zero failures, i.e. ~4 minutes for the full Nifty 500.
REQUEST_DELAY = 0.4

# Promoter stake below this is treated as "no identified promoter" rather than
# "promoters sold everything". ITC sits at 0.02% purely from legacy holdings.
NO_PROMOTER_THRESHOLD = 0.5

# Disagreement between the two endpoints worth reporting (percentage points).
HOLDING_DISAGREEMENT_PP = 1.0


def _to_float(value):
    """NSE pads numbers with spaces and uses '-' for missing."""
    if value is None:
        return np.nan
    text = str(value).strip()
    if text in ("", "-", "NA", "None"):
        return np.nan
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return np.nan


def _session():
    import requests
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    return s


def _get_json(session, url, timeout=20):
    response = session.get(url, timeout=timeout)
    if response.status_code != 200 or not response.text.strip():
        return None
    try:
        return response.json()
    except ValueError:
        return None


def parse_holding_history(payload, symbol: str) -> pd.DataFrame:
    """Quarterly promoter/public shareholding from the SHP endpoint."""
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    rows = []
    for record in payload:
        promoter = _to_float(record.get("pr_and_prgrp"))
        if np.isnan(promoter):
            continue
        stamp = pd.to_datetime(record.get("date"), format="%d-%b-%Y", errors="coerce")
        if pd.isna(stamp):
            continue
        rows.append({
            "symbol": symbol,
            "quarter_end": stamp.date().isoformat(),
            "promoter_pct": promoter,
            "public_pct": _to_float(record.get("public_val")),
            "employee_trusts_pct": _to_float(record.get("employeeTrusts")),
        })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .drop_duplicates(subset=["symbol", "quarter_end"], keep="last")
            .sort_values("quarter_end")
            .reset_index(drop=True))


def parse_pledge(payload, symbol: str) -> dict:
    """Current promoter encumbrance from the pledge endpoint."""
    empty = {
        "symbol": symbol,
        "promoter_pledge_pct": np.nan,
        "pledge_as_of": None,
        "total_encumbered_shares": np.nan,
        "pledge_api_promoter_pct": np.nan,
    }
    if not isinstance(payload, dict):
        return empty
    data = payload.get("data") or []
    if not data:
        # No disclosure filed. For most companies this means nothing is
        # pledged; it is recorded as 0 rather than NaN only when the SHP shows
        # a real promoter, which the caller resolves.
        return empty
    record = data[0]
    return {
        "symbol": symbol,
        # THE correct pledging field -- see module docstring.
        "promoter_pledge_pct": _to_float(record.get("percPromoterShares")),
        "pledge_as_of": record.get("shp") or record.get("broadcastDt"),
        # Total encumbrance incl. non-promoter/NBFC. Retained for context only;
        # never scored, because it false-flags ITC/HDFCBANK/YESBANK.
        "total_encumbered_shares": _to_float(record.get("numSharesPledged")),
        # Kept purely to cross-check against the authoritative SHP figure.
        "pledge_api_promoter_pct": _to_float(record.get("percPromoterHolding")),
    }


def _summarise(history: pd.DataFrame, pledge: dict, symbol: str) -> dict:
    """Collapse history + pledge into one scoreable row."""
    row = {
        "symbol": symbol,
        "promoter_holding_pct": np.nan,
        "promoter_holding_as_of": None,
        "promoter_change_1q_pp": np.nan,
        "promoter_change_4q_pp": np.nan,
        "promoter_change_max_pp": np.nan,
        "quarters_of_history": 0,
        "has_promoter": False,
        "promoter_pledge_pct": pledge.get("promoter_pledge_pct", np.nan),
        "pledge_as_of": pledge.get("pledge_as_of"),
        "total_encumbered_shares": pledge.get("total_encumbered_shares", np.nan),
        "holding_source_disagreement_pp": np.nan,
        "fetched_at": date.today().isoformat(),
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
    }

    if history is None or history.empty:
        row["status"] = "no_holding_history"
        return row

    series = history.sort_values("quarter_end")
    latest = float(series["promoter_pct"].iloc[-1])
    row["promoter_holding_pct"] = latest
    row["promoter_holding_as_of"] = series["quarter_end"].iloc[-1]
    row["quarters_of_history"] = int(len(series))
    row["has_promoter"] = bool(latest >= NO_PROMOTER_THRESHOLD)

    # Trend: falling promoter stake is the signal, so deltas are signed
    # (negative = promoters selling down).
    if len(series) >= 2:
        row["promoter_change_1q_pp"] = latest - float(series["promoter_pct"].iloc[-2])
    if len(series) >= 5:
        row["promoter_change_4q_pp"] = latest - float(series["promoter_pct"].iloc[-5])
    row["promoter_change_max_pp"] = latest - float(series["promoter_pct"].iloc[0])

    # A company with no promoter cannot have promoter pledging; NSE sometimes
    # still returns a stale non-zero. Null it rather than score a phantom.
    if not row["has_promoter"]:
        row["promoter_pledge_pct"] = np.nan
    elif np.isnan(row["promoter_pledge_pct"]):
        # Promoter exists, no encumbrance disclosure filed => nothing pledged.
        row["promoter_pledge_pct"] = 0.0

    api_pct = pledge.get("pledge_api_promoter_pct", np.nan)
    if not np.isnan(api_pct):
        row["holding_source_disagreement_pp"] = abs(api_pct - latest)

    return row


def fetch_one(session, symbol: str) -> tuple[dict, pd.DataFrame]:
    try:
        shp = _get_json(session, SHP_URL.format(sym=symbol))
        history = parse_holding_history(shp, symbol)
    except Exception:
        history = pd.DataFrame()
    try:
        pledge = parse_pledge(_get_json(session, PLEDGE_URL.format(sym=symbol)), symbol)
    except Exception:
        pledge = parse_pledge(None, symbol)
    return _summarise(history, pledge, symbol), history


def _load_cache(name, key="symbol") -> pd.DataFrame:
    path = cache_file(name)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, OSError):
        return pd.DataFrame()
    return df if key in df.columns else pd.DataFrame()


def _is_fresh(row, ttl_days: int) -> bool:
    if int(row.get("schema_version", -1) or -1) != SCHEMA_VERSION:
        return False
    stamp = row.get("fetched_at")
    if not isinstance(stamp, str):
        return False
    try:
        age = (date.today() - datetime.fromisoformat(stamp).date()).days
    except ValueError:
        return False
    return age <= ttl_days and row.get("status") == "ok"


def fetch_ownership(symbols: list[str], ttl_days: int = DEFAULT_TTL_DAYS,
                    force: bool = False, verbose: bool = True,
                    delay: float = REQUEST_DELAY):
    """Fetch promoter holding + pledging for `symbols` (bare NSE symbols)."""
    def _say(msg):
        if verbose:
            print(msg)

    own_cache = _load_cache(OWNERSHIP_CACHE)
    hist_cache = _load_cache(HOLDING_HISTORY_CACHE)

    fresh = set()
    if not force and not own_cache.empty:
        fresh = {r["symbol"] for _, r in own_cache.iterrows() if _is_fresh(r, ttl_days)}

    todo = [s for s in symbols if s not in fresh]
    _say(f"[INFO] {len(fresh)} cached & fresh, {len(todo)} to fetch "
         f"(~{len(todo) * delay * 2 / 60:.1f} min)")
    if not todo:
        return own_cache, hist_cache

    if not own_cache.empty:
        own_cache = own_cache[~own_cache["symbol"].isin(todo)]
    if not hist_cache.empty:
        hist_cache = hist_cache[~hist_cache["symbol"].isin(todo)]

    session = _session()
    rows, histories = [], []
    counts = {"ok": 0, "no_holding_history": 0}
    started = time.time()

    for i, symbol in enumerate(todo, 1):
        row, history = fetch_one(session, symbol)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        rows.append(row)
        if not history.empty:
            histories.append(history)

        if verbose and (i % 50 == 0 or i == len(todo)):
            rate = i / max(time.time() - started, 1e-9)
            _say(f"  {i}/{len(todo)}  ok={counts.get('ok',0)} "
                 f"no_history={counts.get('no_holding_history',0)}  "
                 f"eta {(len(todo)-i)/max(rate,1e-9)/60:.1f}m")
        time.sleep(delay)

    ownership = pd.concat([own_cache, pd.DataFrame(rows)], ignore_index=True)
    history_all = pd.concat([hist_cache] + histories, ignore_index=True) \
        if histories else hist_cache

    _save(ownership, history_all)
    _say(f"\n[OK] fetched {len(todo)} in {(time.time()-started)/60:.1f}m -- "
         + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return ownership, history_all


def _save(ownership: pd.DataFrame, history: pd.DataFrame) -> None:
    if ownership is not None and not ownership.empty:
        path = ensure_parent(cache_file(OWNERSHIP_CACHE))
        (ownership.drop_duplicates(subset=["symbol"], keep="last")
                  .sort_values("symbol").to_csv(path, index=False))
    if history is not None and not history.empty:
        path = ensure_parent(cache_file(HOLDING_HISTORY_CACHE))
        (history.drop_duplicates(subset=["symbol", "quarter_end"], keep="last")
                .sort_values(["symbol", "quarter_end"]).to_csv(path, index=False))


def load_ownership() -> pd.DataFrame:
    df = _load_cache(OWNERSHIP_CACHE)
    if df.empty:
        raise FileNotFoundError(
            f"No ownership cache at {cache_file(OWNERSHIP_CACHE)}. "
            "Run: python -m src.data.fetch_ownership")
    return df


def load_holding_history() -> pd.DataFrame:
    return _load_cache(HOLDING_HISTORY_CACHE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", metavar="SYMBOL",
                        help="bare NSE symbols (INFY, not INFY.NS)")
    parser.add_argument("--ttl", type=int, default=DEFAULT_TTL_DAYS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    args = parser.parse_args()

    enable_utf8()

    if args.symbols:
        symbols = [s.replace(".NS", "").upper() for s in args.symbols]
    else:
        symbols = load_universe(refresh=False, verbose=True)["symbol"].tolist()
    if args.limit:
        symbols = symbols[:args.limit]

    ownership, history = fetch_ownership(symbols, ttl_days=args.ttl,
                                         force=args.force, delay=args.delay)

    scope = ownership[ownership["symbol"].isin(symbols)]
    have = scope[scope["promoter_holding_pct"].notna()]
    promoted = have[have["has_promoter"]]

    print(f"\nOwnership rows     : {len(scope)}")
    print(f"With holding data  : {len(have)}")
    print(f"  has promoter     : {len(promoted)}")
    print(f"  no promoter      : {len(have) - len(promoted)}  "
          "(widely held / professionally managed -- not a red flag)")
    print(f"Holding history    : {len(history):,} quarterly rows")

    if len(promoted):
        pledged = promoted[promoted["promoter_pledge_pct"] > 0]
        print(f"\nPromoter pledging  : {len(pledged)} with any encumbrance")
        top = pledged.nlargest(10, "promoter_pledge_pct")
        for _, r in top.iterrows():
            print(f"    {r['symbol']:<14s} {r['promoter_pledge_pct']:6.2f}% pledged "
                  f"| holding {r['promoter_holding_pct']:5.2f}%")

        selling = promoted[promoted["promoter_change_4q_pp"] < -1.0]
        if len(selling):
            print(f"\nPromoter sell-down (>1pp over 4 quarters): {len(selling)}")
            for _, r in selling.nsmallest(10, "promoter_change_4q_pp").iterrows():
                print(f"    {r['symbol']:<14s} {r['promoter_change_4q_pp']:+6.2f}pp "
                      f"-> {r['promoter_holding_pct']:5.2f}%")

    disagree = scope[scope["holding_source_disagreement_pp"] > HOLDING_DISAGREEMENT_PP]
    if len(disagree):
        print(f"\n[INFO] {len(disagree)} ticker(s) where NSE's pledge endpoint "
              "disagrees with the official shareholding pattern by >1pp. "
              "The SHP figure is used (it is the authoritative filing).")

    print(f"\nCached to: {cache_file(OWNERSHIP_CACHE)}")
    print(f"           {cache_file(HOLDING_HISTORY_CACHE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
