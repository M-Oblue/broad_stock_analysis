#!/usr/bin/env python3
"""analyze.py -- multi-horizon stock analysis for the Nifty 500.

    python analyze.py --horizon long      # 2-5 year holds
    python analyze.py --horizon medium    # 1-2 year holds
    python analyze.py --horizon short     # weeks to months
    python analyze.py --horizon all

    python analyze.py --horizon long --top 30 --detail 10
    python analyze.py --horizon short --refresh      # refetch data first
    python analyze.py --horizon long --ticker INFY.NS

The pipeline is data -> metrics -> scoring -> report. Acquisition is separate
and cached, so repeat runs are fast and reproducible: scoring 500 stocks reads
local CSVs and takes seconds, while a full refresh takes roughly half an hour
and is only needed when the underlying filings change.

Run --refresh (or the individual fetchers) before the first run:
    python -m src.data.fetch_prices
    python -m src.data.fetch_fundamentals
    python -m src.data.fetch_ownership

Educational tool. Not investment advice.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common.io_console import enable_utf8
from src.common.run_paths import run_output
from src.data.fetch_fundamentals import _load_info_cache, _load_statements_cache
from src.data.fetch_ownership import OWNERSHIP_CACHE
from src.data.fetch_ownership import _load_cache as _load_ownership_cache
from src.data.fetch_prices import _load_cache as _load_price_cache
from src.data.universe import load_universe, tickers as universe_tickers
from src.data.validate import validate_all
from src.metrics.fundamental import compute_all as compute_fundamentals
from src.metrics.governance import compute_all as compute_governance
from src.metrics.technical import compute_all as compute_technical
from src.metrics.valuation import compute_all as compute_valuation
from src.report.console import render_report, to_csv
from src.scoring.base import rank_scores
from src.scoring.financials import lender_tickers, model_map, score_any
from src.scoring.medium_term import score_medium_term
from src.scoring.short_term import score_short_term

HORIZONS = ("long", "medium", "short")


def _row(frame: pd.DataFrame, key: str) -> dict:
    """One row as a dict, or empty when the key is absent."""
    if frame is None or frame.empty or key not in frame.index:
        return {}
    row = frame.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row.to_dict()


def load_everything(refresh: bool = False, verbose: bool = True) -> dict:
    """Load caches and compute every metric family once.

    Metrics are computed once and shared across horizons rather than per
    horizon: the three engines read the same underlying measurements and differ
    only in which they consult and how they weight them.
    """
    def say(message):
        if verbose:
            print(message)

    if refresh:
        say("[INFO] Refreshing data (this takes ~30 minutes)...")
        from src.data import fetch_fundamentals, fetch_ownership, fetch_prices
        universe = load_universe(refresh=True, verbose=verbose)
        fetch_prices.save_prices(
            fetch_prices.fetch_prices(universe_tickers(universe), verbose=verbose))
        fetch_fundamentals.fetch_fundamentals(universe_tickers(universe),
                                              verbose=verbose)
        fetch_ownership.fetch_ownership(universe["symbol"].tolist(), verbose=verbose)

    universe = load_universe(refresh=False, verbose=False)
    statements = _load_statements_cache()
    info = _load_info_cache()
    prices = _load_price_cache()
    ownership = _load_ownership_cache(OWNERSHIP_CACHE)

    missing = [name for name, frame in (("prices", prices),
                                        ("fundamentals", statements),
                                        ("info", info)) if frame is None or frame.empty]
    if missing:
        raise SystemExit(
            f"[FAIL] Missing cached data: {', '.join(missing)}.\n"
            "       Run: python analyze.py --refresh   (or the individual "
            "fetchers in src/data/)")

    say(f"[INFO] Universe {len(universe)} | prices {len(prices):,} rows | "
        f"statements {len(statements):,} rows | ownership {len(ownership)} rows")

    # Validation runs before scoring so suppressed fields never reach a score.
    report = validate_all(statements=statements, info=info, prices=prices,
                          expected_tickers=universe_tickers(universe))
    if report.errors:
        say(f"[WARN] {len(report.errors)} data-quality error(s); affected "
            "fields are suppressed rather than scored. "
            "Run `python -m src.data.validate` for detail.")

    financial_tickers = set(universe[universe["is_financial_sector"]]["yf_ticker"])
    models = model_map(info, financial_tickers)
    lenders = lender_tickers(info, financial_tickers)
    sector_map = dict(zip(universe["yf_ticker"], universe["nse_industry"]))

    say(f"[INFO] Scoring models: "
        f"{pd.Series(list(models.values())).value_counts().to_dict()}")
    say("[INFO] Computing metrics...")

    fundamentals = compute_fundamentals(statements, lender_symbols=lenders)
    valuation = compute_valuation(statements, info, prices=prices,
                                  sector_map=sector_map,
                                  suppressed=report.suppressed)
    technical = compute_technical(prices)
    governance = compute_governance(ownership)

    return {
        "universe": universe, "statements": statements, "info": info,
        "prices": prices, "ownership": ownership,
        "fundamentals": fundamentals.set_index("ticker"),
        "valuation": valuation.set_index("ticker"),
        "technical": technical.set_index("symbol"),
        "governance": (governance.set_index("symbol")
                       if not governance.empty else pd.DataFrame()),
        "info_indexed": info.set_index("ticker"),
        "models": models, "lenders": lenders, "sector_map": sector_map,
        "validation": report, "financial_tickers": financial_tickers,
    }


def score_universe(data: dict, horizon: str, only: list[str] | None = None) -> pd.DataFrame:
    """Score every ticker for one horizon."""
    fundamentals = data["fundamentals"]
    tickers = only or list(fundamentals.index)

    rows = []
    for ticker in tickers:
        if ticker not in fundamentals.index:
            continue
        symbol = ticker.replace(".NS", "")
        fund = _row(fundamentals, ticker) | {"ticker": ticker}
        val = _row(data["valuation"], ticker)
        tech = _row(data["technical"], ticker)
        gov = _row(data["governance"], symbol)
        info_row = _row(data["info_indexed"], ticker)

        if horizon == "long":
            score = score_any(fund, val, gov, info_row, tech,
                              nse_is_financial=ticker in data["financial_tickers"])
        elif horizon == "medium":
            score = score_medium_term(fund, val, gov, tech)
        else:
            score = score_short_term(fund, val, gov, tech)

        record = score.to_dict()
        record.setdefault("sector", data["sector_map"].get(ticker))
        rows.append(record)

    return rank_scores(rows)


def _regime(verbose: bool = True) -> dict | None:
    """Market regime, tolerating a network failure."""
    try:
        from src.metrics.regime import fetch_regime
        return fetch_regime()
    except Exception as exc:
        if verbose:
            print(f"[WARN] Regime unavailable ({type(exc).__name__}); "
                  "proceeding without a regime gate")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", choices=(*HORIZONS, "all"), default="long",
                        help="holding period to screen for (default: long)")
    parser.add_argument("--top", type=int, default=20,
                        help="candidates in the ranked table (default 20)")
    parser.add_argument("--detail", type=int, default=5,
                        help="candidates explained in full (default 5)")
    parser.add_argument("--ticker", nargs="+", metavar="TICKER",
                        help="score only these tickers (e.g. INFY.NS)")
    parser.add_argument("--refresh", action="store_true",
                        help="refetch all data before scoring (~30 min)")
    parser.add_argument("--min-coverage", type=float, default=0.0,
                        help="drop candidates below this data coverage (0-1)")
    parser.add_argument("--no-regime", action="store_true",
                        help="skip the market-regime lookup")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress progress output")
    args = parser.parse_args()

    enable_utf8()
    verbose = not args.quiet
    started = time.time()

    data = load_everything(refresh=args.refresh, verbose=verbose)
    regime = None if args.no_regime else _regime(verbose)

    horizons = HORIZONS if args.horizon == "all" else (args.horizon,)
    universe_size = len(data["universe"])

    for horizon in horizons:
        ranked = score_universe(data, horizon, only=args.ticker)
        if args.min_coverage > 0 and "coverage" in ranked.columns:
            ranked = ranked[ranked["coverage"] >= args.min_coverage].reset_index(drop=True)
            ranked["rank"] = ranked.index + 1

        print()
        print(render_report(ranked, horizon, regime=regime, top=args.top,
                            detail=args.detail, universe_size=universe_size))

        path = to_csv(ranked, run_output(f"ranked_{horizon}.csv"))
        print(f"\nFull ranking written to: {path}")

    if verbose:
        print(f"\n[OK] Completed in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
