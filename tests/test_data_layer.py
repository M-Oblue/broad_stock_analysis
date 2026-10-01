"""Tests for the universe loader and price fetcher.

These focus on the two failure modes that are silent rather than loud, because
those are the ones that reach production output:

1. The universe falling back to the seed list with a DIFFERENT schema, so
   downstream code half-works instead of failing cleanly.
2. Incremental price refresh appending correctly-adjusted new bars onto stale
   pre-split history, producing a price cliff mid-series that corrupts every
   momentum/MA/volatility figure computed across it -- with no exception raised.
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import universe as U
from src.data import fetch_prices as FP


def _prices(symbol, n=30, start_price=100.0, start_date="2026-01-01"):
    dates = pd.bdate_range(start_date, periods=n)
    close = pd.Series([start_price + i for i in range(n)], dtype=float)
    return pd.DataFrame({
        "symbol": symbol,
        "date": dates,
        "open": close.values, "high": close.values + 1,
        "low": close.values - 1, "close": close.values,
        "volume": 1000,
    })


class TestUniverseNormalise(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "Company Name": ["Infosys Ltd.", "HDFC Bank Ltd.", "Tata Steel Ltd."],
            "Industry": ["Information Technology", "Financial Services", "Metals & Mining"],
            "Symbol": ["INFY", "HDFCBANK", "TATASTEEL"],
            "Series": ["EQ", "EQ", "BE"],
            "ISIN Code": ["INE009A01021", "INE040A01034", "INE081A01020"],
        })

    def test_builds_yfinance_tickers(self):
        df = U._normalise(self.raw)
        self.assertEqual(sorted(df["yf_ticker"]),
                         ["HDFCBANK.NS", "INFY.NS", "TATASTEEL.NS"])

    def test_flags_financial_sector_for_lender_routing(self):
        df = U._normalise(self.raw).set_index("symbol")
        self.assertTrue(df.loc["HDFCBANK", "is_financial_sector"])
        self.assertFalse(df.loc["INFY", "is_financial_sector"])

    def test_flags_trade_to_trade_series(self):
        df = U._normalise(self.raw).set_index("symbol")
        self.assertTrue(df.loc["TATASTEEL", "is_trade_to_trade"])
        self.assertFalse(df.loc["INFY", "is_trade_to_trade"])

    def test_deduplicates_symbols(self):
        dupe = pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True)
        self.assertEqual(len(U._normalise(dupe)), 3)

    def test_drops_nse_dummy_placeholders(self):
        """NSE holds index slots during demergers with untradeable
        'Dummy <Company> Ltd.' rows that 404 on every price fetch."""
        withdummy = pd.concat([self.raw, pd.DataFrame({
            "Company Name": ["Dummy HEG Ltd."], "Industry": ["Metals & Mining"],
            "Symbol": ["DUMMYHEG"], "Series": ["EQ"], "ISIN Code": ["INE000000000"],
        })], ignore_index=True)
        out = U._normalise(withdummy)
        self.assertNotIn("DUMMYHEG", set(out["symbol"]))
        self.assertEqual(len(out), 3)

    def test_metadata_complete_detects_seed_fallback(self):
        full = U._normalise(self.raw)
        self.assertTrue(U.metadata_complete(full))
        seedlike = full.copy()
        seedlike["nse_industry"] = pd.NA
        self.assertFalse(U.metadata_complete(seedlike))


class TestPriceFlatten(unittest.TestCase):
    def test_flatten_multiindex(self):
        dates = pd.date_range("2026-01-01", periods=3)
        cols = pd.MultiIndex.from_product(
            [["INFY.NS", "TCS.NS"], ["Open", "High", "Low", "Close", "Volume"]])
        raw = pd.DataFrame(np.arange(30).reshape(3, 10), index=dates, columns=cols)
        out = FP._flatten(raw, ["INFY.NS", "TCS.NS"])
        self.assertEqual(set(out["symbol"]), {"INFY.NS", "TCS.NS"})
        self.assertEqual(list(out.columns), FP.COLUMNS)
        self.assertEqual(len(out), 6)

    def test_flatten_empty_returns_schema(self):
        out = FP._flatten(pd.DataFrame(), ["INFY.NS"])
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), FP.COLUMNS)

    def test_flatten_drops_rows_without_close(self):
        dates = pd.date_range("2026-01-01", periods=3)
        cols = pd.MultiIndex.from_product([["INFY.NS"], ["Open", "High", "Low", "Close", "Volume"]])
        raw = pd.DataFrame(np.arange(15, dtype=float).reshape(3, 5), index=dates, columns=cols)
        raw.iloc[1, 3] = np.nan  # Close
        out = FP._flatten(raw, ["INFY.NS"])
        self.assertEqual(len(out), 2)


class TestSplitReadjustmentDetection(unittest.TestCase):
    """The core safety net: catching retroactively re-adjusted history."""

    def test_detects_split_readjustment(self):
        cached = _prices("INFY.NS")
        fresh = cached.copy()
        # A 1:5 split re-adjusts all prior bars downward at the source.
        fresh["close"] = fresh["close"] / 5
        self.assertEqual(FP._detect_readjusted(cached, fresh), ["INFY.NS"])

    def test_ignores_floating_point_noise(self):
        cached = _prices("INFY.NS")
        fresh = cached.copy()
        fresh["close"] = fresh["close"] * (1 + 1e-9)
        self.assertEqual(FP._detect_readjusted(cached, fresh), [])

    def test_ignores_sub_tolerance_restatement(self):
        cached = _prices("INFY.NS")
        fresh = cached.copy()
        fresh["close"] = fresh["close"] * 1.005  # 0.5%, under the 1% tolerance
        self.assertEqual(FP._detect_readjusted(cached, fresh), [])

    def test_flags_only_the_affected_symbol(self):
        cached = pd.concat([_prices("INFY.NS"), _prices("TCS.NS")], ignore_index=True)
        fresh = cached.copy()
        mask = fresh["symbol"] == "TCS.NS"
        fresh.loc[mask, "close"] = fresh.loc[mask, "close"] / 2
        self.assertEqual(FP._detect_readjusted(cached, fresh), ["TCS.NS"])

    def test_no_overlap_is_not_a_readjustment(self):
        """Disjoint date ranges must not be misread as a split."""
        cached = _prices("INFY.NS", n=10, start_date="2026-01-01")
        fresh = _prices("INFY.NS", n=10, start_price=500.0, start_date="2026-06-01")
        self.assertEqual(FP._detect_readjusted(cached, fresh), [])

    def test_empty_inputs_are_safe(self):
        self.assertEqual(FP._detect_readjusted(pd.DataFrame(), _prices("A")), [])
        self.assertEqual(FP._detect_readjusted(_prices("A"), pd.DataFrame()), [])

    def test_zero_close_does_not_divide_by_zero(self):
        cached = _prices("INFY.NS", n=5)
        cached.loc[0, "close"] = 0.0
        fresh = cached.copy()
        self.assertEqual(FP._detect_readjusted(cached, fresh), [])


class TestBatching(unittest.TestCase):
    def test_covers_all_items_without_overlap(self):
        items = [f"S{i}" for i in range(145)]
        batches = list(FP._batched(items, 60))
        self.assertEqual([len(b) for b in batches], [60, 60, 25])
        flat = [x for b in batches for x in b]
        self.assertEqual(flat, items)


if __name__ == "__main__":
    unittest.main()
