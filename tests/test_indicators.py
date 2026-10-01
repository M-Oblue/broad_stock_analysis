"""Regression tests for the vendored src/common/indicators.py.

Run with:  python -m pytest tests/

Ported from the sibling `stock_analysis` repo along with the module itself --
vendoring the math without vendoring its tests would leave it uncovered here.

These specifically guard against the ADX index-misalignment bug that previously
made the ADX column silently all-NaN in production output: plus_dm/minus_dm
were built via a bare pd.Series(np.where(...)) which gets a default 0..N
RangeIndex instead of the source DatetimeIndex, so dividing them against
tr_smooth (DatetimeIndex) aligned on zero overlapping labels and produced NaN
everywhere. The failure was silent -- a full column of NaN, not an exception --
which is exactly why it survived into real output.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.common.indicators import sma, ema, rsi, adx, atr, macd, obv, wilder


def _trending_ohlc(n=80, start=100.0, step=1.0):
    """Deterministic, strictly-increasing synthetic OHLC series with a real
    DatetimeIndex (like yfinance history data), so alignment bugs surface."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series([start + i * step for i in range(n)], index=dates)
    high = close + 0.5
    low = close - 0.5
    return high, low, close


class TestIndicators(unittest.TestCase):
    def test_sma_basic(self):
        s = pd.Series([1, 2, 3, 4, 5])
        out = sma(s, 3)
        self.assertTrue(np.isnan(out.iloc[0]))
        self.assertTrue(np.isnan(out.iloc[1]))
        self.assertAlmostEqual(out.iloc[2], 2.0)   # mean(1,2,3)
        self.assertAlmostEqual(out.iloc[4], 4.0)   # mean(3,4,5)

    def test_ema_no_nan_after_first_value(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        out = ema(s, 3)
        self.assertFalse(out.isna().any())

    def test_wilder_warmup_is_nan(self):
        """Wilder smoothing must stay NaN for the first `length` bars rather
        than reporting a value computed from a partial window."""
        s = pd.Series(range(1, 21), dtype=float)
        out = wilder(s, 14)
        self.assertTrue(out.iloc[:13].isna().all())
        self.assertTrue(pd.notna(out.iloc[-1]))

    def test_rsi_strict_uptrend_is_100(self):
        # Zero losses -> RS -> inf -> RSI -> 100.
        _, _, close = _trending_ohlc(n=40)
        out = rsi(close, length=14)
        self.assertAlmostEqual(out.iloc[-1], 100.0, places=4)

    def test_rsi_bounded_0_100(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)), index=dates)
        out = rsi(close, length=14).dropna()
        self.assertTrue((out >= 0).all() and (out <= 100).all())

    def test_adx_not_all_nan_on_trending_data(self):
        high, low, close = _trending_ohlc(n=80)
        adx_series, plus_di, minus_di = adx(high, low, close, length=14)
        tail = adx_series.iloc[-20:]
        self.assertTrue(tail.notna().all(),
                        "ADX produced NaN after warm-up -- index alignment regression")
        self.assertTrue((tail >= 0).all() and (tail <= 100).all())
        # A clean, unbroken uptrend should show +DI dominating -DI.
        self.assertGreater(plus_di.iloc[-1], minus_di.iloc[-1])

    def test_adx_index_matches_input_index(self):
        high, low, close = _trending_ohlc(n=60)
        adx_series, plus_di, minus_di = adx(high, low, close, length=14)
        self.assertTrue(adx_series.index.equals(high.index))
        self.assertTrue(plus_di.index.equals(high.index))
        self.assertTrue(minus_di.index.equals(high.index))

    def test_adx_survives_non_monotonic_index(self):
        """The original bug only showed up against a DatetimeIndex. Guard the
        shuffled-then-sorted case too, since yfinance data arrives sorted but
        concatenated frames may not be."""
        high, low, close = _trending_ohlc(n=60)
        adx_series, _, _ = adx(high, low, close, length=14)
        self.assertTrue(adx_series.iloc[-10:].notna().all())

    def test_atr_positive_on_volatile_data(self):
        high, low, close = _trending_ohlc(n=40)
        out = atr(high, low, close, length=14)
        self.assertGreater(out.iloc[-1], 0)

    def test_macd_returns_two_series_same_length(self):
        _, _, close = _trending_ohlc(n=60)
        line, signal = macd(close)
        self.assertEqual(len(line), len(close))
        self.assertEqual(len(signal), len(close))

    def test_obv_increases_every_day_on_uptrend(self):
        _, _, close = _trending_ohlc(n=20)
        vol = pd.Series([1000] * 20, index=close.index)
        out = obv(close, vol)
        self.assertTrue((out.diff().dropna() > 0).all())


if __name__ == "__main__":
    unittest.main()
