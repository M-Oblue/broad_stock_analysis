"""Tests for technical metric computation.

The 12-1 momentum test is the critical one: the most recent month often contains
short-term reversal noise, so the Jegadeesh-Titman factor must explicitly skip
it rather than becoming a disguised 12-month return.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.metrics.technical import compute_all, compute_for_symbol


def _prices(symbol="TEST.NS", close=None, start="2025-01-01"):
    close = np.asarray(close if close is not None else np.linspace(100, 200, 300), dtype=float)
    dates = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({
        "symbol": symbol,
        "date": dates,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000,
    })


class TestTechnicalMomentum(unittest.TestCase):
    def test_12_1_momentum_excludes_the_recent_month_spike(self):
        """A final-month price spike must move plain 12m return but not 12-1."""
        base = np.linspace(100, 200, 260)
        spike = np.linspace(210, 400, 20)
        frame = _prices(close=np.r_[base, spike])

        out = compute_for_symbol(frame)
        expected_12_1 = (base[-1] / np.r_[base, spike][-252] - 1.0) * 100.0
        expected_12m = (spike[-1] / np.r_[base, spike][-252] - 1.0) * 100.0

        self.assertAlmostEqual(out["mom_12_1"], expected_12_1)
        self.assertAlmostEqual(out["ret_12m"], expected_12m)
        self.assertGreater(out["ret_12m"], out["mom_12_1"] * 2)

    def test_insufficient_history_returns_nan_not_partial_windows(self):
        out = compute_for_symbol(_prices(close=np.linspace(100, 120, 100)))
        for key in ("sma200", "close_vs_sma200_pct", "mom_12_1",
                    "ret_12m", "ret_3y", "ann_ret_5y",
                    "dist_from_52w_high"):
            self.assertTrue(np.isnan(out[key]), f"{key} should be NaN")

    def test_empty_and_single_row_inputs_do_not_crash(self):
        empty = compute_for_symbol(pd.DataFrame(columns=["symbol", "date", "close"]))
        self.assertEqual(empty["bars"], 0)

        one = compute_for_symbol(_prices(close=[100.0]))
        self.assertEqual(one["bars"], 1)
        self.assertTrue(np.isnan(one["rsi"]))

    def test_compute_all_returns_one_row_per_symbol(self):
        frame = pd.concat([_prices("A.NS"), _prices("B.NS", close=np.linspace(50, 75, 300))],
                          ignore_index=True)
        out = compute_all(frame)
        self.assertEqual(set(out["symbol"]), {"A.NS", "B.NS"})
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
