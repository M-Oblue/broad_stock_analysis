"""Tests for risk metrics and market-regime classification."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.metrics.regime import classify_regime
from src.metrics.risk import beta, compute_all, max_drawdown


def _close(values, start="2026-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


class TestRiskMetrics(unittest.TestCase):
    def test_beta_aligns_on_dates_not_positions(self):
        """Different start dates must be intersected by date before returns."""
        index = _close([100, 102, 104, 106, 108, 110], "2026-01-01")
        stock = _close([50, 54, 58, 62, 66, 70], "2026-01-05")

        aligned = pd.concat([stock.rename("stock"), index.rename("index")],
                            axis=1, join="inner").dropna()
        returns = np.log(aligned / aligned.shift(1)).dropna()
        expected = returns["stock"].cov(returns["index"]) / returns["index"].var()

        self.assertAlmostEqual(beta(stock, index), expected)

    def test_max_drawdown_is_negative_peak_to_trough(self):
        self.assertAlmostEqual(max_drawdown(_close([100, 120, 90, 150])), -0.25)

    def test_compute_all_missing_benchmark_returns_nan_beta(self):
        dates = pd.bdate_range("2026-01-01", periods=5)
        frame = pd.DataFrame({
            "symbol": "A.NS",
            "date": dates,
            "close": [100, 101, 99, 102, 103],
        })
        out = compute_all(frame, benchmark_symbol="^NSEI")
        self.assertTrue(np.isnan(out.loc[0, "beta"]))

    def test_empty_inputs_do_not_crash(self):
        self.assertTrue(np.isnan(beta(pd.Series(dtype=float), pd.Series(dtype=float))))
        self.assertTrue(np.isnan(max_drawdown(pd.Series(dtype=float))))


class TestRegimeClassification(unittest.TestCase):
    def test_clear_bull_market_is_risk_on(self):
        out = classify_regime(_close(np.linspace(100, 200, 240)))
        self.assertEqual(out["regime"], "risk_on")
        self.assertTrue(out["index_above_200dma"])
        self.assertTrue(np.isnan(out["vix_level"]))
        self.assertIn("VIX unavailable", out["rationale"])

    def test_clear_bear_market_is_risk_off(self):
        out = classify_regime(_close(np.linspace(200, 100, 240)))
        self.assertEqual(out["regime"], "risk_off")
        self.assertFalse(out["index_above_200dma"])
        self.assertLess(out["drawdown_from_high"], 0)

    def test_elevated_vix_pushes_toward_risk_off(self):
        out = classify_regime(_close(np.linspace(100, 200, 240)),
                              _close([30.0] * 240))
        self.assertEqual(out["regime"], "risk_off")
        self.assertEqual(out["vix_level"], 30.0)

    def test_short_history_is_neutral(self):
        out = classify_regime(_close([100.0]))
        self.assertEqual(out["regime"], "neutral")
        self.assertIn("insufficient", out["rationale"])


if __name__ == "__main__":
    unittest.main()
