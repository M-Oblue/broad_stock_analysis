"""Tests for fundamental metric computation.

The quarterly-momentum tests carry the most weight. yfinance's quarterly series
has holes -- RELIANCE is missing 2025-09-30 entirely and INFY reports revenue
for only 5 of its 7 quarters -- so locating "the same quarter last year" by
position silently compares the wrong interval. That bug produced a plausible
18.4% YoY for RELIANCE where the truth was 27.0%: no exception, no obvious
tell, just a wrong number feeding the medium-horizon engine.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.metrics.fundamental import (
    _safe_div, _cagr, _slope_per_year, _quarterly_momentum,
    compute_roce_series, compute_for_ticker, YEAR_AGO_TOLERANCE_DAYS,
)


def _annual(ticker="T.NS", periods=("2022-03-31", "2023-03-31", "2024-03-31",
                                    "2025-03-31", "2026-03-31"), **items):
    """Build long-format annual statement rows."""
    rows = []
    for item, values in items.items():
        for period, value in zip(periods, values):
            if value is None:
                continue
            rows.append({"ticker": ticker, "statement": "income",
                         "period_type": "annual", "period_end": period,
                         "item": item, "value": float(value)})
    return pd.DataFrame(rows)


def _quarterly_frame(pairs, column="total_revenue"):
    """Build a wide quarterly frame from (period_end, value) pairs."""
    index = [p for p, _ in pairs]
    return pd.DataFrame({column: [v for _, v in pairs]}, index=index)


class TestSafeDiv(unittest.TestCase):
    def test_normal_division(self):
        self.assertAlmostEqual(_safe_div(10, 4), 2.5)

    def test_zero_denominator_is_nan(self):
        self.assertTrue(np.isnan(_safe_div(10, 0)))

    def test_negative_denominator_rejected_by_default(self):
        """ROE on negative equity yields a positive-looking number that would
        rank well in a screen. That silent sign flip is the danger."""
        self.assertTrue(np.isnan(_safe_div(-10, -5)))

    def test_negative_denominator_allowed_when_requested(self):
        self.assertAlmostEqual(_safe_div(-10, -5, positive_denominator=False), 2.0)

    def test_nan_inputs(self):
        self.assertTrue(np.isnan(_safe_div(np.nan, 5)))
        self.assertTrue(np.isnan(_safe_div(5, np.nan)))
        self.assertTrue(np.isnan(_safe_div(None, None)))


class TestCagr(unittest.TestCase):
    def test_doubling_over_3_years(self):
        self.assertAlmostEqual(_cagr(100, 200, 3), 25.99, places=1)

    def test_flat(self):
        self.assertAlmostEqual(_cagr(100, 100, 5), 0.0)

    def test_negative_start_is_undefined(self):
        """Loss-to-profit has no meaningful growth RATE; forcing one would
        report spectacular growth for a turnaround off a tiny base."""
        self.assertTrue(np.isnan(_cagr(-50, 100, 3)))
        self.assertTrue(np.isnan(_cagr(0, 100, 3)))

    def test_collapse_to_loss_reported_as_minus_100(self):
        self.assertEqual(_cagr(100, -20, 3), -100.0)


class TestSlope(unittest.TestCase):
    def test_rising_series_positive_slope(self):
        self.assertGreater(_slope_per_year(pd.Series([10.0, 12, 14, 16])), 0)

    def test_falling_series_negative_slope(self):
        self.assertLess(_slope_per_year(pd.Series([20.0, 18, 16, 14])), 0)

    def test_too_few_points_is_nan(self):
        self.assertTrue(np.isnan(_slope_per_year(pd.Series([10.0, 12]))))

    def test_regression_resists_single_outlier(self):
        """First-vs-last would invert on a final-year outlier; least squares
        across all years should still see the underlying uptrend."""
        self.assertGreater(_slope_per_year(pd.Series([10.0, 12, 14, 16, 15])), 0)


class TestRoce(unittest.TestCase):
    def test_roce_is_ebit_over_debt_plus_equity(self):
        annual = pd.DataFrame({"ebit": [100.0], "total_debt": [200.0],
                               "stockholders_equity": [800.0]},
                              index=["2026-03-31"])
        self.assertAlmostEqual(compute_roce_series(annual).iloc[0], 10.0)

    def test_missing_debt_treated_as_zero(self):
        """A debt-free company has capital employed equal to its equity."""
        annual = pd.DataFrame({"ebit": [100.0], "total_debt": [np.nan],
                               "stockholders_equity": [500.0]},
                              index=["2026-03-31"])
        self.assertAlmostEqual(compute_roce_series(annual).iloc[0], 20.0)

    def test_no_ebit_yields_empty(self):
        """Lenders report no EBIT at all."""
        annual = pd.DataFrame({"stockholders_equity": [500.0]}, index=["2026-03-31"])
        self.assertTrue(compute_roce_series(annual).empty)


class TestQuarterlyMomentumDateMatching(unittest.TestCase):
    """The RELIANCE gap bug, pinned down."""

    def test_matches_same_quarter_one_year_earlier(self):
        q = _quarterly_frame([
            ("2025-06-30", 100.0), ("2025-09-30", 110.0),
            ("2025-12-31", 120.0), ("2026-03-31", 130.0), ("2026-06-30", 125.0),
        ])
        out = _quarterly_momentum(q)
        self.assertAlmostEqual(out["q_revenue_yoy"], 25.0)   # 125 vs 100

    def test_gap_in_series_does_not_shift_the_comparison(self):
        """RELIANCE's real shape: 2025-09-30 missing. Positional indexing
        compared Jun-2026 against Mar-2025 and called it YoY."""
        q = _quarterly_frame([
            ("2025-03-31", 261.0), ("2025-06-30", 243.0),
            ("2025-12-31", 264.0), ("2026-03-31", 294.0), ("2026-06-30", 309.0),
        ])
        out = _quarterly_momentum(q)
        self.assertAlmostEqual(out["q_revenue_yoy"], (309 / 243 - 1) * 100, places=2)
        # The buggy positional answer would have been 309/261.
        self.assertNotAlmostEqual(out["q_revenue_yoy"], (309 / 261 - 1) * 100, places=2)

    def test_missing_year_ago_quarter_returns_nan_not_a_guess(self):
        q = _quarterly_frame([
            ("2025-12-31", 120.0), ("2026-03-31", 130.0), ("2026-06-30", 125.0),
        ])
        self.assertTrue(np.isnan(_quarterly_momentum(q)["q_revenue_yoy"]))

    def test_adjacent_quarter_is_outside_tolerance(self):
        """A quarter 91 days from the target must never be accepted as the
        year-ago comparison."""
        self.assertLess(YEAR_AGO_TOLERANCE_DAYS, 91)

    def test_acceleration_is_change_in_growth_rate(self):
        q = _quarterly_frame([
            ("2025-03-31", 100.0), ("2025-06-30", 100.0),
            ("2025-09-30", 100.0), ("2025-12-31", 100.0),
            ("2026-03-31", 110.0), ("2026-06-30", 125.0),
        ])
        out = _quarterly_momentum(q)
        self.assertAlmostEqual(out["q_revenue_yoy"], 25.0)        # 125 vs 100
        self.assertAlmostEqual(out["q_revenue_yoy_prev"], 10.0)   # 110 vs 100
        self.assertAlmostEqual(out["q_revenue_acceleration"], 15.0)

    def test_growth_off_negative_base_is_nan(self):
        q = _quarterly_frame([
            ("2025-06-30", -50.0), ("2025-09-30", 10.0),
            ("2025-12-31", 20.0), ("2026-03-31", 30.0), ("2026-06-30", 40.0),
        ])
        self.assertTrue(np.isnan(_quarterly_momentum(q)["q_revenue_yoy"]))

    def test_empty_frame_is_safe(self):
        out = _quarterly_momentum(pd.DataFrame())
        self.assertTrue(np.isnan(out["q_revenue_yoy"]))
        self.assertIsNone(out["latest_quarter"])


class TestComputeForTicker(unittest.TestCase):
    def setUp(self):
        self.statements = _annual(
            "GOOD.NS",
            total_revenue=[1000, 1100, 1250, 1400, 1600],
            ebit=[150, 170, 200, 230, 270],
            net_income=[100, 115, 135, 155, 180],
            stockholders_equity=[600, 660, 730, 810, 900],
            total_debt=[200, 190, 180, 160, 140],
            operating_cash_flow=[120, 140, 160, 185, 215],
            free_cash_flow=[80, 95, 110, 130, 155],
            diluted_eps=[10, 11.5, 13.5, 15.5, 18],
            shares_outstanding=[10, 10, 10, 10, 10],
            interest_expense=[20, 19, 18, 16, 14],
            total_assets=[1000, 1100, 1200, 1300, 1450],
        )

    def test_computes_core_quality_metrics(self):
        m = compute_for_ticker(self.statements, "GOOD.NS")
        self.assertTrue(m["has_fundamentals"])
        # ROCE = 270 / (140 + 900) = 25.96%
        self.assertAlmostEqual(m["roce_latest"], 25.96, places=1)
        self.assertGreater(m["roce_mean_5y"], 20)
        self.assertLess(m["roce_std_5y"], 5)          # consistent
        self.assertGreater(m["cfo_to_net_income"], 1) # cash-backed earnings

    def test_detects_deleveraging_as_negative_debt_trend(self):
        m = compute_for_ticker(self.statements, "GOOD.NS")
        self.assertLess(m["debt_to_equity_trend"], 0)

    def test_computes_growth_cagrs(self):
        m = compute_for_ticker(self.statements, "GOOD.NS")
        self.assertAlmostEqual(m["revenue_cagr_3y"],
                               _cagr(1100, 1600, 3), places=2)
        self.assertGreater(m["eps_cagr_3y"], 0)

    def test_no_dilution_detected(self):
        m = compute_for_ticker(self.statements, "GOOD.NS")
        self.assertAlmostEqual(m["share_count_cagr"], 0.0)

    def test_lender_nulls_ebit_derived_metrics(self):
        """A bank must not receive a ROCE or interest-coverage figure -- both
        are meaningless when interest IS revenue."""
        m = compute_for_ticker(self.statements, "GOOD.NS", is_lender=True)
        for key in ("roce_latest", "roce_mean_5y", "interest_coverage",
                    "operating_margin", "net_debt_to_ebitda"):
            self.assertTrue(np.isnan(m[key]), f"{key} should be NaN for a lender")
        # Metrics that remain meaningful for a lender stay populated.
        self.assertFalse(np.isnan(m["roe_latest"]))
        self.assertFalse(np.isnan(m["revenue_cagr_3y"]))

    def test_missing_ticker_reports_no_fundamentals(self):
        m = compute_for_ticker(self.statements, "ABSENT.NS")
        self.assertFalse(m["has_fundamentals"])
        self.assertEqual(m["annual_periods"], 0)

    def test_counts_profitable_and_fcf_positive_years(self):
        m = compute_for_ticker(self.statements, "GOOD.NS")
        self.assertEqual(m["profitable_years"], 5)
        self.assertEqual(m["positive_fcf_years"], 5)


if __name__ == "__main__":
    unittest.main()
