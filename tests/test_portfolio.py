import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.portfolio.sizing import position_size, apply_position_cap, r_multiples, portfolio_heat
from src.portfolio.constraints import (
    sector_exposure, concentration, check_constraints, filter_candidates,
    correlation_filter,
)


class TestPositionSizing(unittest.TestCase):
    def test_position_size_arithmetic(self):
        sized = position_size(100000, 100, 95, risk_pct=1, lot=1)
        self.assertEqual(sized["shares"], 200)
        self.assertEqual(sized["position_value"], 20000)
        self.assertEqual(sized["risk_amount"], 1000)
        self.assertEqual(sized["risk_per_share"], 5)
        self.assertEqual(sized["position_pct_of_capital"], 20)

    def test_stop_at_or_above_entry_returns_zero_with_reason(self):
        for stop in (100, 101):
            sized = position_size(100000, 100, stop)
            self.assertEqual(sized["shares"], 0)
            self.assertIn("stop must be below entry", sized["reason"])

    def test_position_cap_binds_on_tight_stop(self):
        sized = apply_position_cap(100000, 100, 99, risk_pct=1, max_position_pct=15)
        self.assertTrue(sized["cap_applied"])
        self.assertEqual(sized["shares"], 150)
        self.assertEqual(sized["position_value"], 15000)
        self.assertEqual(sized["risk_amount"], 150)

    def test_r_multiples_and_heat(self):
        self.assertEqual(r_multiples(100, 95), {"target_2r": 110.0, "target_3r": 115.0})
        heat = portfolio_heat([
            {"capital": 100000, "risk_amount": 3000, "position_value": 20000},
            {"capital": 100000, "risk_amount": 3500, "position_value": 15000},
        ])
        self.assertAlmostEqual(heat["heat_pct"], 6.5)
        self.assertIn("account-level", heat["note"])


class TestPortfolioConstraints(unittest.TestCase):
    def setUp(self):
        self.holdings = {"A": 40000, "B": 30000, "C": 20000, "D": 10000}
        self.sectors = {"A": "Tech", "B": "Tech", "C": "Banks", "D": "Energy", "E": "Tech", "F": "Banks"}

    def test_sector_exposure(self):
        exposure = sector_exposure(self.holdings, self.sectors)
        self.assertAlmostEqual(exposure["Tech"], 70.0)
        self.assertAlmostEqual(exposure["Banks"], 20.0)

    def test_hhi_effective_stocks_and_top3(self):
        c = concentration(self.holdings)
        self.assertAlmostEqual(c["hhi"], 0.30)
        self.assertAlmostEqual(c["effective_stocks"], 1 / 0.30)
        self.assertAlmostEqual(c["top_3_weight"], 90.0)

    def test_constraint_breaches(self):
        breaches = check_constraints(self.holdings, self.sectors, max_position_pct=35, max_sector_pct=30)
        self.assertTrue(any(b["type"] == "position" and b["symbol"] == "A" for b in breaches))
        self.assertTrue(any(b["type"] == "sector" and b["sector"] == "Tech" for b in breaches))

    def test_sector_headroom_rejection_with_reason(self):
        result = filter_candidates([{"symbol": "E", "score": 5}, {"symbol": "F", "score": 3}],
                                   self.holdings, self.sectors,
                                   max_position_pct=10, max_sector_pct=30)
        self.assertEqual([r["symbol"] for r in result["rejected"]], ["E"])
        self.assertIn("no headroom", result["rejected"][0]["reason"])
        self.assertEqual(result["kept"][0]["symbol"], "F")

    def test_correlation_filter_flags_nominal_diversification(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        base = np.arange(30, dtype=float) + 100
        rows = []
        for sym, values in {
            "A": base,
            "E": base * 2,
            "F": base[::-1] + np.sin(np.arange(30)),
        }.items():
            for d, close in zip(dates, values):
                rows.append({"symbol": sym, "date": d, "close": close})
        result = correlation_filter(["E", "F"], pd.DataFrame(rows), existing=["A"], max_corr=0.8)
        self.assertEqual(result["rejected"][0]["symbol"], "E")
        self.assertIn("correlation", result["rejected"][0]["reason"])

    def test_empty_inputs_do_not_crash(self):
        self.assertEqual(sector_exposure({}, {}), {})
        self.assertEqual(concentration({})["effective_stocks"], 0.0)
        self.assertEqual(check_constraints({}, {}), [])
        self.assertEqual(filter_candidates([], {}, {})["rejected"], [])


if __name__ == "__main__":
    unittest.main()
