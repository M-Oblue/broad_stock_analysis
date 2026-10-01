"""Tests for valuation and governance metrics.

Both modules encode judgements that are easy to get subtly and silently wrong:

- A loss-making company has a negative P/E, which sorts to the top of an
  ascending "cheapest" screen. Negative multiples must become NaN, not bargains.
- NaN is truthy in Python, so a missing `financialCurrency` compared as a string
  against "INR" reads as a currency mismatch. JSWDULUX.NS was suppressed on
  exactly that bug -- absent data must mean "cannot determine", not "different".
- A company with no promoter (HDFCBANK, ITC) must be routed past promoter checks
  entirely, rather than scored as though promoters had sold everything.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.metrics.valuation import (
    _positive, _ratio, _ttm, compute_for_ticker, own_history_percentile,
    add_sector_relative, MIN_BARS_FOR_HISTORY,
)
from src.metrics.governance import (
    classify_pledge, classify_holding, classify_trend, compute_for_symbol,
    PLEDGE_SEVERE, SELLDOWN_SEVERE_PP,
)


def _stmt(ticker="T.NS", period_type="annual", statement="income", **items):
    rows = []
    for item, periods in items.items():
        for period_end, value in periods:
            rows.append({"ticker": ticker, "statement": statement,
                         "period_type": period_type, "period_end": period_end,
                         "item": item, "value": float(value)})
    return pd.DataFrame(rows)


def _info(**kwargs):
    base = {"ticker": "T.NS", "marketCap": 1e11, "enterpriseValue": 1.1e11,
            "currency": "INR", "financialCurrency": "INR",
            "dividendYield": 1.5, "payoutRatio": 0.3, "earningsGrowth": 0.20}
    base.update(kwargs)
    return base


class TestPositiveGuard(unittest.TestCase):
    def test_rejects_zero_and_negative(self):
        self.assertTrue(np.isnan(_positive(0)))
        self.assertTrue(np.isnan(_positive(-5)))

    def test_accepts_positive(self):
        self.assertEqual(_positive(12.5), 12.5)

    def test_ratio_with_negative_denominator_is_nan(self):
        """A loss makes P/E negative, which would rank as the cheapest stock
        in an ascending sort."""
        self.assertTrue(np.isnan(_ratio(1e11, -5e9)))


class TestTTM(unittest.TestCase):
    def test_sums_last_four_quarters(self):
        stmts = _stmt(period_type="quarterly", total_revenue=[
            ("2025-06-30", 100), ("2025-09-30", 110),
            ("2025-12-31", 120), ("2026-03-31", 130), ("2026-06-30", 140)])
        self.assertAlmostEqual(_ttm(stmts, "T.NS", "total_revenue"), 500.0)

    def test_falls_back_to_annual_when_quarters_insufficient(self):
        stmts = pd.concat([
            _stmt(period_type="quarterly", total_revenue=[("2026-06-30", 140)]),
            _stmt(period_type="annual", total_revenue=[("2026-03-31", 480)]),
        ], ignore_index=True)
        self.assertAlmostEqual(_ttm(stmts, "T.NS", "total_revenue"), 480.0)

    def test_missing_item_is_nan(self):
        self.assertTrue(np.isnan(_ttm(_stmt(total_revenue=[("2026-03-31", 1)]),
                                      "T.NS", "ebitda")))


class TestValuationRatios(unittest.TestCase):
    def setUp(self):
        self.stmts = pd.concat([
            _stmt(period_type="quarterly",
                  net_income=[("2025-09-30", 1e9), ("2025-12-31", 1e9),
                              ("2026-03-31", 1e9), ("2026-06-30", 1e9)],
                  total_revenue=[("2025-09-30", 5e9), ("2025-12-31", 5e9),
                                 ("2026-03-31", 5e9), ("2026-06-30", 5e9)],
                  ebitda=[("2025-09-30", 2e9), ("2025-12-31", 2e9),
                          ("2026-03-31", 2e9), ("2026-06-30", 2e9)]),
            _stmt(period_type="annual",
                  stockholders_equity=[("2026-03-31", 2e10)],
                  free_cash_flow=[("2026-03-31", 3e9)]),
        ], ignore_index=True)

    def test_computes_core_multiples(self):
        out = compute_for_ticker("T.NS", self.stmts, _info())
        self.assertAlmostEqual(out["pe"], 1e11 / 4e9)       # 25
        self.assertAlmostEqual(out["pb"], 1e11 / 2e10)      # 5
        self.assertAlmostEqual(out["ps"], 1e11 / 2e10)      # 5
        self.assertAlmostEqual(out["ev_ebitda"], 1.1e11 / 8e9)
        self.assertAlmostEqual(out["earnings_yield"], 4.0)

    def test_loss_making_yields_nan_not_negative_pe(self):
        stmts = _stmt(period_type="quarterly", net_income=[
            ("2025-09-30", -1e9), ("2025-12-31", -1e9),
            ("2026-03-31", -1e9), ("2026-06-30", -1e9)])
        out = compute_for_ticker("T.NS", stmts, _info())
        self.assertTrue(out["is_loss_making"])
        self.assertTrue(np.isnan(out["pe"]))
        self.assertIn("cannot rank as cheap", out["valuation_caveat"])

    def test_peg_requires_positive_growth(self):
        out = compute_for_ticker("T.NS", self.stmts, _info(earningsGrowth=-0.1))
        self.assertTrue(np.isnan(out["peg"]))

    def test_peg_computed_on_positive_growth(self):
        out = compute_for_ticker("T.NS", self.stmts, _info(earningsGrowth=0.25))
        self.assertAlmostEqual(out["peg"], 25.0 / 25.0)


class TestCurrencySuppression(unittest.TestCase):
    def setUp(self):
        self.stmts = _stmt(period_type="quarterly", net_income=[
            ("2025-09-30", 1e9), ("2025-12-31", 1e9),
            ("2026-03-31", 1e9), ("2026-06-30", 1e9)])

    def test_usd_statements_inr_price_suppresses_all_multiples(self):
        out = compute_for_ticker("T.NS", self.stmts,
                                 _info(financialCurrency="USD"))
        self.assertTrue(out["currency_suppressed"])
        for field in ("pe", "pb", "ps", "ev_ebitda", "peg", "fcf_yield"):
            self.assertTrue(np.isnan(out[field]), f"{field} must be suppressed")
        self.assertIn("ROCE", out["valuation_caveat"])

    def test_missing_financial_currency_is_not_a_mismatch(self):
        """NaN is truthy, so the naive check compared "INR" to "nan" and
        suppressed JSWDULUX.NS, which has no financialCurrency at all."""
        out = compute_for_ticker("T.NS", self.stmts,
                                 _info(financialCurrency=np.nan))
        self.assertFalse(out["currency_suppressed"])
        self.assertFalse(np.isnan(out["pe"]))
        self.assertIn("unconfirmed", out["valuation_caveat"])

    def test_matching_currencies_are_not_suppressed(self):
        out = compute_for_ticker("T.NS", self.stmts, _info())
        self.assertFalse(out["currency_suppressed"])
        self.assertFalse(np.isnan(out["pe"]))


class TestOwnHistoryPercentile(unittest.TestCase):
    def test_cheap_versus_own_history_scores_high(self):
        history = pd.Series(np.linspace(20, 40, MIN_BARS_FOR_HISTORY))
        self.assertGreater(own_history_percentile(21.0, history), 90)

    def test_expensive_versus_own_history_scores_low(self):
        history = pd.Series(np.linspace(20, 40, MIN_BARS_FOR_HISTORY))
        self.assertLess(own_history_percentile(39.0, history), 10)

    def test_short_history_is_nan_not_a_confident_answer(self):
        self.assertTrue(np.isnan(own_history_percentile(25.0, pd.Series([20.0, 30.0]))))


class TestSectorRelative(unittest.TestCase):
    def test_cheaper_than_peers_scores_positive(self):
        frame = pd.DataFrame({"ticker": [f"T{i}.NS" for i in range(6)],
                              "pe": [10.0, 20, 22, 24, 26, 28]})
        sectors = {f"T{i}.NS": "IT" for i in range(6)}
        out = add_sector_relative(frame, sectors, metrics=("pe",))
        self.assertGreater(out.loc[0, "pe_sector_z"], 0)   # cheapest
        self.assertLess(out.loc[5, "pe_sector_z"], 0)      # dearest

    def test_thin_sector_yields_nan(self):
        frame = pd.DataFrame({"ticker": ["A.NS", "B.NS"], "pe": [10.0, 30.0]})
        out = add_sector_relative(frame, {"A.NS": "X", "B.NS": "X"},
                                  metrics=("pe",))
        self.assertTrue(out["pe_sector_z"].isna().all())


class TestGovernanceClassification(unittest.TestCase):
    def test_no_promoter_is_not_penalised(self):
        """HDFCBANK and ITC have no promoter; that is a structure, not a fault."""
        band, note = classify_pledge(np.nan, has_promoter=False)
        self.assertEqual(band, "not_applicable")
        self.assertIn("professionally managed", note)
        band, note = classify_holding(0.0, has_promoter=False)
        self.assertEqual(band, "widely_held")

    def test_severe_pledging_band(self):
        band, note = classify_pledge(PLEDGE_SEVERE + 10, has_promoter=True)
        self.assertEqual(band, "severe")
        self.assertIn("force lender selling", note)

    def test_zero_pledge_is_clean(self):
        self.assertEqual(classify_pledge(0.0, True)[0], "none")

    def test_very_high_holding_notes_float_risk(self):
        band, note = classify_holding(85.0, has_promoter=True)
        self.assertEqual(band, "very_high")
        self.assertIn("thin", note)

    def test_selldown_requires_enough_history(self):
        self.assertEqual(classify_trend(-6.0, quarters=2)[0], "unknown")

    def test_heavy_selldown_detected(self):
        self.assertEqual(classify_trend(SELLDOWN_SEVERE_PP - 1, quarters=8)[0],
                         "selling_heavily")

    def test_small_change_is_stable_not_a_selldown(self):
        """Sub-threshold drift is ESOP dilution and reclassification, not exit."""
        self.assertEqual(classify_trend(-0.5, quarters=8)[0], "stable")


class TestGovernanceFlags(unittest.TestCase):
    def test_no_promoter_company_has_no_red_flag(self):
        out = compute_for_symbol({
            "symbol": "HDFCBANK", "promoter_holding_pct": 0.0,
            "promoter_pledge_pct": np.nan, "has_promoter": False,
            "promoter_change_4q_pp": 0.0, "quarters_of_history": 22})
        self.assertFalse(out["has_red_flag"])
        self.assertEqual(out["governance_flags"], [])

    def test_severe_pledging_raises_flag(self):
        out = compute_for_symbol({
            "symbol": "AFCONS", "promoter_holding_pct": 50.17,
            "promoter_pledge_pct": 100.0, "has_promoter": True,
            "promoter_change_4q_pp": 0.0, "quarters_of_history": 20})
        self.assertTrue(out["has_red_flag"])
        self.assertTrue(any("SEVERE" in f for f in out["governance_flags"]))

    def test_pledging_plus_selldown_is_called_out_as_compounding(self):
        out = compute_for_symbol({
            "symbol": "X", "promoter_holding_pct": 40.0,
            "promoter_pledge_pct": 30.0, "has_promoter": True,
            "promoter_change_4q_pp": -6.0, "quarters_of_history": 20})
        self.assertTrue(any("combined with an active sell-down" in f
                            for f in out["governance_flags"]))

    def test_clean_company_has_no_flags(self):
        out = compute_for_symbol({
            "symbol": "INFY", "promoter_holding_pct": 13.82,
            "promoter_pledge_pct": 0.0, "has_promoter": True,
            "promoter_change_4q_pp": -0.48, "quarters_of_history": 21})
        self.assertFalse(out["has_red_flag"])

    def test_stable_low_holding_is_not_a_red_flag(self):
        """Infosys sits at 13.8% because its founders diluted over decades. A
        small but stable stake must not land in the same bucket as a promoter
        who has pledged their entire holding."""
        out = compute_for_symbol({
            "symbol": "INFY", "promoter_holding_pct": 13.82,
            "promoter_pledge_pct": 0.0, "has_promoter": True,
            "promoter_change_4q_pp": -0.48, "quarters_of_history": 21})
        self.assertEqual(out["holding_band"], "low")
        self.assertFalse(out["has_red_flag"])

    def test_low_holding_that_is_still_shrinking_is_flagged(self):
        """A small stake that keeps shrinking is an exit in progress."""
        out = compute_for_symbol({
            "symbol": "X", "promoter_holding_pct": 12.0,
            "promoter_pledge_pct": 0.0, "has_promoter": True,
            "promoter_change_4q_pp": -6.0, "quarters_of_history": 20})
        self.assertTrue(out["has_red_flag"])
        self.assertTrue(any("still selling" in f for f in out["governance_flags"]))


if __name__ == "__main__":
    unittest.main()
