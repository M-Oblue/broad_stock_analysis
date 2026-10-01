"""Tests for the lender scoring override and financial-sector routing.

NSE's "Financial Services" bucket holds 101 Nifty 500 names spanning at least
four business models. Scoring them all as lenders reproduced, one level down,
the exact error the lender model exists to correct:

  - Life insurers (HDFCLIFE, ICICIPRULI, CANHLIFE) crashed to 0-19/100, each
    carrying a 12-point "thin capital buffer" penalty -- because an insurer's
    balance sheet is mostly policyholder funds, so equity/assets is structurally
    tiny and says nothing about solvency.
  - Fee-based brokers ranked near the top of a lending model that does not
    describe them: ANANDRATHI showed a 29.8% "ROA" and GROWW 11.2%.

EBIT availability confirms the split empirically: 0 of 26 Banks report it,
while 9 of 11 Asset Managers and 3 of 3 Exchanges do.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.scoring.financials import (
    classify_financial, model_map, lender_tickers, score_financial, score_any,
    MODEL_LENDER, MODEL_INSURER, MODEL_GENERIC, ASSET_GROWTH_CAUTION,
    W_PROFITABILITY, W_CAPITALISATION, W_GROWTH, W_VALUATION,
)


def _lender(**overrides):
    base = {
        "ticker": "BANK.NS", "is_lender": True, "years_of_data": 5,
        "roa_latest": 1.8, "roe_latest": 16.0, "roe_mean_5y": 16.5,
        "equity_to_assets": 12.0, "nim_proxy": 3.5,
        "asset_cagr_3y": 18.0, "eps_cagr_3y": 15.0,
        "revenue_cagr_3y": 17.0,
    }
    base.update(overrides)
    return base


class TestWeights(unittest.TestCase):
    def test_lender_weights_sum_to_100(self):
        self.assertAlmostEqual(
            W_PROFITABILITY + W_CAPITALISATION + W_GROWTH + W_VALUATION, 100.0)


class TestClassification(unittest.TestCase):
    def test_banks_and_nbfcs_are_lenders(self):
        for industry in ("Banks - Regional", "Credit Services", "Mortgage Finance"):
            self.assertEqual(classify_financial(industry, True), MODEL_LENDER,
                             f"{industry} should route to the lender model")

    def test_insurers_are_not_lenders(self):
        """An insurer's equity/assets is structurally tiny because the balance
        sheet is policyholder funds -- the lender model reads that as
        insolvency and applies a capital-buffer penalty."""
        for industry in ("Insurance - Life", "Insurance - Diversified",
                         "Insurance - Reinsurance", "Insurance Brokers"):
            self.assertEqual(classify_financial(industry, True), MODEL_INSURER)

    def test_fee_businesses_route_to_generic(self):
        """Asset managers, brokers and exchanges earn fees and report EBIT --
        they are ordinary businesses that happen to sit in a financial bucket."""
        for industry in ("Asset Management", "Capital Markets",
                         "Financial Data & Stock Exchanges",
                         "Software - Infrastructure"):
            self.assertEqual(classify_financial(industry, True), MODEL_GENERIC)

    def test_non_financial_always_generic(self):
        self.assertEqual(classify_financial("Banks - Regional", False), MODEL_GENERIC)

    def test_unknown_financial_industry_defaults_to_generic(self):
        """Defaulting to the lender model would risk the insurer failure."""
        self.assertEqual(classify_financial(None, True), MODEL_GENERIC)
        self.assertEqual(classify_financial("", True), MODEL_GENERIC)


class TestModelMap(unittest.TestCase):
    def setUp(self):
        self.info = pd.DataFrame({
            "ticker": ["HDFCBANK.NS", "HDFCLIFE.NS", "HDFCAMC.NS", "INFY.NS"],
            "industry": ["Banks - Regional", "Insurance - Life",
                         "Asset Management", "Information Technology Services"],
        })
        self.financials = {"HDFCBANK.NS", "HDFCLIFE.NS", "HDFCAMC.NS"}

    def test_routes_each_ticker_correctly(self):
        mapping = model_map(self.info, self.financials)
        self.assertEqual(mapping["HDFCBANK.NS"], MODEL_LENDER)
        self.assertEqual(mapping["HDFCLIFE.NS"], MODEL_INSURER)
        self.assertEqual(mapping["HDFCAMC.NS"], MODEL_GENERIC)
        self.assertEqual(mapping["INFY.NS"], MODEL_GENERIC)

    def test_lender_set_excludes_insurers_and_fee_businesses(self):
        """The lender set drives which companies get their EBIT metrics
        nulled, so over-including here silently strips ROCE from businesses
        that legitimately report it."""
        lenders = lender_tickers(self.info, self.financials)
        self.assertEqual(lenders, {"HDFCBANK.NS"})


class TestLenderScoring(unittest.TestCase):
    def test_strong_lender_scores_well(self):
        s = score_financial(_lender(), {"pb": 1.5}, {}, {})
        self.assertGreater(s.score, 75)

    def test_weak_lender_scores_poorly(self):
        s = score_financial(
            _lender(roa_latest=0.15, roe_mean_5y=3.0, equity_to_assets=4.0,
                    nim_proxy=1.0, asset_cagr_3y=-5.0, eps_cagr_3y=-15.0),
            {"pb": 5.5}, {}, {})
        self.assertLess(s.score, 30)

    def test_thin_capital_is_penalised(self):
        s = score_financial(_lender(equity_to_assets=3.0), {"pb": 1.5}, {}, {})
        self.assertTrue(any("thin buffer" in p.lower() or "buffer" in p.lower()
                            for p in s.to_dict()["penalties"]))

    def test_rapid_asset_growth_is_a_warning_not_a_reward(self):
        """Fast loan growth is how a lender buys future bad debts; the defaults
        surface two to three years later, inside the holding period."""
        steady = score_financial(_lender(asset_cagr_3y=18.0), {"pb": 1.5}, {}, {})
        rapid = score_financial(
            _lender(asset_cagr_3y=ASSET_GROWTH_CAUTION + 25), {"pb": 1.5}, {}, {})
        self.assertLess(rapid.score, steady.score)
        self.assertTrue(any("rapid" in w.lower() for w in rapid.warnings))

    def test_asset_quality_caveat_always_present(self):
        """The model cannot see NPAs, and must say so every time."""
        s = score_financial(_lender(), {"pb": 1.5}, {}, {})
        self.assertTrue(any("Asset quality is NOT assessed" in w
                            for w in s.warnings))

    def test_uses_price_to_book_not_pe(self):
        cheap = score_financial(_lender(), {"pb": 0.9}, {}, {})
        dear = score_financial(_lender(), {"pb": 6.0}, {}, {})
        self.assertGreater(cheap.score, dear.score)

    def test_pledging_penalised(self):
        clean = score_financial(_lender(), {"pb": 1.5}, {}, {})
        pledged = score_financial(
            _lender(), {"pb": 1.5},
            {"pledge_band": "severe", "promoter_pledge_pct": 60.0}, {})
        self.assertLess(pledged.score, clean.score)

    def test_missing_metrics_do_not_crash(self):
        s = score_financial({"ticker": "X.NS", "is_lender": True}, {}, {}, {})
        self.assertFalse(s.confident)


class TestScoreAnyRouting(unittest.TestCase):
    def test_lender_uses_lender_model(self):
        s = score_any(_lender(), {"pb": 1.5}, {},
                      {"industry": "Banks - Regional"}, nse_is_financial=True)
        self.assertEqual(s.to_dict().get("model"), "financials")

    def test_insurer_gets_generic_model_plus_caveat(self):
        s = score_any(
            _lender(ticker="HDFCLIFE.NS", is_lender=False, roce_mean_5y=15.0,
                    roce_std_5y=3.0, revenue_cagr_5y=12.0, eps_cagr_5y=14.0,
                    cfo_to_net_income_mean=1.1, net_margin_trend=0.2,
                    profitable_years=5, debt_to_equity=0.3,
                    interest_coverage=12.0, debt_to_equity_trend=-0.02,
                    share_count_cagr=0.0, positive_fcf_years=5),
            {"pe_own_pctile": 50.0}, {},
            {"industry": "Insurance - Life"}, nse_is_financial=True)
        self.assertEqual(s.to_dict().get("model"), MODEL_INSURER)
        self.assertTrue(any("Insurer" in w for w in s.warnings))

    def test_fee_financial_gets_generic_model(self):
        s = score_any(
            _lender(ticker="HDFCAMC.NS", is_lender=False, roce_mean_5y=30.0,
                    roce_std_5y=2.0, revenue_cagr_5y=15.0, eps_cagr_5y=16.0,
                    cfo_to_net_income_mean=1.1, net_margin_trend=0.3,
                    profitable_years=5, debt_to_equity=0.05,
                    interest_coverage=50.0, debt_to_equity_trend=0.0,
                    share_count_cagr=0.0, positive_fcf_years=5),
            {"pe_own_pctile": 60.0}, {},
            {"industry": "Asset Management"}, nse_is_financial=True)
        self.assertEqual(s.to_dict().get("model"), MODEL_GENERIC)
        self.assertFalse(any("Insurer" in w for w in s.warnings))

    def test_generic_path_clears_the_lender_flag(self):
        """Otherwise the generic engine nulls the very metrics an asset
        manager legitimately reports."""
        s = score_any(
            _lender(is_lender=True, roce_mean_5y=30.0, roce_std_5y=2.0,
                    revenue_cagr_5y=15.0, eps_cagr_5y=16.0,
                    cfo_to_net_income_mean=1.1, net_margin_trend=0.3,
                    profitable_years=5, debt_to_equity=0.05,
                    interest_coverage=50.0, debt_to_equity_trend=0.0,
                    share_count_cagr=0.0, positive_fcf_years=5),
            {"pe_own_pctile": 60.0}, {},
            {"industry": "Capital Markets"}, nse_is_financial=True)
        self.assertFalse(any("Lender:" in w for w in s.warnings))


if __name__ == "__main__":
    unittest.main()
