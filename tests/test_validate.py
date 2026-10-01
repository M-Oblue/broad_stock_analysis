"""Tests for the data-quality validation gates.

The currency-mismatch tests carry the most weight. That bug is the archetype of
what this layer exists to catch: only 2 of 500 tickers affected, no exception
raised, and the corrupted value (EV/EBITDA of 936 instead of 10.6) is extreme
enough to dominate any screen that sorts on it.
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.validate import (
    ValidationReport, check_currency_mismatch, check_statement_coverage,
    check_staleness, check_value_sanity, check_info_sanity,
    check_price_coverage, validate_all, PRICE_OVER_STATEMENT_FIELDS,
)


def _info(rows):
    base = {
        "ticker": None, "currency": "INR", "financialCurrency": "INR",
        "sector": "Technology", "marketCap": 1e12, "trailingPE": 25.0,
        "priceToBook": 4.0, "enterpriseToEbitda": 15.0, "status": "ok",
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _statements(ticker="INFY.NS", period_end="2026-03-31", **items):
    defaults = {"total_revenue": 1e11, "net_income": 1.5e10,
                "total_assets": 2e11, "stockholders_equity": 8e10}
    defaults.update(items)
    return pd.DataFrame([
        {"ticker": ticker, "statement": "income", "period_type": "annual",
         "period_end": period_end, "item": k, "value": v}
        for k, v in defaults.items()
    ])


class TestCurrencyMismatch(unittest.TestCase):
    def test_detects_usd_statements_with_inr_price(self):
        info = _info([
            {"ticker": "INFY.NS", "financialCurrency": "USD",
             "enterpriseToEbitda": 936.0},
            {"ticker": "TCS.NS"},
        ])
        report = ValidationReport()
        bad = check_currency_mismatch(info, report)
        self.assertEqual(list(bad["ticker"]), ["INFY.NS"])
        self.assertFalse(report.ok)
        self.assertIn("INFY.NS", report.suppressed)

    def test_suppresses_only_price_over_statement_ratios(self):
        """ROCE/margins/CAGR divide statement by statement, so currency
        cancels -- they must stay usable. Only price/statement ratios break."""
        info = _info([{"ticker": "INFY.NS", "financialCurrency": "USD"}])
        report = ValidationReport()
        check_currency_mismatch(info, report)
        suppressed = report.suppressed["INFY.NS"]
        for field in ["trailingPE", "priceToBook", "enterpriseToEbitda"]:
            self.assertIn(field, suppressed)
        for safe in ["returnOnEquity", "profitMargins", "debtToEquity"]:
            self.assertNotIn(safe, suppressed)

    def test_clean_universe_produces_no_error(self):
        info = _info([{"ticker": "TCS.NS"}, {"ticker": "WIPRO.NS"}])
        report = ValidationReport()
        bad = check_currency_mismatch(info, report)
        self.assertTrue(bad.empty)
        self.assertTrue(report.ok)
        self.assertEqual(report.suppressed, {})

    def test_missing_currency_columns_warns_not_crashes(self):
        report = ValidationReport()
        out = check_currency_mismatch(pd.DataFrame({"ticker": ["A"]}), report)
        self.assertTrue(out.empty)
        self.assertTrue(report.warnings)

    def test_null_currency_rows_are_skipped_not_flagged(self):
        info = _info([{"ticker": "A.NS", "financialCurrency": None}])
        report = ValidationReport()
        self.assertTrue(check_currency_mismatch(info, report).empty)
        self.assertTrue(report.ok)

    def test_error_message_names_the_safe_metrics(self):
        """The report must tell the user what is still usable, or they will
        discard the whole ticker."""
        info = _info([{"ticker": "INFY.NS", "financialCurrency": "USD"}])
        report = ValidationReport()
        check_currency_mismatch(info, report)
        self.assertIn("ROCE", report.errors[0])


class TestCoverageAndStaleness(unittest.TestCase):
    def test_reports_missing_statement_tickers(self):
        stmts = _statements("INFY.NS")
        report = ValidationReport()
        check_statement_coverage(stmts, _info([{"ticker": "INFY.NS"}]),
                                 ["INFY.NS", "TATAMOTORS.NS"], report)
        self.assertTrue(any("TATAMOTORS.NS" in w for w in report.warnings))

    def test_full_coverage_produces_no_warning(self):
        stmts = _statements("INFY.NS")
        report = ValidationReport()
        check_statement_coverage(stmts, _info([{"ticker": "INFY.NS"}]),
                                 ["INFY.NS"], report)
        self.assertEqual(report.warnings, [])

    def test_flags_stale_annual_statements(self):
        old = (date.today() - timedelta(days=900)).isoformat()
        report = ValidationReport()
        check_staleness(_statements("OLD.NS", period_end=old), report)
        self.assertTrue(any("stale" in w for w in report.warnings))

    def test_recent_statements_not_flagged(self):
        recent = (date.today() - timedelta(days=120)).isoformat()
        report = ValidationReport()
        check_staleness(_statements("NEW.NS", period_end=recent), report)
        self.assertEqual(report.warnings, [])


class TestValueSanity(unittest.TestCase):
    def test_negative_revenue_is_an_error(self):
        report = ValidationReport()
        check_value_sanity(_statements("BAD.NS", total_revenue=-5e10), report)
        self.assertTrue(any("negative revenue" in e for e in report.errors))

    def test_nonpositive_assets_is_an_error(self):
        report = ValidationReport()
        check_value_sanity(_statements("BAD.NS", total_assets=0.0), report)
        self.assertTrue(any("total assets" in e for e in report.errors))

    def test_negative_equity_warns_because_roe_breaks(self):
        report = ValidationReport()
        check_value_sanity(_statements("LEV.NS", stockholders_equity=-1e10), report)
        self.assertTrue(any("negative shareholders" in w for w in report.warnings))

    def test_absurd_margin_is_flagged(self):
        report = ValidationReport()
        check_value_sanity(
            _statements("WEIRD.NS", total_revenue=1e6, net_income=1e10), report)
        self.assertTrue(any("margins outside" in w for w in report.warnings))

    def test_healthy_statements_are_clean(self):
        report = ValidationReport()
        check_value_sanity(_statements("GOOD.NS"), report)
        self.assertTrue(report.ok)
        self.assertEqual(report.warnings, [])


class TestInfoSanity(unittest.TestCase):
    def test_nonpositive_market_cap_is_an_error(self):
        report = ValidationReport()
        check_info_sanity(_info([{"ticker": "BAD.NS", "marketCap": 0}]), report)
        self.assertFalse(report.ok)

    def test_negative_pe_is_noted_not_errored(self):
        """Loss-making is legitimate; the risk is ranking it as 'cheap'."""
        report = ValidationReport()
        check_info_sanity(_info([{"ticker": "LOSS.NS", "trailingPE": -12.0}]), report)
        self.assertTrue(report.ok)
        self.assertTrue(any("negative trailing P/E" in n for n in report.info))

    def test_failed_status_warns(self):
        report = ValidationReport()
        check_info_sanity(_info([{"ticker": "X.NS", "status": "no_statements"}]), report)
        self.assertTrue(any("no_statements" in w for w in report.warnings))


class TestPriceCoverage(unittest.TestCase):
    def test_flags_recently_listed_symbols(self):
        prices = pd.concat([
            pd.DataFrame({"symbol": "OLD.NS", "date": pd.bdate_range("2021-01-01", periods=600)}),
            pd.DataFrame({"symbol": "NEW.NS", "date": pd.bdate_range("2026-01-01", periods=100)}),
        ], ignore_index=True)
        report = ValidationReport()
        check_price_coverage(prices, min_bars=500, report=report)
        self.assertTrue(any("NEW.NS" in w for w in report.warnings))
        self.assertFalse(any("OLD.NS" in w for w in report.warnings))

    def test_empty_prices_warns(self):
        report = ValidationReport()
        check_price_coverage(pd.DataFrame(), 500, report)
        self.assertTrue(report.warnings)


class TestValidateAll(unittest.TestCase):
    def test_runs_end_to_end_and_aggregates(self):
        info = _info([{"ticker": "INFY.NS", "financialCurrency": "USD"},
                      {"ticker": "TCS.NS"}])
        stmts = pd.concat([_statements("INFY.NS"), _statements("TCS.NS")],
                          ignore_index=True)
        prices = pd.DataFrame({
            "symbol": "INFY.NS", "date": pd.bdate_range("2021-01-01", periods=600)})
        report = validate_all(statements=stmts, info=info, prices=prices,
                              expected_tickers=["INFY.NS", "TCS.NS"])
        self.assertFalse(report.ok)                 # currency mismatch
        self.assertIn("INFY.NS", report.suppressed)
        self.assertIn("[FAIL]", report.render())

    def test_clean_inputs_report_ok(self):
        info = _info([{"ticker": "TCS.NS"}])
        stmts = _statements("TCS.NS")
        report = validate_all(statements=stmts, info=info,
                              expected_tickers=["TCS.NS"])
        self.assertTrue(report.ok)

    def test_handles_all_empty_inputs(self):
        report = validate_all()
        self.assertTrue(report.ok)
        self.assertIn("No data-quality issues", report.render())


if __name__ == "__main__":
    unittest.main()
