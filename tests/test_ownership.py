"""Tests for NSE ownership/pledging acquisition.

Two field-selection traps dominate this module, and both were verified against
live NSE data before being encoded here:

1. `numSharesPledged` counts non-promoter and NBFC encumbrances, so scoring on
   it red-flags ITC, HDFCBANK, YESBANK and SUZLON -- companies whose promoters
   have pledged nothing. SUZLON reports 943,080,314 encumbered shares against
   0.00% promoter pledging.

2. The pledge endpoint's `percPromoterHolding` contradicts the official
   shareholding-pattern filing (HDFCBANK 13.32 vs 0.00, INFY 20.76 vs 13.82).
   HDFC Bank has no promoter at all post-merger, so the pledge endpoint would
   invent a 13% stake that does not exist.
"""
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import fetch_ownership as FO


def _shp(pairs, symbol="TEST"):
    """Build an SHP-endpoint payload from (date, promoter_pct) pairs."""
    return [{"date": d, "pr_and_prgrp": str(v), "public_val": "50.0",
             "employeeTrusts": "0.1"} for d, v in pairs]


def _pledge(perc_promoter_shares="0.00", num_pledged="1000",
            perc_promoter_holding="50.00"):
    return {"data": [{
        "percPromoterShares": perc_promoter_shares,
        "numSharesPledged": num_pledged,
        "percPromoterHolding": perc_promoter_holding,
        "percSharesPledged": "1.21",
        "shp": "30-Jun-2026",
    }]}


class TestFloatParsing(unittest.TestCase):
    def test_strips_nse_padding(self):
        self.assertEqual(FO._to_float("    20.76"), 20.76)

    def test_handles_thousands_separator(self):
        self.assertEqual(FO._to_float("1,234,567"), 1234567.0)

    def test_missing_markers_become_nan(self):
        for marker in ["-", "", "NA", None, "  "]:
            self.assertTrue(np.isnan(FO._to_float(marker)))


class TestHoldingHistory(unittest.TestCase):
    def test_parses_and_sorts_quarters(self):
        payload = _shp([("30-Jun-2026", 13.82), ("31-Mar-2026", 14.38),
                        ("31-Dec-2025", 14.52)], "INFY")
        out = FO.parse_holding_history(payload, "INFY")
        self.assertEqual(len(out), 3)
        self.assertEqual(out["quarter_end"].tolist(),
                         ["2025-12-31", "2026-03-31", "2026-06-30"])

    def test_skips_unparseable_rows(self):
        payload = _shp([("30-Jun-2026", 13.82)]) + [{"date": "bogus", "pr_and_prgrp": "5"}]
        self.assertEqual(len(FO.parse_holding_history(payload, "T")), 1)

    def test_empty_payload_returns_empty_frame(self):
        self.assertTrue(FO.parse_holding_history([], "T").empty)
        self.assertTrue(FO.parse_holding_history(None, "T").empty)


class TestPledgeFieldSelection(unittest.TestCase):
    """The single most important correctness test in this module."""

    def test_uses_percPromoterShares_not_numSharesPledged(self):
        payload = _pledge(perc_promoter_shares="0.00", num_pledged="943080314")
        out = FO.parse_pledge(payload, "SUZLON")
        self.assertEqual(out["promoter_pledge_pct"], 0.0)
        self.assertEqual(out["total_encumbered_shares"], 943080314.0)

    def test_real_pledging_is_captured(self):
        out = FO.parse_pledge(_pledge(perc_promoter_shares="5.38"), "ZEEL")
        self.assertAlmostEqual(out["promoter_pledge_pct"], 5.38)

    def test_no_disclosure_yields_nan_not_zero(self):
        """Absent data and confirmed-zero are different; the summariser
        resolves them using whether a promoter exists at all."""
        out = FO.parse_pledge({"data": []}, "T")
        self.assertTrue(np.isnan(out["promoter_pledge_pct"]))


class TestSummarise(unittest.TestCase):
    def test_detects_no_promoter_company(self):
        """HDFCBANK: SHP says 0.00 and the pledge endpoint says 13.32.
        The SHP filing wins, and zero promoter is not a red flag."""
        history = FO.parse_holding_history(
            _shp([("31-Mar-2026", 0.0), ("30-Jun-2026", 0.0)]), "HDFCBANK")
        pledge = FO.parse_pledge(
            _pledge(num_pledged="374653401", perc_promoter_holding="13.32"), "HDFCBANK")
        row = FO._summarise(history, pledge, "HDFCBANK")

        self.assertEqual(row["promoter_holding_pct"], 0.0)
        self.assertFalse(row["has_promoter"])
        # Phantom pledging must be nulled, not scored.
        self.assertTrue(np.isnan(row["promoter_pledge_pct"]))
        self.assertAlmostEqual(row["holding_source_disagreement_pp"], 13.32)

    def test_prefers_shp_over_pledge_endpoint_for_holding(self):
        history = FO.parse_holding_history(_shp([("30-Jun-2026", 13.82)]), "INFY")
        pledge = FO.parse_pledge(_pledge(perc_promoter_holding="20.76"), "INFY")
        row = FO._summarise(history, pledge, "INFY")
        self.assertAlmostEqual(row["promoter_holding_pct"], 13.82)
        self.assertAlmostEqual(row["holding_source_disagreement_pp"], 6.94, places=2)

    def test_promoter_exists_with_no_disclosure_means_zero_pledged(self):
        history = FO.parse_holding_history(_shp([("30-Jun-2026", 50.0)]), "T")
        row = FO._summarise(history, FO.parse_pledge({"data": []}, "T"), "T")
        self.assertTrue(row["has_promoter"])
        self.assertEqual(row["promoter_pledge_pct"], 0.0)

    def test_computes_signed_holding_trend(self):
        pairs = [("30-Sep-2025", 75.0), ("31-Dec-2025", 74.5), ("31-Mar-2026", 74.0),
                 ("30-Jun-2026", 73.5), ("30-Sep-2026", 72.0)]
        history = FO.parse_holding_history(_shp(pairs), "T")
        row = FO._summarise(history, FO.parse_pledge(_pledge(), "T"), "T")
        self.assertAlmostEqual(row["promoter_change_1q_pp"], -1.5)   # vs 73.5
        self.assertAlmostEqual(row["promoter_change_4q_pp"], -3.0)   # vs 75.0
        self.assertAlmostEqual(row["promoter_change_max_pp"], -3.0)
        self.assertEqual(row["quarters_of_history"], 5)

    def test_itc_style_residual_stake_counts_as_no_promoter(self):
        """ITC holds 0.02% from legacy holdings -- effectively promoterless."""
        history = FO.parse_holding_history(_shp([("30-Jun-2026", 0.02)]), "ITC")
        row = FO._summarise(history, FO.parse_pledge(_pledge(num_pledged="254334331"), "ITC"), "ITC")
        self.assertFalse(row["has_promoter"])
        self.assertTrue(np.isnan(row["promoter_pledge_pct"]))

    def test_missing_history_is_reported_as_status(self):
        row = FO._summarise(pd.DataFrame(), FO.parse_pledge(_pledge(), "T"), "T")
        self.assertEqual(row["status"], "no_holding_history")
        self.assertTrue(np.isnan(row["promoter_holding_pct"]))

    def test_single_quarter_history_has_no_deltas(self):
        history = FO.parse_holding_history(_shp([("30-Jun-2026", 60.0)]), "T")
        row = FO._summarise(history, FO.parse_pledge(_pledge(), "T"), "T")
        self.assertTrue(np.isnan(row["promoter_change_1q_pp"]))
        self.assertTrue(np.isnan(row["promoter_change_4q_pp"]))
        self.assertEqual(row["promoter_change_max_pp"], 0.0)


class TestFreshness(unittest.TestCase):
    def test_current_schema_and_recent_is_fresh(self):
        row = {"schema_version": FO.SCHEMA_VERSION,
               "fetched_at": date.today().isoformat(), "status": "ok"}
        self.assertTrue(FO._is_fresh(row, 30))

    def test_old_schema_is_stale_even_if_recent(self):
        row = {"schema_version": FO.SCHEMA_VERSION - 1,
               "fetched_at": date.today().isoformat(), "status": "ok"}
        self.assertFalse(FO._is_fresh(row, 30))

    def test_failed_rows_are_retried(self):
        row = {"schema_version": FO.SCHEMA_VERSION,
               "fetched_at": date.today().isoformat(), "status": "no_holding_history"}
        self.assertFalse(FO._is_fresh(row, 30))


if __name__ == "__main__":
    unittest.main()
