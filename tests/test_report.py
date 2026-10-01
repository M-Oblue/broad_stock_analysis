"""Tests for console/CSV reporting.

The report is the product, not the score. A ranked list of numbers invites
sorting and buying the top rows; the score is a summary of reasoning that can be
wrong in ways the number cannot express. These tests therefore check that the
reasoning and -- especially -- the caveats actually survive into the output,
since a suppressed caveat is how a reader ends up trusting a score built on a
loan book nobody verified.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.report.console import (
    to_csv, summary_table, explain, render_report, _risk_plan_line, _fmt, _join,
    CSV_COLUMNS,
)


def _ranked(**overrides):
    base = {
        "rank": 1, "ticker": "TEST.NS", "score": 87.5, "tier": "strong",
        "raw_score": 92.5, "penalty": 5.0, "coverage": 0.87, "confident": True,
        "sector": "Information Technology", "model": "generic",
        "reasons": ["ROCE above 25% (5y avg)", "Net margin expanding"],
        "misses": ["Leverage rising"],
        "penalties": ["Elevated promoter pledging (-5)"],
        "warnings": ["Asset quality is NOT assessed"],
    }
    base.update(overrides)
    return pd.DataFrame([base])


class TestFormatting(unittest.TestCase):
    def test_handles_missing_values(self):
        self.assertEqual(_fmt(None), "n/a")
        self.assertEqual(_fmt(np.nan), "n/a")

    def test_formats_numbers_and_percentages(self):
        self.assertEqual(_fmt(87.456), "87.5")
        self.assertEqual(_fmt(0.87, "{:.0%}"), "87%")

    def test_joins_list_columns_readably(self):
        self.assertEqual(_join(["a", "b"]), "a | b")
        self.assertEqual(_join("plain"), "plain")


class TestSummaryTable(unittest.TestCase):
    def test_renders_rows(self):
        out = summary_table(_ranked())
        self.assertIn("TEST.NS", out)
        self.assertIn("87.5", out)
        self.assertIn("strong", out)

    def test_empty_frame_is_handled(self):
        self.assertIn("no candidates", summary_table(pd.DataFrame()))

    def test_respects_top_limit(self):
        frame = pd.concat([_ranked(ticker=f"T{i}.NS", rank=i)
                           for i in range(1, 11)], ignore_index=True)
        out = summary_table(frame, top=3)
        self.assertIn("T1.NS", out)
        self.assertNotIn("T9.NS", out)


class TestExplain(unittest.TestCase):
    def test_includes_reasons_misses_and_penalties(self):
        out = explain(_ranked().iloc[0])
        self.assertIn("ROCE above 25%", out)
        self.assertIn("Leverage rising", out)
        self.assertIn("Elevated promoter pledging", out)

    def test_caveats_are_never_truncated(self):
        """Caveats are the part a reader skips and most needs to see, so they
        print in full rather than being capped like the reasons list."""
        warnings = [f"caveat number {i}" for i in range(6)]
        out = explain(_ranked(warnings=warnings).iloc[0])
        for warning in warnings:
            self.assertIn(warning, out)

    def test_truncates_long_reason_lists(self):
        reasons = [f"reason {i}" for i in range(20)]
        out = explain(_ranked(reasons=reasons).iloc[0], max_reasons=5)
        self.assertIn("and 15 more", out)

    def test_handles_missing_optional_fields(self):
        row = pd.Series({"ticker": "X.NS", "score": 50.0, "tier": "moderate",
                         "coverage": 0.5})
        self.assertIn("X.NS", explain(row))


class TestRiskPlan(unittest.TestCase):
    def test_formats_stop_and_targets(self):
        line = _risk_plan_line({"stop_loss": 95.0, "target_2r": 110.0,
                                "target_3r": 115.0, "suggested_hold": "4-10 weeks"})
        self.assertIn("stop 95.00", line)
        self.assertIn("2R 110.00", line)
        self.assertIn("4-10 weeks", line)

    def test_absent_when_no_stop(self):
        self.assertIsNone(_risk_plan_line({"ticker": "X"}))


class TestRenderReport(unittest.TestCase):
    def test_states_the_method_for_the_horizon(self):
        out = render_report(_ranked(), "long")
        self.assertIn("LONG TERM (2-5 year hold)", out)
        self.assertIn("mean-revert", out)   # why momentum is excluded

    def test_includes_disclaimer(self):
        self.assertIn("Not investment advice", render_report(_ranked(), "long"))

    def test_risk_off_regime_warns_momentum_horizons(self):
        """A momentum screen in a falling market lists things dropping slightly
        less than everything else -- the reader must be told."""
        regime = {"regime": "risk_off", "rationale": "index below a falling 200DMA"}
        out = render_report(_ranked(), "short", regime=regime)
        self.assertIn("RISK-OFF", out)
        self.assertIn("may still be a losing trade", out)

    def test_risk_off_does_not_warn_long_horizon(self):
        """A 2-5 year investor is not gated by the 200DMA the way a swing
        trader is, so the momentum warning would be noise."""
        regime = {"regime": "risk_off", "rationale": "index below a falling 200DMA"}
        out = render_report(_ranked(), "long", regime=regime)
        self.assertIn("RISK-OFF", out)
        self.assertNotIn("may still be a losing trade", out)

    def test_risk_on_regime_has_no_warning(self):
        regime = {"regime": "risk_on", "rationale": "index above a rising 200DMA"}
        out = render_report(_ranked(), "short", regime=regime)
        self.assertNotIn("may still be a losing trade", out)

    def test_empty_ranking_does_not_crash(self):
        self.assertIn("no candidates", render_report(pd.DataFrame(), "long"))


class TestCsvOutput(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()

    def test_writes_expected_columns_and_flattens_lists(self):
        path = to_csv(_ranked(), Path(self.dir) / "out.csv")
        self.assertTrue(path.exists())
        frame = pd.read_csv(path)
        for column in ("rank", "ticker", "score", "tier", "reasons", "warnings"):
            self.assertIn(column, frame.columns)
        # Lists must be readable in a spreadsheet, not Python repr.
        self.assertNotIn("[", str(frame.loc[0, "reasons"]))
        self.assertIn("|", str(frame.loc[0, "reasons"]))

    def test_creates_parent_directory(self):
        path = to_csv(_ranked(), Path(self.dir) / "nested" / "deep" / "out.csv")
        self.assertTrue(path.exists())

    def test_survives_missing_optional_columns(self):
        minimal = pd.DataFrame([{"rank": 1, "ticker": "X.NS", "score": 50.0}])
        frame = pd.read_csv(to_csv(minimal, Path(self.dir) / "min.csv"))
        self.assertEqual(list(frame["ticker"]), ["X.NS"])


if __name__ == "__main__":
    unittest.main()
