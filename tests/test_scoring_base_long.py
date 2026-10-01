"""Tests for the shared scoring engine and the long-term (2-5y) model.

The normalisation tests matter most. Scoring a missing metric as zero would
conflate "this company is bad" with "we could not measure it", and would push
recently-listed companies and lenders -- which legitimately lack whole
categories of metric -- to the bottom of every screen.

But the opposite failure is just as real, and was observed live: INFY topped the
long-term screen at 94/100 precisely BECAUSE its valuation is
currency-suppressed, so the one dimension that might have held it back never
applied. Normalisation had turned "we cannot price it" into "it is priced well".
Both behaviours are pinned down below.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.scoring.base import (
    Score, conviction_tier, rank_scores, _missing,
    MIN_COVERAGE_FOR_CONFIDENCE,
)
from src.scoring.long_term import (
    score_long_term, _valuation_unmeasurable,
    W_QUALITY, W_GROWTH, W_HEALTH, W_VALUATION,
)


def _good_fundamentals(**overrides):
    base = {
        "ticker": "GOOD.NS", "is_lender": False, "years_of_data": 5,
        "roce_mean_5y": 28.0, "roce_std_5y": 2.0, "roce_latest": 29.0,
        "roe_mean_5y": 25.0, "roe_latest": 26.0,
        "net_margin_trend": 0.8, "cfo_to_net_income_mean": 1.2,
        "revenue_cagr_5y": 18.0, "eps_cagr_5y": 22.0,
        "profitable_years": 5, "positive_fcf_years": 5,
        "debt_to_equity": 0.15, "interest_coverage": 25.0,
        "debt_to_equity_trend": -0.08, "share_count_cagr": 0.0,
    }
    base.update(overrides)
    return base


def _good_valuation(**overrides):
    base = {"pe_own_pctile": 80.0, "pe_sector_z": 1.2, "fcf_yield": 7.0,
            "peg": 0.9, "currency_suppressed": False, "is_loss_making": False}
    base.update(overrides)
    return base


class TestMissingDetection(unittest.TestCase):
    def test_detects_none_and_nan(self):
        self.assertTrue(_missing(None))
        self.assertTrue(_missing(np.nan))
        self.assertTrue(_missing(pd.NaT))

    def test_real_values_are_present(self):
        self.assertFalse(_missing(0))      # zero is a value, not absence
        self.assertFalse(_missing(0.0))
        self.assertFalse(_missing(False))


class TestScoreNormalisation(unittest.TestCase):
    def test_unavailable_criterion_leaves_denominator_untouched(self):
        """The core rule: we must not punish a company for data we lack."""
        s = Score("T", "long")
        s.add(True, 50, "earned")
        s.add(False, 50, "missing", available=False)
        self.assertEqual(s.available, 50)
        self.assertAlmostEqual(s.raw_score, 100.0)

    def test_failed_criterion_does_count_against(self):
        s = Score("T", "long")
        s.add(True, 50, "earned")
        s.add(False, 50, "failed", miss="did not meet")
        self.assertAlmostEqual(s.raw_score, 50.0)

    def test_add_metric_skips_on_missing_input(self):
        s = Score("T", "long")
        s.add_metric(np.nan, 40, lambda v: v > 0, "metric")
        self.assertEqual(s.available, 0)
        self.assertIn("metric", s.skipped)

    def test_coverage_reflects_how_much_applied(self):
        s = Score("T", "long")
        s.add(True, 25, "a")
        s.add(False, 25, "b", miss="b missed")
        s.add_metric(np.nan, 25, lambda v: True, "c")
        s.add_metric(np.nan, 25, lambda v: True, "d")
        self.assertAlmostEqual(s.coverage, 0.5)
        self.assertFalse(s.confident)

    def test_full_coverage_is_confident(self):
        s = Score("T", "long")
        s.add(True, 50, "a")
        s.add(True, 50, "b")
        self.assertAlmostEqual(s.coverage, 1.0)
        self.assertTrue(s.confident)

    def test_no_available_criteria_yields_nan(self):
        s = Score("T", "long")
        s.add_metric(np.nan, 50, lambda v: True, "a")
        self.assertTrue(np.isnan(s.raw_score))
        self.assertTrue(np.isnan(s.score))


class TestBandedScoring(unittest.TestCase):
    def test_first_matching_band_wins(self):
        s = Score("T", "long")
        s.add_banded(30.0, 100, [
            (lambda v: v >= 25, 1.0, "excellent"),
            (lambda v: v >= 15, 0.5, "ok"),
            (lambda v: True, 0.0, "poor")], "test")
        self.assertAlmostEqual(s.earned, 100.0)
        self.assertIn("excellent", s.reasons)

    def test_partial_credit_at_middle_band(self):
        s = Score("T", "long")
        s.add_banded(18.0, 100, [
            (lambda v: v >= 25, 1.0, "excellent"),
            (lambda v: v >= 15, 0.5, "ok"),
            (lambda v: True, 0.0, "poor")], "test")
        self.assertAlmostEqual(s.earned, 50.0)

    def test_zero_band_records_a_miss_not_a_reason(self):
        s = Score("T", "long")
        s.add_banded(5.0, 100, [
            (lambda v: v >= 25, 1.0, "excellent"),
            (lambda v: True, 0.0, "poor")], "test")
        self.assertEqual(s.earned, 0.0)
        self.assertIn("poor", s.misses)
        self.assertEqual(s.reasons, [])

    def test_missing_value_is_skipped(self):
        s = Score("T", "long")
        s.add_banded(np.nan, 100, [(lambda v: True, 1.0, "any")], "test")
        self.assertEqual(s.available, 0)


class TestPenalties(unittest.TestCase):
    def test_penalty_subtracts_outside_the_average(self):
        """A pledged promoter must not be offset by a good margin trend."""
        s = Score("T", "long")
        s.add(True, 100, "perfect on every criterion")
        self.assertAlmostEqual(s.raw_score, 100.0)
        s.penalise(15, "severe pledging")
        self.assertAlmostEqual(s.score, 85.0)

    def test_score_floors_at_zero(self):
        s = Score("T", "long")
        s.add(True, 100, "fine")
        s.penalise(150, "catastrophic")
        self.assertEqual(s.score, 0.0)

    def test_penalties_appear_in_output(self):
        s = Score("T", "long")
        s.add(True, 100, "fine")
        s.penalise(10, "pledging")
        self.assertIn("pledging (-10)", s.to_dict()["penalties"])


class TestConvictionTier(unittest.TestCase):
    def test_tiers_by_score(self):
        self.assertEqual(conviction_tier(85), "strong")
        self.assertEqual(conviction_tier(70), "good")
        self.assertEqual(conviction_tier(55), "moderate")
        self.assertEqual(conviction_tier(40), "weak")
        self.assertEqual(conviction_tier(20), "avoid")

    def test_low_confidence_overrides_a_high_score(self):
        self.assertEqual(conviction_tier(90, confident=False), "insufficient data")

    def test_nan_is_unrated(self):
        self.assertEqual(conviction_tier(np.nan), "unrated")


class TestLongTermWeights(unittest.TestCase):
    def test_weights_sum_to_100(self):
        self.assertAlmostEqual(W_QUALITY + W_GROWTH + W_HEALTH + W_VALUATION, 100.0)


class TestLongTermScoring(unittest.TestCase):
    def test_high_quality_compounder_scores_well(self):
        s = score_long_term(_good_fundamentals(), _good_valuation())
        self.assertGreater(s.score, 80)
        self.assertTrue(s.confident)

    def test_weak_company_scores_poorly(self):
        weak = _good_fundamentals(
            roce_mean_5y=5.0, roce_std_5y=15.0, revenue_cagr_5y=1.0,
            eps_cagr_5y=-8.0, cfo_to_net_income_mean=0.4,
            debt_to_equity=3.0, interest_coverage=1.2,
            debt_to_equity_trend=0.3, net_margin_trend=-2.0,
            profitable_years=2, positive_fcf_years=0)
        s = score_long_term(weak, _good_valuation(pe_own_pctile=10.0,
                                                  pe_sector_z=-1.5,
                                                  fcf_yield=0.2, peg=5.0))
        self.assertLess(s.score, 30)

    def test_excludes_momentum_and_technical_signals(self):
        """At 3-5 years equities mean-revert, so importing 12-month momentum
        from the medium-term engine would be actively harmful."""
        base = score_long_term(_good_fundamentals(), _good_valuation())
        with_tech = score_long_term(
            _good_fundamentals(), _good_valuation(),
            technical={"mom_12_1": 95.0, "rsi": 72.0, "adx": 40.0})
        self.assertAlmostEqual(base.score, with_tech.score)

    def test_pledging_penalises_an_otherwise_excellent_company(self):
        clean = score_long_term(_good_fundamentals(), _good_valuation())
        pledged = score_long_term(
            _good_fundamentals(), _good_valuation(),
            governance={"pledge_band": "severe", "promoter_pledge_pct": 80.0,
                        "trend_band": "stable"})
        self.assertLess(pledged.score, clean.score)
        self.assertTrue(any("pledging" in p.lower() for p in pledged.to_dict()["penalties"]))

    def test_promoter_selldown_penalised(self):
        s = score_long_term(
            _good_fundamentals(), _good_valuation(),
            governance={"pledge_band": "none", "trend_band": "selling_heavily",
                        "promoter_change_4q_pp": -8.0})
        self.assertGreater(s.penalty_total, 0)

    def test_lender_skips_roce_and_leverage_criteria(self):
        s = score_long_term(
            _good_fundamentals(is_lender=True, roce_mean_5y=np.nan,
                               roce_std_5y=np.nan, interest_coverage=np.nan),
            _good_valuation())
        self.assertFalse(np.isnan(s.score))
        self.assertTrue(any("Lender" in w for w in s.warnings))

    def test_accruals_failure_penalised(self):
        s = score_long_term(_good_fundamentals(cfo_to_net_income_mean=0.3),
                            _good_valuation())
        self.assertTrue(any("cash" in p.lower() for p in s.to_dict()["penalties"]))


class TestUnmeasurableValuation(unittest.TestCase):
    """The INFY case: a data gap must not win the screen."""

    def test_currency_suppression_detected(self):
        self.assertTrue(_valuation_unmeasurable({"currency_suppressed": True}))

    def test_all_inputs_missing_detected(self):
        self.assertTrue(_valuation_unmeasurable(
            {"pe_own_pctile": np.nan, "pe_sector_z": np.nan,
             "fcf_yield": np.nan, "peg": np.nan}))

    def test_partial_valuation_is_measurable(self):
        self.assertFalse(_valuation_unmeasurable(
            {"pe_own_pctile": np.nan, "pe_sector_z": 0.5,
             "fcf_yield": np.nan, "peg": np.nan}))

    def test_suppressed_valuation_scores_neutral_not_skipped(self):
        """Scored neutral, so an excellent business with unmeasurable
        valuation lands below one whose cheapness is demonstrated."""
        cheap = score_long_term(_good_fundamentals(), _good_valuation())
        suppressed = score_long_term(
            _good_fundamentals(),
            _good_valuation(currency_suppressed=True))
        self.assertLess(suppressed.score, cheap.score)
        self.assertTrue(any("neutral" in w for w in suppressed.warnings))

    def test_suppressed_still_beats_a_demonstrably_expensive_stock(self):
        expensive = score_long_term(
            _good_fundamentals(),
            _good_valuation(pe_own_pctile=5.0, pe_sector_z=-2.0,
                            fcf_yield=0.1, peg=6.0))
        suppressed = score_long_term(
            _good_fundamentals(), _good_valuation(currency_suppressed=True))
        self.assertGreater(suppressed.score, expensive.score)


class TestRanking(unittest.TestCase):
    def test_ranks_descending_and_assigns_tiers(self):
        rows = [Score("A", "long"), Score("B", "long")]
        rows[0].add(True, 100, "great")
        rows[1].add(False, 100, "poor", miss="poor")
        frame = rank_scores([r.to_dict() for r in rows])
        self.assertEqual(list(frame["ticker"]), ["A", "B"])
        self.assertEqual(list(frame["rank"]), [1, 2])
        self.assertEqual(frame.loc[0, "tier"], "strong")

    def test_empty_input(self):
        self.assertTrue(rank_scores([]).empty)


if __name__ == "__main__":
    unittest.main()
