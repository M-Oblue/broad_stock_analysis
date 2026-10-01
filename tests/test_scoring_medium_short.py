"""Tests for medium- and short-term scoring engines."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring.medium_term import (
    score_medium_term, W_EARNINGS_MOMENTUM, W_PRICE_MOMENTUM as W_MED_PRICE,
    W_VALUATION, W_QUALITY, W_TREND,
)
from src.scoring.short_term import (
    score_short_term, risk_plan, W_TREND_STRUCTURE, W_PRICE_MOMENTUM,
    W_CONFIRMATION, W_RELATIVE_STRENGTH,
)


def _medium_fundamentals(**kwargs):
    base = {"ticker": "STRONG", "q_revenue_yoy": 28.0, "q_eps_yoy": 35.0,
            "q_revenue_acceleration": 12.0, "q_eps_acceleration": 14.0,
            "roce_mean_5y": 26.0, "debt_to_equity": 0.2,
            "is_lender": False}
    base.update(kwargs)
    return base


def _medium_valuation(**kwargs):
    base = {"pe_own_pctile": 82.0, "pe_sector_z": 1.2, "peg": 0.7,
            "is_loss_making": False, "currency_suppressed": False}
    base.update(kwargs)
    return base


def _medium_technical(**kwargs):
    base = {"symbol": "STRONG", "mom_12_1": 42.0, "ret_6m": 26.0,
            "close_vs_sma200_pct": 12.0, "weekly_close_vs_ema20_pct": 7.0,
            "adx": 31.0, "dist_from_52w_high": -8.0, "rsi": 62.0}
    base.update(kwargs)
    return base


def _short_technical(**kwargs):
    base = {"symbol": "SETUP", "latest_close": 100.0, "atr": 5.0,
            "close_vs_sma20_pct": 4.0, "close_vs_sma50_pct": 8.0,
            "close_vs_sma200_pct": 12.0, "sma20_slope_10d": 2.5,
            "sma50_slope_10d": 1.4, "weekly_close_vs_ema20_pct": 5.0,
            "structure_hh_hl_60d": 1, "adx": 28.0,
            "adx_trajectory_10d": 6.0, "roc_20d": 8.0,
            "roc_60d": 18.0, "ret_3m": 22.0, "rsi": 58.0,
            "macd_vs_signal": 1.5, "volume_vs_20d_avg": 1.6,
            "relative_strength_3m": 9.0, "avg_turnover_20d": 50_000_000.0}
    base.update(kwargs)
    return base


class TestMediumTermScoring(unittest.TestCase):
    def test_strong_candidate_scores_high_and_weak_scores_low(self):
        strong = score_medium_term(_medium_fundamentals(), _medium_valuation(), {},
                                   _medium_technical())
        weak = score_medium_term(
            _medium_fundamentals(q_revenue_yoy=-5, q_eps_yoy=-20,
                                 q_revenue_acceleration=-12, q_eps_acceleration=-18,
                                 roce_mean_5y=7, debt_to_equity=2.2),
            _medium_valuation(pe_own_pctile=5, pe_sector_z=-1.2, peg=4.0), {},
            _medium_technical(mom_12_1=-18, ret_6m=-12, close_vs_sma200_pct=-10,
                              weekly_close_vs_ema20_pct=-6, adx=11))
        self.assertGreater(strong.score, 85)
        self.assertLess(weak.score, 25)

    def test_missing_metrics_are_skipped_not_zero_scored(self):
        """Individual missing metrics leave the denominator alone, so a
        partially-measured company is not punished for data we lack.

        The valuation block is the deliberate exception: when it is wholly
        unmeasurable it scores NEUTRAL rather than being skipped, which is why
        this lands near 82 rather than ~100. Skipping a fifth of the score
        would let a data gap outrank demonstrated cheapness -- see
        base.valuation_unmeasurable.
        """
        partial = score_medium_term(
            {"ticker": "PART", "q_revenue_acceleration": 15.0,
             "q_eps_acceleration": 18.0, "is_lender": False},
            {}, {}, {"symbol": "PART", "mom_12_1": 40.0})
        self.assertGreater(partial.raw_score, 75)
        self.assertLess(partial.coverage, 0.5)
        self.assertIn("Quarterly revenue growth", partial.skipped)
        self.assertTrue(any("neutral" in w for w in partial.warnings))

    def test_unmeasurable_valuation_ranks_below_demonstrated_cheapness(self):
        """The INFY failure mode, in the medium engine: a suppressed valuation
        scored 97.4 against 96.4 for a stock whose cheapness was measured."""
        cheap = score_medium_term(_medium_fundamentals(), _medium_valuation(),
                                  {}, _medium_technical())
        suppressed = score_medium_term(
            _medium_fundamentals(),
            _medium_valuation(currency_suppressed=True), {}, _medium_technical())
        expensive = score_medium_term(
            _medium_fundamentals(),
            _medium_valuation(pe_own_pctile=5.0, pe_sector_z=-2.0, peg=6.0),
            {}, _medium_technical())
        self.assertGreater(cheap.score, suppressed.score)
        self.assertGreater(suppressed.score, expensive.score)

    def test_penalties_reduce_final_score_and_are_reported(self):
        clean = score_medium_term(_medium_fundamentals(), _medium_valuation(), {},
                                  _medium_technical())
        penalised = score_medium_term(
            _medium_fundamentals(), _medium_valuation(is_loss_making=True),
            {"pledge_band": "severe", "promoter_pledge_pct": 30.0,
             "trend_band": "selling_heavily", "promoter_change_4q_pp": -6.0},
            _medium_technical(dist_from_52w_high=-1.0, rsi=74.0))
        out = penalised.to_dict()
        self.assertLess(penalised.score, clean.score)
        self.assertTrue(any("Severe promoter pledging" in p for p in out["penalties"]))
        self.assertTrue(any("Overextended" in p for p in out["penalties"]))

    def test_12_1_momentum_drives_price_momentum_component(self):
        high = score_medium_term(_medium_fundamentals(), _medium_valuation(), {},
                                 _medium_technical(mom_12_1=45.0, ret_6m=12.0))
        low = score_medium_term(_medium_fundamentals(), _medium_valuation(), {},
                                _medium_technical(mom_12_1=-20.0, ret_6m=12.0))
        self.assertGreater(high.raw_score - low.raw_score, 12.0)

    def test_lender_uses_roe_without_crashing(self):
        lender = score_medium_term(
            _medium_fundamentals(is_lender=True, roce_mean_5y=None,
                                 roe_mean_5y=17.0, debt_to_equity=None),
            _medium_valuation(), {}, _medium_technical())
        self.assertGreater(lender.score, 70)
        self.assertTrue(any("ROE" in r for r in lender.reasons))


class TestShortTermScoring(unittest.TestCase):
    def test_strong_candidate_scores_high_and_weak_scores_low(self):
        strong = score_short_term({}, {}, {}, _short_technical())
        weak = score_short_term({}, {}, {}, _short_technical(
            close_vs_sma20_pct=-6, close_vs_sma50_pct=-10, close_vs_sma200_pct=-20,
            sma20_slope_10d=-2, sma50_slope_10d=-1, weekly_close_vs_ema20_pct=-8,
            structure_hh_hl_60d=0, adx=10, adx_trajectory_10d=-5,
            roc_20d=-8, roc_60d=-18, ret_3m=-22, rsi=35,
            macd_vs_signal=-2, volume_vs_20d_avg=0.5, relative_strength_3m=-12,
            avg_turnover_20d=50_000_000.0))
        self.assertGreater(strong.score, 85)
        self.assertLess(weak.score, 20)

    def test_missing_metrics_are_skipped_not_zero_scored(self):
        partial = score_short_term({}, {}, {}, {"symbol": "PART", "latest_close": 100,
                                                "atr": 4, "rsi": 55,
                                                "macd_vs_signal": 1.0})
        self.assertGreater(partial.raw_score, 95)
        self.assertLess(partial.coverage, 0.25)
        self.assertIn("Price vs SMA20", partial.skipped)

    def test_overbought_and_blowoff_penalise_rather_than_reward(self):
        normal = score_short_term({}, {}, {}, _short_technical())
        hot = score_short_term({}, {}, {}, _short_technical(rsi=82.0,
                                                            volume_vs_20d_avg=4.2))
        out = hot.to_dict()
        self.assertLess(hot.score, normal.score)
        self.assertTrue(any("overbought" in p.lower() for p in out["penalties"]))
        self.assertTrue(any("Blow-off" in p for p in out["penalties"]))

    def test_risk_plan_arithmetic(self):
        plan = risk_plan({"latest_close": 100.0, "atr": 5.0})
        self.assertAlmostEqual(plan["stop_loss"], 90.0)
        self.assertAlmostEqual(plan["risk_per_share"], 10.0)
        self.assertAlmostEqual(plan["target_2r"], 120.0)
        self.assertAlmostEqual(plan["target_3r"], 130.0)
        self.assertEqual(plan["suggested_hold"], "4-10 weeks")

    def test_risk_plan_is_attached_to_score(self):
        out = score_short_term({}, {}, {}, _short_technical()).to_dict()
        self.assertEqual(out["stop_loss"], 90.0)
        self.assertEqual(out["target_2r"], 120.0)


class TestWeights(unittest.TestCase):
    def test_medium_weights_sum_to_100(self):
        self.assertEqual(W_EARNINGS_MOMENTUM + W_MED_PRICE + W_VALUATION
                         + W_QUALITY + W_TREND, 100.0)

    def test_short_weights_sum_to_100(self):
        self.assertEqual(W_TREND_STRUCTURE + W_PRICE_MOMENTUM + W_CONFIRMATION
                         + W_RELATIVE_STRENGTH, 100.0)


if __name__ == "__main__":
    unittest.main()
