import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.backtest.price_backtest import (
    walk_forward, information_coefficient, quantile_spread, summarise, _verdict,
)


def _records(scores, returns, date="2024-01-31"):
    return pd.DataFrame({
        "rebalance_date": [pd.Timestamp(date)] * len(scores),
        "symbol": [f"S{i}" for i in range(len(scores))],
        "score": scores,
        "forward_return": returns,
        "rank": np.arange(1, len(scores) + 1),
        "selected": [i < 2 for i in range(len(scores))],
    })


class TestWalkForward(unittest.TestCase):
    def test_no_lookahead_future_spike_does_not_influence_score(self):
        dates = pd.date_range("2024-01-01", "2024-03-15", freq="B")
        rows = []
        for d in dates:
            rows.append({"symbol": "A", "date": d, "open": 10, "high": 10, "low": 10, "close": 10, "volume": 1})
            close = 1000 if d == pd.Timestamp("2024-02-05") else 10
            rows.append({"symbol": "B", "date": d, "open": close, "high": close, "low": close, "close": close, "volume": 1})
        seen_max_dates = []

        def score_fn(history):
            seen_max_dates.append(history["date"].max())
            return history["close"].max()

        recs = walk_forward(pd.DataFrame(rows), score_fn, rebalance_freq="M",
                            forward_days=3, top_n=1, min_bars=5)
        jan_b = recs[(recs["rebalance_date"] == pd.Timestamp("2024-01-31")) & (recs["symbol"] == "B")].iloc[0]
        self.assertEqual(jan_b["score"], 10)
        self.assertGreater(jan_b["forward_return"], 1000)
        self.assertTrue(all(d <= pd.Timestamp("2024-02-29") for d in seen_max_dates))

    def test_walk_forward_selects_top_n_but_keeps_universe_records(self):
        dates = pd.date_range("2024-01-01", periods=45, freq="B")
        rows = []
        for sym, start in (("A", 10), ("B", 20), ("C", 30)):
            for i, d in enumerate(dates):
                close = start + i
                rows.append({"symbol": sym, "date": d, "close": close})
        recs = walk_forward(pd.DataFrame(rows), lambda h: h["close"].iloc[-1],
                            rebalance_freq="M", forward_days=3, top_n=2, min_bars=5)
        jan = recs[recs["rebalance_date"] == pd.Timestamp("2024-01-31")]
        self.assertEqual(len(jan), 3)
        self.assertEqual(int(jan["selected"].sum()), 2)


class TestBacktestAnalytics(unittest.TestCase):
    def test_information_coefficient_predictive_and_random(self):
        predictive = information_coefficient(_records([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
        randomish = information_coefficient(_records([1, 2, 3, 4, 5], [2, 5, 1, 4, 3]))
        self.assertGreater(predictive["overall"], 0.9)
        self.assertLess(abs(randomish["overall"]), 0.3)

    def test_quantile_spread_ordering(self):
        recs = _records(list(range(10)), list(range(10)))
        spread = quantile_spread(recs, quantiles=5)
        self.assertTrue(spread["monotonic"])
        self.assertGreater(spread["top_minus_bottom"], 0)
        self.assertEqual(set(spread["by_quantile"].keys()), {1, 2, 3, 4, 5})

    def test_verdict_thresholds_match_real_equity_ics(self):
        """Verdict bands are calibrated to what an IC actually looks like on
        liquid equities: 0.02 useful, 0.05 good, 0.10 strong. An earlier
        version required 0.15, which reported "no predictive power" for the
        medium engine despite a mean IC of 0.074, 61% positive periods,
        monotonic quintiles and +8.5pp over the universe -- a false negative on
        the best-validated model in the repo.

        Tested directly rather than through hand-built permutations: with only
        a handful of points a "random" ordering routinely produces an IC of
        0.4+, so a permutation cannot reliably represent noise.
        """
        monotonic = {"monotonic": True, "top_minus_bottom": 12.0}
        flat = {"monotonic": False, "top_minus_bottom": 0.0}

        self.assertIn("no predictive power", _verdict(0.005, 50.0, flat, 0.1))
        self.assertIn("weak but positive", _verdict(0.03, 58.0, monotonic, 3.0))
        self.assertIn("useful edge", _verdict(0.074, 61.1, monotonic, 8.5))
        self.assertIn("strong edge", _verdict(0.15, 70.0, monotonic, 12.0))
        self.assertIn("NEGATIVELY predictive",
                      _verdict(-0.03, 37.8, flat, -1.0))

    def test_verdict_flags_thin_corroboration_as_provisional(self):
        """A positive IC with nothing else supporting it is not the same as one
        confirmed by monotonic quintiles and consistency across periods."""
        thin = _verdict(0.06, 50.0, {"monotonic": False}, -0.5)
        self.assertIn("provisional", thin)
        strong = _verdict(0.06, 65.0, {"monotonic": True}, 5.0)
        self.assertNotIn("provisional", strong)

    def test_verdict_handles_missing_inputs(self):
        self.assertIn("Not enough data",
                      _verdict(float("nan"), float("nan"), {}, float("nan")))

    def test_summary_reports_all_fields(self):
        text = summarise(_records([1, 2, 3, 4, 5, 6], [3, 1, 5, 2, 6, 4]))
        for field in ("Periods:", "Mean IC:", "Overall IC:",
                      "Quintile monotonic:", "Top-N mean return:"):
            self.assertIn(field, text)

    def test_empty_and_degenerate_inputs_do_not_crash(self):
        self.assertTrue(walk_forward(pd.DataFrame(), lambda h: 1).empty)
        self.assertTrue(np.isnan(information_coefficient([])["overall"]))
        self.assertEqual(quantile_spread([])["by_quantile"], {})
        text = summarise([])
        self.assertIn("No walk-forward records", text)
        const = information_coefficient(_records([1, 1, 1], [1, 2, 3]))
        self.assertTrue(np.isnan(const["overall"]))


if __name__ == "__main__":
    unittest.main()
