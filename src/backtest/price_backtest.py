#!/usr/bin/env python3
"""Walk-forward price-signal validation for short and medium horizons only.

Fundamental backtests are deliberately NOT implemented here because yfinance
provides no point-in-time fundamentals. Replaying today's financial statements
through past dates would carry look-ahead and survivorship bias. Long-horizon
(2-5y) validation is deliberately NOT implemented either: five years of daily
price data yields too few independent multi-year observations to distinguish
skill from luck. A backtest that cannot be trusted is worse than none because it
lends false confidence to the weights.

This module also does not model slippage, transaction costs or position sizing.
It is a signal-validation tool: at each rebalance it scores using only bars that
existed on that date, then measures the later return. Portfolio simulation is a
separate problem with different assumptions.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr


PRICE_COLUMNS = ("symbol", "date", "close")


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty or not set(PRICE_COLUMNS).issubset(prices.columns):
        return pd.DataFrame(columns=list(prices.columns) if prices is not None else list(PRICE_COLUMNS))
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "date", "close"])
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def _score_value(result) -> float:
    if hasattr(result, "score"):
        value = result.score
    elif isinstance(result, dict) and "score" in result:
        value = result["score"]
    else:
        value = result
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan


def _rebalance_dates(dates: pd.Series, freq: str) -> list[pd.Timestamp]:
    if dates.empty:
        return []
    unique = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    periods = unique.dt.to_period(freq)
    return unique.groupby(periods).max().tolist()


def walk_forward(prices: pd.DataFrame, score_fn, rebalance_freq: str = "M",
                 forward_days: int = 63, top_n: int = 20,
                 min_bars: int = 250) -> pd.DataFrame:
    """Score each symbol at each rebalance using only data available then."""
    frame = _clean_prices(prices)
    if frame.empty or forward_days <= 0:
        return pd.DataFrame(columns=["rebalance_date", "symbol", "score", "forward_return", "rank", "selected"])

    by_symbol = {symbol: sub.reset_index(drop=True)
                 for symbol, sub in frame.groupby("symbol", sort=True)}
    rebalances = _rebalance_dates(frame["date"], rebalance_freq)
    records = []

    for rebalance_date in rebalances:
        period_records = []
        for symbol, sub in by_symbol.items():
            dates = sub["date"]
            pos = int(dates.searchsorted(rebalance_date, side="right") - 1)
            if pos < min_bars - 1 or pos + forward_days >= len(sub):
                continue
            history = sub.iloc[:pos + 1].copy()
            if not history.empty:
                assert history["date"].max() <= rebalance_date
            try:
                score = _score_value(score_fn(history))
            except Exception:
                score = np.nan
            if pd.isna(score):
                continue
            entry = float(sub.loc[pos, "close"])
            future = float(sub.loc[pos + forward_days, "close"])
            if entry <= 0 or not math.isfinite(entry) or not math.isfinite(future):
                continue
            period_records.append({"rebalance_date": pd.Timestamp(rebalance_date),
                                   "symbol": symbol,
                                   "score": score,
                                   "forward_return": (future / entry - 1.0) * 100.0})
        if period_records:
            period = pd.DataFrame(period_records).sort_values("score", ascending=False, kind="mergesort")
            period["rank"] = np.arange(1, len(period) + 1)
            period["selected"] = period["rank"] <= int(top_n or len(period))
            records.extend(period.to_dict("records"))

    if not records:
        return pd.DataFrame(columns=["rebalance_date", "symbol", "score", "forward_return", "rank", "selected"])
    return pd.DataFrame(records)


def _spearman(x, y) -> float:
    if len(x) < 2 or pd.Series(x).nunique(dropna=True) < 2 or pd.Series(y).nunique(dropna=True) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        corr = spearmanr(x, y, nan_policy="omit").correlation
    return float(corr) if corr is not None and math.isfinite(corr) else np.nan


def information_coefficient(records) -> dict:
    """Spearman rank correlation between score and forward return."""
    frame = pd.DataFrame(records)
    if frame.empty or not {"rebalance_date", "score", "forward_return"}.issubset(frame.columns):
        return {"periods": pd.DataFrame(columns=["rebalance_date", "ic", "n"]), "overall": np.nan}
    rows = []
    for date, group in frame.groupby("rebalance_date", sort=True):
        clean = group[["score", "forward_return"]].dropna()
        rows.append({"rebalance_date": pd.Timestamp(date), "ic": _spearman(clean["score"], clean["forward_return"]), "n": int(len(clean))})
    clean_all = frame[["score", "forward_return"]].dropna()
    return {"periods": pd.DataFrame(rows),
            "overall": _spearman(clean_all["score"], clean_all["forward_return"])}


def quantile_spread(records, quantiles: int = 5) -> dict:
    """Mean forward return by score quantile and top-minus-bottom spread."""
    frame = pd.DataFrame(records)
    if frame.empty or not {"rebalance_date", "score", "forward_return"}.issubset(frame.columns):
        return {"by_quantile": {}, "top_minus_bottom": np.nan, "monotonic": False}
    labelled = []
    for _, group in frame.dropna(subset=["score", "forward_return"]).groupby("rebalance_date", sort=True):
        n = min(int(quantiles), len(group), group["score"].nunique())
        if n < 2:
            continue
        ranks = group["score"].rank(method="first")
        part = group.copy()
        part["quantile"] = pd.qcut(ranks, q=n, labels=False, duplicates="drop") + 1
        labelled.append(part)
    if not labelled:
        return {"by_quantile": {}, "top_minus_bottom": np.nan, "monotonic": False}
    out = pd.concat(labelled, ignore_index=True)
    means = out.groupby("quantile")["forward_return"].mean().sort_index()
    spread = float(means.iloc[-1] - means.iloc[0]) if len(means) >= 2 else np.nan
    monotonic = bool((means.diff().dropna() >= 0).all()) if len(means) >= 2 else False
    return {"by_quantile": {int(k): float(v) for k, v in means.items()},
            "top_minus_bottom": spread,
            "monotonic": monotonic}


def summarise(records) -> str:
    """Readable signal-validation summary with honest zero-edge language."""
    frame = pd.DataFrame(records)
    if frame.empty:
        return "No walk-forward records: not enough data to validate the signal."
    ic = information_coefficient(frame)
    periods = ic["periods"].dropna(subset=["ic"]) if not ic["periods"].empty else pd.DataFrame()
    mean_ic = float(periods["ic"].mean()) if not periods.empty else np.nan
    positive_pct = float((periods["ic"] > 0).mean() * 100.0) if not periods.empty else np.nan
    spread = quantile_spread(frame)
    selected = frame[frame.get("selected", False)] if "selected" in frame.columns else pd.DataFrame()
    top_mean = float(selected["forward_return"].mean()) if not selected.empty else np.nan
    universe_mean = float(frame["forward_return"].mean()) if "forward_return" in frame else np.nan
    top_vs_universe = top_mean - universe_mean if math.isfinite(top_mean) and math.isfinite(universe_mean) else np.nan

    # ---- verdict ----------------------------------------------------------- #
    # Thresholds are calibrated to what an Information Coefficient actually
    # looks like on liquid equities. A mean IC of 0.02-0.05 is a useful signal,
    # 0.05-0.10 is good, and above 0.10 is excellent; sustained values near
    # 0.15 are rare enough that requiring them declares every real signal
    # worthless. An earlier version used abs(mean_ic) >= 0.15 and reported "no
    # predictive power" for the medium-horizon engine despite a mean IC of
    # 0.074, 61% positive periods, perfectly monotonic quintiles and +8.5pp
    # over the universe -- a false negative on the tool's best-validated model.
    #
    # The verdict weighs three independent pieces of evidence rather than IC
    # alone. Monotonic quintiles matter because they show the relationship
    # holding across the whole score range instead of being driven by a handful
    # of outliers, and consistency across periods separates a signal from one
    # lucky window.
    verdict = _verdict(mean_ic, positive_pct, spread, top_vs_universe)

    return (f"Periods: {frame['rebalance_date'].nunique()}, records: {len(frame)}\n"
            f"Mean IC: {mean_ic:.3f}, positive IC periods: {positive_pct:.1f}%\n"
            f"Overall IC: {ic['overall']:.3f}\n"
            f"Quintile monotonic: {spread['monotonic']}, top-bottom spread: {spread['top_minus_bottom']:.2f}%\n"
            f"Top-N mean return: {top_mean:.2f}%, universe mean: {universe_mean:.2f}%, difference: {top_vs_universe:.2f}%\n"
            f"{verdict}")


# Mean-IC bands for liquid equities. See the note in summarise().
IC_USEFUL = 0.02
IC_GOOD = 0.05
IC_STRONG = 0.10
# Share of rebalance periods with a positive IC that distinguishes a persistent
# signal from one good window. 50% is a coin flip.
CONSISTENT_PCT = 55.0


def _verdict(mean_ic: float, positive_pct: float, spread: dict,
             top_vs_universe: float) -> str:
    """Plain-language read on whether the scoring predicted anything."""
    if not math.isfinite(mean_ic):
        return "Not enough data to judge predictive power."

    if mean_ic <= -IC_USEFUL and math.isfinite(positive_pct) and positive_pct < 45:
        return ("Scoring is NEGATIVELY predictive over the tested window -- "
                "higher scores went with lower forward returns. Do not trade "
                "this ranking; the weights need rethinking.")

    if mean_ic < IC_USEFUL:
        return ("Scoring shows no predictive power over the tested window. "
                "Treat the ranking as a screen to investigate, not a signal.")

    evidence = []
    if math.isfinite(positive_pct) and positive_pct >= CONSISTENT_PCT:
        evidence.append(f"{positive_pct:.0f}% of periods positive")
    if spread.get("monotonic"):
        evidence.append("returns rise monotonically across score quintiles")
    if math.isfinite(top_vs_universe) and top_vs_universe > 0:
        evidence.append(f"top-N beat the universe by {top_vs_universe:.1f}pp")

    if mean_ic >= IC_STRONG:
        strength = "a strong edge"
    elif mean_ic >= IC_GOOD:
        strength = "a useful edge"
    else:
        strength = "a weak but positive edge"

    detail = ("; ".join(evidence)) if evidence else "on mean IC alone"
    caution = ("" if len(evidence) >= 2 else
               " Corroborating evidence is thin, so treat this as provisional.")
    return (f"Scoring shows {strength} over the tested window ({detail})."
            f"{caution} This measures signal quality only -- it models no "
            "costs, slippage or position sizing.")
