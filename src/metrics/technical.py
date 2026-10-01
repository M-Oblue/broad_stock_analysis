#!/usr/bin/env python3
"""Technical price metrics across short, medium and long horizons.

The module deliberately separates three timeframes because they answer
different questions. Short-horizon measures describe current trend, momentum,
market structure, volatility and liquidity; medium-horizon returns include the
classic 12-1 momentum factor, which avoids the noisy most-recent month; long
horizon returns measure whether a stock has actually compounded over years.

Every full-window metric returns NaN until the full lookback exists. A short
listed stock should be missing a 5-year return, not receive a 184-day return
silently labelled as 5 years.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.indicators import adx as _adx
from src.common.indicators import atr as _atr
from src.common.indicators import ema, macd as _macd
from src.common.indicators import rsi as _rsi
from src.common.indicators import sma

TRADING_DAYS_PER_YEAR = 252

SMA_LENGTHS = (20, 50, 200)
RETURN_YEARS = (3, 5)


def _nan_metrics() -> dict:
    out = {"latest_close": np.nan}
    for length in SMA_LENGTHS:
        out[f"sma{length}"] = np.nan
        out[f"close_vs_sma{length}_pct"] = np.nan
        out[f"sma{length}_slope_10d"] = np.nan
    out.update({
        "weekly_close_vs_ema20_pct": np.nan,
        "weekly_close_vs_ema50_pct": np.nan,
        "rsi": np.nan,
        "macd": np.nan,
        "macd_signal": np.nan,
        "macd_vs_signal": np.nan,
        "adx": np.nan,
        "adx_trajectory_10d": np.nan,
        "atr": np.nan,
        "atr_pct": np.nan,
        "roc_20d": np.nan,
        "roc_60d": np.nan,
        "dist_from_52w_high": np.nan,
        "dist_from_52w_low": np.nan,
        "structure_hh_hl_60d": np.nan,
        "volume_vs_20d_avg": np.nan,
        "avg_turnover_20d": np.nan,
        "mom_12_1": np.nan,
        "ret_3m": np.nan,
        "ret_6m": np.nan,
        "ret_12m": np.nan,
    })
    for years in RETURN_YEARS:
        out[f"ret_{years}y"] = np.nan
        out[f"ann_ret_{years}y"] = np.nan
    return out


def _safe_div(numerator, denominator):
    if numerator is None or denominator is None:
        return np.nan
    try:
        n, d = float(numerator), float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(n) or np.isnan(d) or d == 0:
        return np.nan
    return n / d


def _last(series: pd.Series):
    clean = series.dropna() if series is not None else pd.Series(dtype=float)
    return float(clean.iloc[-1]) if len(clean) else np.nan


def _return_pct(close: pd.Series, bars: int, end_offset: int = 0):
    """Return from `bars` ago to `end_offset` bars ago, requiring both points."""
    clean = close.dropna()
    if len(clean) < bars + end_offset:
        return np.nan
    end_pos = -1 - end_offset
    start_pos = end_pos - bars + 1
    start = clean.iloc[start_pos]
    end = clean.iloc[end_pos]
    ratio = _safe_div(end, start)
    return (ratio - 1.0) * 100.0 if not np.isnan(ratio) else np.nan


def _annualised_return(close: pd.Series, years: int):
    total = _return_pct(close, TRADING_DAYS_PER_YEAR * years)
    if np.isnan(total):
        return np.nan
    return ((1.0 + total / 100.0) ** (1.0 / years) - 1.0) * 100.0


def _position_vs(value, reference):
    ratio = _safe_div(value, reference)
    return (ratio - 1.0) * 100.0 if not np.isnan(ratio) else np.nan


def _slope_pct(series: pd.Series, lookback: int = 10):
    clean = series.dropna()
    if len(clean) <= lookback:
        return np.nan
    ratio = _safe_div(clean.iloc[-1], clean.iloc[-(lookback + 1)])
    return (ratio - 1.0) * 100.0 if not np.isnan(ratio) else np.nan


def _weekly_ema_position(close: pd.Series, length: int):
    weekly = close.resample("W-FRI").last().dropna()
    if len(weekly) < length:
        return np.nan
    return _position_vs(weekly.iloc[-1], ema(weekly, length).iloc[-1])


def _hh_hl_60(high: pd.Series, low: pd.Series):
    if len(high.dropna()) < 60 or len(low.dropna()) < 60:
        return np.nan
    recent_high = high.iloc[-20:].max()
    prior_high = high.iloc[-60:-20].max()
    recent_low = low.iloc[-20:].min()
    prior_low = low.iloc[-60:-20].min()
    if pd.isna(recent_high) or pd.isna(prior_high) or pd.isna(recent_low) or pd.isna(prior_low):
        return np.nan
    return int(recent_high > prior_high and recent_low > prior_low)


def compute_for_symbol(prices_df_for_one_symbol: pd.DataFrame) -> dict:
    """Compute all technical metrics for one symbol's long-format OHLCV frame."""
    frame = prices_df_for_one_symbol.copy()
    symbol = None
    if "symbol" in frame.columns and frame["symbol"].notna().any():
        symbol = frame["symbol"].dropna().iloc[0]

    out: dict = {"symbol": symbol, "bars": int(len(frame))}
    out.update(_nan_metrics())
    if frame.empty:
        return out

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").set_index("date")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    out["bars"] = int(frame["close"].notna().sum())
    close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
    latest_close = _last(close)
    out["latest_close"] = latest_close

    for length in SMA_LENGTHS:
        ma = sma(close, length)
        out[f"sma{length}"] = _last(ma)
        out[f"close_vs_sma{length}_pct"] = _position_vs(latest_close, out[f"sma{length}"])
        out[f"sma{length}_slope_10d"] = _slope_pct(ma, 10)

    out["weekly_close_vs_ema20_pct"] = _weekly_ema_position(close, 20)
    out["weekly_close_vs_ema50_pct"] = _weekly_ema_position(close, 50)

    rsi_series = _rsi(close, 14)
    out["rsi"] = _last(rsi_series)
    macd_line, signal_line = _macd(close)
    out["macd"] = _last(macd_line) if out["bars"] >= 35 else np.nan
    out["macd_signal"] = _last(signal_line) if out["bars"] >= 35 else np.nan
    out["macd_vs_signal"] = out["macd"] - out["macd_signal"] \
        if not (np.isnan(out["macd"]) or np.isnan(out["macd_signal"])) else np.nan

    adx_series, _, _ = _adx(high, low, close, 14)
    out["adx"] = _last(adx_series)
    out["adx_trajectory_10d"] = (adx_series.dropna().iloc[-1] - adx_series.dropna().iloc[-11]
                                  if len(adx_series.dropna()) > 10 else np.nan)
    atr_series = _atr(high, low, close, 14)
    out["atr"] = _last(atr_series)
    out["atr_pct"] = _position_vs(latest_close + out["atr"], latest_close) \
        if not np.isnan(out["atr"]) else np.nan

    out["roc_20d"] = _return_pct(close, 21)
    out["roc_60d"] = _return_pct(close, 61)

    high_52w = high.rolling(TRADING_DAYS_PER_YEAR, min_periods=TRADING_DAYS_PER_YEAR).max()
    low_52w = low.rolling(TRADING_DAYS_PER_YEAR, min_periods=TRADING_DAYS_PER_YEAR).min()
    out["dist_from_52w_high"] = _position_vs(latest_close, _last(high_52w))
    out["dist_from_52w_low"] = _position_vs(latest_close, _last(low_52w))
    out["structure_hh_hl_60d"] = _hh_hl_60(high, low)

    avg_vol20 = volume.rolling(20, min_periods=20).mean()
    out["volume_vs_20d_avg"] = _safe_div(_last(volume), _last(avg_vol20))
    turnover = close * volume
    out["avg_turnover_20d"] = _last(turnover.rolling(20, min_periods=20).mean())

    out["mom_12_1"] = _return_pct(close, 232, end_offset=20)
    out["ret_3m"] = _return_pct(close, 63)
    out["ret_6m"] = _return_pct(close, 126)
    out["ret_12m"] = _return_pct(close, 252)
    for years in RETURN_YEARS:
        out[f"ret_{years}y"] = _return_pct(close, TRADING_DAYS_PER_YEAR * years)
        out[f"ann_ret_{years}y"] = _annualised_return(close, years)

    return out


def compute_all(prices_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Technical metrics for every symbol in the long-format price frame."""
    if prices_df is None or prices_df.empty or "symbol" not in prices_df.columns:
        return pd.DataFrame()
    records = []
    symbols = sorted(prices_df["symbol"].dropna().unique())
    for i, symbol in enumerate(symbols, 1):
        records.append(compute_for_symbol(prices_df[prices_df["symbol"] == symbol]))
        if verbose and i % 100 == 0:
            print(f"  {i}/{len(symbols)}")
    return pd.DataFrame(records)
