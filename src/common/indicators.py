# indicators.py
"""Shared technical-indicator math.

VENDORED from the sibling `stock_analysis` repo (scripts/indicators.py) and now
owned here. Copied rather than imported: a cross-repo import would couple the
two projects and break the moment either moves.

Safe to carry over because these are standard, objective indicator definitions
-- they encode no strategy opinion, which is the thing this repo deliberately
avoids inheriting. All scoring logic was rebuilt from first principles instead.

Single source of truth for every consumer in src/metrics/ -- fix a formula here
and all horizons get it.
"""
import pandas as pd
import numpy as np

def sma(series, length):
    """Simple Moving Average"""
    return series.rolling(window=length).mean()

def ema(series, length):
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def wilder(series, length):
    """Wilder's smoothing -- the EMA variant (alpha=1/length) underlying
    RSI/ATR/ADX. NaN for the first `length` bars (proper warm-up period)."""
    return series.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

def rsi(series, length=14):
    """Relative Strength Index (Wilder-smoothed -- the standard definition)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = wilder(gain, length) / wilder(loss, length)
    return 100 - (100 / (1 + rs))

def obv(close, volume):
    """On-Balance Volume"""
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()

def macd(close, fast=12, slow=26, signal=9):
    """MACD Line and Signal Line"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def atr(high, low, close, length=14):
    """Average True Range (Wilder-smoothed)."""
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return wilder(tr, length)

def adx(high, low, close, length=14):
    """Average Directional Index (ADX) - returns (adx, plus_di, minus_di).

    NOTE: plus_dm/minus_dm must be built as pd.Series(..., index=high.index).
    A previous version built them via bare pd.Series(np.where(...)), which
    gets a default 0..N RangeIndex instead of the source DatetimeIndex; the
    later division against tr_smooth (DatetimeIndex) then aligned on zero
    overlapping labels and silently produced all-NaN output.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr_smooth = atr(high, low, close, length)
    plus_dm_smooth = wilder(plus_dm, length)
    minus_dm_smooth = wilder(minus_dm, length)

    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = wilder(dx, length)

    return adx_series, plus_di, minus_di