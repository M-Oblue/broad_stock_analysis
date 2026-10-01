#!/usr/bin/env python3
"""Market-regime gate for momentum signals.

Cross-sectional momentum can work well in healthy markets and fail abruptly in
bear phases. The classifier therefore emits an explicit regime and rationale so
downstream scoring can scale signals down instead of silently treating every
market backdrop as equally friendly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.indicators import sma

DMA_LENGTH = 200
SLOPE_LOOKBACK = 20
VIX_ELEVATED = 20.0
VIX_HIGH = 25.0


def _clean_close(close) -> pd.Series:
    if close is None:
        return pd.Series(dtype=float)
    if isinstance(close, pd.DataFrame):
        if isinstance(close.columns, pd.MultiIndex):
            for level in range(close.columns.nlevels):
                if "Close" in close.columns.get_level_values(level):
                    close = close.xs("Close", axis=1, level=level, drop_level=True)
                    break
        elif "Close" in close.columns:
            close = close["Close"]
        elif "close" in close.columns:
            close = close["close"]
        if isinstance(close, pd.DataFrame) and close.shape[1] >= 1:
            close = close.iloc[:, 0]
        elif isinstance(close, pd.DataFrame):
            return pd.Series(dtype=float)
    s = pd.to_numeric(pd.Series(close), errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()]
    return s.sort_index()


def _neutral(rationale: str) -> dict:
    return {
        "regime": "neutral",
        "index_above_200dma": False,
        "index_vs_200dma_pct": np.nan,
        "dma200_slope": np.nan,
        "vix_level": np.nan,
        "drawdown_from_high": np.nan,
        "rationale": rationale,
    }


def classify_regime(index_close, vix_close=None) -> dict:
    """Classify the market as risk_on, neutral or risk_off."""
    close = _clean_close(index_close)
    if len(close) < DMA_LENGTH:
        out = _neutral("Neutral: insufficient index history for a 200DMA regime check.")
        if len(close):
            high = close.max()
            out["drawdown_from_high"] = float(close.iloc[-1] / high - 1.0) if high else np.nan
        return out

    dma200 = sma(close, DMA_LENGTH)
    latest_close = float(close.iloc[-1])
    latest_dma = float(dma200.iloc[-1])
    above = bool(latest_close > latest_dma) if not np.isnan(latest_dma) else False
    index_vs_dma = (latest_close / latest_dma - 1.0) * 100.0 if latest_dma else np.nan

    dma_clean = dma200.dropna()
    dma_slope = (float(dma_clean.iloc[-1] - dma_clean.iloc[-(SLOPE_LOOKBACK + 1)])
                 if len(dma_clean) > SLOPE_LOOKBACK else np.nan)
    rising = bool(dma_slope > 0) if not np.isnan(dma_slope) else False
    high = float(close.max())
    drawdown = float(latest_close / high - 1.0) if high else np.nan

    if above and rising:
        regime = "risk_on"
        reason = "index is above a rising 200DMA"
    elif (not above) and (not rising):
        regime = "risk_off"
        reason = "index is below a falling 200DMA"
    else:
        regime = "neutral"
        reason = "200DMA signals are mixed"

    vix = _clean_close(vix_close)
    vix_level = float(vix.iloc[-1]) if len(vix) else np.nan
    if not np.isnan(vix_level):
        if vix_level >= VIX_HIGH:
            regime = "risk_off"
            reason += f"; VIX is high at {vix_level:.1f}"
        elif vix_level >= VIX_ELEVATED and regime == "risk_on":
            regime = "neutral"
            reason += f"; elevated VIX at {vix_level:.1f} tempers risk-on"
        elif vix_level >= VIX_ELEVATED:
            regime = "risk_off"
            reason += f"; elevated VIX at {vix_level:.1f}"
    else:
        reason += "; VIX unavailable"

    return {
        "regime": regime,
        "index_above_200dma": above,
        "index_vs_200dma_pct": float(index_vs_dma),
        "dma200_slope": float(dma_slope),
        "vix_level": vix_level,
        "drawdown_from_high": drawdown,
        "rationale": f"{regime}: {reason}.",
    }


def fetch_regime(period: str = "2y") -> dict:
    """Fetch Nifty 50 and India VIX history, degrading safely on failure."""
    try:
        import yfinance as yf

        index = yf.download("^NSEI", period=period, auto_adjust=True,
                            progress=False, threads=False)
        if index is None or index.empty:
            return _neutral("Neutral: Nifty 50 history was unavailable from yfinance.")

        vix_close = None
        try:
            vix = yf.download("^INDIAVIX", period=period, auto_adjust=True,
                              progress=False, threads=False)
            if vix is not None and not vix.empty:
                vix_close = vix["Close"] if "Close" in vix.columns else vix
        except Exception as exc:
            vix_close = None

        index_close = index["Close"] if "Close" in index.columns else index
        return classify_regime(index_close, vix_close)
    except Exception as exc:
        return _neutral(f"Neutral: regime fetch failed ({type(exc).__name__}: {exc}).")
