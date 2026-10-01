#!/usr/bin/env python3
"""Risk metrics computed from price history.

Risk is kept separate from trend because the failure modes differ. Volatility
and drawdown say how painful a holding has been, while beta and correlations
must be aligned by date; positional alignment can manufacture a clean-looking
relationship from non-overlapping time periods.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TRADING_DAYS_PER_YEAR = 252


def _clean_close(close: pd.Series) -> pd.Series:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()]
    return s.sort_index()


def _log_returns(close: pd.Series) -> pd.Series:
    s = _clean_close(close)
    return np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()


def annualised_volatility(close, window):
    """Latest rolling stdev of daily log returns, annualised."""
    returns = _log_returns(close)
    if window is None or window <= 1 or len(returns) < window:
        return np.nan
    return float(returns.iloc[-window:].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def beta(stock_close, index_close):
    """Beta of stock daily returns vs index daily returns, aligned by date."""
    stock = _clean_close(stock_close).rename("stock")
    index = _clean_close(index_close).rename("index")
    aligned = pd.concat([stock, index], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        return np.nan
    returns = np.log(aligned / aligned.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        return np.nan
    variance = returns["index"].var(ddof=1)
    if pd.isna(variance) or variance == 0:
        return np.nan
    covariance = returns["stock"].cov(returns["index"])
    return float(covariance / variance)


def max_drawdown(close):
    """Most negative peak-to-trough return."""
    s = _clean_close(close)
    if s.empty:
        return np.nan
    running_peak = s.cummax()
    drawdown = (s / running_peak - 1.0).replace([np.inf, -np.inf], np.nan)
    clean = drawdown.dropna()
    return float(clean.min()) if len(clean) else np.nan


def downside_deviation(close):
    """Annualised stdev of negative daily log returns only."""
    negative = _log_returns(close)
    negative = negative[negative < 0]
    if len(negative) < 2:
        return np.nan
    return float(negative.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def correlation_matrix(prices_df: pd.DataFrame, symbols) -> pd.DataFrame:
    """Pairwise correlation of daily log returns for the requested symbols."""
    if prices_df is None or prices_df.empty or not symbols:
        return pd.DataFrame(index=symbols, columns=symbols, dtype=float)
    frame = prices_df[prices_df["symbol"].isin(symbols)].copy()
    if frame.empty:
        return pd.DataFrame(index=symbols, columns=symbols, dtype=float)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    wide = frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    returns = np.log(wide / wide.shift(1)).replace([np.inf, -np.inf], np.nan)
    return returns.corr().reindex(index=symbols, columns=symbols)


def _series_for_symbol(prices_df: pd.DataFrame, symbol: str) -> pd.Series:
    rows = prices_df[prices_df["symbol"] == symbol].copy()
    if rows.empty:
        return pd.Series(dtype=float)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["close"] = pd.to_numeric(rows["close"], errors="coerce")
    rows = rows.dropna(subset=["date"]).sort_values("date")
    return pd.Series(rows["close"].to_numpy(dtype=float), index=rows["date"], name=symbol)


def compute_all(prices_df: pd.DataFrame, benchmark_symbol: str = "^NSEI",
                verbose: bool = False) -> pd.DataFrame:
    """Per-symbol risk metrics from a long-format OHLCV frame."""
    if prices_df is None or prices_df.empty or "symbol" not in prices_df.columns:
        return pd.DataFrame()
    symbols = sorted(prices_df["symbol"].dropna().unique())
    benchmark = _series_for_symbol(prices_df, benchmark_symbol) if benchmark_symbol in symbols else pd.Series(dtype=float)
    records = []
    for i, symbol in enumerate(symbols, 1):
        close = _series_for_symbol(prices_df, symbol)
        records.append({
            "symbol": symbol,
            "vol_1y": annualised_volatility(close, TRADING_DAYS_PER_YEAR),
            "beta": beta(close, benchmark) if not benchmark.empty and symbol != benchmark_symbol else np.nan,
            "max_drawdown_1y": (max_drawdown(close.iloc[-TRADING_DAYS_PER_YEAR:])
                                if len(close) >= TRADING_DAYS_PER_YEAR else np.nan),
            "max_drawdown_full": max_drawdown(close),
            "downside_deviation": downside_deviation(close),
        })
        if verbose and i % 100 == 0:
            print(f"  {i}/{len(symbols)}")
    return pd.DataFrame(records)
