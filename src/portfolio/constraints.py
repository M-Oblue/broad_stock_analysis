#!/usr/bin/env python3
"""Portfolio concentration, sector and correlation constraints.

A screen ranks individual names, but portfolios fail through groups: one crowded
sector, one duplicated factor exposure, or five stocks that are really the same
bet. Effective-stock count is intentionally based on HHI, not the raw holding
count, because 20 holdings spread across 3 dominant weights is not 20 bets --
that gap is the point of measuring concentration at all.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.metrics.risk import correlation_matrix


VALUE_COLUMNS = ("value", "market_value", "position_value", "amount")
WEIGHT_COLUMNS = ("weight", "weight_pct", "position_pct_of_capital")


def _records(items) -> list[dict]:
    if items is None:
        return []
    if isinstance(items, pd.DataFrame):
        return items.to_dict("records")
    if isinstance(items, dict):
        if all(isinstance(v, dict) for v in items.values()):
            out = []
            for symbol, values in items.items():
                rec = dict(values)
                rec.setdefault("symbol", symbol)
                out.append(rec)
            return out
        return [{"symbol": k, "value": v} for k, v in items.items()]
    out = []
    for item in items:
        if isinstance(item, dict):
            out.append(dict(item))
        else:
            out.append({"symbol": item})
    return out


def _symbol(record) -> str | None:
    for key in ("symbol", "ticker"):
        value = record.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def _record_weight_value(record) -> float:
    for key in WEIGHT_COLUMNS:
        if key in record and record.get(key) is not None:
            value = float(record.get(key) or 0.0)
            return value / 100.0 if key.endswith("pct") or value > 1.5 else value
    for key in VALUE_COLUMNS:
        if key in record and record.get(key) is not None:
            return float(record.get(key) or 0.0)
    shares = record.get("shares")
    price = record.get("price", record.get("entry_price"))
    if shares is not None and price is not None:
        return float(shares or 0.0) * float(price or 0.0)
    return 0.0


def _weights(holdings) -> pd.Series:
    records = _records(holdings)
    if not records:
        return pd.Series(dtype=float)
    values = {}
    explicit_weights = False
    for rec in records:
        sym = _symbol(rec)
        if sym is None:
            continue
        explicit_weights = explicit_weights or any(k in rec for k in WEIGHT_COLUMNS)
        values[sym] = values.get(sym, 0.0) + _record_weight_value(rec)
    s = pd.Series(values, dtype=float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if s.empty:
        return pd.Series(dtype=float)
    if explicit_weights or s.sum() <= 1.5:
        return s
    return s / s.sum()


def _candidate_weight_pct(candidate, default_pct: float) -> float:
    rec = candidate if isinstance(candidate, dict) else {"symbol": candidate}
    for key in WEIGHT_COLUMNS:
        if key in rec and rec.get(key) is not None:
            value = float(rec.get(key) or 0.0)
            return value if value > 1.5 or key.endswith("pct") else value * 100.0
    return float(default_pct)


def sector_exposure(holdings, sector_map) -> dict:
    """Percentage exposure by sector from holding weights or market values."""
    weights = _weights(holdings)
    if weights.empty:
        return {}
    exposure = {}
    for symbol, weight in weights.items():
        sector = sector_map.get(symbol, "Unknown") if sector_map else "Unknown"
        exposure[sector] = exposure.get(sector, 0.0) + float(weight) * 100.0
    return dict(sorted(exposure.items()))


def concentration(holdings) -> dict:
    """HHI concentration, effective stock count and top-three weight.

    Effective stocks is 1 / HHI. It is almost always far lower than the raw count,
    which is the point: 20 holdings across 3 sectors or a few dominant positions
    is not 20 independent bets.
    """
    weights = _weights(holdings)
    if weights.empty:
        return {"hhi": 0.0, "effective_stocks": 0.0, "top_3_weight": 0.0}
    hhi = float((weights ** 2).sum())
    top3 = float(weights.sort_values(ascending=False).head(3).sum() * 100.0)
    return {"hhi": hhi,
            "effective_stocks": (1.0 / hhi) if hhi > 0 else 0.0,
            "top_3_weight": top3}


def check_constraints(holdings, sector_map, max_position_pct: float = 15,
                      max_sector_pct: float = 30) -> list[dict]:
    """Return position and sector limit breaches."""
    breaches = []
    weights = _weights(holdings)
    for symbol, weight in weights.items():
        pct = float(weight) * 100.0
        if pct > max_position_pct:
            breaches.append({"type": "position", "symbol": symbol, "value": pct,
                             "limit": float(max_position_pct),
                             "reason": f"{symbol} is {pct:.1f}% above {max_position_pct:g}% limit"})
    for sector, pct in sector_exposure(holdings, sector_map).items():
        if pct > max_sector_pct:
            breaches.append({"type": "sector", "sector": sector, "value": pct,
                             "limit": float(max_sector_pct),
                             "reason": f"{sector} exposure is {pct:.1f}% above {max_sector_pct:g}% limit"})
    return breaches


def filter_candidates(candidates, holdings, sector_map, max_position_pct: float = 15,
                      max_sector_pct: float = 30, candidate_position_pct: float | None = None) -> dict:
    """Drop candidates whose sector has no room for another addition.

    A 5/5 candidate is still a bad addition if it would become the fourth name in
    an already-overweight sector; portfolio fit is judged after standalone merit.
    """
    exposure = sector_exposure(holdings, sector_map)
    kept, rejected = [], []
    default_add = max_position_pct if candidate_position_pct is None else candidate_position_pct
    for cand in _records(candidates):
        sym = _symbol(cand)
        sector = sector_map.get(sym, "Unknown") if sector_map and sym is not None else "Unknown"
        current = float(exposure.get(sector, 0.0))
        add_pct = _candidate_weight_pct(cand, default_add)
        if current >= max_sector_pct:
            rejected.append({"candidate": cand, "symbol": sym, "sector": sector,
                             "reason": f"{sector} already has no headroom ({current:.1f}% >= {max_sector_pct:g}%)"})
        elif current + add_pct > max_sector_pct:
            rejected.append({"candidate": cand, "symbol": sym, "sector": sector,
                             "reason": f"adding {sym} would take {sector} to {current + add_pct:.1f}% above {max_sector_pct:g}%"})
        else:
            kept.append(cand)
    return {"kept": kept, "rejected": rejected}


def _symbols(items) -> list[str]:
    symbols = []
    for rec in _records(items):
        sym = _symbol(rec)
        if sym is not None:
            symbols.append(sym)
    return symbols


def correlation_filter(candidates, prices, existing, max_corr: float = 0.8) -> dict:
    """Flag candidates highly correlated with existing holdings."""
    candidate_records = _records(candidates)
    candidate_symbols = [_symbol(c) for c in candidate_records if _symbol(c) is not None]
    existing_symbols = _symbols(existing)
    if prices is None or len(candidate_symbols) == 0 or len(existing_symbols) == 0:
        return {"kept": candidate_records, "rejected": []}

    symbols = sorted(set(candidate_symbols + existing_symbols))
    corr = correlation_matrix(prices, symbols)
    kept, rejected = [], []
    for cand in candidate_records:
        sym = _symbol(cand)
        if sym is None or sym not in corr.index:
            kept.append(cand)
            continue
        best_symbol, best_corr = None, np.nan
        for ex in existing_symbols:
            value = corr.loc[sym, ex] if ex in corr.columns else np.nan
            if pd.notna(value) and (pd.isna(best_corr) or abs(value) > abs(best_corr)):
                best_symbol, best_corr = ex, float(value)
        if pd.notna(best_corr) and abs(best_corr) >= max_corr:
            rejected.append({"candidate": cand, "symbol": sym,
                             "existing_symbol": best_symbol, "correlation": best_corr,
                             "reason": f"{sym} correlation {best_corr:.2f} with {best_symbol} exceeds {max_corr:g}"})
        else:
            kept.append(cand)
    return {"kept": kept, "rejected": rejected}
