#!/usr/bin/env python3
"""ATR position sizing and portfolio heat controls.

ATR stops are useful because they scale the trade to the stock's own noise, but
raw ATR sizing has a dangerous failure mode: when the stop is very tight the
formula can demand a position far larger than the account can survive. That is
how a per-trade risk rule turns into hidden concentration. The cap helper is
therefore part of the sizing module rather than a reporting afterthought.

The functions return plain dictionaries so reports, notebooks and tests can use
the same audited numbers without depending on a portfolio object model.
"""
from __future__ import annotations

import math


def _finite_positive(value) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def _floor_to_lot(shares: float, lot: int) -> int:
    lot = max(int(lot or 1), 1)
    if not math.isfinite(shares) or shares <= 0:
        return 0
    return int(shares // lot * lot)


def position_size(capital, entry_price, stop_price, risk_pct: float = 1.0,
                  lot: int = 1) -> dict:
    """Size a trade from account risk and stop distance."""
    if not (_finite_positive(capital) and _finite_positive(entry_price)):
        return {"shares": 0, "position_value": 0.0, "risk_amount": 0.0,
                "risk_per_share": 0.0, "position_pct_of_capital": 0.0,
                "reason": "capital and entry price must be positive finite numbers"}

    capital = float(capital)
    entry = float(entry_price)
    stop = float(stop_price) if stop_price is not None else float("nan")
    risk_amount = capital * max(float(risk_pct or 0.0), 0.0) / 100.0

    if not math.isfinite(stop) or stop >= entry:
        return {"shares": 0, "position_value": 0.0, "risk_amount": risk_amount,
                "risk_per_share": 0.0, "position_pct_of_capital": 0.0,
                "reason": "stop must be below entry; refusing negative or infinite size"}

    risk_per_share = entry - stop
    shares = _floor_to_lot(risk_amount / risk_per_share, lot)
    position_value = shares * entry
    return {"shares": shares,
            "position_value": position_value,
            "risk_amount": shares * risk_per_share,
            "risk_per_share": risk_per_share,
            "position_pct_of_capital": (position_value / capital * 100.0) if capital else 0.0,
            "reason": "sized from per-trade risk"}


def apply_position_cap(capital, entry_price, stop_price, risk_pct: float = 1.0,
                       lot: int = 1, max_position_pct: float = 15.0) -> dict:
    """Apply ATR risk sizing, then cap position value as a share of capital.

    Tight stops can create huge nominal exposure while still risking only 1R on
    paper. The cap prevents that leverage-by-formula failure.
    """
    sized = position_size(capital, entry_price, stop_price, risk_pct, lot)
    if sized.get("shares", 0) <= 0:
        sized["cap_applied"] = False
        return sized

    capital = float(capital)
    entry = float(entry_price)
    stop = float(stop_price)
    max_value = capital * max(float(max_position_pct or 0.0), 0.0) / 100.0
    capped_shares = _floor_to_lot(max_value / entry, lot)
    if capped_shares < sized["shares"]:
        risk_per_share = entry - stop
        position_value = capped_shares * entry
        sized.update({"shares": capped_shares,
                      "position_value": position_value,
                      "risk_amount": capped_shares * risk_per_share,
                      "position_pct_of_capital": position_value / capital * 100.0,
                      "cap_applied": True,
                      "reason": f"position capped at {max_position_pct:g}% of capital"})
    else:
        sized["cap_applied"] = False
    return sized


def r_multiples(entry, stop, targets=(2, 3)) -> dict:
    """Target prices at the requested R multiples."""
    entry = float(entry)
    stop = float(stop)
    risk = entry - stop
    if not math.isfinite(risk) or risk <= 0:
        return {f"target_{r}r": float("nan") for r in targets}
    return {f"target_{r:g}r": entry + float(r) * risk for r in targets}


def portfolio_heat(positions) -> dict:
    """Total open risk as a percentage of capital.

    Heat above roughly 6% means several correlated stops firing together is an
    account-level event, not a collection of independent position-level events.
    """
    if not positions:
        return {"heat_pct": 0.0, "open_risk": 0.0, "capital": 0.0,
                "note": "portfolio heat above ~6% is account-level risk"}
    open_risk = 0.0
    capital = None
    for pos in positions:
        if pos is None:
            continue
        if capital is None and pos.get("capital") is not None:
            capital = float(pos.get("capital") or 0.0)
        if pos.get("risk_amount") is not None:
            open_risk += float(pos.get("risk_amount") or 0.0)
        elif pos.get("shares") is not None and pos.get("entry_price") is not None and pos.get("stop_price") is not None:
            open_risk += max(float(pos["entry_price"]) - float(pos["stop_price"]), 0.0) * float(pos["shares"])
    if capital is None:
        capital = sum(float(pos.get("position_value") or 0.0) for pos in positions if pos)
    heat = open_risk / capital * 100.0 if capital else 0.0
    note = "portfolio heat above ~6% is account-level risk"
    if heat > 6.0:
        note = "heat above ~6%: correlated stops are an account-level event"
    return {"heat_pct": heat, "open_risk": open_risk, "capital": capital, "note": note}
