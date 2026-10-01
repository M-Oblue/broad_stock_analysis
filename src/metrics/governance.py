#!/usr/bin/env python3
"""Governance metrics from NSE promoter ownership and pledging data.

This is the only signal set here that is neither price nor accounting. It earns
its place at the long horizon because promoter behaviour leads reported
numbers: a promoter pledging their stake or steadily selling down is visible
quarters before it shows up in earnings, and it is invisible to every technical
and fundamental metric in this repo.

WHAT PLEDGING ACTUALLY MEANS
    Pledged promoter shares are collateral against borrowing. The risk is
    reflexive rather than linear: if the price falls, the lender issues a margin
    call, the promoter must post more collateral or the lender sells the pledged
    shares into the market, which pushes the price down further. A heavily
    pledged stock therefore has a fragile floor, and the damage lands precisely
    when the stock is already weak. That is why pledging is scored as a
    threshold penalty rather than a smooth negative -- the risk is not
    proportional, it is a cliff.

TWO DISTINCTIONS THAT DECIDE WHETHER THIS IS RIGHT OR NONSENSE
    1. NO PROMOTER IS NOT ZERO PROMOTER QUALITY. Widely-held, professionally
       managed companies -- HDFCBANK, ITC and most large private banks -- have
       no identified promoter at all. Treating a null promoter stake as
       "promoters own nothing, red flag" would penalise some of the highest
       governance quality in the index. `has_promoter=False` routes these past
       promoter checks entirely, neither rewarded nor punished.

    2. HIGH PROMOTER HOLDING IS NOT LINEARLY GOOD. Skin in the game is
       genuinely reassuring, but a very high stake also means a thin float and
       limited minority-shareholder influence. The score rewards a healthy band
       and stops rewarding beyond it, rather than ranking a 90% promoter stake
       above a 55% one.

DIRECTION MATTERS MORE THAN LEVEL
    A promoter at 55% who was at 75% a year ago is a different proposition from
    one who has held 55% for a decade. The trend metrics are signed so that
    negative always means promoters are selling down, and the sell-down
    threshold is deliberately above the noise floor: small changes come from
    ESOP dilution, creeping acquisition and pledge-related reclassification
    rather than a decision to exit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Pledging thresholds (% of the promoter's own stake that is encumbered).
# Calibrated against the live Nifty 500 distribution: 79 of 468 promoter-led
# companies carry any encumbrance at all, and the tail runs to 100% (MPHASIS,
# AFCONS -- both genuine, being PE/LBO and Shapoorji Pallonji structures).
PLEDGE_NONE = 0.0
PLEDGE_MINOR = 5.0      # below this, usually structural rather than distress
PLEDGE_ELEVATED = 15.0  # meaningful margin-call reflexivity
PLEDGE_SEVERE = 25.0    # a genuine red flag at any horizon

# Promoter stake bands. Below MIN, the promoter has little at stake; above
# COMFORTABLE, additional holding adds float/liquidity risk rather than comfort.
PROMOTER_MIN_MEANINGFUL = 25.0
PROMOTER_COMFORTABLE = 50.0
PROMOTER_VERY_HIGH = 75.0

# Change in percentage points over four quarters that counts as a real
# sell-down rather than ESOP dilution or reclassification noise.
SELLDOWN_NOTABLE_PP = -2.0
SELLDOWN_SEVERE_PP = -5.0

# Quarters of filed shareholding history required before a trend is trusted.
MIN_QUARTERS_FOR_TREND = 5


def _f(value):
    """Coerce to float, mapping missing/blank to NaN."""
    if value is None:
        return np.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result


def classify_pledge(pledge_pct: float, has_promoter: bool) -> tuple[str, str]:
    """Map promoter encumbrance to a band and a plain-language explanation."""
    if not has_promoter:
        return ("not_applicable",
                "no identified promoter -- professionally managed, so promoter "
                "pledging does not apply")
    if np.isnan(pledge_pct):
        return "unknown", "no pledging disclosure available"
    if pledge_pct <= PLEDGE_NONE:
        return "none", "no promoter shares pledged"
    if pledge_pct < PLEDGE_MINOR:
        return "minor", f"{pledge_pct:.1f}% of promoter stake pledged (minor)"
    if pledge_pct < PLEDGE_ELEVATED:
        return "moderate", f"{pledge_pct:.1f}% of promoter stake pledged"
    if pledge_pct < PLEDGE_SEVERE:
        return ("elevated",
                f"{pledge_pct:.1f}% of promoter stake pledged -- margin calls on "
                "a price fall could force selling into weakness")
    return ("severe",
            f"{pledge_pct:.1f}% of promoter stake pledged -- a fall in the price "
            "can force lender selling, which pushes it down further")


def classify_holding(holding_pct: float, has_promoter: bool) -> tuple[str, str]:
    """Map promoter stake to a band, recognising that more is not linearly better."""
    if not has_promoter:
        return ("widely_held",
                "no identified promoter -- widely held and professionally managed")
    if np.isnan(holding_pct):
        return "unknown", "promoter holding not available"
    if holding_pct < PROMOTER_MIN_MEANINGFUL:
        return ("low",
                f"promoter holds {holding_pct:.1f}% -- limited skin in the game")
    if holding_pct < PROMOTER_COMFORTABLE:
        return "moderate", f"promoter holds {holding_pct:.1f}%"
    if holding_pct < PROMOTER_VERY_HIGH:
        return "strong", f"promoter holds {holding_pct:.1f}% -- well aligned"
    return ("very_high",
            f"promoter holds {holding_pct:.1f}% -- strong alignment but a thin "
            "float, so liquidity and minority influence are limited")


def classify_trend(change_4q_pp: float, quarters: int) -> tuple[str, str]:
    """Classify promoter buying/selling over the last four quarters."""
    if quarters < MIN_QUARTERS_FOR_TREND or np.isnan(change_4q_pp):
        return "unknown", "insufficient shareholding history for a trend"
    if change_4q_pp <= SELLDOWN_SEVERE_PP:
        return ("selling_heavily",
                f"promoter stake down {abs(change_4q_pp):.1f}pp over four "
                "quarters -- a sustained exit, not incidental dilution")
    if change_4q_pp <= SELLDOWN_NOTABLE_PP:
        return ("selling",
                f"promoter stake down {abs(change_4q_pp):.1f}pp over four quarters")
    if change_4q_pp >= 1.0:
        return ("buying",
                f"promoter stake up {change_4q_pp:.1f}pp over four quarters")
    return "stable", "promoter stake broadly unchanged"


def compute_for_symbol(ownership_row: pd.Series | dict) -> dict:
    """Governance metrics and flags for one company."""
    row = dict(ownership_row) if not isinstance(ownership_row, dict) else ownership_row

    holding = _f(row.get("promoter_holding_pct"))
    pledge = _f(row.get("promoter_pledge_pct"))
    change_1q = _f(row.get("promoter_change_1q_pp"))
    change_4q = _f(row.get("promoter_change_4q_pp"))
    change_max = _f(row.get("promoter_change_max_pp"))
    quarters = int(_f(row.get("quarters_of_history")) or 0)

    has_promoter = bool(row.get("has_promoter", False))

    pledge_band, pledge_note = classify_pledge(pledge, has_promoter)
    holding_band, holding_note = classify_holding(holding, has_promoter)
    trend_band, trend_note = classify_trend(change_4q, quarters)

    out = {
        "symbol": row.get("symbol"),
        "has_promoter": has_promoter,
        "promoter_holding_pct": holding,
        "promoter_pledge_pct": pledge,
        "promoter_change_1q_pp": change_1q,
        "promoter_change_4q_pp": change_4q,
        "promoter_change_max_pp": change_max,
        "quarters_of_history": quarters,
        "pledge_band": pledge_band,
        "holding_band": holding_band,
        "trend_band": trend_band,
        "governance_notes": [n for n in (holding_note, pledge_note, trend_note)
                             if n],
    }

    # Red flags are the actionable output: conditions that should veto or
    # heavily penalise a long-term candidate regardless of how good its
    # financials look.
    flags = []
    if pledge_band == "severe":
        flags.append(f"SEVERE PLEDGING: {pledge:.1f}% of promoter stake encumbered")
    elif pledge_band == "elevated":
        flags.append(f"Elevated pledging: {pledge:.1f}% of promoter stake encumbered")
    if trend_band == "selling_heavily":
        flags.append(f"Promoter sell-down: {change_4q:+.1f}pp over four quarters")
    # A low promoter stake is a characteristic, not a fault, when it is stable.
    # Infosys sits at 13.8% because its founders diluted over three decades; it
    # is one of the better-governed companies in the index. Raising a red flag
    # there would put it in the same bucket as a promoter who has pledged their
    # entire holding. The combination -- a small stake that is ALSO shrinking --
    # is what actually signals an exit, so only that is flagged.
    if has_promoter and holding_band == "low" and trend_band in ("selling",
                                                                 "selling_heavily"):
        flags.append(f"Low promoter holding ({holding:.1f}%) and still selling")
    # The compounding case: pledged AND selling is materially worse than either
    # alone, because a forced sale and a voluntary exit reinforce one another.
    if pledge_band in ("elevated", "severe") and trend_band in ("selling",
                                                                "selling_heavily"):
        flags.append("Pledging combined with an active sell-down")
    out["governance_flags"] = flags
    out["has_red_flag"] = bool(flags)

    return out


def compute_all(ownership: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Governance metrics for every row in the ownership cache."""
    if ownership is None or ownership.empty:
        return pd.DataFrame()
    records = [compute_for_symbol(row) for _, row in ownership.iterrows()]
    frame = pd.DataFrame(records)
    if verbose:
        print(f"  governance computed for {len(frame)} symbols; "
              f"{int(frame['has_red_flag'].sum())} carry a red flag")
    return frame
