#!/usr/bin/env python3
"""Console and CSV reporting for scored candidates.

THE REPORT IS THE PRODUCT, NOT THE SCORE
    A ranked list of numbers invites exactly the wrong behaviour: sorting by
    score and buying the top rows. The score is a summary of reasoning that can
    be wrong in ways the number cannot express -- a lender whose loan book is
    unverifiable, a company whose valuation could not be measured, a business
    with four years of history being judged over five.

    So every row carries its reasoning, its caveats and its coverage, and the
    console report leads with the explanation rather than the rank. For the
    long horizon this is not a nicety: no backtest can honestly validate those
    weights (see README), so the auditability of the reasoning IS the basis for
    trusting the output. A ranking you can read beats a ranking you can only
    sort.

WHY SCORES ARE SHOWN WITH TIERS
    Reporting 73.4 versus 71.8 implies a precision the inputs do not support.
    Tiers ("strong", "good", "moderate") communicate the same ordering without
    inviting false confidence in a 1.6-point gap that is well inside the noise
    of the underlying data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.common.run_paths import ensure_parent

# Columns written to CSV, in order. List-valued columns are joined so the file
# stays readable in a spreadsheet rather than showing Python list syntax.
CSV_COLUMNS = [
    "rank", "ticker", "score", "tier", "raw_score", "penalty", "coverage",
    "confident", "sector", "model", "reasons", "misses", "penalties",
    "warnings",
]

LIST_COLUMNS = ("reasons", "misses", "penalties", "warnings", "skipped")

HORIZON_LABELS = {
    "long": "LONG TERM (2-5 year hold)",
    "medium": "MEDIUM TERM (1-2 year hold)",
    "short": "SHORT TERM (weeks to months)",
}

HORIZON_METHODS = {
    "long": "Quality compounding and GARP. Scored on ROCE durability, growth, "
            "balance-sheet health and entry valuation. Price momentum and "
            "technical signals are deliberately excluded -- equities mean-revert "
            "over 3-5 years, so recent winners are a poor long-term entry.",
    "medium": "Earnings momentum and relative strength. Scored on quarterly "
              "growth and its acceleration, 12-1 momentum, re-rating room and a "
              "quality floor that keeps leveraged junk out of a momentum screen.",
    "short": "Trend following with risk control. Scored on trend structure, "
             "momentum and confirmation. At this horizon risk management is the "
             "edge, so every candidate carries a stop and R-multiple targets.",
}


def _fmt(value, spec="{:.1f}", dash="n/a"):
    if value is None:
        return dash
    try:
        if pd.isna(value):
            return dash
    except (TypeError, ValueError):
        pass
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def _join(value):
    if isinstance(value, (list, tuple)):
        return " | ".join(str(v) for v in value)
    return value


def to_csv(ranked: pd.DataFrame, path: Path) -> Path:
    """Write a ranked frame to CSV, flattening list columns."""
    out = ranked.copy()
    for column in LIST_COLUMNS:
        if column in out.columns:
            out[column] = out[column].map(_join)
    columns = [c for c in CSV_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in columns and c not in LIST_COLUMNS]
    path = ensure_parent(path)
    out[columns + extra].to_csv(path, index=False)
    return path


def summary_table(ranked: pd.DataFrame, top: int = 20) -> str:
    """Compact ranked table for the console."""
    if ranked.empty:
        return "  (no candidates)"

    head = ranked.head(top)
    lines = [
        f"  {'#':>3} {'TICKER':<16} {'SCORE':>6} {'TIER':<10} {'COV':>5} "
        f"{'PEN':>5}  {'SECTOR':<28}",
        f"  {'-'*3} {'-'*16} {'-'*6} {'-'*10} {'-'*5} {'-'*5}  {'-'*28}",
    ]
    for _, row in head.iterrows():
        sector = str(row.get("sector") or "")[:28]
        lines.append(
            f"  {int(row['rank']):>3} {str(row['ticker']):<16} "
            f"{_fmt(row['score']):>6} {str(row.get('tier','')):<10} "
            f"{_fmt(row.get('coverage'), '{:.0%}'):>5} "
            f"{_fmt(row.get('penalty'), '{:.0f}'):>5}  {sector:<28}")
    return "\n".join(lines)


def explain(row, max_reasons: int = 8, max_misses: int = 4) -> str:
    """Full per-stock explanation: what earned points, what did not, and why."""
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    lines = []

    header = (f"{data.get('ticker','?')}  --  score {_fmt(data.get('score'))}/100 "
              f"({data.get('tier','?')})")
    lines.append(header)
    lines.append("-" * len(header))

    facts = []
    if data.get("sector"):
        facts.append(f"sector: {data['sector']}")
    if data.get("model"):
        facts.append(f"model: {data['model']}")
    facts.append(f"coverage: {_fmt(data.get('coverage'), '{:.0%}')}")
    if data.get("penalty"):
        facts.append(f"penalties: -{_fmt(data.get('penalty'), '{:.0f}')}")
    lines.append("  " + "   ".join(facts))

    reasons = data.get("reasons") or []
    if reasons:
        lines.append("  What earned points:")
        for reason in reasons[:max_reasons]:
            lines.append(f"    + {reason}")
        if len(reasons) > max_reasons:
            lines.append(f"    ... and {len(reasons) - max_reasons} more")

    misses = data.get("misses") or []
    if misses:
        lines.append("  What did not:")
        for miss in misses[:max_misses]:
            lines.append(f"    - {miss}")
        if len(misses) > max_misses:
            lines.append(f"    ... and {len(misses) - max_misses} more")

    penalties = data.get("penalties") or []
    if penalties:
        lines.append("  Penalties applied:")
        for penalty in penalties:
            lines.append(f"    ! {penalty}")

    # Caveats are printed last and unabridged. They are the part a reader is
    # most likely to skip and most needs to see.
    warnings = data.get("warnings") or []
    if warnings:
        lines.append("  Caveats:")
        for warning in warnings:
            lines.append(f"    * {warning}")

    plan = _risk_plan_line(data)
    if plan:
        lines.append(f"  Risk plan: {plan}")

    return "\n".join(lines)


def _risk_plan_line(data: dict) -> str | None:
    """Format the short-horizon stop/target plan when present."""
    stop = data.get("stop_loss")
    if stop is None or (isinstance(stop, float) and np.isnan(stop)):
        return None
    parts = [f"stop {_fmt(stop, '{:.2f}')}"]
    for key, label in (("target_2r", "2R"), ("target_3r", "3R")):
        if data.get(key) is not None:
            parts.append(f"{label} {_fmt(data.get(key), '{:.2f}')}")
    if data.get("suggested_hold"):
        parts.append(f"hold {data['suggested_hold']}")
    return "  ".join(parts)


def render_report(ranked: pd.DataFrame, horizon: str, regime: dict | None = None,
                  top: int = 20, detail: int = 5,
                  universe_size: int | None = None) -> str:
    """Full console report for one horizon."""
    width = 78
    lines = ["=" * width, HORIZON_LABELS.get(horizon, horizon.upper()), "=" * width]

    method = HORIZON_METHODS.get(horizon)
    if method:
        lines.append("")
        for chunk in _wrap(method, width - 2):
            lines.append(chunk)

    if regime:
        lines.append("")
        lines.extend(_regime_block(regime, horizon, width))

    lines.append("")
    scored = len(ranked)
    confident = int(ranked["confident"].sum()) if "confident" in ranked else scored
    if universe_size:
        lines.append(f"Scored {scored} of {universe_size} in the universe; "
                     f"{confident} with sufficient data coverage.")
    else:
        lines.append(f"Scored {scored}; {confident} with sufficient data coverage.")

    lines.append("")
    lines.append(f"TOP {min(top, scored)} CANDIDATES")
    lines.append(summary_table(ranked, top))

    if detail and scored:
        lines.append("")
        lines.append("-" * width)
        lines.append(f"WHY THE TOP {min(detail, scored)} SCORED AS THEY DID")
        lines.append("-" * width)
        for _, row in ranked.head(detail).iterrows():
            lines.append("")
            lines.append(explain(row))

    lines.append("")
    lines.append("-" * width)
    lines.append("Educational tool. Not investment advice. Scores rank and "
                 "explain candidates;")
    lines.append("they do not size or place trades. Confirm independently "
                 "before risking money.")
    return "\n".join(lines)


def _regime_block(regime: dict, horizon: str, width: int) -> list[str]:
    """Market-regime banner.

    Placed before the candidate list rather than after it because the regime
    changes how the list should be read: a momentum screen in a risk-off tape
    is a list of things falling slightly less fast than everything else.
    """
    label = str(regime.get("regime", "unknown")).replace("_", "-").upper()
    lines = [f"MARKET REGIME: {label}"]
    rationale = regime.get("rationale")
    if rationale:
        lines.extend(_wrap(f"  {rationale}", width))

    if regime.get("regime") == "risk_off" and horizon in ("short", "medium"):
        lines.append("")
        lines.extend(_wrap(
            "  [WARN] Momentum and trend strategies perform poorly in this "
            "regime. These candidates are ranked relative to each other, not "
            "against cash -- the best of them may still be a losing trade. "
            "Consider smaller positions or waiting for the trend to turn.", width))
    return lines


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width) or [""]
