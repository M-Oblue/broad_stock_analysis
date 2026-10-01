"""Console encoding helper.

VENDORED from the sibling `stock_analysis` repo (scripts/console.py, renamed to
avoid colliding with src/report/console.py) and now owned here.

Windows terminals default to the cp1252 code page. Output from this repo is
plain ASCII, so that alone would be fine -- but the *data* it prints is not
under our control: company, sector and industry strings come from yfinance, and
NSE company names arrive from a live API. Printing a holding named "Nestle"
with an acute accent under cp1252 raises UnicodeEncodeError and kills the run,
typically at the final print of a long batch, with no emoji involved at all.

Entry points call enable_utf8() before printing anything. Library modules should
NOT call it at import time -- reconfiguring a caller's stdio as a side effect of
an import is not theirs to do, and the entry point has already done it.
"""
import sys


def enable_utf8():
    """Force UTF-8 on stdout/stderr where the interpreter supports it.

    errors='replace' rather than 'strict' so an exotic character degrades to
    '?' instead of taking down a long batch run at the final print.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # redirected to something that isn't a TextIOWrapper
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # already detached/closed -- not worth failing the run over
