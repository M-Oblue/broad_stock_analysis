"""Dated run-output directories and cache locations.

VENDORED from the sibling `stock_analysis` repo (scripts/run_paths.py) and now
owned here. Copied rather than imported, and adjusted for this repo's deeper
layout (src/common/ rather than scripts/) plus a dedicated cache directory.

Every pipeline run writes into `data/processed/<YYYY-MM-DD>/`, so re-running
next month archives that month instead of overwriting last month's numbers.
Re-running on the *same* day overwrites in place -- a day's output is a
snapshot of that day, not an append log.

Caches live in `data/cache/` rather than inside a dated folder: they are shared
state that must survive across runs (a fundamentals cache keyed by TTL is
worthless if every run starts empty), not the output of any single run. The old
repo kept caches loose in data/processed/ next to dated folders; giving them
their own directory makes the distinction explicit and lets .gitignore treat
outputs and caches separately.

Nothing here creates directories. Call `ensure_parent()` at the point of
writing, so `--help` and argument parsing never leave stray folders behind and
a user-supplied `--out` path gets its parent created too.
"""
from datetime import date as _date
from pathlib import Path

# src/common/run_paths.py -> src/common -> src -> repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
RUN_DATE_FORMAT = "%Y-%m-%d"


def run_dir(run_date=None):
    """data/processed/<YYYY-MM-DD>/ for `run_date` (default today)."""
    stamp = (run_date or _date.today()).strftime(RUN_DATE_FORMAT)
    return PROCESSED_DATA_DIR / stamp


def run_output(filename, run_date=None):
    """Path for an output file inside today's (or `run_date`'s) run folder."""
    return run_dir(run_date) / filename


def cache_file(filename):
    """Path for a cross-run cache file (fundamentals, ownership, prices)."""
    return CACHE_DIR / filename


def raw_file(filename):
    """Path for a tracked seed input (ticker universe, catalysts)."""
    return RAW_DATA_DIR / filename


def ensure_parent(path):
    """Create the parent directory of `path`. Returns the path for chaining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def iter_run_dirs(newest_first=True):
    """Existing run folders, sorted by date. Ignores non-dated subfolders."""
    if not PROCESSED_DATA_DIR.exists():
        return []
    dirs = []
    for child in PROCESSED_DATA_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            _date.fromisoformat(child.name)
        except ValueError:
            continue  # not a run folder (e.g. a manual archive)
        dirs.append(child)
    # ISO dates sort correctly as strings, which is the point of the format.
    return sorted(dirs, key=lambda p: p.name, reverse=newest_first)


def latest_run_file(filename):
    """Newest dated run containing `filename`, else None.

    Lets each stage pick up whatever the previous stage most recently produced,
    without the caller having to type a dated path.
    """
    for folder in iter_run_dirs():
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None
