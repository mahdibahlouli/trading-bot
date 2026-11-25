"""Time helpers. Everything on disk is UTC; New York time is used only inside rules."""

from __future__ import annotations

import pandas as pd

NY = "America/New_York"


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a timezone-aware UTC DatetimeIndex.

    Accepts either a DatetimeIndex or a ``ts`` column.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" not in df.columns:
            raise ValueError("bars need a DatetimeIndex or a 'ts' column")
        df = df.set_index("ts")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx
    out.index.name = "ts"
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    return out


def to_ny(index: pd.DatetimeIndex, tz: str = NY) -> pd.DatetimeIndex:
    return index.tz_convert(tz)


def utc_timestamp(value) -> pd.Timestamp:
    """Parse ``value`` as a UTC timestamp (naive values are taken as UTC)."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
