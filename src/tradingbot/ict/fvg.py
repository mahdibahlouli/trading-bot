"""Fair value gaps (FVGs), plan section 4.2.

Bullish FVG completed at bar ``i``: ``low[i] > high[i-2]``; the gap runs from ``high[i-2]``
(bottom) to ``low[i]`` (top). Bearish: ``high[i] < low[i-2]``; the gap runs from ``high[i]``
(bottom) to ``low[i-2]`` (top).

The gap is known at the close of bar ``i`` (``ts``); tracking starts with bar ``i + 1``.
For each gap this module records, within ``ttl`` bars:

* ``first_touch_ts``: price first trades back into the gap (gap-edge entry);
* ``ce_ts``: price first reaches the 50% level, "consequent encroachment";
* ``invalidated_ts``: a bar closes through the far side (bullish: close below bottom);
* ``expired_ts``: the last tracked bar, set when the gap lasted ``ttl`` bars without being
  invalidated (it is no longer watched after that).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutils import ensure_utc_index
from .indicators import atr as _atr

FVG_COLS = [
    "ts", "direction", "top", "bottom", "mid", "size", "size_atr", "displacement_body_atr",
    "first_touch_ts", "ce_ts", "invalidated_ts", "expired_ts", "status",
]


def find_fvgs(
    bars: pd.DataFrame,
    min_size_atr: float = 0.0,
    ttl: int = 60,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Detect FVGs and follow each one for ``ttl`` bars.

    ``min_size_atr`` drops gaps smaller than that multiple of ATR (ATR at bar ``i``). Gaps found
    before ATR has warmed up are kept only when ``min_size_atr`` is 0.
    ``status`` is ``invalidated``, ``filled_ce``, ``touched``, ``expired`` or ``open`` (data ended).
    """
    b = ensure_utc_index(bars)
    o, h, l, c = (b[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    a = _atr(b, atr_period).to_numpy()
    idx = b.index
    m = len(b)
    if m < 3:
        return pd.DataFrame(columns=FVG_COLS)
    bull = np.zeros(m, bool)
    bear = np.zeros(m, bool)
    bull[2:] = l[2:] > h[:-2]
    bear[2:] = h[2:] < l[:-2]
    rows = []
    for i in np.flatnonzero(bull | bear):
        is_bull = bool(bull[i])
        top, bottom = (l[i], h[i - 2]) if is_bull else (l[i - 2], h[i])
        size = top - bottom
        ref = a[i]
        size_atr = size / ref if ref and not np.isnan(ref) else np.nan
        if min_size_atr > 0 and (np.isnan(size_atr) or size_atr < min_size_atr):
            continue
        mid = (top + bottom) / 2
        body_atr = abs(c[i - 1] - o[i - 1]) / ref if ref and not np.isnan(ref) else np.nan
        s, e = i + 1, min(i + 1 + ttl, m)
        if is_bull:
            touch = np.flatnonzero(l[s:e] <= top)
            ce = np.flatnonzero(l[s:e] <= mid)
            inval = np.flatnonzero(c[s:e] < bottom)
        else:
            touch = np.flatnonzero(h[s:e] >= bottom)
            ce = np.flatnonzero(h[s:e] >= mid)
            inval = np.flatnonzero(c[s:e] > top)
        inval_k = inval[0] if inval.size else None
        def first(arr):  # events after the invalidating bar do not count
            if not arr.size:
                return pd.NaT
            if inval_k is not None and arr[0] > inval_k:
                return pd.NaT
            return idx[s + arr[0]]
        t_touch, t_ce = first(touch), first(ce)
        t_inval = idx[s + inval_k] if inval_k is not None else pd.NaT
        complete = e - s == ttl
        if not pd.isna(t_inval):
            status = "invalidated"
        elif not pd.isna(t_ce):
            status = "filled_ce"
        elif not pd.isna(t_touch):
            status = "touched"
        elif complete:
            status = "expired"
        else:
            status = "open"
        t_exp = idx[e - 1] if complete and pd.isna(t_inval) else pd.NaT
        rows.append([
            idx[i], "bullish" if is_bull else "bearish", top, bottom, mid, size, size_atr, body_atr,
            t_touch, t_ce, t_inval, t_exp, status,
        ])
    return pd.DataFrame(rows, columns=FVG_COLS)
