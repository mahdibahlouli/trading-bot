"""Liquidity sweeps (raids), plan section 4.2.

A sweep of a **high** level:

1. a bar trades above the level (``high > price``): the *break bar*;
2. within ``max_bars`` bars, counting the break bar, a bar **closes back below** the level:
   the *confirm bar*;
3. the furthest price reached from the break bar to the confirm bar is at least
   ``min_overshoot_atr`` x ATR beyond the level (ATR taken at the bar before the break, so it is
   known in advance).

Lows mirror this. The event is known at the close of the confirm bar (``confirmed_ts``), so a
strategy can act from the next bar on.

Each level is used at most once. If price trades through and does not close back in time, the
level counts as **broken** (taken) and stops being watched. If it pokes through by less than the
minimum and closes back, that was only a touch and the level stays alive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutils import ensure_utc_index
from .indicators import atr as _atr

EVENT_COLS = [
    "level_name", "level_side", "level_price", "level_known_from", "direction", "outcome",
    "break_ts", "confirmed_ts", "extreme", "overshoot", "overshoot_atr", "bars_to_confirm",
]


def detect_sweeps(
    bars: pd.DataFrame,
    levels: pd.DataFrame,
    min_overshoot_atr: float = 0.1,
    max_bars: int = 3,
    atr_period: int = 14,
    include_breaks: bool = False,
) -> pd.DataFrame:
    """Find sweeps of ``levels`` (a level table from :mod:`tradingbot.ict.levels`).

    ``direction`` is the reversal the sweep implies: a swept high is ``"bearish"``, a swept low
    ``"bullish"``. With ``include_breaks=True`` levels that were traded through without closing back
    are returned too, with ``outcome="break"``.
    """
    b = ensure_utc_index(bars)
    idx = b.index
    h, l, c = (b[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
    a_prev = _atr(b, atr_period).shift(1).to_numpy()
    m = len(b)
    events = []
    for lv in levels.itertuples(index=False):
        start = idx.searchsorted(pd.Timestamp(lv.known_from), side="left")
        stop = m if pd.isna(lv.valid_until) else idx.searchsorted(pd.Timestamp(lv.valid_until), side="left")
        price = float(lv.price)
        is_high = lv.side == "high"
        j = start
        while j < stop:
            beyond = (h[j:stop] > price) if is_high else (l[j:stop] < price)
            hits = np.flatnonzero(beyond)
            if hits.size == 0:
                break
            j = j + hits[0]
            end = min(j + max_bars, m)
            back = (c[j:end] < price) if is_high else (c[j:end] > price)
            kb = np.flatnonzero(back)
            if kb.size == 0:
                if include_breaks:
                    k = end - 1
                    ext = h[j:end].max() if is_high else l[j:end].min()
                    events.append(_event(lv, price, is_high, "break", idx[j], idx[k], ext, a_prev[j], k - j))
                break
            k = j + kb[0]
            ext = h[j : k + 1].max() if is_high else l[j : k + 1].min()
            over = (ext - price) if is_high else (price - ext)
            ref = a_prev[j]
            if not np.isnan(ref) and over >= min_overshoot_atr * ref:
                events.append(_event(lv, price, is_high, "sweep", idx[j], idx[k], ext, ref, k - j))
                break
            j = k + 1  # too small (or ATR not warmed up yet): a touch, keep watching
    out = pd.DataFrame(events, columns=EVENT_COLS)
    return out.sort_values("confirmed_ts").reset_index(drop=True)


def _event(lv, price, is_high, outcome, break_ts, confirmed_ts, ext, ref, nbars):
    over = (ext - price) if is_high else (price - ext)
    return [
        lv.name, lv.side, price, lv.known_from, "bearish" if is_high else "bullish", outcome,
        break_ts, confirmed_ts, ext, over, over / ref if ref and not np.isnan(ref) else np.nan, nbars + 1,
    ]
