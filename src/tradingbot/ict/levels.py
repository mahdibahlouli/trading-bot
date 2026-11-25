"""Liquidity levels: previous day/week highs and lows, session highs and lows, swings, equal highs/lows.

Every function returns a *level table* with the same columns, so they can be concatenated and fed
to :func:`tradingbot.ict.sweeps.detect_sweeps`:

``name, side, price, known_from, valid_until, source_ts``

* ``side`` is ``"high"`` (buy-side liquidity above) or ``"low"`` (sell-side liquidity below).
* ``known_from``: first bar open time at which the level may be used.
* ``valid_until``: bar open time from which the level is no longer watched (``NaT`` = end of data).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..timeutils import NY, ensure_utc_index
from .indicators import atr as _atr
from .indicators import bar_delta
from .sessions import DEFAULT_SESSIONS, SessionWindow, session_ranges, trading_day

LEVEL_COLS = ["name", "side", "price", "known_from", "valid_until", "source_ts"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=LEVEL_COLS)


def _day_table(bars: pd.DataFrame, day_start_hour: int, tz: str) -> pd.DataFrame:
    b = ensure_utc_index(bars)
    days = trading_day(b.index, day_start_hour, tz)
    g = b.groupby(days)
    t = pd.DataFrame(
        {
            "high": g["high"].max(),
            "low": g["low"].min(),
            "high_ts": g["high"].idxmax(),
            "low_ts": g["low"].idxmin(),
            "first_ts": g.apply(lambda x: x.index[0]),
            "last_ts": g.apply(lambda x: x.index[-1]),
        }
    )
    return t


def previous_day_levels(bars: pd.DataFrame, day_start_hour: int = 18, tz: str = NY) -> pd.DataFrame:
    """PDH/PDL: the previous trading day's high and low, watched for the whole next trading day."""
    t = _day_table(bars, day_start_hour, tz)
    if len(t) < 2:
        return _empty()
    delta = bar_delta(ensure_utc_index(bars))
    rows = []
    for (_, prev), (_, cur) in zip(t.iloc[:-1].iterrows(), t.iloc[1:].iterrows()):
        valid_until = cur["last_ts"] + delta
        rows.append(["pdh", "high", prev["high"], cur["first_ts"], valid_until, prev["high_ts"]])
        rows.append(["pdl", "low", prev["low"], cur["first_ts"], valid_until, prev["low_ts"]])
    return pd.DataFrame(rows, columns=LEVEL_COLS)


def previous_week_levels(bars: pd.DataFrame, day_start_hour: int = 18, tz: str = NY) -> pd.DataFrame:
    """Previous week's high and low (weeks of trading days, Monday to Sunday), watched all next week."""
    t = _day_table(bars, day_start_hour, tz)
    if t.empty:
        return _empty()
    week = t.index.to_period("W-SUN")
    g = t.groupby(week)
    w = pd.DataFrame(
        {
            "high": g["high"].max(),
            "low": g["low"].min(),
            "high_ts": g.apply(lambda x: x.loc[x["high"].idxmax(), "high_ts"]),
            "low_ts": g.apply(lambda x: x.loc[x["low"].idxmin(), "low_ts"]),
            "first_ts": g["first_ts"].min(),
            "last_ts": g["last_ts"].max(),
        }
    )
    delta = bar_delta(ensure_utc_index(bars))
    rows = []
    for (_, prev), (_, cur) in zip(w.iloc[:-1].iterrows(), w.iloc[1:].iterrows()):
        valid_until = cur["last_ts"] + delta
        rows.append(["pwh", "high", prev["high"], cur["first_ts"], valid_until, prev["high_ts"]])
        rows.append(["pwl", "low", prev["low"], cur["first_ts"], valid_until, prev["low_ts"]])
    return pd.DataFrame(rows, columns=LEVEL_COLS)


def session_levels(
    bars: pd.DataFrame,
    sessions: tuple[SessionWindow, ...] = DEFAULT_SESSIONS,
    which: tuple[str, ...] = ("asia", "london_kz"),
    day_start_hour: int = 18,
    tz: str = NY,
) -> pd.DataFrame:
    """Session highs and lows (default: Asia and London), watched from the session's end to the end of that trading day."""
    b = ensure_utc_index(bars)
    r = session_ranges(b, sessions, day_start_hour, tz)
    r = r[r["session"].isin(which)]
    if r.empty:
        return _empty()
    day_end = _day_table(b, day_start_hour, tz)["last_ts"] + bar_delta(b)
    rows = []
    for _, s in r.iterrows():
        until = day_end.get(s["trading_day"], pd.NaT)
        rows.append([f"{s['session']}_high", "high", s["high"], s["known_from"], until, s["high_ts"]])
        rows.append([f"{s['session']}_low", "low", s["low"], s["known_from"], until, s["low_ts"]])
    return pd.DataFrame(rows, columns=LEVEL_COLS)


def swings(bars: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Swing highs and lows with ``n`` bars on each side.

    A swing high at bar ``i``: ``high[i]`` is above the ``n`` bars before it and at least as high as
    the ``n`` bars after it (so of two equal highs, the first one is the swing). It is only known
    once bar ``i + n`` has closed: ``confirmed_ts`` is that bar's time and ``known_from`` the next
    bar's open time. Using a swing before ``known_from`` is lookahead bias.
    """
    b = ensure_utc_index(bars)
    h, l = b["high"].to_numpy(), b["low"].to_numpy()
    m = len(b)
    delta = bar_delta(b)
    rows = []
    if m < 2 * n + 1:
        return pd.DataFrame(columns=["ts", "side", "price", "confirmed_ts", "known_from"])
    from numpy.lib.stride_tricks import sliding_window_view

    hw = sliding_window_view(h, 2 * n + 1)
    lw = sliding_window_view(l, 2 * n + 1)
    centre_h, centre_l = hw[:, n], lw[:, n]
    is_sh = (centre_h > hw[:, :n].max(axis=1)) & (centre_h >= hw[:, n + 1 :].max(axis=1))
    is_sl = (centre_l < lw[:, :n].min(axis=1)) & (centre_l <= lw[:, n + 1 :].min(axis=1))
    idx = b.index
    for side, mask, vals in (("high", is_sh, centre_h), ("low", is_sl, centre_l)):
        for k in np.flatnonzero(mask):
            i = k + n
            rows.append(
                {
                    "ts": idx[i],
                    "side": side,
                    "price": vals[k],
                    "confirmed_ts": idx[i + n],
                    "known_from": idx[i + n] + delta,
                }
            )
    return pd.DataFrame(rows).sort_values(["confirmed_ts", "side"]).reset_index(drop=True)


def swing_levels(bars: pd.DataFrame, n: int = 3, max_age: pd.Timedelta | str | None = "3D") -> pd.DataFrame:
    """Confirmed swing highs/lows as levels, watched for ``max_age`` after confirmation."""
    s = swings(bars, n)
    if s.empty:
        return _empty()
    until = s["known_from"] + pd.Timedelta(max_age) if max_age is not None else pd.NaT
    return pd.DataFrame(
        {
            "name": "swing_" + s["side"],
            "side": s["side"],
            "price": s["price"],
            "known_from": s["known_from"],
            "valid_until": until,
            "source_ts": s["ts"],
        }
    )[LEVEL_COLS]


def equal_levels(
    bars: pd.DataFrame,
    n: int = 3,
    tol_atr: float = 0.05,
    atr_period: int = 14,
    max_age: pd.Timedelta | str | None = "3D",
) -> pd.DataFrame:
    """Equal highs/lows: two consecutive confirmed swings on the same side within ``tol_atr`` x ATR.

    The level is the more extreme of the two and is known once the second swing is confirmed.
    """
    b = ensure_utc_index(bars)
    s = swings(b, n)
    if s.empty:
        return _empty()
    a = _atr(b, atr_period)
    rows = []
    for side, g in s.groupby("side"):
        g = g.sort_values("ts")
        for (_, first), (_, second) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            ref = a.get(second["confirmed_ts"], np.nan)
            if np.isnan(ref):
                continue
            if abs(first["price"] - second["price"]) <= tol_atr * ref:
                price = max(first["price"], second["price"]) if side == "high" else min(first["price"], second["price"])
                until = second["known_from"] + pd.Timedelta(max_age) if max_age is not None else pd.NaT
                rows.append([f"equal_{side}s", side, price, second["known_from"], until, second["ts"]])
    return pd.DataFrame(rows, columns=LEVEL_COLS) if rows else _empty()
