"""Sessions and killzones in New York time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from ..timeutils import NY, ensure_utc_index
from .indicators import bar_delta


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start: time
    end: time  # exclusive; an end at or before start means the window crosses midnight

    @classmethod
    def parse(cls, name: str, start: str, end: str) -> "SessionWindow":
        return cls(name, time.fromisoformat(start), time.fromisoformat(end))

    def contains(self, minutes_of_day: np.ndarray) -> np.ndarray:
        s = self.start.hour * 60 + self.start.minute
        e = self.end.hour * 60 + self.end.minute
        if e > s:
            return (minutes_of_day >= s) & (minutes_of_day < e)
        return (minutes_of_day >= s) | (minutes_of_day < e)


# Plan defaults (section 4.2). Adjust once, then freeze.
DEFAULT_SESSIONS: tuple[SessionWindow, ...] = (
    SessionWindow.parse("asia", "20:00", "00:00"),
    SessionWindow.parse("london_kz", "02:00", "05:00"),
    SessionWindow.parse("ny_am_kz", "07:00", "10:00"),
    SessionWindow.parse("ny_pm", "13:30", "16:00"),
)


def sessions_from_config(cfg: dict) -> tuple[SessionWindow, ...]:
    return tuple(SessionWindow.parse(k, v[0], v[1]) for k, v in cfg.items())


def trading_day(index: pd.DatetimeIndex, day_start_hour: int = 18, tz: str = NY) -> pd.DatetimeIndex:
    """The trading day each timestamp belongs to.

    With ``day_start_hour=18`` (futures convention) a bar at 18:00 New York time or later belongs
    to the next calendar day, so the Asia session (20:00-00:00) opens the day it is part of.
    """
    wall = index.tz_convert(tz).tz_localize(None)
    if day_start_hour:
        wall = wall + pd.Timedelta(hours=24 - day_start_hour)
    return wall.normalize()


def label_sessions(
    bars: pd.DataFrame,
    sessions: tuple[SessionWindow, ...] = DEFAULT_SESSIONS,
    day_start_hour: int = 18,
    tz: str = NY,
) -> pd.DataFrame:
    """Per bar: New York time, trading day, ``in_<session>`` flags and the first matching ``session`` name.

    A bar belongs to a session when its open time is inside the window.
    """
    b = ensure_utc_index(bars)
    ny = b.index.tz_convert(tz)
    mod = np.asarray(ny.hour * 60 + ny.minute)
    out = pd.DataFrame(index=b.index)
    out["ny_time"] = ny
    out["trading_day"] = trading_day(b.index, day_start_hour, tz)
    label = np.full(len(b), None, dtype=object)
    for s in sessions:
        mask = s.contains(mod)
        out[f"in_{s.name}"] = mask
        label = np.where((label == None) & mask, s.name, label)  # noqa: E711
    out["session"] = label
    return out


def session_ranges(
    bars: pd.DataFrame,
    sessions: tuple[SessionWindow, ...] = DEFAULT_SESSIONS,
    day_start_hour: int = 18,
    tz: str = NY,
) -> pd.DataFrame:
    """High, low, open and close of every session on every trading day.

    ``known_from`` is the close of the session's last bar: the range is only usable from then on.
    """
    b = ensure_utc_index(bars)
    lab = label_sessions(b, sessions, day_start_hour, tz)
    delta = bar_delta(b)
    rows = []
    for s in sessions:
        m = lab[f"in_{s.name}"].to_numpy()
        if not m.any():
            continue
        sub = b[m]
        days = lab.loc[m, "trading_day"]
        for day, g in sub.groupby(days):
            rows.append(
                {
                    "trading_day": day,
                    "session": s.name,
                    "open": g["open"].iloc[0],
                    "high": g["high"].max(),
                    "low": g["low"].min(),
                    "close": g["close"].iloc[-1],
                    "high_ts": g["high"].idxmax(),
                    "low_ts": g["low"].idxmin(),
                    "start_ts": g.index[0],
                    "known_from": g.index[-1] + delta,
                    "bars": len(g),
                }
            )
    cols = ["trading_day", "session", "open", "high", "low", "close", "high_ts", "low_ts", "start_ts", "known_from", "bars"]
    return pd.DataFrame(rows, columns=cols).sort_values(["trading_day", "start_ts"]).reset_index(drop=True)
