import pandas as pd

from tradingbot.ict import label_sessions, session_ranges, trading_day
from conftest import make_bars


def test_killzone_follows_new_york_dst():
    # 07:00 New York is 12:00 UTC in winter (EST) and 11:00 UTC in summer (EDT).
    winter = make_bars("2024-03-08 11:00", 120)
    summer = make_bars("2024-03-11 10:00", 120)
    w = label_sessions(winter)
    s = label_sessions(summer)
    assert w.loc[w["in_ny_am_kz"]].index[0] == pd.Timestamp("2024-03-08 12:00", tz="UTC")
    assert s.loc[s["in_ny_am_kz"]].index[0] == pd.Timestamp("2024-03-11 11:00", tz="UTC")


def test_trading_day_rolls_at_18_new_york():
    idx = pd.DatetimeIndex(["2024-01-09 22:59", "2024-01-09 23:00", "2024-01-10 01:00"], tz="UTC")  # 17:59, 18:00, 20:00 NY
    days = trading_day(idx, 18)
    assert list(days.strftime("%Y-%m-%d")) == ["2024-01-09", "2024-01-10", "2024-01-10"]
    assert list(trading_day(idx, 0).strftime("%Y-%m-%d")) == ["2024-01-09", "2024-01-09", "2024-01-09"]


def test_labels_and_session_names():
    bars = make_bars("2024-01-10 01:00", 60)  # 20:00-21:00 NY = Asia
    lab = label_sessions(bars)
    assert lab["in_asia"].all() and (lab["session"] == "asia").all()
    assert not lab["in_london_kz"].any()
    assert (lab["trading_day"] == pd.Timestamp("2024-01-10")).all()


def test_session_ranges_and_known_from():
    # Bars covering one full trading day, 18:00 NY on Jan 9 to 18:00 NY on Jan 10 (EST: +5h).
    bars = make_bars("2024-01-09 23:00", 24 * 60)
    # Put a high inside Asia (21:30 NY = 02:30 UTC) and a low inside London (03:00 NY = 08:00 UTC).
    bars.loc[pd.Timestamp("2024-01-10 02:30", tz="UTC"), "high"] = 105
    bars.loc[pd.Timestamp("2024-01-10 08:00", tz="UTC"), "low"] = 95
    r = session_ranges(bars).set_index("session")
    assert r.loc["asia", "high"] == 105
    assert r.loc["asia", "known_from"] == pd.Timestamp("2024-01-10 05:00", tz="UTC")  # 00:00 NY
    assert r.loc["london_kz", "low"] == 95
    assert r.loc["london_kz", "known_from"] == pd.Timestamp("2024-01-10 10:00", tz="UTC")  # 05:00 NY
    assert r.loc["ny_am_kz", "bars"] == 180
    assert r.loc["ny_pm", "bars"] == 150
