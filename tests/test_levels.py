import pandas as pd

from tradingbot.ict import equal_levels, previous_day_levels, previous_week_levels, session_levels, swings
from conftest import make_bars


def test_previous_day_levels_are_known_only_next_day():
    bars = make_bars("2024-01-09 23:00", 2 * 24 * 60)  # trading days Jan 10 and Jan 11
    bars.loc[pd.Timestamp("2024-01-10 15:00", tz="UTC"), "high"] = 110
    lv = previous_day_levels(bars).set_index("name")
    assert lv.loc["pdh", "price"] == 110
    assert lv.loc["pdh", "known_from"] == pd.Timestamp("2024-01-10 23:00", tz="UTC")  # 18:00 NY
    assert lv.loc["pdh", "valid_until"] == pd.Timestamp("2024-01-11 23:00", tz="UTC")
    assert lv.loc["pdl", "price"] == 99.5


def test_previous_week_levels():
    bars = make_bars("2024-01-07 23:00", 14 * 24 * 60, freq="1min")
    bars.loc[pd.Timestamp("2024-01-09 15:00", tz="UTC"), "low"] = 90
    lv = previous_week_levels(bars)
    pwl = lv[lv["name"] == "pwl"].iloc[0]
    assert pwl["price"] == 90
    assert pwl["known_from"] > pd.Timestamp("2024-01-13", tz="UTC")


def test_session_levels():
    bars = make_bars("2024-01-09 23:00", 24 * 60)
    lv = session_levels(bars)
    assert set(lv["name"]) == {"asia_high", "asia_low", "london_kz_high", "london_kz_low"}
    assert (lv["valid_until"] == pd.Timestamp("2024-01-10 23:00", tz="UTC")).all()


def test_swing_is_known_only_n_bars_later():
    highs = [1, 2, 3, 6, 3, 2, 1, 1, 1]
    idx = pd.date_range("2024-01-02", periods=len(highs), freq="1min", tz="UTC")
    bars = pd.DataFrame({"open": 0.5, "high": highs, "low": 0.0, "close": 0.5}, index=idx)
    s = swings(bars, n=3)
    sh = s[s["side"] == "high"].iloc[0]
    assert sh["ts"] == idx[3] and sh["price"] == 6
    assert sh["confirmed_ts"] == idx[6]
    assert sh["known_from"] == idx[7]


def test_equal_highs_first_one_is_the_swing():
    highs = [1, 2, 5, 5, 2, 1, 1]
    idx = pd.date_range("2024-01-02", periods=len(highs), freq="1min", tz="UTC")
    bars = pd.DataFrame({"open": 0.5, "high": highs, "low": 0.0, "close": 0.5}, index=idx)
    s = swings(bars, n=2)
    assert list(s[s["side"] == "high"]["ts"]) == [idx[2]]


def test_equal_levels():
    bars = make_bars("2024-01-02", 60)
    for i, h in ((10, 103.0), (30, 103.02)):
        bars.iloc[i, bars.columns.get_loc("high")] = h
    lv = equal_levels(bars, n=3, tol_atr=0.05)
    assert len(lv) == 1 and lv["price"].iloc[0] == 103.02
