import pandas as pd

from tradingbot.ict import detect_sweeps
from tradingbot.ict.levels import LEVEL_COLS
from conftest import make_bars

# Flat bars around 100 with a true range of 1, so ATR(14) = 1 once warmed up.


def level(price, side="high", known_from="2024-01-02 00:20", valid_until=None, name="pdh"):
    return pd.DataFrame(
        [[name, side, price, pd.Timestamp(known_from, tz="UTC"),
          pd.Timestamp(valid_until, tz="UTC") if valid_until else pd.NaT, pd.NaT]],
        columns=LEVEL_COLS,
    )


def set_bar(bars, i, **kw):
    for k, v in kw.items():
        bars.iloc[i, bars.columns.get_loc(k)] = v


def test_sweep_of_high_wick_and_close_back():
    bars = make_bars("2024-01-02", 60)
    set_bar(bars, 30, high=101.2, close=100.0)  # trades 0.2 above 101 and closes back below
    ev = detect_sweeps(bars, level(101.0))
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["direction"] == "bearish" and e["outcome"] == "sweep"
    assert e["break_ts"] == e["confirmed_ts"] == bars.index[30]
    assert abs(e["overshoot"] - 0.2) < 1e-9 and abs(e["overshoot_atr"] - 0.2) < 1e-6


def test_sweep_confirmed_bars_later():
    bars = make_bars("2024-01-02", 60)
    set_bar(bars, 30, high=101.3, close=101.2)
    set_bar(bars, 31, high=101.5, close=101.1)
    set_bar(bars, 32, high=101.2, close=100.8)
    e = detect_sweeps(bars, level(101.0), max_bars=3).iloc[0]
    assert e["confirmed_ts"] == bars.index[32] and e["bars_to_confirm"] == 3
    assert e["extreme"] == 101.5


def test_no_close_back_in_time_is_a_break():
    bars = make_bars("2024-01-02", 60)
    for i in range(30, 60):
        set_bar(bars, i, open=101.5, high=102.0, low=101.2, close=101.6)
    assert detect_sweeps(bars, level(101.0)).empty
    br = detect_sweeps(bars, level(101.0), include_breaks=True)
    assert list(br["outcome"]) == ["break"]


def test_tiny_poke_is_a_touch_and_level_stays_alive():
    bars = make_bars("2024-01-02", 60)
    set_bar(bars, 25, high=101.05)          # 0.05 ATR: below the 0.1 minimum
    set_bar(bars, 40, high=101.4)           # real sweep later
    e = detect_sweeps(bars, level(101.0), min_overshoot_atr=0.1)
    assert len(e) == 1 and e["break_ts"].iloc[0] == bars.index[40]


def test_level_not_used_before_known_from_or_after_valid_until():
    bars = make_bars("2024-01-02", 60)
    set_bar(bars, 18, high=101.5)  # before known_from (00:20)
    assert detect_sweeps(bars, level(101.0)).empty
    set_bar(bars, 50, high=101.5)
    assert detect_sweeps(bars, level(101.0, valid_until="2024-01-02 00:45")).empty
    assert len(detect_sweeps(bars, level(101.0))) == 1


def test_sweep_of_low_is_bullish():
    bars = make_bars("2024-01-02", 60)
    set_bar(bars, 30, low=98.7, close=100.0)
    e = detect_sweeps(bars, level(99.0, side="low", name="asia_low")).iloc[0]
    assert e["direction"] == "bullish" and e["level_name"] == "asia_low"
    assert abs(e["overshoot"] - 0.3) < 1e-9
