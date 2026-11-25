import pandas as pd

from tradingbot.ict import find_fvgs
from conftest import make_bars


def bars_from(ohlc, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(ohlc), freq="1min", tz="UTC")
    return pd.DataFrame(ohlc, columns=["open", "high", "low", "close"], index=idx)


def test_bullish_fvg_touch_ce_and_invalidation():
    b = bars_from([
        (100, 101, 99, 100.5),    # 0: high 101
        (100.5, 104, 100.4, 103.8),  # 1: displacement
        (103.8, 105, 102, 104.5),  # 2: low 102 > 101 -> gap 101..102
        (104.5, 104.8, 102.8, 103),  # 3: no touch
        (103, 103.2, 101.9, 102.5),  # 4: touch (low <= 102)
        (102.5, 102.6, 101.4, 101.8),  # 5: reaches CE 101.5
        (101.8, 101.9, 100.2, 100.5),  # 6: close below 101 -> invalidated
    ])
    f = find_fvgs(b, ttl=10)
    assert list(f["direction"]) == ["bullish", "bearish"]  # bars 3 and 5 also leave a bearish gap
    g = f.iloc[0]
    assert g["direction"] == "bullish" and (g["bottom"], g["top"], g["mid"]) == (101, 102, 101.5)
    assert g["ts"] == b.index[2]
    assert g["first_touch_ts"] == b.index[4]
    assert g["ce_ts"] == b.index[5]
    assert g["invalidated_ts"] == b.index[6]
    assert g["status"] == "invalidated"


def test_bearish_fvg_expires_untouched():
    rows = [
        (100, 101, 99, 99.5),     # 0: low 99
        (99.5, 99.6, 96, 96.2),   # 1
        (96.2, 98, 95, 95.5),     # 2: high 98 < 99 -> gap 98..99
    ] + [(95.5, 96, 95, 95.5)] * 5
    b = bars_from(rows)
    g = find_fvgs(b, ttl=5).iloc[0]
    assert g["direction"] == "bearish" and (g["bottom"], g["top"]) == (98, 99)
    assert g["status"] == "expired" and g["expired_ts"] == b.index[7]
    assert pd.isna(g["first_touch_ts"])


def test_open_when_data_ends_before_ttl():
    rows = [(100, 101, 99, 100.5), (100.5, 104, 100.4, 103.8), (103.8, 105, 102, 104.5), (104.5, 105, 103, 104)]
    g = find_fvgs(bars_from(rows), ttl=10).iloc[0]
    assert g["status"] == "open" and pd.isna(g["expired_ts"])


def test_min_size_filter_uses_atr():
    base = make_bars("2024-01-02", 40)  # ATR = 1
    tail = bars_from([(100, 100.5, 99.5, 100), (100, 100.9, 100, 100.8), (100.8, 101.2, 100.6, 101)],
                     start="2024-01-02 00:40")
    b = pd.concat([base, tail.assign(volume=1.0)])
    assert len(find_fvgs(b, min_size_atr=0.0)) == 1   # gap 100.5..100.6 = 0.1 ATR
    assert find_fvgs(b, min_size_atr=0.5).empty


def test_events_after_invalidation_ignored():
    b = bars_from([
        (100, 101, 99, 100.5), (100.5, 104, 100.4, 103.8), (103.8, 105, 102, 104.5),
        (104, 104.2, 100.5, 100.6),  # touches and invalidates in the same bar
    ])
    g = find_fvgs(b, ttl=10).iloc[0]
    assert g["first_touch_ts"] == b.index[3] and g["invalidated_ts"] == b.index[3]
