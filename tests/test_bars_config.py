import pandas as pd

from tradingbot.config import load_config
from tradingbot.data.bars import load_bars, quality_report, resample_bars, ticks_to_bars
from tradingbot.storage import write_parquet
from conftest import make_bars


def test_config_loads_defaults_and_override(tmp_path):
    cfg = load_config()
    assert cfg["hyperliquid"]["positions_interval"] == 300
    assert cfg["ict"]["sessions"]["asia"] == ["20:00", "00:00"]
    extra = tmp_path / "x.toml"
    extra.write_text('[hyperliquid]\ncoins = ["BTC"]\n')
    cfg2 = load_config(extra)
    assert cfg2["hyperliquid"]["coins"] == ["BTC"] and cfg2["hyperliquid"]["positions_interval"] == 300


def test_resample_5m():
    b = make_bars("2024-01-02", 10)
    b.iloc[3, b.columns.get_loc("high")] = 105
    r = resample_bars(b, "5min")
    assert len(r) == 2 and r["high"].iloc[0] == 105 and r["volume"].iloc[0] == 5


def test_ticks_to_bars_mid():
    ticks = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02 00:00:10", "2024-01-02 00:00:50"], utc=True),
                          "bid": [10.0, 12.0], "ask": [10.2, 12.2]})
    bars = ticks_to_bars(ticks, price="mid")
    assert bars["open"].iloc[0] == 10.1 and bars["close"].iloc[0] == 12.1
    assert abs(bars["spread"].iloc[0] - 0.2) < 1e-9


def test_quality_report_counts_problems():
    b = make_bars("2024-01-02 15:00", 60)  # Tuesday
    b = b.drop(b.index[10:13])                 # 3 missing minutes
    b.iloc[20, b.columns.get_loc("high")] = 99  # high below open -> bad OHLC
    raw = pd.concat([b.reset_index(), b.reset_index().iloc[[5]]])  # one duplicate
    rep = quality_report(raw)
    assert rep.missing_minutes == 3 and rep.duplicate_bars == 1 and rep.bad_ohlc == 1 and rep.weekend_bars == 0


def test_load_bars_from_store(tmp_path):
    b = make_bars("2024-01-02", 30).reset_index()
    write_parquet(b, tmp_path / "bars/dukascopy/XAUUSD/1m/date=2024-01-02.parquet")
    out = load_bars("dukascopy", "XAUUSD", "15min", start="2024-01-02 00:00", root=tmp_path)
    assert len(out) == 2 and out.index.tz is not None
