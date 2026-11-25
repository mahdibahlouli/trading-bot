import asyncio

import pandas as pd
import pytest

from tradingbot.data.hyperliquid.analysis import gap_report, infer_liquidations, load_stream
from tradingbot.data.hyperliquid.client import WeightLimiter, request_weight
from tradingbot.data.hyperliquid.parsers import (
    parse_asset_ctxs,
    parse_clearinghouse,
    parse_leaderboard,
    parse_trades,
    parse_user_fills,
    select_addresses,
)
from tradingbot.data.hyperliquid.recorder import Recorder
from tradingbot.data.hyperliquid.sink import ParquetSink, compact_day

TS = pd.Timestamp("2026-11-25 12:00", tz="UTC")

META = [
    {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40}, {"name": "ETH", "szDecimals": 4, "maxLeverage": 25}]},
    [
        {"funding": "0.0000125", "openInterest": "25000.5", "markPx": "95000.0", "oraclePx": "95010.0",
         "midPx": "95001.5", "premium": "-0.0001", "dayNtlVlm": "1500000000", "prevDayPx": "94000",
         "impactPxs": ["95000", "95003"]},
        {"funding": "0.00001", "openInterest": "500000", "markPx": "3500", "oraclePx": "3501",
         "midPx": None, "premium": "0", "dayNtlVlm": "800000000", "prevDayPx": "3400", "impactPxs": None},
    ],
]

CLEARING = {
    "assetPositions": [
        {"type": "oneWay", "position": {
            "coin": "BTC", "szi": "-2.5", "entryPx": "96000", "positionValue": "237500", "unrealizedPnl": "2500",
            "returnOnEquity": "0.1", "leverage": {"type": "cross", "value": 10}, "liquidationPx": "120000",
            "marginUsed": "23750", "cumFunding": {"allTime": "-50", "sinceOpen": "-10", "sinceChange": "-1"}}},
    ],
    "marginSummary": {"accountValue": "500000", "totalNtlPos": "237500", "totalMarginUsed": "23750"},
    "withdrawable": "476250",
}


def test_request_weights():
    assert request_weight("clearinghouseState") == 2
    assert request_weight("metaAndAssetCtxs") == 20


def test_weight_limiter_waits_when_empty():
    clock = {"t": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    lim = WeightLimiter(60, clock=lambda: clock["t"], sleep=sleep)  # 1 weight per second
    lim.acquire(60)
    assert slept == []
    lim.acquire(20)
    assert pytest.approx(sum(slept), rel=1e-6) == 20


def test_parse_asset_ctxs_filters_coins():
    rows = parse_asset_ctxs(META, TS, ["BTC"])
    assert len(rows) == 1
    r = rows[0]
    assert r["coin"] == "BTC" and r["funding"] == 0.0000125 and r["open_interest"] == 25000.5
    assert r["impact_ask_px"] == 95003.0
    eth = parse_asset_ctxs(META, TS)[1]
    assert eth["mid_px"] is None and eth["impact_bid_px"] is None


def test_parse_clearinghouse():
    pos, acct = parse_clearinghouse("0xABC", CLEARING, TS)
    assert pos[0]["address"] == "0xabc" and pos[0]["szi"] == -2.5 and pos[0]["liquidation_px"] == 120000
    assert pos[0]["leverage"] == 10 and pos[0]["leverage_type"] == "cross"
    assert acct["account_value"] == 500000 and acct["n_positions"] == 1


def test_parse_trades_and_fills():
    msg = {"channel": "trades", "data": [{"coin": "BTC", "side": "B", "px": "95000", "sz": "0.1", "time": 1764072000000,
                                          "tid": 1, "hash": "0x1", "users": ["0xBUY", "0xSELL"]}]}
    t = parse_trades(msg, TS)[0]
    assert t["buyer"] == "0xbuy" and t["seller"] == "0xsell" and t["ts"] == pd.Timestamp(1764072000000, unit="ms", tz="UTC")
    fills = {"channel": "userFills", "data": {"user": "0xABC", "isSnapshot": False, "fills": [
        {"coin": "ETH", "px": "3400", "sz": "10", "side": "A", "time": 1764072000000, "dir": "Close Long",
         "closedPnl": "-5000", "fee": "1", "tid": 7, "oid": 8, "hash": "0x2", "crossed": True, "startPosition": "10",
         "liquidation": {"liquidatedUser": "0xABC", "markPx": "3399", "method": "market"}},
        {"coin": "ETH", "px": "3500", "sz": "1", "side": "B", "time": 1764072001000, "tid": 9},
    ]}}
    rows = parse_user_fills(fills, TS)
    assert rows[0]["is_liquidation"] and rows[0]["liq_mark_px"] == 3399 and rows[0]["liquidated_user"] == "0xabc"
    assert not rows[1]["is_liquidation"]


def test_leaderboard_selection():
    raw = [
        {"ethAddress": "0xA", "accountValue": "1000000", "windowPerformances": [["month", {"pnl": "10"}]]},
        {"ethAddress": "0xB", "accountValue": "200000", "windowPerformances": [["month", {"pnl": "500"}]]},
        {"ethAddress": "0xC", "accountValue": "10", "windowPerformances": [["month", {"pnl": "9999"}]]},
        {"ethAddress": "0xD", "accountValue": "90000", "windowPerformances": [["month", {"pnl": "100"}]]},
    ]
    lb = parse_leaderboard(raw, TS)
    assert lb[0]["address"] == "0xa"
    picked = select_addresses(lb, 2, extra=["0xE"])
    assert picked == ["0xe", "0xb", "0xd"]  # 0xC is below the minimum account value


def test_sink_partitions_by_day_and_compacts(tmp_path):
    sink = ParquetSink(tmp_path)
    sink.add("trades", [{"ts": pd.Timestamp("2026-11-24 23:59", tz="UTC"), "px": 1.0},
                        {"ts": pd.Timestamp("2026-11-25 00:01", tz="UTC"), "px": 2.0}])
    paths = sink.flush()
    assert len(paths) == 2
    sink.add("trades", [{"ts": pd.Timestamp("2026-11-24 10:00", tz="UTC"), "px": 3.0}])
    sink.flush()
    assert compact_day(tmp_path, "trades", "2026-11-24") is not None
    day = tmp_path / "raw/hyperliquid/trades/date=2026-11-24"
    assert [p.name for p in day.glob("*.parquet")] == ["compacted.parquet"]
    df = load_stream(tmp_path, "trades")
    assert sorted(df["px"]) == [1.0, 2.0, 3.0]


def test_gap_report_detects_gap():
    end = pd.Timestamp("2026-11-25 12:00", tz="UTC")
    ts = pd.date_range(end - pd.Timedelta(hours=1), end, freq="60s")
    hb = pd.DataFrame({"ts": ts, "task": "asset_ctxs", "ok": True})
    rep = gap_report(hb, hours=1, end=end, expected={"asset_ctxs": 60})
    assert bool(rep["pass"].iloc[0])
    hb2 = hb[(hb["ts"] < end - pd.Timedelta(minutes=40)) | (hb["ts"] > end - pd.Timedelta(minutes=30))]
    rep2 = gap_report(hb2, hours=1, end=end, expected={"asset_ctxs": 60})
    assert rep2["gaps"].iloc[0] == 1 and not rep2["pass"].iloc[0]


def test_infer_liquidations():
    t0, t1 = pd.Timestamp("2026-11-25 12:00", tz="UTC"), pd.Timestamp("2026-11-25 12:05", tz="UTC")
    positions = pd.DataFrame([
        {"ts": t0, "address": "0xa", "coin": "BTC", "szi": 1.0, "liquidation_px": 90000.0},   # liquidated
        {"ts": t0, "address": "0xb", "coin": "BTC", "szi": 1.0, "liquidation_px": 80000.0},   # closed by choice
        {"ts": t0, "address": "0xc", "coin": "BTC", "szi": 1.0, "liquidation_px": 85000.0},   # still open
        {"ts": t1, "address": "0xc", "coin": "BTC", "szi": 1.0, "liquidation_px": 85000.0},
    ])
    marks = pd.DataFrame({"ts": [t0 + pd.Timedelta(minutes=m) for m in range(1, 6)], "coin": "BTC",
                          "mark_px": [95000, 92000, 89900, 91000, 93000]})
    liq = infer_liquidations(positions, marks)
    assert list(liq["address"]) == ["0xa"]
    assert liq["side"].iloc[0] == "long"


class FakeClient:
    def __init__(self):
        self.calls = []

    def meta_and_asset_ctxs(self):
        return META

    def clearinghouse_state(self, addr):
        self.calls.append(addr)
        if addr == "0xbad":
            raise RuntimeError("boom")
        return CLEARING

    def leaderboard(self):
        return [{"ethAddress": f"0x{i}", "accountValue": "1000000",
                 "windowPerformances": [["month", {"pnl": str(i)}]]} for i in range(5)]


def test_recorder_poll_cycles(tmp_path):
    cfg = {"coins": ["BTC"], "top_addresses": 2, "fill_stream_addresses": 2, "extra_addresses": ["0xbad"],
           "api_url": "", "leaderboard_url": "", "ws_url": ""}
    sink = ParquetSink(tmp_path)

    async def go():
        rec = Recorder(cfg, sink, client=FakeClient())
        await rec.refresh_leaderboard()
        assert rec.addresses == ["0xbad", "0x4", "0x3"]
        assert rec.addresses_changed.is_set()
        assert await rec.poll_positions() == 2  # one address failed, the others still recorded
        assert await rec.poll_asset_ctxs() == 1

    asyncio.run(go())
    sink.flush()
    assert len(load_stream(tmp_path, "positions")) == 2
    assert len(load_stream(tmp_path, "accounts")) == 2
    assert len(load_stream(tmp_path, "leaderboard")) == 5


def test_recorder_websocket_end_to_end(tmp_path):
    """Run the real websocket loop against a local server that mimics Hyperliquid."""
    import json

    import websockets

    received = []

    async def handler(ws):
        async for raw in ws:
            msg = json.loads(raw)
            received.append(msg)
            if msg.get("method") == "subscribe":
                coin = msg["subscription"]["coin"]
                await ws.send(json.dumps({"channel": "subscriptionResponse", "data": msg}))
                await ws.send(json.dumps({"channel": "trades", "data": [
                    {"coin": coin, "side": "A", "px": "95000", "sz": "0.5", "time": 1764072000000,
                     "tid": 42, "hash": "0x9", "users": ["0xb", "0xs"]}]}))

    async def go():
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            cfg = {"coins": ["BTC", "ETH"], "ws_url": f"ws://127.0.0.1:{port}", "api_url": "", "leaderboard_url": ""}
            rec = Recorder(cfg, ParquetSink(tmp_path), client=FakeClient())
            task = asyncio.create_task(rec.trades_ws())
            for _ in range(100):
                if rec.ws_trade_count >= 2:
                    break
                await asyncio.sleep(0.05)
            rec.stop.set()
            await asyncio.wait_for(task, 10)
            assert await rec.count_ws_trades() == 2
            rec.sink.flush()

    asyncio.run(go())
    df = load_stream(tmp_path, "trades")
    assert sorted(df["coin"]) == ["BTC", "ETH"]
    assert {m["subscription"]["coin"] for m in received if m.get("method") == "subscribe"} == {"BTC", "ETH"}
