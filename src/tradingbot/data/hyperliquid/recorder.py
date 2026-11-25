"""Continuous recorder for Hyperliquid public data.

Streams (each a folder under ``data/raw/hyperliquid/``):

* ``asset_ctxs``   every 60 s: funding, open interest, mark/oracle/mid price per coin
* ``positions``    every 5 min: open positions (size, entry, leverage, liquidation price) of the
                   tracked addresses
* ``accounts``     every 5 min: account value and margin of the tracked addresses
* ``leaderboard``  every 6 h: the public leaderboard (kept whole, to study survivorship bias later)
* ``trades``       live websocket trades for the configured coins (with buyer/seller addresses)
* ``fills``        live websocket fills of the top tracked addresses; liquidation fills are
                   flagged (``is_liquidation``), which is the liquidation feed
* ``heartbeat``    one row per completed task cycle, used by ``tb hl-status`` to find gaps
* ``events``       websocket connects/disconnects and errors

Every task catches its own errors, backs off and keeps going, so the recorder survives network
blips. Stop it with Ctrl+C or SIGTERM; buffers are flushed before exit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from typing import Any

import pandas as pd

from .client import InfoClient
from .parsers import (
    parse_asset_ctxs,
    parse_clearinghouse,
    parse_leaderboard,
    parse_trades,
    parse_user_fills,
    select_addresses,
)
from .sink import ParquetSink

log = logging.getLogger(__name__)


def now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


class Recorder:
    def __init__(self, cfg: dict[str, Any], sink: ParquetSink, client: InfoClient | None = None):
        self.cfg = cfg
        self.sink = sink
        self.client = client or InfoClient(
            api_url=cfg["api_url"],
            leaderboard_url=cfg["leaderboard_url"],
            max_weight_per_minute=cfg.get("max_weight_per_minute", 600),
        )
        self.coins: list[str] = list(cfg.get("coins", []))
        self.addresses: list[str] = [a.lower() for a in cfg.get("extra_addresses", [])]
        self.addresses_changed = asyncio.Event()
        self.stop = asyncio.Event()
        self.ws_trade_count = 0

    # ---- helpers -------------------------------------------------------------------
    def beat(self, task: str, ok: bool = True, rows: int = 0, detail: str = "") -> None:
        self.sink.add("heartbeat", [{"ts": now(), "task": task, "ok": ok, "rows": rows, "detail": detail[:500]}])

    def event(self, kind: str, detail: str) -> None:
        log.info("%s: %s", kind, detail)
        self.sink.add("events", [{"ts": now(), "kind": kind, "detail": detail[:2000]}])

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _every(self, name: str, interval: float, fn, skip_first: bool = False) -> None:
        """Run ``fn`` every ``interval`` seconds, aligned to the clock, never dying on errors."""
        backoff = 5.0
        if skip_first:
            await self._sleep(interval - (time.time() % interval))
        while not self.stop.is_set():
            try:
                rows = await fn()
                self.beat(name, True, rows or 0)
                backoff = 5.0
                wait = interval - (time.time() % interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep recording whatever happens
                log.exception("%s failed", name)
                self.beat(name, False, 0, repr(exc))
                wait = backoff
                backoff = min(backoff * 2, interval)
            await self._sleep(max(wait, 0.5))

    # ---- polling tasks -------------------------------------------------------------
    async def poll_asset_ctxs(self) -> int:
        resp = await asyncio.to_thread(self.client.meta_and_asset_ctxs)
        rows = parse_asset_ctxs(resp, now(), self.coins or None)
        self.sink.add("asset_ctxs", rows)
        return len(rows)

    async def refresh_leaderboard(self) -> int:
        raw = await asyncio.to_thread(self.client.leaderboard)
        rows = parse_leaderboard(raw, now())
        self.sink.add("leaderboard", rows)
        picked = select_addresses(
            rows,
            int(self.cfg.get("top_addresses", 300)),
            rank_by=self.cfg.get("rank_by", "pnl_month"),
            min_account_value=float(self.cfg.get("min_account_value", 50_000)),
            extra=self.cfg.get("extra_addresses", []),
        )
        if picked and picked != self.addresses:
            old = set(self.addresses[: self._n_fill()])
            self.addresses = picked
            if set(picked[: self._n_fill()]) != old:
                self.addresses_changed.set()
            self.event("addresses", f"tracking {len(picked)} addresses")
        return len(rows)

    async def poll_positions(self) -> int:
        ts = now()
        positions, accounts, failures = [], [], 0
        for addr in list(self.addresses):
            if self.stop.is_set():
                break
            try:
                resp = await asyncio.to_thread(self.client.clearinghouse_state, addr)
            except Exception as exc:
                failures += 1
                log.warning("clearinghouseState %s failed: %s", addr, exc)
                continue
            p, a = parse_clearinghouse(addr, resp, ts)
            positions.extend(p)
            accounts.append(a)
        self.sink.add("positions", positions)
        self.sink.add("accounts", accounts)
        if failures and failures == len(self.addresses):
            raise RuntimeError("every clearinghouseState request failed")
        return len(positions)

    async def flush_loop(self) -> None:
        interval = float(self.cfg.get("flush_seconds", 300))
        while not self.stop.is_set():
            await self._sleep(interval)
            await asyncio.to_thread(self.sink.flush)

    # ---- websocket tasks -----------------------------------------------------------
    def _n_fill(self) -> int:
        return int(self.cfg.get("fill_stream_addresses", 50))

    async def _ws_session(self, name: str, subscriptions: list[dict], handle, restart: asyncio.Event | None = None):
        import websockets

        backoff = 1.0
        while not self.stop.is_set():
            try:
                async with websockets.connect(self.cfg["ws_url"], ping_interval=None, max_size=2**24) as ws:
                    for sub in subscriptions:
                        await ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
                    self.event("ws_connect", f"{name}: {len(subscriptions)} subscriptions")
                    backoff = 1.0
                    last_ping = time.time()
                    last_msg = time.time()
                    while not self.stop.is_set():
                        if restart is not None and restart.is_set():
                            restart.clear()
                            self.event("ws_restart", name)
                            return
                        if time.time() - last_ping > 30:  # server drops idle connections after ~60 s
                            await ws.send(json.dumps({"method": "ping"}))
                            last_ping = time.time()
                        if time.time() - last_msg > 120:
                            raise TimeoutError("no websocket message for 120 s")
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1)
                        except asyncio.TimeoutError:
                            continue
                        last_msg = time.time()
                        msg = json.loads(raw)
                        handle(msg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.event("ws_error", f"{name}: {exc!r}")
                await self._sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _handle_trades(self, msg: dict) -> None:
        if msg.get("channel") == "trades":
            rows = parse_trades(msg, now())
            self.ws_trade_count += len(rows)
            self.sink.add("trades", rows)

    def _handle_fills(self, msg: dict) -> None:
        if msg.get("channel") == "userFills":
            self.sink.add("fills", parse_user_fills(msg, now()))

    async def trades_ws(self) -> None:
        subs = [{"type": "trades", "coin": c} for c in self.coins]
        if subs:
            await self._ws_session("trades", subs, self._handle_trades)

    async def count_ws_trades(self) -> int:
        """Heartbeat for the trades websocket: number of trades received in the last minute."""
        n, self.ws_trade_count = self.ws_trade_count, 0
        if n == 0 and self.coins:
            raise RuntimeError("no trades received in the last minute")
        return n

    async def fills_ws(self) -> None:
        n = self._n_fill()
        while not self.stop.is_set():
            addrs = self.addresses[:n]
            if not addrs:
                await self._sleep(10)
                continue
            subs = [{"type": "userFills", "user": a} for a in addrs]
            await self._ws_session("fills", subs, self._handle_fills, restart=self.addresses_changed)

    # ---- run -----------------------------------------------------------------------
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop.set)
            except (NotImplementedError, RuntimeError):  # Windows
                pass
        self.event("start", f"coins={self.coins}")
        try:
            await self.refresh_leaderboard()
        except Exception as exc:
            self.event("leaderboard_error", repr(exc))
        tasks = [
            asyncio.create_task(self._every("asset_ctxs", self.cfg.get("asset_ctx_interval", 60), self.poll_asset_ctxs)),
            asyncio.create_task(self._every("positions", self.cfg.get("positions_interval", 300), self.poll_positions)),
            asyncio.create_task(self._every("leaderboard", self.cfg.get("leaderboard_refresh", 21600), self.refresh_leaderboard, skip_first=True)),
            asyncio.create_task(self._every("ws_trades", 60, self.count_ws_trades, skip_first=True)),
            asyncio.create_task(self.trades_ws()),
            asyncio.create_task(self.fills_ws()),
            asyncio.create_task(self.flush_loop()),
        ]
        await self.stop.wait()
        self.event("stop", "shutting down")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.sink.flush()
        log.info("recorder stopped; rows written: %s", dict(self.sink.rows_written))


def run_recorder(cfg: dict[str, Any], sink: ParquetSink) -> None:
    asyncio.run(Recorder(cfg, sink).run())
