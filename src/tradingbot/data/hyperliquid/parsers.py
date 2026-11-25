"""Pure functions turning Hyperliquid JSON into flat rows. Kept separate so they are easy to test.

Field names follow the public API docs [verify]; parsers use ``.get`` so a missing field becomes
``None`` instead of crashing the recorder.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def _f(x) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _ms(x) -> pd.Timestamp | None:
    return None if x is None else pd.Timestamp(int(x), unit="ms", tz="UTC")


def parse_asset_ctxs(resp: Any, ts: pd.Timestamp, coins: Iterable[str] | None = None) -> list[dict]:
    """Rows of funding, open interest and prices per coin from ``metaAndAssetCtxs``."""
    meta, ctxs = resp[0], resp[1]
    wanted = set(coins) if coins else None
    rows = []
    for asset, ctx in zip(meta.get("universe", []), ctxs):
        coin = asset.get("name")
        if wanted is not None and coin not in wanted:
            continue
        impact = ctx.get("impactPxs") or [None, None]
        rows.append(
            {
                "ts": ts,
                "coin": coin,
                "funding": _f(ctx.get("funding")),
                "open_interest": _f(ctx.get("openInterest")),
                "mark_px": _f(ctx.get("markPx")),
                "oracle_px": _f(ctx.get("oraclePx")),
                "mid_px": _f(ctx.get("midPx")),
                "premium": _f(ctx.get("premium")),
                "day_ntl_vlm": _f(ctx.get("dayNtlVlm")),
                "prev_day_px": _f(ctx.get("prevDayPx")),
                "impact_bid_px": _f(impact[0]) if len(impact) > 0 else None,
                "impact_ask_px": _f(impact[1]) if len(impact) > 1 else None,
                "max_leverage": asset.get("maxLeverage"),
            }
        )
    return rows


def parse_clearinghouse(address: str, resp: dict, ts: pd.Timestamp) -> tuple[list[dict], dict]:
    """Position rows and one account row from ``clearinghouseState``."""
    positions = []
    for ap in resp.get("assetPositions", []) or []:
        p = ap.get("position", {})
        lev = p.get("leverage") or {}
        cum = p.get("cumFunding") or {}
        positions.append(
            {
                "ts": ts,
                "address": address.lower(),
                "coin": p.get("coin"),
                "szi": _f(p.get("szi")),
                "entry_px": _f(p.get("entryPx")),
                "position_value": _f(p.get("positionValue")),
                "unrealized_pnl": _f(p.get("unrealizedPnl")),
                "return_on_equity": _f(p.get("returnOnEquity")),
                "leverage_type": lev.get("type"),
                "leverage": _f(lev.get("value")),
                "liquidation_px": _f(p.get("liquidationPx")),
                "margin_used": _f(p.get("marginUsed")),
                "cum_funding_since_open": _f(cum.get("sinceOpen")),
            }
        )
    ms = resp.get("marginSummary") or {}
    account = {
        "ts": ts,
        "address": address.lower(),
        "account_value": _f(ms.get("accountValue")),
        "total_ntl_pos": _f(ms.get("totalNtlPos")),
        "total_margin_used": _f(ms.get("totalMarginUsed")),
        "withdrawable": _f(resp.get("withdrawable")),
        "n_positions": len(positions),
    }
    return positions, account


def parse_trades(msg: dict, recv_ts: pd.Timestamp) -> list[dict]:
    """Rows from a websocket ``trades`` message."""
    rows = []
    for t in msg.get("data", []) or []:
        users = t.get("users") or [None, None]
        rows.append(
            {
                "ts": _ms(t.get("time")),
                "recv_ts": recv_ts,
                "coin": t.get("coin"),
                "side": t.get("side"),  # "B" = buyer aggressor, "A" = seller aggressor
                "px": _f(t.get("px")),
                "sz": _f(t.get("sz")),
                "tid": t.get("tid"),
                "hash": t.get("hash"),
                "buyer": (users[0] or "").lower() or None,
                "seller": (users[1] or "").lower() if len(users) > 1 and users[1] else None,
            }
        )
    return rows


def parse_user_fills(msg: dict, recv_ts: pd.Timestamp) -> list[dict]:
    """Rows from a websocket ``userFills`` message. Liquidation fills carry a ``liquidation`` object."""
    data = msg.get("data") or {}
    user = (data.get("user") or "").lower()
    snapshot = bool(data.get("isSnapshot", False))
    rows = []
    for f in data.get("fills", []) or []:
        liq = f.get("liquidation") or {}
        rows.append(
            {
                "ts": _ms(f.get("time")),
                "recv_ts": recv_ts,
                "address": user,
                "coin": f.get("coin"),
                "side": f.get("side"),
                "dir": f.get("dir"),
                "px": _f(f.get("px")),
                "sz": _f(f.get("sz")),
                "start_position": _f(f.get("startPosition")),
                "closed_pnl": _f(f.get("closedPnl")),
                "fee": _f(f.get("fee")),
                "crossed": f.get("crossed"),
                "tid": f.get("tid"),
                "oid": f.get("oid"),
                "hash": f.get("hash"),
                "is_liquidation": bool(liq),
                "liquidated_user": (liq.get("liquidatedUser") or "").lower() or None,
                "liq_mark_px": _f(liq.get("markPx")),
                "liq_method": liq.get("method"),
                "is_snapshot": snapshot,
            }
        )
    return rows


def _window_pnl(row: dict, window: str) -> float:
    for name, perf in row.get("windowPerformances", []) or []:
        if name == window:
            return _f(perf.get("pnl")) or 0.0
    return 0.0


def parse_leaderboard(rows: list[dict], ts: pd.Timestamp, keep: int = 2000) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "ts": ts,
                "address": (r.get("ethAddress") or "").lower(),
                "display_name": r.get("displayName"),
                "account_value": _f(r.get("accountValue")),
                "pnl_day": _window_pnl(r, "day"),
                "pnl_week": _window_pnl(r, "week"),
                "pnl_month": _window_pnl(r, "month"),
                "pnl_all_time": _window_pnl(r, "allTime"),
            }
        )
    out.sort(key=lambda x: x["account_value"] or 0.0, reverse=True)
    return out[:keep]


def select_addresses(
    leaderboard: list[dict],
    n: int,
    rank_by: str = "pnl_month",
    min_account_value: float = 50_000,
    extra: Iterable[str] = (),
) -> list[str]:
    """Pick the addresses to track: ``extra`` first, then the top ``n`` by ``rank_by``.

    Ranking by recent PnL picks lucky wallets as well as skilled ones (survivorship bias);
    the full leaderboard is stored too, so that can be corrected later.
    """
    extra = [a.lower() for a in extra]
    picked = list(dict.fromkeys(extra))
    pool = [r for r in leaderboard if (r.get("account_value") or 0) >= min_account_value and r.get("address")]
    pool.sort(key=lambda r: r.get(rank_by) or 0.0, reverse=True)
    for r in pool:
        if len(picked) >= n + len(extra):
            break
        if r["address"] not in picked:
            picked.append(r["address"])
    return picked
