"""Checks and derived data built from what the recorder saved."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...storage import read_parquet_glob

EXPECTED_INTERVALS = {"asset_ctxs": 60, "positions": 300, "ws_trades": 60}


def gap_report(heartbeat: pd.DataFrame, hours: float = 72.0, end: pd.Timestamp | None = None,
               tolerance: float = 2.5, expected: dict[str, float] | None = None) -> pd.DataFrame:
    """Per task: successful cycles, failures, and gaps over the last ``hours``.

    A gap is a stretch between successful cycles longer than ``tolerance`` x the task's interval
    (plus the stretch before the first and after the last success inside the window).
    The plan's Phase 0 gate is: 72 hours with no gaps.
    """
    expected = expected or EXPECTED_INTERVALS
    end = end or pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(hours=hours)
    hb = heartbeat.copy()
    hb["ts"] = pd.to_datetime(hb["ts"], utc=True)
    hb = hb[(hb["ts"] >= start) & (hb["ts"] <= end)]
    rows = []
    for task, interval in expected.items():
        t = hb[hb["task"] == task]
        ok = t[t["ok"]].sort_values("ts")["ts"]
        limit = pd.Timedelta(seconds=interval * tolerance)
        edges = pd.concat([pd.Series([start]), ok, pd.Series([end])], ignore_index=True)
        d = edges.diff().dropna()
        gaps = d[d > limit]
        rows.append(
            {
                "task": task,
                "ok_cycles": int(len(ok)),
                "failed_cycles": int((~t["ok"]).sum()),
                "gaps": int(len(gaps)),
                "longest_gap_min": round(d.max().total_seconds() / 60, 1) if len(d) else None,
                "pass": len(gaps) == 0,
            }
        )
    return pd.DataFrame(rows)


def load_stream(root: Path, stream: str) -> pd.DataFrame:
    return read_parquet_glob(Path(root) / "raw" / "hyperliquid" / stream / "*" / "*.parquet")


def infer_liquidations(positions: pd.DataFrame, marks: pd.DataFrame) -> pd.DataFrame:
    """Guess liquidations between two position snapshots of the same address and coin.

    A position counts as probably liquidated when it is gone (or shrank) at the next snapshot and
    the coin's mark price touched its last known ``liquidation_px`` in between. This only covers
    tracked addresses and is an estimate; liquidation fills in the ``fills`` stream are exact.

    ``positions``: rows from the ``positions`` stream. ``marks``: rows with ``ts, coin, mark_px``
    (the ``asset_ctxs`` stream).
    """
    pos = positions.copy()
    pos["ts"] = pd.to_datetime(pos["ts"], utc=True)
    marks = marks[["ts", "coin", "mark_px"]].copy()
    marks["ts"] = pd.to_datetime(marks["ts"], utc=True)
    snaps = list(pd.DatetimeIndex(pos["ts"].unique()).sort_values())
    nxt = dict(zip(snaps[:-1], snaps[1:]))
    out = []
    by_key = {k: g.set_index("ts") for k, g in pos.groupby(["address", "coin"])}
    marks_by_coin = {c: g.set_index("ts")["mark_px"].sort_index() for c, g in marks.groupby("coin")}
    for (addr, coin), g in by_key.items():
        for ts, row in g.iterrows():
            t1 = nxt.get(ts)
            if t1 is None or pd.isna(row["liquidation_px"]) or not row["szi"]:
                continue
            later = g["szi"].get(t1, 0.0)
            same_side = np.sign(later) == np.sign(row["szi"])
            reduced = abs(later) < abs(row["szi"]) or not same_side
            if not reduced:
                continue
            m = marks_by_coin.get(coin)
            if m is None:
                continue
            window = m[(m.index > ts) & (m.index <= t1)]
            if window.empty:
                continue
            long = row["szi"] > 0
            touched = window.min() <= row["liquidation_px"] if long else window.max() >= row["liquidation_px"]
            if touched:
                out.append(
                    {
                        "ts_before": ts,
                        "ts_after": t1,
                        "address": addr,
                        "coin": coin,
                        "side": "long" if long else "short",
                        "size_before": row["szi"],
                        "size_after": later,
                        "liquidation_px": row["liquidation_px"],
                        "notional": abs(row["szi"]) * row["liquidation_px"],
                    }
                )
    return pd.DataFrame(out)
