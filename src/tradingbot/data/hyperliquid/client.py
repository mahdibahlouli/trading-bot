"""Small client for Hyperliquid's public ``/info`` endpoint and the public leaderboard file."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

# Request weights from the rate-limit docs [verify]: the IP budget is 1200 weight per minute.
# A handful of light requests cost 2; most other info requests cost 20.
LIGHT_TYPES = {"l2Book", "allMids", "clearinghouseState", "orderStatus", "spotClearinghouseState", "exchangeStatus"}


def request_weight(req_type: str) -> int:
    return 2 if req_type in LIGHT_TYPES else 20


class WeightLimiter:
    """Thread-safe token bucket measured in request weight per minute."""

    def __init__(self, per_minute: float, clock=time.monotonic, sleep=time.sleep):
        self.capacity = float(per_minute)
        self.tokens = float(per_minute)
        self.rate = per_minute / 60.0
        self.clock, self.sleep = clock, sleep
        self.last = clock()
        self.lock = threading.Lock()

    def acquire(self, weight: float) -> None:
        weight = min(weight, self.capacity)
        while True:
            with self.lock:
                now = self.clock()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= weight:
                    self.tokens -= weight
                    return
                wait = (weight - self.tokens) / self.rate
            self.sleep(wait)


class InfoClient:
    def __init__(
        self,
        api_url: str = "https://api.hyperliquid.xyz/info",
        leaderboard_url: str = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard",
        max_weight_per_minute: float = 600,
        session: requests.Session | None = None,
        retries: int = 4,
    ):
        self.api_url = api_url
        self.leaderboard_url = leaderboard_url
        self.limiter = WeightLimiter(max_weight_per_minute)
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "tradingbot-research/0.1")
        self.retries = retries

    def info(self, payload: dict[str, Any]) -> Any:
        self.limiter.acquire(request_weight(payload["type"]))
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                r = self.session.post(self.api_url, json=payload, timeout=20)
                if r.status_code == 429:
                    raise requests.HTTPError("429 rate limited", response=r)
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.retries:
                    raise
                log.warning("info %s failed (%s), retrying in %.0fs", payload["type"], exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)

    def meta_and_asset_ctxs(self) -> Any:
        return self.info({"type": "metaAndAssetCtxs"})

    def clearinghouse_state(self, address: str) -> Any:
        return self.info({"type": "clearinghouseState", "user": address})

    def leaderboard(self) -> list[dict]:
        r = self.session.get(self.leaderboard_url, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("leaderboardRows", data) if isinstance(data, dict) else data
