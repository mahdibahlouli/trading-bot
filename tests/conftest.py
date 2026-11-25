import pandas as pd
import pytest


def make_bars(start: str, n: int, freq: str = "1min", base: float = 100.0, rng: float = 1.0) -> pd.DataFrame:
    """Flat bars: open=close=base, high=base+rng/2, low=base-rng/2 (true range = rng)."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="ts")
    return pd.DataFrame(
        {"open": base, "high": base + rng / 2, "low": base - rng / 2, "close": base, "volume": 1.0},
        index=idx,
    )


@pytest.fixture
def bars_factory():
    return make_bars


class FakeResponse:
    def __init__(self, content=b"", status=200, json_data=None):
        self.content = content
        self.status_code = status
        self._json = json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._json


class FakeSession:
    """Maps URL -> FakeResponse; anything else is a 404."""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self.routes.get(url, FakeResponse(status=404))

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        key = (url, json["type"])
        return self.routes.get(key, FakeResponse(status=404))


@pytest.fixture
def fake_session():
    return FakeSession
