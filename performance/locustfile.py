"""CW-011 无界面 Locust 场景：仅向 runner 给定的回环服务发送合成请求。"""

from __future__ import annotations

import itertools
import json
import os
import time
from pathlib import Path

from locust import HttpUser, constant, events, task

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = _ROOT / "tests/fixtures/cw009/model_regression_cases.json"
_FEATURES = json.loads(_CORPUS.read_text(encoding="utf-8"))["cases"][0]["features"]
_ROUTE_CYCLE = (
    "/v1/score", "/v1/score", "/v1/score", "/v1/score",
    "/v1/risk/evaluate", "/v1/risk/evaluate", "/v1/risk/evaluate",
    "/v1/decisions/evaluate", "/v1/decisions/evaluate", "/v1/decisions/evaluate",
)
_START: float | None = None
_STREAM = None


@events.test_start.add_listener
def begin_measurement(environment, **kwargs) -> None:
    global _START, _STREAM
    event_path = Path(os.environ["CW011_EVENTS_PATH"])
    _STREAM = event_path.open("w", encoding="utf-8", newline="\n")
    _START = time.monotonic()


@events.request.add_listener
def record_request(request_type, name, response_time, exception, **kwargs) -> None:
    if _START is None or _STREAM is None:
        return
    elapsed = time.monotonic() - _START - float(os.environ["CW011_WARMUP_SECONDS"])
    measure_seconds = float(os.environ["CW011_MEASURE_SECONDS"])
    if not 0 <= elapsed <= measure_seconds or name not in _ROUTE_CYCLE:
        return
    _STREAM.write(json.dumps({
        "name": name,
        "elapsed_s": elapsed,
        "response_ms": float(response_time),
        "ok": exception is None,
    }, separators=(",", ":")) + "\n")
    _STREAM.flush()


@events.test_stop.add_listener
def end_measurement(environment, **kwargs) -> None:
    global _STREAM
    if _STREAM is not None:
        _STREAM.close()
        _STREAM = None


class RiskUser(HttpUser):
    """确定性 4:3:3 请求循环；每个用户均独立计数。"""

    wait_time = constant(0.05)

    def on_start(self) -> None:
        self._routes = itertools.cycle(_ROUTE_CYCLE)

    @task
    def evaluate(self) -> None:
        route = next(self._routes)
        expected = 201 if route == "/v1/decisions/evaluate" else 200
        with self.client.post(
            route,
            json={"features": _FEATURES},
            name=route,
            catch_response=True,
            timeout=5,
        ) as response:
            if response.status_code != expected:
                response.failure("unexpected_status")
