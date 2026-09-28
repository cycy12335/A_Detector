"""从 Locust 的安全事件流计算本机测量统计。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ENDPOINTS = (
    "/v1/score",
    "/v1/risk/evaluate",
    "/v1/decisions/evaluate",
)


class MeasurementError(ValueError):
    """事件流或样本量不能形成有效测量。"""


@dataclass(frozen=True)
class EndpointStats:
    name: str
    requests: int
    successes: int
    failures: int
    failure_rate: float
    rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def as_dict(self) -> dict[str, str | int | float]:
        return vars(self).copy()


def percentile(values: list[float], fraction: float) -> float:
    """最近秩分位数，精度不随显示格式丢失。"""
    if not values or not 0 < fraction <= 1:
        raise MeasurementError("分位数输入无效")
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def _stats(name: str, values: list[tuple[float, bool]], seconds: float) -> EndpointStats:
    times = [response_time for response_time, _ in values]
    failures = sum(not ok for _, ok in values)
    total = len(values)
    return EndpointStats(
        name=name,
        requests=total,
        successes=total - failures,
        failures=failures,
        failure_rate=failures / total,
        rps=total / seconds,
        p50_ms=percentile(times, 0.50),
        p95_ms=percentile(times, 0.95),
        p99_ms=percentile(times, 0.99),
    )


def analyze_events(path: Path, *, seconds: float, minimum: int) -> dict[str, EndpointStats]:
    """严格验证 Locust 事件流后计算各接口和总体统计。

    事件只含白名单接口、测量窗口内时间、耗时和成功布尔值。失败请求保留在
    分位数和吞吐统计中，同时使正式测量失败关闭。
    """
    if not math.isfinite(seconds) or seconds <= 0 or minimum < 1:
        raise MeasurementError("测量配置无效")
    grouped: dict[str, list[tuple[float, bool]]] = {name: [] for name in ENDPOINTS}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if not isinstance(item, dict) or set(item) != {"name", "elapsed_s", "response_ms", "ok"}:
                    raise MeasurementError("事件字段无效")
                name = item["name"]
                elapsed = item["elapsed_s"]
                response_ms = item["response_ms"]
                ok = item["ok"]
                if name not in grouped or type(ok) is not bool:
                    raise MeasurementError("事件类别无效")
                if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                        or not 0 <= elapsed <= seconds + 0.25):
                    raise MeasurementError("事件时间无效")
                if (type(response_ms) not in (int, float) or not math.isfinite(response_ms)
                        or response_ms < 0):
                    raise MeasurementError("响应耗时无效")
                grouped[name].append((float(response_ms), ok))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise MeasurementError("事件流不可读取") from error
    if any(len(values) < minimum for values in grouped.values()):
        raise MeasurementError("有效样本不足")
    result = {name: _stats(name, grouped[name], seconds) for name in ENDPOINTS}
    all_values = [value for values in grouped.values() for value in values]
    result["aggregate"] = _stats("aggregate", all_values, seconds)
    return result
