"""CW-011 的样本窗口、分位数和失败关闭反例。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from performance.analysis import ENDPOINTS, MeasurementError, analyze_events, percentile


def _events(path: Path, *, count: int = 10, failures: int = 0) -> Path:
    rows = []
    for name in ENDPOINTS:
        for index in range(count):
            rows.append({
                "name": name,
                "elapsed_s": index / 10,
                "response_ms": index + 1,
                "ok": index >= failures,
            })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_statistics_keep_failures_and_use_nearest_rank(tmp_path: Path) -> None:
    stats = analyze_events(_events(tmp_path / "events.jsonl", failures=2), seconds=2, minimum=10)
    assert stats[ENDPOINTS[0]].requests == 10
    assert stats[ENDPOINTS[0]].failures == 2
    assert stats[ENDPOINTS[0]].failure_rate == 0.2
    assert stats[ENDPOINTS[0]].p50_ms == 5
    assert stats[ENDPOINTS[0]].p95_ms == 10
    assert stats["aggregate"].requests == 30
    assert stats["aggregate"].failures == 6
    assert stats["aggregate"].rps == 15


def test_percentile_requires_samples() -> None:
    with pytest.raises(MeasurementError):
        percentile([], 0.95)


def test_one_missing_endpoint_invalidates_measurement(tmp_path: Path) -> None:
    path = _events(tmp_path / "events.jsonl")
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if ENDPOINTS[2] not in line]
    path.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(MeasurementError, match="样本不足"):
        analyze_events(path, seconds=2, minimum=10)


@pytest.mark.parametrize("change", [
    lambda row: {**row, "name": "http://elsewhere/private"},
    lambda row: {**row, "elapsed_s": -1},
    lambda row: {**row, "response_ms": float("nan")},
    lambda row: {**row, "ok": 1},
    lambda row: {**row, "payload": "secret"},
])
def test_untrusted_event_fields_fail_closed(tmp_path: Path, change) -> None:
    path = _events(tmp_path / "events.jsonl")
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[0] = json.dumps(change(json.loads(rows[0])))
    path.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(MeasurementError):
        analyze_events(path, seconds=2, minimum=10)


def test_missing_or_invalid_event_stream_fails(tmp_path: Path) -> None:
    with pytest.raises(MeasurementError):
        analyze_events(tmp_path / "absent.jsonl", seconds=2, minimum=1)
    path = tmp_path / "events.jsonl"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(MeasurementError):
        analyze_events(path, seconds=2, minimum=1)
