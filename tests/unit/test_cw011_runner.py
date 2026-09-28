"""CW-011 的本机边界、进程清理和失败报告。"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from performance import runner


def _valid_events_text() -> str:
    """一份**完全有效**的事件流（每个白名单接口各一条成功样本）。"""
    names = ("/v1/score", "/v1/risk/evaluate", "/v1/decisions/evaluate")
    return "\n".join(
        json.dumps({"name": name, "elapsed_s": 0.5, "response_ms": 10.0, "ok": True})
        for name in names
    ) + "\n"


def _round_files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.name.endswith(".round"))


@pytest.mark.parametrize("target", [
    "https://127.0.0.1:8000",
    "http://localhost:8000",
    "http://192.168.1.1:8000",
    "http://example.com:8000",
    "http://127.0.0.1:8000/private",
    "http://user:pass@127.0.0.1:8000",
])
def test_remote_or_ambiguous_target_is_rejected(target: str) -> None:
    with pytest.raises(runner.RunError, match="non_loopback_target"):
        runner.validate_target(target)


def test_exact_loopback_target_is_accepted() -> None:
    runner.validate_target("http://127.0.0.1:8000")


class _Process:
    def __init__(self, *, timeout: bool = False) -> None:
        self.timeout = timeout
        self.terminated = False
        self.killed = False
        self.waited = 0

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, *, timeout):
        self.waited += 1
        if self.timeout and self.waited == 1:
            raise subprocess.TimeoutExpired("child", timeout)
        return 0


def test_cleanup_terminates_and_reaps() -> None:
    process = _Process()
    runner.stop_process(process)
    assert process.terminated and process.waited == 1 and not process.killed


def test_cleanup_kills_after_terminate_timeout() -> None:
    process = _Process(timeout=True)
    runner.stop_process(process)
    assert process.terminated and process.killed and process.waited == 2


def test_locust_nonzero_exit_is_failure_even_with_valid_samples(tmp_path: Path, monkeypatch) -> None:
    class FailedProcess:
        def wait(self, timeout):
            return 3

        def poll(self):
            return 3

    def spawn(command, *, env):
        assert "--headless" in command
        assert command[command.index("--host") + 1] == "http://127.0.0.1:8123"
        path = Path(env["CW011_EVENTS_PATH"])
        names = ("/v1/score", "/v1/risk/evaluate", "/v1/decisions/evaluate")
        path.write_text("\n".join(json.dumps({
            "name": name, "elapsed_s": 0.5, "response_ms": 10.0, "ok": True
        }) for name in names) + "\n", encoding="utf-8")
        return FailedProcess()

    monkeypatch.setattr(runner, "_spawn", spawn)
    with pytest.raises(runner.RunError, match="request_failures"):
        runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", tmp_path / "events.jsonl")


def test_locust_timeout_reaps_child(tmp_path: Path, monkeypatch) -> None:
    class SlowProcess:
        def __init__(self):
            self.terminated = False
            self.reaped = False

        def wait(self, timeout):
            if not self.terminated:
                raise subprocess.TimeoutExpired("locust", timeout)
            self.reaped = True
            return 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = SlowProcess()
    monkeypatch.setattr(runner, "_spawn", lambda command, env: process)
    with pytest.raises(runner.RunError, match="locust_timeout"):
        runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", tmp_path / "events.jsonl")
    assert process.terminated and process.reaped


def test_service_start_failure_writes_report_and_cleans_temp(tmp_path: Path, monkeypatch) -> None:
    report_dir = tmp_path / "reports"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    default_db = runtime_dir / "default.sqlite3"
    default_db.write_bytes(b"frozen")

    class ServiceProcess:
        stopped = False

        def poll(self):
            return None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            return 0

    service = ServiceProcess()
    monkeypatch.setattr(runner, "REPORT_DIR", report_dir)
    monkeypatch.setattr(runner, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runner, "DEFAULT_DB", default_db)
    monkeypatch.setattr(runner, "_spawn", lambda command, env=None: service)
    monkeypatch.setattr(
        runner, "_wait_ready", lambda target, process: (_ for _ in ()).throw(runner.RunError("service_start_failed"))
    )
    assert runner.run(smoke=True) == 2
    report = json.loads((report_dir / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["error_code"] == "service_start_failed"
    assert service.stopped
    assert default_db.read_bytes() == b"frozen"
    assert list(runtime_dir.iterdir()) == [default_db]
    assert not any(path.name.startswith(".service_") for path in report_dir.iterdir())


def test_report_contains_metrics_but_no_event_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "REPORT_DIR", tmp_path)
    report = {
        "status": "passed", "created_at_utc": "2026-09-27T00:00:00+00:00",
        "error_code": None, "command_shape": "python scripts/run_performance.py",
        "environment": {"python": "3.14.6", "locust": "2.46.6"},
        "service_api_version": "v1", "model_version": "synthetic-test-model",
        "inputs": {"bundle_sha256": "a" * 64, "policy_sha256": "b" * 64},
        "profiles": [{
            "profile": {"name": "low", "users": 1, "spawn_rate": 1,
                        "warmup_s": 5, "measure_s": 25, "minimum_per_endpoint": 10},
            "stats": {"/v1/score": {
                "requests": 10, "failures": 0, "failure_rate": 0.0, "rps": 1.0,
                "p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0,
            }},
        }],
    }
    runner._write_report(report, "report")
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "P95 ms" in markdown and "| 2.00 |" in markdown


# --- 本轮事件文件隔离与新鲜度（首次验收阻断项） --------------------------


class _SilentProcess:
    """退出码可控但**什么都不写**的 Locust 替身。"""

    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code

    def wait(self, timeout):
        return self.exit_code

    def poll(self):
        return self.exit_code


def test_stale_events_are_never_counted_as_this_round(tmp_path: Path, monkeypatch) -> None:
    """预先放置一份**有效**旧事件文件、本轮退出 0 但不写事件：必须失败。

    这正是首次验收复现的反例：若能读到旧文件，样本门槛会被旧样本满足，
    "零新样本"的运行就会被判成功，报告来源因此不可信。
    """
    events = tmp_path / "low_events.jsonl"
    events.write_text(_valid_events_text(), encoding="utf-8")
    before = events.read_bytes()

    monkeypatch.setattr(runner, "_spawn", lambda command, *, env: _SilentProcess(0))
    with pytest.raises(runner.RunError) as caught:
        runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", events)
    assert caught.value.code == "no_new_events"
    assert caught.value.profile_result is None, "不得把旧样本算成本轮统计"
    assert events.read_bytes() == before, "失败的一轮不得破坏既有的可追溯原始事件"
    assert _round_files(tmp_path) == [], "本轮临时事件文件必须被清理"


def test_nonzero_exit_never_falls_back_to_previous_events(tmp_path: Path, monkeypatch) -> None:
    """本轮非零退出且未写事件时，不得引用旧事件充当本轮统计。"""
    events = tmp_path / "low_events.jsonl"
    events.write_text(_valid_events_text(), encoding="utf-8")
    before = events.read_bytes()

    monkeypatch.setattr(runner, "_spawn", lambda command, *, env: _SilentProcess(3))
    with pytest.raises(runner.RunError) as caught:
        runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", events)
    assert caught.value.code == "locust_failed"
    assert caught.value.profile_result is None, "非零退出不得产出任何本轮统计"
    assert events.read_bytes() == before
    assert _round_files(tmp_path) == []


def test_successful_exit_without_any_events_is_a_failure(tmp_path: Path, monkeypatch) -> None:
    """没有任何旧文件的干净场景下，退出 0 但不写事件同样是失败。"""
    monkeypatch.setattr(runner, "_spawn", lambda command, *, env: _SilentProcess(0))
    with pytest.raises(runner.RunError, match="no_new_events"):
        runner._run_profile(
            runner.SMOKE_PROFILE, "http://127.0.0.1:8123", tmp_path / "events.jsonl"
        )


def test_this_round_writes_to_a_file_of_its_own(tmp_path: Path, monkeypatch) -> None:
    """本轮事件写在独占文件里；成功后**才**发布到正式文件名，保留可追溯性。"""
    events = tmp_path / "low_events.jsonl"
    seen: dict[str, Path] = {}

    def spawn(command, *, env):
        seen["path"] = Path(env["CW011_EVENTS_PATH"])
        seen["path"].write_text(_valid_events_text(), encoding="utf-8")
        return _SilentProcess(0)

    monkeypatch.setattr(runner, "_spawn", spawn)
    result = runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", events)

    assert seen["path"] != events, "本轮必须写独占文件，不得直接复用正式文件名"
    assert result["stats"]["/v1/score"]["requests"] == 1
    assert events.read_text(encoding="utf-8") == _valid_events_text(), (
        "成功后原始事件必须发布到正式文件名"
    )
    assert _round_files(tmp_path) == []


def test_freshness_check_rejects_an_old_timestamp(tmp_path: Path) -> None:
    """新鲜度校验本身：时间戳早于本轮开始的文件（旧样本）必须被拒绝。"""
    path = tmp_path / "events.jsonl"
    path.write_text(_valid_events_text(), encoding="utf-8")
    old = time.time() - 3600
    os.utime(path, (old, old))
    with pytest.raises(runner.RunError, match="stale_events"):
        runner._fresh_events(path, round_started=time.time())


def test_freshness_check_rejects_a_missing_or_empty_file(tmp_path: Path) -> None:
    with pytest.raises(runner.RunError, match="no_new_events"):
        runner._fresh_events(tmp_path / "absent.jsonl", round_started=time.time())
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(runner.RunError, match="no_new_events"):
        runner._fresh_events(empty, round_started=time.time())


def test_freshness_check_accepts_a_file_written_this_round(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(_valid_events_text(), encoding="utf-8")
    runner._fresh_events(path, round_started=time.time() - 1.0)


# --- 失败路径也必须清理本轮 .round（首次整改复验阻断项） ----------------


class _WriteThenHangProcess:
    """先让本轮 `.round` 真的落盘，再"挂住"直到被回收的 Locust 替身。"""

    def __init__(self) -> None:
        self.terminated = False
        self.reaped = False
        self.round_path: Path | None = None

    def wait(self, timeout):
        if not self.terminated:
            raise subprocess.TimeoutExpired("locust", timeout)
        self.reaped = True
        return 0

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


def test_timeout_after_events_are_written_still_cleans_the_round_file(
    tmp_path: Path, monkeypatch
) -> None:
    """本轮 `.round` 已写入后 Locust 超时：报 `locust_timeout`、回收进程、
    删除本轮 `.round`，且既有的正式事件文件字节不变。

    这是首次整改复验的阻断项：清理逻辑若只挂在后续统计段，超时路径就会遗留
    已经写入的本轮临时事件。
    """
    events = tmp_path / "low_events.jsonl"
    events.write_text(_valid_events_text(), encoding="utf-8")
    before = events.read_bytes()

    process = _WriteThenHangProcess()

    def spawn(command, *, env):
        process.round_path = Path(env["CW011_EVENTS_PATH"])
        process.round_path.write_text(_valid_events_text(), encoding="utf-8")
        return process

    monkeypatch.setattr(runner, "_spawn", spawn)
    with pytest.raises(runner.RunError) as caught:
        runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", events)

    assert caught.value.code == "locust_timeout"
    assert process.round_path is not None
    assert process.round_path != events, "本轮事件必须写在独占文件上"
    assert process.terminated and process.reaped, "超时后必须回收进程"
    assert not process.round_path.exists(), "超时后必须删除本轮 .round 事件文件"
    assert events.read_bytes() == before, "不得破坏既有的可追溯原始事件"
    assert _round_files(tmp_path) == []


def test_invalid_measurement_still_cleans_the_round_file(tmp_path: Path, monkeypatch) -> None:
    """本轮写出了事件但样本不足：报 `invalid_measurement` 并清理本轮 `.round`。"""
    written: dict[str, Path] = {}

    def spawn(command, *, env):
        path = Path(env["CW011_EVENTS_PATH"])
        written["path"] = path
        path.write_text(
            json.dumps({"name": "/v1/score", "elapsed_s": 0.5, "response_ms": 1.0, "ok": True})
            + "\n",
            encoding="utf-8",
        )
        return _SilentProcess(0)

    monkeypatch.setattr(runner, "_spawn", spawn)
    with pytest.raises(runner.RunError, match="invalid_measurement"):
        runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", tmp_path / "events.jsonl")
    assert not written["path"].exists(), "统计失败后必须删除本轮 .round"
    assert _round_files(tmp_path) == []


def test_request_failures_discards_the_round_file(tmp_path: Path, monkeypatch) -> None:
    """本轮有请求失败：报 `request_failures` 并丢弃本轮 `.round`（不发布为正式事件）。"""
    events = tmp_path / "low_events.jsonl"

    def spawn(command, *, env):
        Path(env["CW011_EVENTS_PATH"]).write_text(_valid_events_text(), encoding="utf-8")
        return _SilentProcess(3)

    monkeypatch.setattr(runner, "_spawn", spawn)
    with pytest.raises(runner.RunError) as caught:
        runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", events)
    assert caught.value.code == "request_failures"
    assert caught.value.profile_result is not None, "失败轮仍保留已算出的本轮统计"
    assert not events.exists(), "失败的一轮不得发布到正式文件名"
    assert _round_files(tmp_path) == []


def test_spawn_failure_cleans_the_round_file(tmp_path: Path, monkeypatch) -> None:
    """spawn 阶段就失败：报 `locust_spawn_failed`，不得留下本轮 `.round`。"""
    events = tmp_path / "low_events.jsonl"

    def spawn(command, *, env):
        Path(env["CW011_EVENTS_PATH"]).write_bytes(b"partial")
        raise OSError("spawn failed")

    monkeypatch.setattr(runner, "_spawn", spawn)
    with pytest.raises(runner.RunError, match="locust_spawn_failed"):
        runner._run_profile(runner.SMOKE_PROFILE, "http://127.0.0.1:8123", events)
    assert _round_files(tmp_path) == []


def test_run_does_not_pass_on_a_stale_event_file(tmp_path: Path, monkeypatch) -> None:
    """端到端：正式文件名上已有上一轮的有效事件，本轮零新样本时整体必须失败。

    报告不得把旧样本呈现为本轮数据，也不得留下任何档位统计。
    """
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    default_db = runtime_dir / "default.sqlite3"
    default_db.write_bytes(b"frozen")
    (report_dir / "smoke_events.jsonl").write_text(_valid_events_text(), encoding="utf-8")

    class ServiceProcess:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            return 0

    spawned: list[_SilentProcess | ServiceProcess] = []

    def spawn(command, *, env=None):
        process = ServiceProcess() if not spawned else _SilentProcess(0)
        spawned.append(process)
        return process

    monkeypatch.setattr(runner, "REPORT_DIR", report_dir)
    monkeypatch.setattr(runner, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(runner, "DEFAULT_DB", default_db)
    monkeypatch.setattr(runner, "_spawn", spawn)
    monkeypatch.setattr(runner, "_wait_ready", lambda target, process: None)

    assert runner.run(smoke=True) == 2
    report = json.loads((report_dir / "smoke_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error_code"] == "no_new_events"
    assert report["profiles"] == [], "零新样本不得产出任何档位统计"
    assert default_db.read_bytes() == b"frozen"
    assert list(runtime_dir.iterdir()) == [default_db]
    assert not any(path.name.startswith(".service_") for path in report_dir.iterdir())
    assert _round_files(report_dir) == []
