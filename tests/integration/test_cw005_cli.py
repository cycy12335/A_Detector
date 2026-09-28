"""CW-005 CLI 层测试：Uvicorn 参数、启动诊断、以及**真实进程**的原始 URL 探针。

首轮独立验收指出：真实 Uvicorn 进程会额外输出默认 access log，把未知 path 与 query
原样记录下来（例如 ``GET /SECRET?token=...``），从而绕过本服务 completion log 的
模板路径 / ``<unmatched>`` 脱敏。单元测试看不到这一点——它只在**真实进程**里发生。
因此本文件启动真实 CLI，用 sentinel URL 打它，再扫描进程的全部输出。
"""

from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

# 服务专项依赖来自 `service` extra：未安装时清晰跳过本模块，安装后必须正常执行（不得跳过）。
pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from fastapi.testclient import TestClient  # noqa: E402

from ai_detector.config import PROJECT_ROOT  # noqa: E402
from tests.cw007_fixtures import staged_db_relative  # noqa: E402
from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.errors import BundleErrorCode, BundleLoadError  # noqa: E402
from ai_detector.service.logging import STARTUP_EVENT  # noqa: E402
from tests.cw005_fixtures import FakeRuntime, RecordingLoader, make_config  # noqa: E402

#: 只出现在探针 URL / query 里的哨兵：若它们出现在进程输出中，说明原始 URL 被记录了。
PATH_SENTINEL = "SECRET_PATH_SENTINEL_987"
QUERY_SENTINEL = "QUERY_SENTINEL_654"

RUN_API = PROJECT_ROOT / "scripts/run_api.py"
SERVICE_CONFIG = PROJECT_ROOT / "configs/service.yaml"


# --- Uvicorn 启动参数 ---------------------------------------------------


def test_cli_disables_the_default_uvicorn_access_log(monkeypatch) -> None:
    """``run_api`` 必须显式关闭 Uvicorn access log，否则原始 URL 会绕过本服务的脱敏日志。"""
    import uvicorn

    captured: dict = {}

    def _fake_run(app, **kwargs):  # noqa: ANN001, ANN202
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    import importlib.util

    spec = importlib.util.spec_from_file_location("cw005_run_api", RUN_API)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.main([]) == 0
    assert captured["access_log"] is False, "必须显式关闭 Uvicorn access log"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000


def test_cli_rejects_an_unsafe_config(tmp_path: Path) -> None:
    """配置不合法时 CLI 以退出码 2 失败，而不是带病启动。"""
    bad = tmp_path / "bad.yaml"
    raw = yaml.safe_load(SERVICE_CONFIG.read_text(encoding="utf-8"))
    raw["service"]["host"] = "0.0.0.0"
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    import importlib.util

    spec = importlib.util.spec_from_file_location("cw005_run_api_bad", RUN_API)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.main(["--config", str(bad)]) == 2


# --- 启动诊断 -----------------------------------------------------------


def _log_records(caplog, event: str) -> list[dict]:
    found = []
    for record in caplog.records:
        if record.name != "ai_detector.service":
            continue
        try:
            payload = json.loads(record.getMessage())
        except json.JSONDecodeError:  # pragma: no cover
            continue
        if payload.get("event") == event:
            found.append(payload)
    return found


def test_startup_diagnostic_is_emitted_on_success(tmp_path: Path, caplog) -> None:
    config = make_config(tmp_path)
    runtime = FakeRuntime(features=["a", "b", "c"])
    app = create_app(config, runtime_loader=RecordingLoader(runtime))

    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        caplog.clear()
        with TestClient(app):
            pass

    records = _log_records(caplog, STARTUP_EVENT)
    assert len(records) == 1, "启动成功应恰好一条结构化诊断"
    payload = records[0]
    assert payload["outcome"] == "ready"
    assert payload["service"] == config.name
    assert payload["api_version"] == config.api_version
    assert payload["model_version"] == runtime.model_version
    assert payload["n_source_features"] == 3
    assert "error_code" not in payload


def test_startup_diagnostic_is_emitted_on_failure_without_leaking(tmp_path: Path, caplog) -> None:
    config = make_config(tmp_path)
    loader = RecordingLoader(
        error=BundleLoadError(BundleErrorCode.HASH_MISMATCH, "bundle 指纹不符：实际 abc… 期望 def…")
    )
    app = create_app(config, runtime_loader=loader)

    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        caplog.clear()
        with TestClient(app):
            pass

    records = _log_records(caplog, STARTUP_EVENT)
    assert len(records) == 1
    payload = records[0]
    assert payload["outcome"] == "not_ready"
    assert payload["error_code"] == BundleErrorCode.HASH_MISMATCH
    assert "model_version" not in payload

    blob = json.dumps(payload, ensure_ascii=False)
    for banned in ("C:\\", "D:\\", "/home/", "/Users/", "Traceback", ".joblib", "bundle 文件"):
        assert banned not in blob, banned


def test_startup_diagnostic_does_not_count_as_a_completion_log(tmp_path: Path, caplog) -> None:
    """启动诊断是独立事件，不影响"每请求恰好一条 completion log"。"""
    config = make_config(tmp_path)
    runtime = FakeRuntime()
    app = create_app(config, runtime_loader=RecordingLoader(runtime))

    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        caplog.clear()
        with TestClient(app) as client:
            client.get("/health/live")
            client.get("/health/live")

    assert len(_log_records(caplog, STARTUP_EVENT)) == 1
    assert len(_log_records(caplog, "request_completed")) == 2


# --- 真实进程的原始 URL 探针 --------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_live(port: int, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


def test_real_process_does_not_log_raw_urls(tmp_path: Path) -> None:
    """真实 CLI 进程：sentinel path/query 不得出现在**任何**输出里。

    这是唯一能发现"Uvicorn 默认 access log 泄露原始 URL"的测试层级：
    TestClient 不经过 Uvicorn 的 logging 配置，看不到那条日志。
    """
    port = _free_port()
    raw = yaml.safe_load(SERVICE_CONFIG.read_text(encoding="utf-8"))
    raw["service"]["port"] = port
    # 存储隔离到临时库：本测试只关心日志，但真实 CLI 进程会打开配置里的存储路径；
    # 不改的话干净检出里会**凭空创建默认决策数据库**（CW-014 的会话级守卫抓到了这一点）。
    staged = staged_db_relative(tmp_path)
    raw.setdefault("storage", {})["path"] = staged
    # bundle 路径保持项目相对（配置层的硬要求）；缺失时服务会降级为未就绪，
    # 这对本测试无影响——我们只关心"未知 path/query 有没有被写进日志"。
    config_path = tmp_path / "service_cli_probe.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(RUN_API), "--config", str(config_path)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _wait_until_live(port), "真实进程未在超时内就绪"
        for url in (
            f"http://127.0.0.1:{port}/{PATH_SENTINEL}",
            f"http://127.0.0.1:{port}/health/live?secret={QUERY_SENTINEL}",
        ):
            try:
                urllib.request.urlopen(url, timeout=5).read()
            except urllib.error.HTTPError as error:  # 404 是预期结果
                error.read()
    finally:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - 防御性
            proc.kill()
            stdout, stderr = proc.communicate(timeout=15)

    output = (stdout or "") + (stderr or "")
    assert PATH_SENTINEL not in output, "原始 path 出现在进程输出里"
    assert QUERY_SENTINEL not in output, "query string 出现在进程输出里"
    assert "GET /" not in output, "出现了 Uvicorn 默认 access log 的请求行"

    completions = [
        json.loads(line)
        for line in output.splitlines()
        if line.strip().startswith("{") and '"request_completed"' in line
    ]
    assert len(completions) >= 3, f"completion log 数量不足：{len(completions)}"
    routes = {entry["route"] for entry in completions}
    assert "<unmatched>" in routes, "未知路径应记为 <unmatched>"
    assert all(PATH_SENTINEL not in json.dumps(entry) for entry in completions)
    assert not any(re.search(r"secret=", json.dumps(entry)) for entry in completions)
