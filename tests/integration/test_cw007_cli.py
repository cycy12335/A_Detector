"""CW-007 CLI 层测试：**真实进程**下的存储就绪语义与日志安全。

为什么必须跑到真实进程这一层：``/health/ready`` 是否把存储算进去、数据库是否真的落在
``artifacts/runtime/``、Uvicorn 默认 access log 是否仍然关闭，都取决于进程入口与实际运行的
服务器，``TestClient`` 看不到。

刻意把探针配置的 ``storage.path`` 指到一个**本次运行专用**的测试数据库：真实进程测试不应改写
生产默认库 ``cw007_decisions.sqlite3``。生产路径本身的正确性由
``test_cw007_storage_config.py`` 直接断言。
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from ai_detector.config import PROJECT_ROOT  # noqa: E402
from tests.cw007_fixtures import (  # noqa: E402
    remove_database,
    staged_db_path,
)

RUN_API = PROJECT_ROOT / "scripts/run_api.py"
SERVICE_CONFIG = PROJECT_ROOT / "configs/service.yaml"
BUNDLE_PATH = PROJECT_ROOT / "artifacts/models/cw004/lightgbm_bundle.joblib"

#: 只出现在探针 URL / query 里的哨兵：出现即说明原始 URL 被写进了进程输出。
#: 形状合法但一定不存在的 decision ID（规范小写 UUID4）。
ZERO_UUID = "00000000-0000-4000-8000-000000000000"

PATH_SENTINEL = "SECRET_DECISION_PATH_318"
QUERY_SENTINEL = "SECRET_DECISION_QUERY_904"

pytestmark = pytest.mark.skipif(
    not BUNDLE_PATH.is_file(), reason="正式 CW-004 bundle 不在位（真实产物门禁）"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ours(port: int, marker: str, timeout: float = 60.0) -> bool:
    """等待端口上应答的**确实是我们刚启动的那个进程**。

    只探测"端口活了"是不够的：``_free_port()`` 先绑 0 号端口再释放，若上一个测试的进程尚未
    完全退出，操作系统可能把同一端口再给一次，于是新进程绑定失败而**旧进程仍在应答**——
    请求会打到另一个服务上，表现为随机失败。这里用注入到 ``service.name`` 的一次性标记
    确认身份（``/health/live`` 会回显它）。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live", timeout=2) as r:
                body = json.loads(r.read().decode("utf-8"))
            if body.get("service") == marker:
                return True
            # 端口上是别人：等它让出端口。
        except (urllib.error.URLError, OSError, json.JSONDecodeError, AttributeError):
            pass
        time.sleep(0.4)
    return False


def _request(port: int, method: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=data)
    if data is not None:
        request.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:  # pragma: no cover - 不应出现
            return error.code, {}


def _probe_config(
    tmp_path: Path, name: str, port: int, *, marker: str, **patches: object
) -> Path:
    """基于仓库配置生成探针配置：改端口与服务名、改存储路径（隔离到测试库）或注入故障。"""
    raw = yaml.safe_load(SERVICE_CONFIG.read_text(encoding="utf-8"))
    raw["service"]["port"] = port
    # 一次性服务名：用于确认端口上应答的确实是本次启动的进程。
    raw["service"]["name"] = marker
    raw.setdefault("storage", {})
    raw["storage"]["path"] = patches.pop("storage_path")
    for dotted, value in patches.items():
        section, _, key = dotted.partition("__")
        raw[section][key] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


class _Server:
    """上下文管理器：拉起真实 CLI 进程，退出时终止并回收输出。

    ## 为什么要"确认端口上回答的确实是我们起的进程"

    ``_free_port()`` 先绑 0 号端口拿到一个空闲端口再释放，随后才让新进程去绑。若上一个
    测试的进程尚未完全退出，操作系统可能把同一个端口再给我们一次：新进程绑定失败，而
    **旧进程仍在应答**，于是"端口活了"的探测通过、请求却打到了另一个服务上。这类竞态会
    表现为随机失败，因此这里**用一个一次性服务名确认端口上应答的就是本次启动的进程**，
    不匹配就换端口重试。
    """

    def __init__(self, config_factory, tmp_path: Path, name: str) -> None:
        self._config_factory = config_factory
        self._tmp_path = tmp_path
        self._name = name
        self.config_path: Path | None = None
        self.marker = ""
        self.port = 0
        self.output = ""
        self.proc: subprocess.Popen | None = None

    def _terminate(self) -> str:
        if self.proc is None:  # pragma: no cover - 防御性
            return ""
        self.proc.terminate()
        try:
            stdout, stderr = self.proc.communicate(timeout=25)
        except subprocess.TimeoutExpired:  # pragma: no cover - 防御性
            self.proc.kill()
            stdout, stderr = self.proc.communicate(timeout=25)
        return (stdout or "") + (stderr or "")

    def __enter__(self) -> "_Server":
        for attempt in range(4):
            self.port = _free_port()
            # 一次性服务名：用于确认端口上应答的确实是本次启动的进程。
            self.marker = f"cw007-probe-{uuid.uuid4().hex[:12]}"
            self.config_path = self._config_factory(self.port, self.marker)
            self.proc = subprocess.Popen(
                [sys.executable, str(RUN_API), "--config", str(self.config_path)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if _wait_until_ours(self.port, self.marker) and self.proc.poll() is None:
                return self
            # 端口被别人占用（或进程起不来）：回收输出、换端口重试。
            self.output = self._terminate()
        raise AssertionError(
            f"真实进程未在超时内就绪；最后一次输出：\n{self.output[-1500:]}"
        )

    def __exit__(self, *exc: object) -> None:
        self.output += self._terminate()


@pytest.fixture
def cli_db(tmp_path: Path) -> Path:
    """本次运行专用的测试数据库（项目相对、位于 artifacts/runtime/ 下）。"""
    path = staged_db_path(tmp_path, "cli.sqlite3")
    yield path
    remove_database(path)


def _relative_db(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def test_real_cli_serves_decisions_with_a_real_database(tmp_path: Path, cli_db: Path) -> None:
    """真实进程：存储就绪 -> ready 200；POST 201 后可按 ID 读回同一份记录。"""
    assert not cli_db.exists()
    with _Server(
        lambda port, marker: _probe_config(
            tmp_path, "cw007_cli.yaml", port, marker=marker, storage_path=_relative_db(cli_db)
        ),
        tmp_path,
        "cw007_cli.yaml",
    ) as server:
        port = server.port
        live_code, _ = _request(port, "GET", "/health/live")
        ready_code, _ = _request(port, "GET", "/health/ready")
        model_code, model = _request(port, "GET", "/v1/model")
        features = {name: 0.0 for name in model["required_features"]}
        features["TransactionAmt"] = 99999.0
        score_code, _ = _request(port, "POST", "/v1/score", {"features": features})
        created_code, created = _request(
            port, "POST", "/v1/decisions/evaluate", {"features": features}
        )
        decision_id = created.get("decision", {}).get("decision_id")
        fetched_code, fetched = _request(port, "GET", f"/v1/decisions/{decision_id}")
    assert live_code == 200 and ready_code == 200, server.output[-2000:]
    assert model_code == 200 and score_code == 200
    assert created_code == 201
    assert created["decision"]["risk_level"] == "elevated_signal"
    assert fetched_code == 200
    assert fetched["decision"] == created["decision"], "GET 必须原样读回存储快照"
    assert cli_db.is_file(), "真实进程应在项目内的 artifacts/runtime 下建库"


def test_real_cli_readiness_reflects_a_broken_storage(tmp_path: Path, cli_db: Path) -> None:
    """存储打不开时：旧接口照常，readiness 与 decisions 接口降级为 503。"""
    cli_db.parent.mkdir(parents=True, exist_ok=True)
    cli_db.write_bytes(b"definitely not a sqlite database" * 8)
    before = cli_db.read_bytes()
    with _Server(
        lambda port, marker: _probe_config(
            tmp_path,
            "cw007_cli_broken.yaml",
            port,
            marker=marker,
            storage_path=_relative_db(cli_db),
        ),
        tmp_path,
        "cw007_cli_broken.yaml",
    ) as server:
        port = server.port
        live_code, _ = _request(port, "GET", "/health/live")
        ready_code, ready = _request(port, "GET", "/health/ready")
        policy_code, _ = _request(port, "GET", "/v1/policy")
        decisions_code, decisions = _request(port, "GET", f"/v1/decisions/{ZERO_UUID}")
    assert live_code == 200
    assert ready_code == 503
    assert ready["error"]["code"] == "storage_not_ready"
    assert policy_code == 200, "存储故障不影响政策接口"
    assert decisions_code == 503
    assert decisions["error"]["code"] == "storage_not_ready"
    assert cli_db.read_bytes() == before, "损坏的数据库不得被改写或删除"
    assert "storage_" in server.output, "启动诊断应报出安全的存储分类码"
    for banned in ("configs/", "cw007_test", "Traceback", ".sqlite3"):
        assert banned not in server.output, (
            f"启动输出泄漏了 {banned}；实际输出尾部：\n{server.output[-1500:]}"
        )


def test_real_cli_does_not_log_raw_urls_on_decision_routes(tmp_path: Path, cli_db: Path) -> None:
    """未知路径与查询串不得进入进程输出（Uvicorn access log 继续关闭）。"""
    with _Server(
        lambda port, marker: _probe_config(
            tmp_path, "cw007_cli_probe.yaml", port, marker=marker, storage_path=_relative_db(cli_db)
        ),
        tmp_path,
        "cw007_cli_probe.yaml",
    ) as server:
        port = server.port
        for url in (
            f"/v1/decisions/{PATH_SENTINEL}",
            f"/v1/decisions/{ZERO_UUID}?q={QUERY_SENTINEL}",
        ):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{url}", timeout=5).read()
            except urllib.error.HTTPError as error:
                error.read()
    assert PATH_SENTINEL not in server.output, "原始 path 出现在进程输出里"
    assert QUERY_SENTINEL not in server.output, "query string 出现在进程输出里"
    assert "GET /" not in server.output, "出现了 Uvicorn 默认 access log 的请求行"
