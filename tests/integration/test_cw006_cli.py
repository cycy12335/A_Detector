"""CW-006 CLI 层测试：**真实进程**下的政策就绪语义与日志安全。

为什么必须跑到真实进程这一层：``/health/ready`` 是否反映政策状态、Uvicorn 默认 access log
是否仍然关闭，都取决于进程入口的启动参数与实际运行的服务器，``TestClient`` 看不到。

本文件需要**正式 CW-004 bundle**（被 Git 忽略）。bundle 不在位时清晰跳过——这属于既有的
"真实数据/产物门禁"跳过，与 CW-005 的处理方式一致；安装了 ``.[test,tree,service]`` 且
产物在位时，这里**不得**跳过。
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from ai_detector.config import PROJECT_ROOT  # noqa: E402
from tests.cw007_fixtures import staged_db_relative  # noqa: E402

RUN_API = PROJECT_ROOT / "scripts/run_api.py"
SERVICE_CONFIG = PROJECT_ROOT / "configs/service.yaml"
BUNDLE_PATH = PROJECT_ROOT / "artifacts/models/cw004/lightgbm_bundle.joblib"

#: 只出现在探针 URL 里的哨兵：出现即说明原始 URL 被写进了进程输出。
PATH_SENTINEL = "SECRET_RISK_PATH_551"
POLICY_SENTINEL = "SECRET_POLICY_NAME_772"

pytestmark = pytest.mark.skipif(
    not BUNDLE_PATH.is_file(), reason="正式 CW-004 bundle 不在位（真实产物门禁）"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_live(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health/live", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


def _get(port: int, path: str) -> tuple[int, dict]:
    """返回 ``(状态码, JSON 体)``；4xx/5xx 也读出来而不是抛异常。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:  # pragma: no cover - 不应出现
            return error.code, {}


class _Server:
    """上下文管理器：拉起真实 CLI 进程，退出时终止并回收输出。"""

    def __init__(self, config_path: Path, port: int) -> None:
        self.config_path = config_path
        self.port = port
        self.output = ""

    def __enter__(self) -> "_Server":
        self.proc = subprocess.Popen(
            [sys.executable, str(RUN_API), "--config", str(self.config_path)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert _wait_until_live(self.port), "真实进程未在超时内就绪"
        return self

    def __exit__(self, *exc: object) -> None:
        self.proc.terminate()
        try:
            stdout, stderr = self.proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover - 防御性
            self.proc.kill()
            stdout, stderr = self.proc.communicate(timeout=20)
        self.output = (stdout or "") + (stderr or "")


def _config_with(tmp_path: Path, name: str, **patches: object) -> Path:
    raw = yaml.safe_load(SERVICE_CONFIG.read_text(encoding="utf-8"))
    raw["service"]["port"] = patches.pop("port")
    # 存储隔离到临时库：真实 CLI 进程会打开配置里的存储路径，不改就会在干净检出里
    # 凭空创建默认决策数据库（CW-014 的会话级守卫抓到了这一点）。
    raw.setdefault("storage", {})["path"] = staged_db_relative(tmp_path)
    for dotted, value in patches.items():
        section, _, key = dotted.partition("__")
        raw[section][key] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_real_cli_is_ready_and_serves_the_frozen_policy(tmp_path: Path) -> None:
    """正式配置下：模型与政策都就绪，``/v1/policy`` 报出冻结版本。"""
    port = _free_port()
    config_path = _config_with(tmp_path, "cw006_ready.yaml", port=port)
    with _Server(config_path, port) as server:
        live_code, _ = _get(port, "/health/live")
        ready_code, ready = _get(port, "/health/ready")
        policy_code, policy = _get(port, "/v1/policy")
    assert live_code == 200
    assert ready_code == 200, server.output[-2000:]
    assert ready["status"] == "ready"
    assert policy_code == 200
    assert policy["policy_version"] == "cw006-risk-policy-v1"
    assert policy["policy_semantics"] == "research_signal_band"
    assert policy["threshold_source"] == "cw003_train_only_q0.99"
    assert [rule["threshold"] for rule in policy["rules"]] == [1092.0, 580.0]


def test_real_cli_readiness_reflects_a_broken_policy(tmp_path: Path) -> None:
    """政策指纹不符时：旧评分接口照常，``/health/ready`` 与 ``/v1/policy`` 503。

    这是本工作包最要紧的真实进程断言——它证明"政策故障"在生产入口上是**可见**的，
    而不是只让新接口静默 503 而已。
    """
    port = _free_port()
    config_path = _config_with(
        tmp_path, "cw006_broken_policy.yaml", port=port, policy__sha256="0" * 64
    )
    with _Server(config_path, port) as server:
        ready_code, ready = _get(port, "/health/ready")
        policy_code, policy = _get(port, "/v1/policy")
        model_code, _ = _get(port, "/v1/model")
        risk_code, risk = _get(port, "/v1/risk/evaluate")
    assert ready_code == 503
    assert ready["error"]["code"] == "policy_not_ready"
    assert policy_code == 503
    assert policy["error"]["code"] == "policy_not_ready"
    # 模型侧不受影响：元数据仍可取，说明降级是**按组件**的。
    assert model_code == 200
    assert risk_code in (405, 422), "GET 不该被风险求值接口接受"
    blob = server.output
    assert "policy_hash_mismatch" in blob, "启动诊断应报出安全的政策分类码"
    for banned in (".yaml", "risk_policy", "configs/", "Traceback"):
        assert banned not in blob, f"启动输出泄漏了 {banned}"


def test_real_cli_does_not_log_raw_urls_on_risk_routes(tmp_path: Path) -> None:
    """未知路径与查询串不得进入进程输出（Uvicorn access log 继续关闭）。"""
    port = _free_port()
    config_path = _config_with(tmp_path, "cw006_probe.yaml", port=port)
    with _Server(config_path, port) as server:
        for url in (
            f"/v1/risk/{PATH_SENTINEL}",
            f"/v1/policy?name={POLICY_SENTINEL}",
        ):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}{url}", timeout=5).read()
            except urllib.error.HTTPError as error:
                error.read()
    assert PATH_SENTINEL not in server.output, "原始 path 出现在进程输出里"
    assert POLICY_SENTINEL not in server.output, "query string 出现在进程输出里"
    assert "GET /" not in server.output, "出现了 Uvicorn 默认 access log 的请求行"


def test_cli_still_disables_the_uvicorn_access_log(monkeypatch) -> None:
    """进程入口的 ``access_log=False`` 不得被改动。"""
    import uvicorn

    captured: dict = {}

    def _fake_run(app, **kwargs):  # noqa: ANN001, ANN202
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    import importlib.util

    spec = importlib.util.spec_from_file_location("cw006_run_api", RUN_API)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.main([]) == 0
    assert captured["access_log"] is False
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
