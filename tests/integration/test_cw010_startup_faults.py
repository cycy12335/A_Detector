"""CW-010 集成测试（一）：启动期故障、组合降级与按组件的接口隔离。

本模块只碰**加载阶段**的故障：模型/政策/存储在 lifespan 里加载失败时，应用仍必须能起来，
且每个接口只按**自己的依赖**降级。被反复验证的不变量：

- ``/health/live`` 在任何故障组合下都是 200（它只表示进程存活）；
- ``/health/ready`` 报"模型 → 政策 → 存储"里最靠前的那个阻塞组件；
- 一个组件坏掉**不得**连带停用与它无关的接口：政策坏掉时纯模型评分照常，模型坏掉时
  ``/v1/policy`` 照常，存储坏掉时前八个接口照常；
- 每个接口报的码必须指向**真正坏掉**的组件，不能张冠李戴；
- 已持久化的决策在模型与政策都坏掉时仍然读得出来（GET 只依赖存储）；
- 加载器抛出的**未预期**异常（含 ``TimeoutError``）与"已分类失败"走同一条降级路径；
- 启动诊断只写安全分类码，不写路径、配置载荷、异常原文或堆栈。

受影响接口的预期状态码与错误码直接取自 :data:`tests.cw010_fixtures.FAULT_MATRIX`，
因此"矩阵里声明的"与"测试实际验证的"是同一份数据，不会各自漂移。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from fastapi.testclient import TestClient  # noqa: E402

from ai_detector.rules.config import PolicyErrorCode, PolicyLoadError  # noqa: E402
from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.errors import (  # noqa: E402
    ERROR_MESSAGES,
    BundleErrorCode,
    BundleLoadError,
)
from ai_detector.service.logging import LOGGER_NAME  # noqa: E402
from ai_detector.storage.config import StorageErrorCode, StorageLoadError  # noqa: E402
from tests.cw005_fixtures import FakeRuntime, RecordingLoader  # noqa: E402
from tests.cw006_fixtures import FakePolicyEngine, RecordingPolicyLoader, risk_features  # noqa: E402
from tests.cw007_fixtures import (  # noqa: E402
    CREATED_AT,
    DECISION_ID_A,
    DECISION_ID_B,
    DECISION_ID_C,
    DECISION_ID_D,
    FakeDecisionRepository,
    FixedClock,
    RecordingStorageLoader,
    SequentialIdFactory,
    remove_database,
    staged_db_path,
    storage_section,
)
from tests.cw010_fixtures import (  # noqa: E402
    FAULT_MATRIX,
    SENTINEL_FEATURE_VALUE,
    SENTINEL_ITEM_ID,
    SENTINEL_KEY,
    SENTINEL_MESSAGE,
    FaultingStorageLoader,
    assert_error_body_is_safe,
    assert_no_leak,
    build_leaky_error,
    staged_relative,
)

FEATURES = ["TransactionAmt", "C13", "feat_c"]

#: 三个组件各自的"已分类"启动失败（形态与生产中真实会发生的失败一致）。
CLASSIFIED_FAULTS: dict[str, BaseException] = {
    "model": BundleLoadError(BundleErrorCode.MISSING, "bundle 文件不存在"),
    "policy": PolicyLoadError(PolicyErrorCode.NOT_CONFIGURED, "服务配置未声明 policy 段"),
    "storage": StorageLoadError(StorageErrorCode.NOT_CONFIGURED, "服务配置未声明 storage 段"),
}

#: 组件 → 未就绪时对外使用的错误码。
COMPONENT_CODE = {
    "model": "model_not_ready",
    "policy": "policy_not_ready",
    "storage": "storage_not_ready",
}

STARTUP_ROWS = [row for row in FAULT_MATRIX if row.stage == "startup"]
STARTUP_ROW_IDS = [row.key for row in STARTUP_ROWS]


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(features=FEATURES, scores=[0.05, 0.9, 0.11, 0.4])


def _make_config(tmp_path: Path, **kwargs):
    from tests.cw005_fixtures import make_config

    kwargs.setdefault("features", len(FEATURES))
    kwargs.setdefault("storage", storage_section(staged_relative(tmp_path)))
    return make_config(tmp_path, **kwargs)


def _components(component: str) -> list[str]:
    return [name for name in component.split("+") if name]


def _build(
    config,
    runtime,
    *,
    model_error: BaseException | None = None,
    policy_error: BaseException | None = None,
    storage_error: BaseException | None = None,
    storage_loader=None,
):
    """按三个"加载失败异常"构造 app；``None`` 表示该组件正常加载。

    默认存储用**内存 fake**（本模块只关心降级语义，不碰磁盘）；需要真实 SQLite 的用例
    自行传入 ``storage_loader``。
    """
    if storage_loader is None:
        storage_loader = RecordingStorageLoader(FakeDecisionRepository(), error=storage_error)
    return create_app(
        config,
        runtime_loader=RecordingLoader(runtime, error=model_error),
        policy_loader=RecordingPolicyLoader(FakePolicyEngine(), error=policy_error),
        storage_loader=storage_loader,
        decision_id_factory=SequentialIdFactory(
            [DECISION_ID_A, DECISION_ID_B, DECISION_ID_C, DECISION_ID_D]
        ),
        utc_clock=FixedClock([CREATED_AT]),
    )


def _build_for(config, runtime, component: str, *, faults=None, **kwargs):
    """让 ``component`` 里列出的组件加载失败。

    Args:
        component: ``"model"`` / ``"policy"`` / ``"storage"`` 或其 ``+`` 组合；空串表示全部正常。
        faults: 覆盖注入的异常（缺省用 :data:`CLASSIFIED_FAULTS`）。未列出的组件不注入，
            因此"未预期异常"变体只影响目标组件。
    """
    names = _components(component)
    source = dict(CLASSIFIED_FAULTS) if faults is None else dict(faults)
    return _build(
        config,
        runtime,
        model_error=source.get("model") if "model" in names else None,
        policy_error=source.get("policy") if "policy" in names else None,
        storage_error=source.get("storage") if "storage" in names else None,
        **kwargs,
    )


def _request(client: TestClient, path: str, runtime):
    """按矩阵里的路径发一次请求（合成输入，取值带哨兵）。"""
    features = risk_features(runtime, transaction_amt=SENTINEL_FEATURE_VALUE)
    if path in ("/health/live", "/health/ready", "/v1/model", "/v1/policy"):
        return client.get(path)
    if path == "/v1/decisions/{id}":
        return client.get(f"/v1/decisions/{DECISION_ID_A}")
    if path.endswith("/batch"):
        return client.post(
            path, json={"items": [{"client_item_id": SENTINEL_ITEM_ID, "features": features}]}
        )
    return client.post(path, json={"features": features})


def _assert_available(response) -> None:
    """接口按自身依赖正常工作：要么成功，要么是"查不到"这类业务性结果。

    不得出现 5xx，也不得出现任何"未就绪"降级码——那正是"无关接口被错误停用"。
    """
    assert response.status_code in (200, 201, 404), response.text
    if response.status_code == 404:
        assert response.json()["error"]["code"] == "decision_not_found"
    assert_no_leak(response.text)


# --- 正常对照 ------------------------------------------------------------


def test_healthy_control_has_every_endpoint_available(tmp_path: Path, runtime) -> None:
    """先证明"没有故障时全部可用"——否则后面的降级断言可能只是接口本来就坏。"""
    config = _make_config(tmp_path)
    app = _build_for(config, runtime, "", storage_loader=FaultingStorageLoader())
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json()["model_version"] == runtime.model_version
        for row in STARTUP_ROWS:
            for path in (item[0] for item in row.affected):
                _assert_available(_request(client, path, runtime))


# --- 受影响接口：状态码与稳定错误码与矩阵逐项一致 ------------------------


@pytest.mark.parametrize("row", STARTUP_ROWS, ids=STARTUP_ROW_IDS)
def test_affected_endpoints_match_the_matrix(tmp_path: Path, runtime, row) -> None:
    config = _make_config(tmp_path)
    app = _build_for(config, runtime, row.component)
    with TestClient(app) as client:
        for path, status, code in row.affected:
            response = _request(client, path, runtime)
            assert response.status_code == status, f"{row.key} {path}: {response.text}"
            body = response.json()["error"]
            assert body["code"] == code, f"{row.key} {path}"
            assert body["message"] == ERROR_MESSAGES[code], f"{row.key} {path}"
            assert body["request_id"], "错误体必须带请求 ID"
            assert_error_body_is_safe(response.text)


@pytest.mark.parametrize("row", STARTUP_ROWS, ids=STARTUP_ROW_IDS)
def test_unaffected_endpoints_keep_working(tmp_path: Path, runtime, row) -> None:
    """隔离不变量：一个（或两个）组件坏掉，与它无关的接口必须照常工作。"""
    config = _make_config(tmp_path)
    app = _build_for(config, runtime, row.component)
    with TestClient(app) as client:
        for path in row.unaffected:
            _assert_available(_request(client, path, runtime))


@pytest.mark.parametrize("row", STARTUP_ROWS, ids=STARTUP_ROW_IDS)
def test_reported_code_names_a_genuinely_broken_component(tmp_path: Path, runtime, row) -> None:
    """每个接口报的码必须指向**真正坏掉**的组件，不能张冠李戴。

    刻意不断言"哪个组件先被报到"这种实现顺序，只断言"报出来的那个确实坏了、没坏的不报"
    ——这正是"接口按自身依赖隔离"可检验的含义。
    """
    broken = {COMPONENT_CODE[name] for name in _components(row.component)}
    config = _make_config(tmp_path)
    app = _build_for(config, runtime, row.component)
    with TestClient(app) as client:
        for path, _, code in row.affected:
            if code not in COMPONENT_CODE.values():
                continue  # storage_unavailable 之类不是"组件未就绪"码
            assert code in broken, f"{row.key} {path} 报了没坏掉的组件：{code}"


# --- readiness 优先级 ----------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["startup.model+policy", "startup.policy+storage", "startup.model+storage"],
)
def test_readiness_priority_is_model_then_policy_then_storage(
    tmp_path: Path, runtime, key: str
) -> None:
    row = next(item for item in STARTUP_ROWS if item.key == key)
    expected = {path: code for path, _, code in row.affected}["/health/ready"]
    config = _make_config(tmp_path)
    app = _build_for(config, runtime, row.component)
    with TestClient(app) as client:
        ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == expected


def test_live_is_200_under_every_combined_fault(tmp_path: Path, runtime) -> None:
    """``/health/live`` 只表示进程存活：任何组合故障下都必须 200。"""
    for row in STARTUP_ROWS:
        config = _make_config(tmp_path)
        app = _build_for(config, runtime, row.component)
        with TestClient(app) as client:
            live = client.get("/health/live")
        assert live.status_code == 200, f"{row.key} 让 /health/live 降级了"
        assert_no_leak(live.text)


# --- 真实存储：模型与政策都坏掉时，历史决策仍可读 ------------------------


def test_stored_decisions_are_still_readable_when_model_and_policy_are_broken(
    tmp_path: Path, runtime
) -> None:
    """GET 只依赖存储：模型与政策都不可用时，已提交的决策仍必须读得出来。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        healthy = _build_for(config, runtime, "", storage_loader=FaultingStorageLoader())
        with TestClient(healthy) as client:
            created = client.post(
                "/v1/decisions/evaluate",
                json={"features": risk_features(runtime, transaction_amt=12.5)},
            )
            assert created.status_code == 201, created.text
            snapshot = created.json()["decision"]

        degraded = _build_for(
            config, runtime, "model+policy", storage_loader=FaultingStorageLoader()
        )
        with TestClient(degraded) as client:
            fetched = client.get(f"/v1/decisions/{snapshot['decision_id']}")
            assert fetched.status_code == 200, fetched.text
            assert fetched.json()["decision"] == snapshot
            assert_no_leak(fetched.text)
    finally:
        remove_database(db_file)


# --- 存储未配置：既有 readiness 语义不得被误改 ---------------------------


def test_unconfigured_storage_does_not_change_readiness_semantics(
    tmp_path: Path, runtime
) -> None:
    """未配置存储时 readiness 只由模型与政策决定（保持 CW-006 语义）。"""
    config = _make_config(tmp_path, storage=None)
    with TestClient(_build_for(config, runtime, "")) as client:
        assert client.get("/health/ready").status_code == 200

    with TestClient(_build_for(config, runtime, "policy")) as client:
        ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "policy_not_ready"


def test_failed_storage_load_never_falls_back_to_a_database(tmp_path: Path, runtime) -> None:
    """存储加载失败时不得回退到内存库或临时文件：一个字节都不该落盘。"""
    config = _make_config(tmp_path)
    app = _build_for(config, runtime, "storage")
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/decisions/evaluate", json={"features": risk_features(runtime)}
            ).status_code
            == 503
        )
    assert getattr(app.state, "decision_repository", None) is None
    assert not staged_db_path(tmp_path).exists(), "启动失败时不得创建任何数据库文件"


# --- 未预期异常与 TimeoutError 走同一条降级路径 --------------------------


@pytest.mark.parametrize("component", ["model", "policy", "storage"])
@pytest.mark.parametrize(
    "make_fault",
    [lambda: TimeoutError(SENTINEL_MESSAGE), lambda: build_leaky_error(RuntimeError)],
    ids=["timeout", "unexpected"],
)
def test_unexpected_startup_faults_degrade_exactly_like_classified_ones(
    tmp_path: Path, runtime, component: str, make_fault
) -> None:
    """加载器抛出的**未预期**异常（含 ``TimeoutError``）必须折叠成同一条降级路径。

    对外码与"已分类失败"完全一致：进程照常起来，只有该组件及其下游接口 503。
    """
    fault = make_fault()
    config = _make_config(tmp_path)
    app = _build_for(config, runtime, component, faults={component: fault})
    row = next(item for item in STARTUP_ROWS if item.component == component)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["error"]["code"] == COMPONENT_CODE[component]
        for path, status, code in row.affected:
            response = _request(client, path, runtime)
            assert response.status_code == status, f"{component} {path}: {response.text}"
            assert response.json()["error"]["code"] == code, f"{component} {path}"
            assert_error_body_is_safe(response.text)


# --- 降级路径上的密钥/特征脱敏 -------------------------------------------


@pytest.mark.parametrize(
    "component,use_real_storage",
    [("storage", False), ("model", True), ("model+policy", True)],
    ids=["storage-load-failed", "model-broken", "model+policy-broken"],
)
def test_keyed_fault_responses_never_echo_the_key_or_features(
    tmp_path: Path, runtime, component: str, use_real_storage: bool
) -> None:
    """带 ``Idempotency-Key`` 的请求在降级路径上也不得回显 key 或任何特征取值。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    loader = FaultingStorageLoader() if use_real_storage else None
    try:
        app = _build_for(config, runtime, component, storage_loader=loader)
        with TestClient(app) as client:
            response = client.post(
                "/v1/decisions/evaluate",
                json={"features": risk_features(runtime, transaction_amt=SENTINEL_FEATURE_VALUE)},
                headers={"Idempotency-Key": SENTINEL_KEY},
            )
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] in COMPONENT_CODE.values()
        assert_error_body_is_safe(response.text)
    finally:
        remove_database(db_file)


# --- 启动诊断日志 --------------------------------------------------------


def test_startup_diagnostic_names_every_blocked_component_without_leaking(
    tmp_path: Path, runtime, caplog
) -> None:
    """三个组件同时坏掉时，启动诊断必须逐个报出安全分类码，且不含任何原文/路径。"""
    config = _make_config(tmp_path)
    leaky = {name: build_leaky_error(RuntimeError) for name in CLASSIFIED_FAULTS}
    app = _build_for(config, runtime, "model+policy+storage", faults=leaky)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        with TestClient(app):
            pass
    lines = [record.message for record in caplog.records if "service_startup" in record.message]
    assert len(lines) == 1, "启动诊断恰好一条"
    payload = json.loads(lines[0])
    assert payload["outcome"] == "not_ready"
    assert payload["error_code"] == "bundle_unexpected_error"
    assert payload["policy_error_code"] == "policy_unexpected_error"
    assert payload["storage_error_code"] == "storage_unexpected_error"
    assert "model_version" not in payload, "未就绪时不得宣称模型版本"
    assert_no_leak(lines[0])


def test_startup_diagnostic_is_ready_only_when_all_components_loaded(
    tmp_path: Path, runtime, caplog
) -> None:
    """全部就绪时 ``outcome`` 为 ``ready``，且不含任何 error 码字段。"""
    config = _make_config(tmp_path)
    app = _build_for(config, runtime, "", storage_loader=FaultingStorageLoader())
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        with TestClient(app):
            pass
    lines = [record.message for record in caplog.records if "service_startup" in record.message]
    payload = json.loads(lines[0])
    assert payload["outcome"] == "ready"
    for key in ("error_code", "policy_error_code", "storage_error_code"):
        assert key not in payload, f"就绪时不得出现 {key}"
    assert_no_leak(lines[0])


def test_startup_diagnostic_is_skipped_when_storage_is_unconfigured(
    tmp_path: Path, runtime, caplog
) -> None:
    """未配置存储不属于降级：``storage_ready`` 保持 True，也不写 storage_error_code。"""
    config = _make_config(tmp_path, storage=None)
    app = _build_for(config, runtime, "")
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        with TestClient(app):
            pass
    lines = [record.message for record in caplog.records if "service_startup" in record.message]
    payload = json.loads(lines[0])
    assert payload["outcome"] == "ready"
    assert payload["storage_ready"] is True
    assert "storage_error_code" not in payload
