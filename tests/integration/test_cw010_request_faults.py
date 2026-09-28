"""CW-010 集成测试（二）：请求期故障、持久化原子性与故障解除后的重试。

本模块只碰**请求期**故障：模型预测、规则求值、存储读/写在一次请求里抛错时，接口必须
折叠成稳定错误、不得回显载荷，并且**不留下任何部分结果**。被反复验证的不变量：

- 未预期异常（含注入的 ``TimeoutError``）一律折叠为安全错误：裸异常 → 500
  ``internal_error``，存储层**已分类**的失败 → 503 ``storage_unavailable``；
- 批量在中途失败时不返回"看起来完整"的部分结果，且**零记录落库**；规则求值失败时
  **零次模型调用**（求值顺序在模型调用之前）；
- 持久化请求在 insert/commit 失败时不留部分 decision、不留部分幂等映射，
  也不返回未提交的成功响应；
- 失败**不会毒化**幂等键：故障解除后用同一个 key 重试仍然正常创建，之后重放不重跑
  模型/政策、也不追加写入；
- 每个场景都有正常对照，且能区分"故障中"与"故障解除后"。

存储故障在**真实 SQLite** 上注入（``FaultInjectingRepository`` 包住生产加载器打开的库），
因此"回滚后零残留"是在真库上验证的。注入点与预期状态码取自
:data:`tests.cw010_fixtures.FAULT_MATRIX`。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from fastapi.testclient import TestClient  # noqa: E402

from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.errors import ERROR_MESSAGES  # noqa: E402
from ai_detector.service.logging import LOGGER_NAME  # noqa: E402
from ai_detector.storage.idempotency import IDEMPOTENCY_KEY_HEADER, REPLAYED_HEADER  # noqa: E402
from ai_detector.storage.sqlite import StorageUnavailableError  # noqa: E402
from tests.cw005_fixtures import FakeRuntime, RecordingLoader  # noqa: E402
from tests.cw006_fixtures import FakePolicyEngine, RecordingPolicyLoader, risk_features  # noqa: E402
from tests.cw007_fixtures import (  # noqa: E402
    CREATED_AT,
    CREATED_AT_LATER,
    DECISION_ID_A,
    DECISION_ID_B,
    DECISION_ID_C,
    DECISION_ID_D,
    FixedClock,
    SequentialIdFactory,
    remove_database,
    sqlite_connection,
    staged_db_path,
    storage_section,
)
from tests.cw010_fixtures import (  # noqa: E402
    FAULT_MATRIX,
    SENTINEL_FEATURE_VALUE,
    SENTINEL_KEY,
    SENTINEL_MESSAGE,
    FaultingStorageLoader,
    FlakyPolicyEngine,
    NthCallFault,
    OperationFault,
    assert_error_body_is_safe,
    assert_no_leak,
    build_leaky_error,
    staged_relative,
)

FEATURES = ["TransactionAmt", "C13", "feat_c"]
DEFAULT_IDS = (DECISION_ID_A, DECISION_ID_B, DECISION_ID_C, DECISION_ID_D)

#: 异常名（矩阵 ``operation`` 里 ``->`` 右侧）→ 该注入用的异常实例。
FAULT_BY_NAME = {
    "TimeoutError": lambda: TimeoutError(SENTINEL_MESSAGE),
    "RuntimeError": lambda: build_leaky_error(RuntimeError),
    "StorageUnavailableError": lambda: StorageUnavailableError(SENTINEL_MESSAGE),
}

#: 需要携带 ``Idempotency-Key`` 才能触发的行：查表与提交都只在带 key 的路径上发生。
KEYED_ROW_KEYS = frozenset(
    {
        "request.storage.lookup.timeout",
        "request.storage.lookup.unavailable",
        "request.storage.commit.timeout",
        "request.storage.commit.rollback",
    }
)

#: 需要专用夹具（真实重复 ID 触发真库回滚）而无法通用注入的行。
DEDICATED_ROW_KEYS = frozenset({"request.storage.commit.rollback"})

REQUEST_ROWS = [row for row in FAULT_MATRIX if row.stage == "request"]
INJECTABLE_ROWS = [row for row in REQUEST_ROWS if row.key not in DEDICATED_ROW_KEYS]


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(features=FEATURES, scores=[0.05, 0.9, 0.11, 0.4])


def _make_config(tmp_path: Path, **kwargs):
    from tests.cw005_fixtures import make_config

    kwargs.setdefault("features", len(FEATURES))
    kwargs.setdefault("storage", storage_section(staged_relative(tmp_path)))
    return make_config(tmp_path, **kwargs)


def _split_operation(row) -> tuple[str, str]:
    """把矩阵里的 ``"op -> ErrorName"`` 拆成两部分（测试与矩阵共用同一份描述）。"""
    operation, _, name = row.operation.partition(" -> ")
    return operation.strip(), name.strip()


def _build(config, runtime, engine=None, storage_loader=None, ids=DEFAULT_IDS):
    """构造 app：模型/政策/存储全部就绪，故障只在**请求期**按需触发。"""
    return create_app(
        config,
        runtime_loader=RecordingLoader(runtime),
        policy_loader=RecordingPolicyLoader(engine if engine is not None else FakePolicyEngine()),
        storage_loader=storage_loader if storage_loader is not None else FaultingStorageLoader(),
        decision_id_factory=SequentialIdFactory(list(ids)),
        utc_clock=FixedClock([CREATED_AT, CREATED_AT_LATER]),
    )


def _build_for_row(config, row, *, ids=DEFAULT_IDS):
    """按矩阵行把注入点接到对应的 fake 上：``operation`` 字段直接驱动注入。"""
    operation, name = _split_operation(row)
    assert name in FAULT_BY_NAME, f"{row.key} 无法通用注入，应列入 DEDICATED_ROW_KEYS"
    if row.component == "model":
        assert operation == "score_many", row.operation
        runtime = FakeRuntime(features=FEATURES, scores=[0.05, 0.9, 0.11, 0.4], error=FAULT_BY_NAME[name]())
        return _build(config, runtime, ids=ids)
    if row.component == "policy":
        assert operation == "evaluate_rules", row.operation
        engine = FakePolicyEngine(error=FAULT_BY_NAME[name]())
        return _build(config, runtime=FakeRuntime(features=FEATURES), engine=engine, ids=ids)
    loader = FaultingStorageLoader(
        faults={operation: OperationFault(FAULT_BY_NAME[name](), times=OperationFault.ALWAYS)}
    )
    return _build(config, runtime=FakeRuntime(features=FEATURES), storage_loader=loader, ids=ids)


def _features(runtime, **kwargs):
    kwargs.setdefault("transaction_amt", SENTINEL_FEATURE_VALUE)
    return risk_features(runtime, **kwargs)


def _body(runtime, **kwargs) -> dict:
    return {"features": _features(runtime, **kwargs)}


def _batch_body(runtime, item_ids=("item-a", "item-b"), **kwargs) -> dict:
    return {
        "items": [
            {"client_item_id": item_id, "features": _features(runtime, **kwargs)}
            for item_id in item_ids
        ]
    }


def _headers(row_key: str) -> dict[str, str]:
    return {IDEMPOTENCY_KEY_HEADER: SENTINEL_KEY} if row_key in KEYED_ROW_KEYS else {}


#: 只接受 GET 的接口。
GET_PATHS = frozenset({"/health/live", "/health/ready", "/v1/model", "/v1/policy"})


def _request(client: TestClient, path: str, runtime, row_key: str):
    """按路径形状发一次请求（合成输入，取值带哨兵）。"""
    headers = _headers(row_key)
    if path in GET_PATHS:
        return client.get(path, headers=headers)
    if path == "/v1/decisions/{id}":
        return client.get(f"/v1/decisions/{DECISION_ID_A}", headers=headers)
    if path.endswith("/batch"):
        return client.post(path, json=_batch_body(runtime), headers=headers)
    return client.post(path, json=_body(runtime), headers=headers)


def _rows(db_file: Path, table: str) -> int:
    with sqlite_connection(db_file) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _all_counts(db_file: Path) -> dict[str, int]:
    return {
        table: _rows(db_file, table)
        for table in ("decisions", "idempotency_requests", "idempotency_results")
    }


def _completion_logs(caplog) -> list[dict]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if "request_completed" in record.message
    ]


# --- 正常对照 ------------------------------------------------------------


def test_healthy_control_succeeds_on_every_endpoint(tmp_path: Path, runtime) -> None:
    """先证明没有故障时全部可用，且确实写入了记录。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        app = _build(config, runtime)
        with TestClient(app) as client:
            assert client.get("/health/live").status_code == 200
            assert client.get("/health/ready").status_code == 200
            assert client.post("/v1/score", json=_body(runtime)).status_code == 200
            assert client.post("/v1/score/batch", json=_batch_body(runtime)).status_code == 200
            assert client.post("/v1/risk/evaluate", json=_body(runtime)).status_code == 200
            created = client.post("/v1/decisions/evaluate", json=_body(runtime))
            assert created.status_code == 201, created.text
            decision_id = created.json()["decision"]["decision_id"]
            assert client.get(f"/v1/decisions/{decision_id}").status_code == 200
        assert _rows(db_file, "decisions") == 1
    finally:
        remove_database(db_file)


# --- 矩阵驱动：请求期故障的折叠与隔离 ------------------------------------


@pytest.mark.parametrize("row", INJECTABLE_ROWS, ids=[row.key for row in INJECTABLE_ROWS])
def test_request_faults_fold_exactly_as_the_matrix_declares(
    tmp_path: Path, runtime, row
) -> None:
    """受影响接口报出矩阵声明的状态码与稳定码，未受影响的接口照常工作。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        app = _build_for_row(config, row)
        with TestClient(app) as client:
            for path, status, code in row.affected:
                response = _request(client, path, runtime, row.key)
                assert response.status_code == status, f"{row.key} {path}: {response.text}"
                body = response.json()["error"]
                assert body["code"] == code, f"{row.key} {path}"
                assert body["message"] == ERROR_MESSAGES[code], f"{row.key} {path}"
                assert_error_body_is_safe(response.text)
            for path in row.unaffected:
                response = _request(client, path, runtime, row.key)
                # 允许"查不到"这类业务性 404（GET 只读存储快照，没有记录就是 404）；
                # 不允许任何降级码——那才是"无关接口被故障连带停用"。
                assert response.status_code in (200, 201, 404), f"{row.key} {path}: {response.text}"
                if response.status_code == 404:
                    assert response.json()["error"]["code"] == "decision_not_found"
                assert_no_leak(response.text)
    finally:
        remove_database(db_file)


def test_the_only_row_needing_a_dedicated_fixture_is_the_real_rollback() -> None:
    """跳过通用注入的行必须恰好是"真库回滚"那一行，且它由专用用例覆盖。

    否则新增一行矩阵时可能因为无法注入而被静默漏测。
    """
    assert {row.key for row in REQUEST_ROWS} - {row.key for row in INJECTABLE_ROWS} == set(
        DEDICATED_ROW_KEYS
    )


# --- 原子性：批量中途失败不出部分结果 ------------------------------------


def test_policy_fault_on_the_second_item_fails_the_batch_with_zero_model_calls(
    tmp_path: Path, runtime
) -> None:
    """2 笔批量在第 2 条的规则求值上失败：整批失败、**零次模型调用**、零记录。

    规则求值在模型调用之前，因此"走到一半"的失败不可能已经消耗过模型，也不可能
    已经产生任何可返回的部分结果。
    """
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        engine = FlakyPolicyEngine(FakePolicyEngine(), NthCallFault(2, TimeoutError(SENTINEL_MESSAGE)))
        app = _build(config, runtime, engine=engine)
        with TestClient(app) as client:
            response = client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime))
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        assert engine.calls == 2, "第 2 条确实被求值过，失败发生在中途"
        assert runtime.call_count == 0, "规则求值失败时模型一次都不该被调用"
        assert _all_counts(db_file) == {
            "decisions": 0,
            "idempotency_requests": 0,
            "idempotency_results": 0,
        }
        assert_error_body_is_safe(response.text)
    finally:
        remove_database(db_file)


def test_model_fault_after_full_rule_evaluation_writes_nothing(tmp_path: Path, runtime) -> None:
    """规则全部求值完毕后模型才失败：仍然零记录，且错误体不回显条目 ID。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        failing = FakeRuntime(features=FEATURES, error=TimeoutError(SENTINEL_MESSAGE))
        engine = FakePolicyEngine()
        app = _build(config, failing, engine=engine)
        with TestClient(app) as client:
            response = client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime))
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"
        assert engine.call_count == 2, "两条规则求值都已完成"
        assert _all_counts(db_file)["decisions"] == 0
        assert "item-a" not in response.text and "item-b" not in response.text
        assert_error_body_is_safe(response.text)
    finally:
        remove_database(db_file)


def test_single_model_fault_writes_nothing(tmp_path: Path, runtime) -> None:
    """单笔评分失败同样零记录：不能把"评分失败"变成"写了一条空记录"。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        failing = FakeRuntime(features=FEATURES, error=TimeoutError(SENTINEL_MESSAGE))
        app = _build(config, failing)
        with TestClient(app) as client:
            response = client.post("/v1/decisions/evaluate", json=_body(runtime))
        assert response.status_code == 500
        assert _all_counts(db_file)["decisions"] == 0
        assert_error_body_is_safe(response.text)
    finally:
        remove_database(db_file)


# --- 原子性：提交失败同时回滚记录与幂等映射 ------------------------------


def test_keyed_commit_rollback_leaves_no_decision_and_no_mapping(
    tmp_path: Path, runtime
) -> None:
    """2 笔带 key 批量在提交事务里失败（真实 SQLite 唯一约束冲突）：

    ``decisions``、``idempotency_requests``、``idempotency_results`` 必须**同时**回滚为空，
    且返回的是稳定 503 而不是未提交的 201。
    """
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        # 两个条目拿到同一个 decision_id：真实 INSERT 会在事务内违反主键约束。
        app = _build(config, runtime, ids=(DECISION_ID_A, DECISION_ID_A, DECISION_ID_B, DECISION_ID_C))
        with TestClient(app) as client:
            response = client.post(
                "/v1/decisions/evaluate/batch",
                json=_batch_body(runtime),
                headers={IDEMPOTENCY_KEY_HEADER: SENTINEL_KEY},
            )
        assert response.status_code == 503, response.text
        assert response.json()["error"]["code"] == "storage_unavailable"
        assert _all_counts(db_file) == {
            "decisions": 0,
            "idempotency_requests": 0,
            "idempotency_results": 0,
        }, "提交失败必须同时回滚记录与幂等映射"
        assert_error_body_is_safe(response.text)
    finally:
        remove_database(db_file)


def test_failed_keyed_request_is_not_poisoned_and_retry_replays(
    tmp_path: Path, runtime
) -> None:
    """失败不得毒化幂等键：同一 key 重试应正常创建，随后重放不重跑模型/政策。

    这正是"部分幂等映射被写进去"会造成的后果——重试会误报 409 冲突，或者重放出一个
    并不存在的决策。这里用"先失败、后成功、再重放"三阶段把三者区分开。
    """
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        app = _build(
            config,
            runtime,
            ids=(DECISION_ID_A, DECISION_ID_A, DECISION_ID_B, DECISION_ID_C),
        )
        body = _batch_body(runtime)
        headers = {IDEMPOTENCY_KEY_HEADER: SENTINEL_KEY}
        with TestClient(app) as client:
            # 阶段一：提交失败
            failed = client.post("/v1/decisions/evaluate/batch", json=body, headers=headers)
            assert failed.status_code == 503
            assert _all_counts(db_file)["idempotency_requests"] == 0
            created = client.post("/v1/decisions/evaluate/batch", json=body, headers=headers)
            assert created.status_code == 201, created.text
            assert created.headers[REPLAYED_HEADER] == "false"
            snapshot = created.json()["decisions"]
            assert _all_counts(db_file) == {
                "decisions": 2,
                "idempotency_requests": 1,
                "idempotency_results": 2,
            }
            calls_before = (runtime.call_count, app.state.policy_engine.call_count)
            # 阶段二：重放
            replayed = client.post("/v1/decisions/evaluate/batch", json=body, headers=headers)
            assert replayed.status_code == 201
            assert replayed.headers[REPLAYED_HEADER] == "true"
            assert replayed.json()["decisions"] == snapshot
            calls_after = (runtime.call_count, app.state.policy_engine.call_count)
        assert calls_after == calls_before, "重放不得重新运行模型或政策"
        assert _all_counts(db_file)["decisions"] == 2, "重放不得追加写入"
    finally:
        remove_database(db_file)


# --- 查表失败发生在求值之前 ----------------------------------------------


@pytest.mark.parametrize(
    "error_factory,expected",
    [
        (lambda: TimeoutError(SENTINEL_MESSAGE), (500, "internal_error")),
        (lambda: StorageUnavailableError(SENTINEL_MESSAGE), (503, "storage_unavailable")),
    ],
    ids=["timeout", "classified-unavailable"],
)
def test_lookup_fault_happens_before_any_evaluation(
    tmp_path: Path, runtime, error_factory, expected
) -> None:
    """查表失败时模型与政策**一次都不运行**，也没有任何写入。

    这证明失败发生在求值之前，因此不可能留下"算过了但没存"的中间状态。
    """
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        loader = FaultingStorageLoader(
            faults={
                "lookup_idempotent": OperationFault(
                    error_factory(), times=OperationFault.ALWAYS
                )
            }
        )
        engine = FakePolicyEngine()
        app = _build(config, runtime, engine=engine, storage_loader=loader)
        with TestClient(app) as client:
            response = client.post(
                "/v1/decisions/evaluate",
                json=_body(runtime),
                headers={IDEMPOTENCY_KEY_HEADER: SENTINEL_KEY},
            )
        status, code = expected
        assert response.status_code == status, response.text
        assert response.json()["error"]["code"] == code
        assert runtime.call_count == 0 and engine.call_count == 0
        assert loader.repository.calls["lookup_idempotent"] == 1
        assert loader.repository.delivered["lookup_idempotent"] == 0
        assert _all_counts(db_file) == {
            "decisions": 0,
            "idempotency_requests": 0,
            "idempotency_results": 0,
        }
        assert_error_body_is_safe(response.text)
    finally:
        remove_database(db_file)


def test_commit_path_corrupt_mapping_is_a_safe_500_and_adds_nothing(
    tmp_path: Path, runtime
) -> None:
    """提交事务内才发现映射损坏：必须是 500 ``idempotency_contract_violation``，且不新增记录。

    CW-008 只覆盖了**查表**阶段的损坏。这里走竞态输家的那条路径：查表临时 miss，
    于是提交事务里二次读取才撞上损坏的映射。
    """
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        app = _build(config, runtime)
        body = _body(runtime)
        headers = {IDEMPOTENCY_KEY_HEADER: SENTINEL_KEY}
        with TestClient(app) as client:
            assert client.post("/v1/decisions/evaluate", json=body, headers=headers).status_code == 201
            with sqlite_connection(db_file) as conn:
                conn.execute("PRAGMA foreign_keys = OFF")
                conn.execute("PRAGMA ignore_check_constraints = ON")
                conn.execute("DELETE FROM idempotency_results")
            before = _all_counts(db_file)

            from ai_detector.storage.sqlite import IDEMPOTENCY_MISS, IdempotencyHit

            repository = app.state.decision_repository
            original = type(repository).lookup_idempotent
            type(repository).lookup_idempotent = (
                lambda self, *args, **kwargs: IdempotencyHit(IDEMPOTENCY_MISS)
            )
            try:
                response = client.post("/v1/decisions/evaluate", json=body, headers=headers)
            finally:
                type(repository).lookup_idempotent = original

        assert response.status_code == 500, response.text
        assert response.json()["error"]["code"] == "idempotency_contract_violation"
        assert _all_counts(db_file) == before, "损坏映射不得导致任何新写入"
        assert_error_body_is_safe(response.text)
    finally:
        remove_database(db_file)


# --- 故障解除后恢复：区分故障前 / 故障中 / 故障解除后 --------------------


def test_releasing_the_fault_restores_full_behaviour(tmp_path: Path, runtime) -> None:
    """同一 app 生命周期内：故障中失败、故障解除后成功——证明降级是**可恢复**的。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        loader = FaultingStorageLoader(
            faults={"insert_many": OperationFault(StorageUnavailableError(SENTINEL_MESSAGE), times=1)}
        )
        app = _build(config, runtime, storage_loader=loader)
        with TestClient(app) as client:
            during = client.post("/v1/decisions/evaluate", json=_body(runtime))
            after = client.post("/v1/decisions/evaluate", json=_body(runtime))
            fetched = client.get(f"/v1/decisions/{after.json()['decision']['decision_id']}")
        assert during.status_code == 503
        assert during.json()["error"]["code"] == "storage_unavailable"
        assert after.status_code == 201, after.text
        assert fetched.status_code == 200
        assert _rows(db_file, "decisions") == 1, "失败那次不得留下记录"
        assert loader.repository.calls["insert_many"] == 2
        assert loader.repository.delivered["insert_many"] == 1
    finally:
        remove_database(db_file)


def test_read_fault_does_not_disable_writes_within_the_same_component(
    tmp_path: Path, runtime
) -> None:
    """同一个组件的读故障不得连带停写：GET 失败，POST 仍须正常创建。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        loader = FaultingStorageLoader(
            faults={"get": OperationFault(TimeoutError(SENTINEL_MESSAGE), times=OperationFault.ALWAYS)}
        )
        app = _build(config, runtime, storage_loader=loader)
        with TestClient(app) as client:
            created = client.post("/v1/decisions/evaluate", json=_body(runtime))
            fetched = client.get(f"/v1/decisions/{created.json()['decision']['decision_id']}")
        assert created.status_code == 201, created.text
        assert fetched.status_code == 500
        assert fetched.json()["error"]["code"] == "internal_error"
        assert _rows(db_file, "decisions") == 1
        assert_error_body_is_safe(fetched.text)
    finally:
        remove_database(db_file)


# --- 请求期故障不得影响健康检查 ------------------------------------------


def test_health_endpoints_stay_healthy_during_request_time_faults(
    tmp_path: Path, runtime
) -> None:
    """请求期故障是**单次请求**的失败：进程与 readiness 都必须保持就绪。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        for row in INJECTABLE_ROWS:
            app = _build_for_row(config, row)
            with TestClient(app) as client:
                assert client.get("/health/live").status_code == 200, row.key
                assert client.get("/health/ready").status_code == 200, row.key
    finally:
        remove_database(db_file)


# --- 日志脱敏 ------------------------------------------------------------


def test_request_faults_log_one_safe_completion_line_without_payload(
    tmp_path: Path, runtime, caplog
) -> None:
    """故障请求仍然恰好一条 completion log，且不含特征值、幂等键或异常原文。"""
    config = _make_config(tmp_path)
    db_file = staged_db_path(tmp_path)
    try:
        for row in INJECTABLE_ROWS:
            app = _build_for_row(config, row)
            with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
                with TestClient(app) as client:
                    response = _request(client, row.affected[0][0], runtime, row.key)
            logs = _completion_logs(caplog)
            assert len(logs) == 1, f"{row.key} 应恰好一条 completion log"
            entry = logs[0]
            assert entry["status_code"] == response.status_code
            assert entry["outcome"] == "error"
            assert_no_leak(json.dumps(entry, ensure_ascii=False), extra=[SENTINEL_KEY])
            caplog.clear()
    finally:
        remove_database(db_file)
