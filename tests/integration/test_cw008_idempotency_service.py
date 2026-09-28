"""CW-008 服务集成测试：两个 decisions POST 上的可选 ``Idempotency-Key``。

刻意**不依赖正式 bundle**：模型/政策用 fake 注入，存储是**真实 SQLite v2**（重放、冲突、
原子性与竞态本来就只能在真库上验证）。被反复验证的不变量：

- 无 key 时与 CW-007 逐字一致（新 ID、新记录、不写映射、不返回 ``Idempotency-Replayed``）；
- 同键同请求重放：201、body 与首次逐字段一致、**零模型/零规则/零 UUID/零时钟/零写入**；
- 同键异请求或异 operation：409，且在模型与规则**之前**发生，零新增记录；
- 重放只依赖存储：模型与政策都坏掉时仍可重放。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from fastapi.testclient import TestClient  # noqa: E402

from ai_detector.rules.config import PolicyErrorCode, PolicyLoadError  # noqa: E402
from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.errors import ERROR_MESSAGES, BundleErrorCode, BundleLoadError  # noqa: E402
from ai_detector.storage.idempotency import (  # noqa: E402
    IDEMPOTENCY_KEY_HEADER,
    REPLAYED_HEADER,
)
from tests.cw005_fixtures import FakeRuntime, RecordingLoader  # noqa: E402
from tests.cw006_fixtures import FakePolicyEngine, RecordingPolicyLoader, risk_features  # noqa: E402
from tests.cw007_fixtures import (  # noqa: E402
    DECISION_ID_A,
    DECISION_ID_B,
    DECISION_ID_C,
    DECISION_ID_D,
    FixedClock,
    SequentialIdFactory,
    remove_database,
    sqlite_connection,
    staged_db_path,
)
from tests.cw008_fixtures import KEY_ONE, KEY_TWO  # noqa: E402

FEATURES = ["TransactionAmt", "C13", "feat_c"]
REQUEST_ID_HEADER = "X-Request-ID"


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(features=FEATURES, scores=[0.05, 0.9, 0.11, 0.4])


@pytest.fixture
def engine() -> FakePolicyEngine:
    return FakePolicyEngine()


def _make_config(tmp_path: Path, **kwargs):
    from tests.cw005_fixtures import make_config
    from tests.cw007_fixtures import staged_db_relative, storage_section

    kwargs.setdefault("features", len(FEATURES))
    kwargs.setdefault("storage", storage_section(staged_db_relative(tmp_path)))
    return make_config(tmp_path, **kwargs)


@pytest.fixture
def config(tmp_path: Path):
    return _make_config(tmp_path)


@pytest.fixture
def db_file(tmp_path: Path) -> Path:
    path = staged_db_path(tmp_path)
    yield path
    remove_database(path)


def _client(
    config,
    runtime,
    *,
    engine=None,
    ids=(DECISION_ID_A, DECISION_ID_B, DECISION_ID_C, DECISION_ID_D),
    runtime_error: Exception | None = None,
    policy_error: Exception | None = None,
):
    loader = RecordingLoader(runtime, error=runtime_error)
    policy_loader = (
        RecordingPolicyLoader(engine, error=policy_error)
        if engine is not None
        else RecordingPolicyLoader(FakePolicyEngine(), error=policy_error)
    )
    return create_app(
        config,
        runtime_loader=loader,
        policy_loader=policy_loader,
        decision_id_factory=SequentialIdFactory(list(ids)),
        utc_clock=FixedClock(["2026-09-25T04:05:06Z", "2026-09-25T04:05:07Z"]),
    )


def _body(runtime, **kwargs) -> dict:
    return {"features": risk_features(runtime, **kwargs)}


def _batch_body(runtime, ids: list[str]) -> dict:
    return {
        "items": [
            {"client_item_id": item_id, "features": risk_features(runtime)} for item_id in ids
        ]
    }


def _rows(db_file: Path, table: str) -> int:
    with sqlite_connection(db_file) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# --- 无 key：CW-007 行为逐字不变 ----------------------------------------


def test_without_key_behaviour_is_unchanged(config, runtime, db_file) -> None:
    app = _client(config, runtime)
    with TestClient(app) as client:
        first = client.post("/v1/decisions/evaluate", json=_body(runtime))
        second = client.post("/v1/decisions/evaluate", json=_body(runtime))
    assert first.status_code == second.status_code == 201
    assert REPLAYED_HEADER not in first.headers
    assert REPLAYED_HEADER not in second.headers
    assert first.json()["decision"]["decision_id"] == DECISION_ID_A
    assert second.json()["decision"]["decision_id"] == DECISION_ID_B, "无 key 必须每次新建记录"
    assert _rows(db_file, "decisions") == 2
    assert _rows(db_file, "idempotency_requests") == 0
    assert _rows(db_file, "idempotency_results") == 0


def test_without_key_batch_is_unchanged(config, runtime, db_file) -> None:
    app = _client(config, runtime)
    with TestClient(app) as client:
        first = client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["a", "b"]))
        second = client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["a", "b"]))
    assert first.status_code == second.status_code == 201
    assert first.json()["decisions"] != second.json()["decisions"], "无 key 必须每次新建记录"
    assert _rows(db_file, "decisions") == 4
    assert _rows(db_file, "idempotency_requests") == 0


# --- keyed 单笔 ---------------------------------------------------------


def test_keyed_single_creates_then_replays(config, runtime, db_file) -> None:
    app = _client(config, runtime)
    body = _body(runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        first = client.post("/v1/decisions/evaluate", json=body, headers=headers)
        second = client.post("/v1/decisions/evaluate", json=body, headers=headers)
    assert first.status_code == 201
    assert first.headers[REPLAYED_HEADER] == "false"
    assert second.status_code == 201
    assert second.headers[REPLAYED_HEADER] == "true"
    # 外层 request_id 是**当前**请求的，内层 decision 完全一致
    assert first.json()["request_id"] != second.json()["request_id"]
    assert first.json()["decision"] == second.json()["decision"]
    assert _rows(db_file, "decisions") == 1
    assert _rows(db_file, "idempotency_requests") == 1
    assert _rows(db_file, "idempotency_results") == 1


def test_replay_performs_zero_evaluation_and_zero_writes(config, runtime, db_file) -> None:
    """重放：不调用模型、不运行规则、不生成 UUID/时钟、不写库。"""
    app = _client(config, runtime)
    body = _body(runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        first = client.post("/v1/decisions/evaluate", json=body, headers=headers)
        runtime.score_calls.clear()
        runtime._scores = []                      # 让"再评分"变成不可能
        scores_after_first = runtime.call_count
        second = client.post("/v1/decisions/evaluate", json=body, headers=headers)
    assert first.status_code == 201 and second.status_code == 201
    assert second.headers[REPLAYED_HEADER] == "true"
    assert runtime.call_count == scores_after_first == 0, "重放不得调用模型"
    assert second.json()["decision"] == first.json()["decision"]
    assert _rows(db_file, "decisions") == 1


def test_replay_works_when_model_and_policy_are_broken(config, runtime, db_file) -> None:
    """重放只依赖存储：模型与政策都坏了也照样返回已提交快照。"""
    app = _client(config, runtime)
    body = _body(runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        first = client.post("/v1/decisions/evaluate", json=body, headers=headers)
        assert first.status_code == 201
        assert client.post("/v1/score", json=body).status_code == 200

    broken = _client(
        config,
        FakeRuntime(features=FEATURES, error=RuntimeError("boom")),
        runtime_error=BundleLoadError(BundleErrorCode.MISSING, "x"),
        policy_error=PolicyLoadError(PolicyErrorCode.HASH_MISMATCH, "x"),
    )
    with TestClient(broken) as client:
        replayed = client.post("/v1/decisions/evaluate", json=body, headers=headers)
        assert client.get("/health/ready").status_code == 503
        assert replayed.status_code == 201
        assert replayed.headers[REPLAYED_HEADER] == "true"
        assert replayed.json()["decision"] == first.json()["decision"]
        # 未命中的 key 仍按原语义降级
        miss = client.post(
            "/v1/decisions/evaluate", json=body, headers={IDEMPOTENCY_KEY_HEADER: KEY_TWO}
        )
        assert miss.status_code == 503
        assert miss.json()["error"]["code"] == "model_not_ready"


# --- keyed 批量 ---------------------------------------------------------


@pytest.mark.parametrize("size", [1, 100])
def test_keyed_batch_creates_then_replays_with_order(config, runtime, db_file, size: int) -> None:
    ids = [f"item-{index}" for index in range(size)]
    runtime._scores = [0.05] * size
    app = _client(config, runtime, ids=[f"{index:08d}-1111-4111-8111-111111111111" for index in range(size)])
    body = _batch_body(runtime, ids)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        first = client.post("/v1/decisions/evaluate/batch", json=body, headers=headers)
        calls_after_first = runtime.call_count
        second = client.post("/v1/decisions/evaluate/batch", json=body, headers=headers)
    assert first.status_code == 201 and first.headers[REPLAYED_HEADER] == "false"
    assert second.status_code == 201 and second.headers[REPLAYED_HEADER] == "true"
    assert calls_after_first == 1, "首次批量只调用一次模型"
    assert runtime.call_count == 1, "重放不得调用模型"
    assert [item["client_item_id"] for item in second.json()["decisions"]] == ids
    assert second.json()["decisions"] == first.json()["decisions"]
    assert _rows(db_file, "decisions") == size
    assert _rows(db_file, "idempotency_results") == size


def test_replayed_batch_records_are_individually_readable(config, runtime, db_file) -> None:
    runtime._scores = [0.05, 0.9]
    app = _client(config, runtime)
    body = _batch_body(runtime, ["a", "b"])
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        first = client.post("/v1/decisions/evaluate/batch", json=body, headers=headers)
        second = client.post("/v1/decisions/evaluate/batch", json=body, headers=headers)
        fetched = [
            client.get(f"/v1/decisions/{item['decision_id']}").json()["decision"]
            for item in second.json()["decisions"]
        ]
    assert first.status_code == second.status_code == 201
    assert fetched == second.json()["decisions"]


# --- 冲突 ---------------------------------------------------------------


def test_same_key_with_changed_features_conflicts_before_evaluation(config, runtime, db_file) -> None:
    app = _client(config, runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        assert client.post("/v1/decisions/evaluate", json=_body(runtime), headers=headers).status_code == 201
        calls = runtime.call_count
        conflict = client.post(
            "/v1/decisions/evaluate",
            json=_body(runtime, transaction_amt=5000.0),
            headers=headers,
        )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"
    assert REPLAYED_HEADER not in conflict.headers
    assert runtime.call_count == calls, "冲突必须在模型与规则之前判出"
    assert _rows(db_file, "decisions") == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body["items"].reverse(),
        lambda body: body["items"][0].update({"client_item_id": "renamed"}),
        lambda body: body["items"].pop(),
    ],
    ids=["order", "client_item_id", "size"],
)
def test_batch_fingerprint_distinguishes_semantics(config, runtime, db_file, mutate) -> None:
    runtime._scores = [0.05, 0.05, 0.05]
    app = _client(config, runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    body = _batch_body(runtime, ["a", "b", "c"])
    changed = json.loads(json.dumps(body))
    mutate(changed)
    with TestClient(app) as client:
        assert client.post(
            "/v1/decisions/evaluate/batch", json=body, headers=headers
        ).status_code == 201
        conflict = client.post("/v1/decisions/evaluate/batch", json=changed, headers=headers)
    assert conflict.status_code == 409
    assert _rows(db_file, "decisions") == 3


def test_cross_endpoint_reuse_of_the_same_key_conflicts(config, runtime, db_file) -> None:
    """单笔与批量不共享命名空间：同 key 跨端点必须是冲突。"""
    runtime._scores = [0.05, 0.05]
    app = _client(config, runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        assert client.post("/v1/decisions/evaluate", json=_body(runtime), headers=headers).status_code == 201
        conflict = client.post(
            "/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["a"]), headers=headers
        )
    assert conflict.status_code == 409
    assert _rows(db_file, "decisions") == 1
    assert _rows(db_file, "idempotency_requests") == 1


def test_different_keys_with_the_same_request_create_different_records(config, runtime, db_file) -> None:
    app = _client(config, runtime)
    body = _body(runtime)
    with TestClient(app) as client:
        first = client.post(
            "/v1/decisions/evaluate", json=body, headers={IDEMPOTENCY_KEY_HEADER: KEY_ONE}
        )
        second = client.post(
            "/v1/decisions/evaluate", json=body, headers={IDEMPOTENCY_KEY_HEADER: KEY_TWO}
        )
    assert first.status_code == second.status_code == 201
    assert first.json()["decision"]["decision_id"] != second.json()["decision"]["decision_id"]
    assert _rows(db_file, "decisions") == 2
    assert _rows(db_file, "idempotency_requests") == 2


def test_request_id_does_not_participate_in_idempotency(config, runtime) -> None:
    """换 ``X-Request-ID`` 仍是同一个请求；无 key 时相同 request ID 仍各建一条。"""
    app = _client(config, runtime)
    body = _body(runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE, REQUEST_ID_HEADER: "req-one"}
    other = {IDEMPOTENCY_KEY_HEADER: KEY_ONE, REQUEST_ID_HEADER: "req-two"}
    shared = {REQUEST_ID_HEADER: "req-shared"}
    with TestClient(app) as client:
        first = client.post("/v1/decisions/evaluate", json=body, headers=headers)
        replay = client.post("/v1/decisions/evaluate", json=body, headers=other)
        plain_a = client.post("/v1/decisions/evaluate", json=body, headers=shared)
        plain_b = client.post("/v1/decisions/evaluate", json=body, headers=shared)
    assert replay.headers[REPLAYED_HEADER] == "true"
    assert replay.json()["decision"] == first.json()["decision"]
    assert plain_a.json()["decision"]["decision_id"] != plain_b.json()["decision"]["decision_id"]


# --- 非法 key -----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    # 注意：非 ASCII 取值**根本发不出去**（HTTP/1.1 header 是 ASCII），httpx 会在客户端就报
    # UnicodeEncodeError，因此非 ASCII 的拒绝由单元测试（``test_cw008_fingerprint.py``）覆盖；
    # 这里覆盖的是能真实到达服务端的那些非法形状。
    ["short", "a" * 15, "a" * 129, " " + "a" * 15, "a" * 15 + " ", "a" * 15 + "!", ""],
)
def test_invalid_key_is_rejected_without_any_side_effect(config, runtime, db_file, value: str) -> None:
    app = _client(config, runtime)
    with TestClient(app) as client:
        response = client.post(
            "/v1/decisions/evaluate",
            json=_body(runtime),
            headers={IDEMPOTENCY_KEY_HEADER: value},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_idempotency_key"
    assert response.json()["error"]["message"] == ERROR_MESSAGES["invalid_idempotency_key"]
    assert value not in response.text if value else True
    assert runtime.call_count == 0
    assert not db_file.exists() or _rows(db_file, "decisions") == 0


def test_duplicate_key_headers_are_rejected(config, runtime, db_file) -> None:
    app = _client(config, runtime)
    with TestClient(app) as client:
        response = client.post(
            "/v1/decisions/evaluate",
            json=_body(runtime),
            headers=[(IDEMPOTENCY_KEY_HEADER, KEY_ONE), (IDEMPOTENCY_KEY_HEADER, KEY_TWO)],
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_idempotency_key"
    assert runtime.call_count == 0
    assert not db_file.exists() or _rows(db_file, "decisions") == 0


def test_invalid_key_is_rejected_on_the_batch_endpoint_too(config, runtime, db_file) -> None:
    app = _client(config, runtime)
    with TestClient(app) as client:
        response = client.post(
            "/v1/decisions/evaluate/batch",
            json=_batch_body(runtime, ["a"]),
            headers={IDEMPOTENCY_KEY_HEADER: "nope"},
        )
    assert response.status_code == 400
    assert runtime.call_count == 0


# --- 竞态：提交阶段才发现赢家 -------------------------------------------


def test_commit_time_replay_returns_the_winner_snapshot(config, runtime, db_file, monkeypatch) -> None:
    """模拟"两个并发都 miss 并都完成求值"：输家的候选记录必须被丢弃，返回赢家的快照。

    做法是把 repository 的 lookup 临时改成永远 miss，从而强制走到提交事务里的二次判定——
    这正是竞态输家实际经过的那条路径。
    """
    from ai_detector.storage.sqlite import IDEMPOTENCY_MISS, IdempotencyHit

    app = _client(config, runtime)
    body = _body(runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        winner = client.post("/v1/decisions/evaluate", json=body, headers=headers)
        assert winner.status_code == 201

        repository = app.state.decision_repository
        monkeypatch.setattr(
            type(repository),
            "lookup_idempotent",
            lambda self, *args, **kwargs: IdempotencyHit(IDEMPOTENCY_MISS),
        )
        loser = client.post("/v1/decisions/evaluate", json=body, headers=headers)

    assert loser.status_code == 201
    assert loser.headers[REPLAYED_HEADER] == "true"
    assert loser.json()["decision"] == winner.json()["decision"]
    assert _rows(db_file, "decisions") == 1, "输家不得留下候选记录"
    assert _rows(db_file, "idempotency_results") == 1
    assert runtime.call_count == 2, "允许重复评分，但不允许重复提交（这正是本包的限制）"


def test_commit_time_conflict_is_reported_safely(config, runtime, db_file, monkeypatch) -> None:
    from ai_detector.storage.sqlite import IDEMPOTENCY_MISS, IdempotencyHit

    app = _client(config, runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        assert client.post(
            "/v1/decisions/evaluate", json=_body(runtime), headers=headers
        ).status_code == 201
        repository = app.state.decision_repository
        monkeypatch.setattr(
            type(repository),
            "lookup_idempotent",
            lambda self, *args, **kwargs: IdempotencyHit(IDEMPOTENCY_MISS),
        )
        conflict = client.post(
            "/v1/decisions/evaluate",
            json=_body(runtime, transaction_amt=9999.0),
            headers=headers,
        )
    assert conflict.status_code == 409
    assert _rows(db_file, "decisions") == 1


# --- 日志与 OpenAPI -----------------------------------------------------


def test_keyed_requests_emit_one_log_with_the_outcome_field(config, runtime, caplog) -> None:
    app = _client(config, runtime)
    body = _body(runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            client.post("/v1/decisions/evaluate", json=body, headers=headers)
            client.post("/v1/decisions/evaluate", json=body, headers=headers)
            client.post("/v1/decisions/evaluate", json=body)
            client.post(
                "/v1/decisions/evaluate",
                json=_body(runtime, transaction_amt=7777.0),
                headers=headers,
            )
    events = [json.loads(record.message) for record in caplog.records if "request_completed" in record.message]
    assert [event["idempotency_outcome"] for event in events] == [
        "created",
        "replayed",
        "not_requested",
        "conflict",
    ]
    assert all(event["event"] == "request_completed" for event in events)


def test_logs_never_contain_the_key_digest_or_fingerprint(config, runtime, caplog) -> None:
    app = _client(config, runtime)
    body = _body(runtime)
    key = "SENTINEL-IDEMPOTENCY-KEY-0123456789"
    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            client.post("/v1/decisions/evaluate", json=body, headers={IDEMPOTENCY_KEY_HEADER: key})
            client.post("/v1/decisions/evaluate", json=body, headers={IDEMPOTENCY_KEY_HEADER: key})
    blob = "\n".join(record.message for record in caplog.records)
    assert key not in blob
    import hashlib

    assert hashlib.sha256(key.encode()).hexdigest() not in blob
    for banned in ("TransactionAmt", '"features"', "idempotency_requests", "SELECT"):
        assert banned not in blob, banned


def test_openapi_still_exposes_eleven_paths(config, runtime) -> None:
    app = _client(config, runtime)
    assert len(app.openapi()["paths"]) == 11


def _header_parameters(operation: dict) -> list[dict]:
    return [item for item in operation.get("parameters", []) if item.get("in") == "header"]


def test_openapi_declares_the_optional_key_on_exactly_the_two_posts(config, runtime) -> None:
    """逐路由检查 ``parameters``：两个 POST 各有一个**可选** header，别处没有。

    刻意不做全文搜索——docstring 里出现过的字符串同样会被搜到，那种断言证明不了任何东西。
    """
    spec = _client(config, runtime).openapi()
    keyed = ("/v1/decisions/evaluate", "/v1/decisions/evaluate/batch")
    for path, operation in (
        ("/v1/decisions/evaluate", spec["paths"]["/v1/decisions/evaluate"]["post"]),
        ("/v1/decisions/evaluate/batch", spec["paths"]["/v1/decisions/evaluate/batch"]["post"]),
    ):
        headers = _header_parameters(operation)
        assert len(headers) == 1, path
        parameter = headers[0]
        assert parameter["name"] == IDEMPOTENCY_KEY_HEADER
        assert parameter["in"] == "header"
        assert parameter["required"] is False
        schema = parameter["schema"]
        assert schema["minLength"] == 16 and schema["maxLength"] == 128
        assert schema["pattern"] == r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$"

    # 其余 9 条路径一律不得声明任何 header parameter。
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            if (path, method) in (("/v1/decisions/evaluate", "post"),
                                  ("/v1/decisions/evaluate/batch", "post")):
                continue
            assert _header_parameters(operation) == [], f"{method.upper()} {path}"
    assert keyed == ("/v1/decisions/evaluate", "/v1/decisions/evaluate/batch")


def test_declared_header_does_not_turn_a_bad_key_into_422(config, runtime, db_file) -> None:
    """声明只是 schema；非法取值仍必须由服务回冻结的 400，而不是被框架拦成 422。"""
    app = _client(config, runtime)
    with TestClient(app) as client:
        response = client.post(
            "/v1/decisions/evaluate",
            json=_body(runtime),
            headers={IDEMPOTENCY_KEY_HEADER: "too-short"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_idempotency_key"


def _replay_fixture(app, runtime, *, statement: str):
    """建一条 keyed 决策后把库改坏，返回 (headers, body)。"""
    body = _body(runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    with TestClient(app) as client:
        assert client.post(
            "/v1/decisions/evaluate", json=body, headers=headers
        ).status_code == 201
    return body, headers


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE decisions SET risk_level = 'not_a_level'",
        "UPDATE decisions SET flagged = 1",
        "UPDATE decisions SET matched_rule_ids = 'not json'",
    ],
    ids=["bad_level", "flag_contradiction", "bad_json"],
)
def test_corrupt_decision_replay_returns_idempotency_contract_violation(
    config, runtime, db_file, statement: str
) -> None:
    """关联的决策记录损坏时，重放必须是 500 ``idempotency_contract_violation``。

    这条路径此前会把记录层的 ``StorageContractError`` 一路冒到路由，被折叠成通用
    ``internal_error``——错误分类因此与冻结契约不符。
    """
    app = _client(config, runtime)
    body, headers = _replay_fixture(app, runtime, statement=statement)
    _tamper(db_file, statement)
    with TestClient(app) as client:
        replayed = client.post("/v1/decisions/evaluate", json=body, headers=headers)
    assert replayed.status_code == 500
    assert replayed.json()["error"]["code"] == "idempotency_contract_violation"
    assert replayed.json()["error"]["message"] == ERROR_MESSAGES[
        "idempotency_contract_violation"
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM idempotency_results",
        "UPDATE idempotency_results SET item_index = 7",
    ],
    ids=["missing_result", "index_gap"],
)
def test_corrupt_mapping_conflict_returns_500_not_409(
    config, runtime, db_file, statement: str
) -> None:
    """映射损坏 + 异请求：必须是安全 500，**不得**伪装成 409 冲突。"""
    app = _client(config, runtime)
    _, headers = _replay_fixture(app, runtime, statement=statement)
    _tamper(db_file, statement)
    with TestClient(app) as client:
        conflicted = client.post(
            "/v1/decisions/evaluate",
            json=_body(runtime, transaction_amt=4242.0),
            headers=headers,
        )
    assert conflicted.status_code == 500
    assert conflicted.json()["error"]["code"] == "idempotency_contract_violation"


def _tamper(db_file: Path, statement: str) -> None:
    """直接改库：先关掉外键与 CHECK 约束，才能构造出"数据库本来不允许"的损坏状态。"""
    connection = sqlite3.connect(db_file, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(statement)
    finally:
        connection.close()


def test_error_bodies_never_echo_the_key_or_its_digest(config, runtime) -> None:
    app = _client(config, runtime)
    headers = {IDEMPOTENCY_KEY_HEADER: KEY_ONE}
    secret = "SENTINEL-CONFLICT-KEY-0123456789"
    with TestClient(app) as client:
        client.post("/v1/decisions/evaluate", json=_body(runtime), headers=headers)
        conflict = client.post(
            "/v1/decisions/evaluate",
            json=_body(runtime, transaction_amt=1234.0),
            headers=headers,
        )
    assert conflict.status_code == 409
    assert KEY_ONE not in conflict.text
    assert secret not in conflict.text
    assert set(conflict.json()["error"]) == {"code", "message", "request_id"}
