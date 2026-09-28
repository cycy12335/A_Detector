"""CW-007 服务集成测试：三个 `/v1/decisions/*` 接口、原子事务、故障隔离与旧接口回归。

刻意**不依赖正式 bundle**：模型与政策分别用 fake runtime / 真实 policy engine（必要时用
fake engine）注入，只有存储是**真实 SQLite**（写入按 ID 读回、重启持久、事务回滚本来就必须
在真库上验证）。需要精确控制存储故障时才注入 fake repository。

被反复验证的不变量：

- 单笔/批量只在 **commit 成功后**才返回；批量一个事务，全成全败；
- GET 只读存储快照：**零**模型调用、**零**规则求值，model/policy 故障时仍可读；
- storage 故障只让新接口与 readiness 降级，旧八接口按原能力工作，且**不**回退到内存或临时库；
- 每请求恰好一条 completion log，旧八接口日志字段集合不变。
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
from ai_detector.storage.config import RECORD_SCHEMA_VERSION, StorageErrorCode, StorageLoadError  # noqa: E402
from ai_detector.storage.records import StoredDecision  # noqa: E402
from tests.cw005_fixtures import FakeRuntime, RecordingLoader  # noqa: E402
from tests.cw006_fixtures import FakePolicyEngine, RecordingPolicyLoader, risk_features  # noqa: E402
from tests.cw007_fixtures import (  # noqa: E402
    DECISION_ID_A,
    DECISION_ID_D,
    DECISION_ID_B,
    DECISION_ID_C,
    FakeDecisionRepository,
    FixedClock,
    RecordingStorageLoader,
    SequentialIdFactory,
    evaluation_result,
    remove_database,
    sqlite_connection,
    staged_db_path,
)

FEATURES = ["TransactionAmt", "C13", "feat_c"]
SENTINEL = "SENTINEL_CW007_LEAK_4b2e"
REQUEST_ID_HEADER = "X-Request-ID"

OLD_ENDPOINTS = (
    "/health/live",
    "/health/ready",
    "/v1/model",
    "/v1/score",
    "/v1/score/batch",
    "/v1/policy",
    "/v1/risk/evaluate",
    "/v1/risk/evaluate/batch",
)


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(features=FEATURES, scores=[0.05, 0.9, 0.11])


def _make_config(tmp_path: Path, **kwargs):
    from tests.cw005_fixtures import make_config
    from tests.cw007_fixtures import storage_section, staged_db_relative

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
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():  # pragma: no cover - 防御性
            sidecar.unlink()


def _client(config, runtime, *, ids=(DECISION_ID_A, DECISION_ID_B, DECISION_ID_C), storage_loader=None):
    """构造 app；默认用真实 storage loader（真实 SQLite）。"""
    kwargs = {}
    if storage_loader is not None:
        kwargs["storage_loader"] = storage_loader
    app = create_app(
        config,
        runtime_loader=RecordingLoader(runtime),
        decision_id_factory=SequentialIdFactory(ids),
        utc_clock=FixedClock(["2026-09-25T04:05:06Z"]),
        **kwargs,
    )
    return TestClient(app)


def _decision_body(runtime, **kwargs):
    return {"features": risk_features(runtime, **kwargs)}


# --- 工厂语义与生命周期 -------------------------------------------------


def test_create_app_performs_no_storage_io(config, tmp_path) -> None:
    """``create_app()`` 只记录配置，不建目录、不建库、不读库。"""
    db = staged_db_path(tmp_path)
    app = create_app(config, runtime_loader=RecordingLoader(FakeRuntime(features=FEATURES)))
    assert not db.exists()
    assert getattr(app.state, "decision_repository", None) is None


def test_storage_is_initialized_once_per_app_lifecycle(config, runtime) -> None:
    loaded: list[int] = []

    from ai_detector.service import runtime as runtime_module

    real_loader = runtime_module.load_storage_repository

    def _counting(config_):  # noqa: ANN001, ANN202
        loaded.append(1)
        return real_loader(config_)

    client = _client(config, runtime, storage_loader=_counting)
    with client:
        for _ in range(3):
            client.post("/v1/decisions/evaluate", json=_decision_body(runtime))
            client.get(f"/v1/decisions/{DECISION_ID_A}")
    assert len(loaded) == 1, "storage 只能初始化一次"


def test_requests_do_not_recreate_the_schema(config, runtime, tmp_path, monkeypatch) -> None:
    """请求期间不得再执行 schema 初始化。"""
    import ai_detector.storage.sqlite as sqlite_module

    calls: list[int] = []
    real = sqlite_module.SqliteDecisionRepository.initialize

    def _counting(self):  # noqa: ANN001, ANN202
        calls.append(1)
        return real(self)

    monkeypatch.setattr(sqlite_module.SqliteDecisionRepository, "initialize", _counting)
    client = _client(config, runtime)
    with client:
        for _ in range(3):
            client.post("/v1/decisions/evaluate", json=_decision_body(runtime))
    assert len(calls) == 1


# --- 单笔持久化 ---------------------------------------------------------


def test_single_decision_returns_201_and_persists(config, runtime, db_file) -> None:
    """单笔：``TransactionAmt`` 高于冻结阈值 -> 命中一条规则 -> 1 点 -> elevated_signal。"""
    client = _client(config, runtime)
    with client:
        response = client.post(
            "/v1/decisions/evaluate",
            json=_decision_body(runtime, transaction_amt=2000.0),
        )
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"request_id", "decision"}
    decision = body["decision"]
    assert decision["decision_id"] == DECISION_ID_A
    assert decision["created_at"] == "2026-09-25T04:05:06Z"
    assert decision["record_schema_version"] == RECORD_SCHEMA_VERSION
    assert decision["client_item_id"] is None
    assert decision["evaluation_request_id"] == body["request_id"]
    assert decision["score"] == pytest.approx(0.05)
    assert decision["flagged"] is False
    assert decision["rule_score"] == pytest.approx(0.5)
    assert decision["matched_rule_ids"] == ["transaction_amt_train_q99_tail"]
    assert decision["signal_points"] == 1
    assert decision["risk_level"] == "elevated_signal"
    assert decision["score_semantics"] == "uncalibrated_model_score"
    assert decision["threshold_policy"] == "validation_f1"
    assert decision["policy_semantics"] == "research_signal_band"
    assert "features" not in json.dumps(decision)

    # 数据库里确实只有一条，且 GET 读回逐字段一致
    with sqlite_connection(db_file) as conn:
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    with client:
        fetched = client.get(f"/v1/decisions/{DECISION_ID_A}")
    assert fetched.status_code == 200
    assert fetched.json()["decision"] == decision


def test_single_decision_response_matches_the_stored_row(config, runtime, db_file) -> None:
    client = _client(config, runtime)
    with client:
        response = client.post(
            "/v1/decisions/evaluate",
            json=_decision_body(runtime, transaction_amt=2000.0, c13=900.0),
        )
    decision = response.json()["decision"]
    with sqlite_connection(db_file) as conn:
        row = conn.execute("SELECT * FROM decisions").fetchone()
    stored = StoredDecision.from_row(dict(row))
    assert stored.to_public_dict() == decision


def test_single_decision_does_not_write_the_old_score_endpoints(config, runtime, db_file) -> None:
    """旧接口保持无持久化副作用：只有 `/v1/decisions/*` POST 写库。"""
    client = _client(config, runtime)
    with client:
        assert client.post("/v1/score", json=_decision_body(runtime)).status_code == 200
        assert client.post("/v1/risk/evaluate", json=_decision_body(runtime)).status_code == 200
        assert client.get("/v1/model").status_code == 200
    assert not db_file.exists() or _row_count(db_file) == 0


def _row_count(path: Path) -> int:
    with sqlite_connection(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])


def test_single_decision_rejects_a_bad_feature_contract(config, runtime, db_file) -> None:
    client = _client(config, runtime)
    payload = _decision_body(runtime)
    payload["features"].pop("TransactionAmt")
    with client:
        response = client.post("/v1/decisions/evaluate", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "feature_contract_violation"
    assert not db_file.exists() or _row_count(db_file) == 0


def test_indeterminate_decision_is_persisted(config, runtime) -> None:
    """``indeterminate`` 是合法结果，同样要落库并原样读回。"""
    client = _client(config, runtime)
    with client:
        response = client.post(
            "/v1/decisions/evaluate",
            json=_decision_body(runtime, transaction_amt=None, c13=900.0),
        )
        fetched = client.get(f"/v1/decisions/{DECISION_ID_A}")
    decision = response.json()["decision"]
    assert response.status_code == 201
    assert decision["risk_level"] == "indeterminate"
    assert decision["signal_points"] is None
    assert decision["unevaluable_rule_ids"] == ["transaction_amt_train_q99_tail"]
    assert fetched.json()["decision"] == decision


# --- flagged 是模型阈值结果：与规则是否可评估无关 -----------------------


def _indeterminate_body(runtime) -> dict:
    """令 ``transaction_amt`` 规则不可评估（``TransactionAmt=null``），其余列合法。"""
    return _decision_body(runtime, transaction_amt=None, c13=900.0)


@pytest.mark.parametrize(
    ("score", "expected_flag"),
    [(0.9, True), (0.05, False)],
    ids=["above_threshold", "below_threshold"],
)
def test_indeterminate_decision_keeps_the_model_flag(
    tmp_path, score: float, expected_flag: bool
) -> None:
    """规则不可评估（``signal_points=null``）时，``flagged`` 仍必须是 ``score >= threshold``。

    这是"从可选点数反推 flagged"会出错的分支：点数缺失时旧实现无条件返回 ``False``，
    于是 ``score=0.9 >= 0.11`` 的记录被写成 ``flagged=false``。旧 ``/v1/risk/evaluate``、
    新 POST 响应、数据库行与 GET 读回必须四者一致。
    """
    runtime = FakeRuntime(features=FEATURES, scores=[score])
    config = _make_config(tmp_path)
    db = staged_db_path(tmp_path)
    client = _client(config, runtime)
    try:
        with client:
            legacy = client.post("/v1/risk/evaluate", json=_indeterminate_body(runtime))
            response = client.post("/v1/decisions/evaluate", json=_indeterminate_body(runtime))
            fetched = client.get(f"/v1/decisions/{DECISION_ID_A}")
        decision = response.json()["decision"]

        # 旧接口是这条不变量的"真值来源"：flagged 由 score >= threshold 决定。
        assert legacy.status_code == 200
        assert legacy.json()["flagged"] is expected_flag
        assert legacy.json()["signal_points"] is None
        assert legacy.json()["risk_level"] == "indeterminate"

        assert response.status_code == 201
        assert decision["flagged"] is expected_flag
        assert decision["score"] == pytest.approx(score)
        assert decision["threshold"] == pytest.approx(0.11)
        assert decision["signal_points"] is None
        assert decision["risk_level"] == "indeterminate"
        assert decision["rule_evaluation_complete"] is False
        assert decision["unevaluable_rule_ids"] == ["transaction_amt_train_q99_tail"]

        with sqlite_connection(db) as conn:
            row = conn.execute(
                "SELECT * FROM decisions WHERE decision_id = ?", (DECISION_ID_A,)
            ).fetchone()
        assert row["flagged"] == (1 if expected_flag else 0), "数据库行的 flagged 必须一致"
        assert StoredDecision.from_row(dict(row)).to_public_dict() == decision

        assert fetched.status_code == 200
        assert fetched.json()["decision"] == decision
    finally:
        remove_database(db)


def test_batch_indeterminate_decisions_keep_each_model_flag(tmp_path) -> None:
    """批量里每一笔各自保留自己的模型标记，不因为规则不可评估而被抹平。"""
    runtime = FakeRuntime(features=FEATURES, scores=[0.9, 0.05])
    config = _make_config(tmp_path)
    db = staged_db_path(tmp_path)
    client = _client(config, runtime, ids=(DECISION_ID_A, DECISION_ID_B))
    try:
        with client:
            response = client.post(
                "/v1/decisions/evaluate/batch",
                json={
                    "items": [
                        {"client_item_id": "a", "features": risk_features(runtime, transaction_amt=None)},
                        {"client_item_id": "b", "features": risk_features(runtime, transaction_amt=None)},
                    ]
                },
            )
            readback = [
                client.get(f"/v1/decisions/{decision_id}").json()["decision"]
                for decision_id in (DECISION_ID_A, DECISION_ID_B)
            ]
        assert response.status_code == 201
        decisions = response.json()["decisions"]
        assert [item["flagged"] for item in decisions] == [True, False]
        assert all(item["signal_points"] is None for item in decisions)
        assert all(item["risk_level"] == "indeterminate" for item in decisions)
        assert readback == decisions
    finally:
        remove_database(db)


@pytest.mark.parametrize("column", ["flagged", "score", "threshold"])
def test_tampered_flag_cross_field_contradiction_returns_500(
    config, runtime, db_file, column: str
) -> None:
    """直接改库制造 ``flagged`` 与 score/threshold 矛盾后，GET 必须安全失败而不是返回错快照。

    改的方向由**行内现值**决定：把 ``flagged`` 取反，或把 ``score``/``threshold`` 挪到阈值
    的另一侧，因此无论原记录是标记还是未标记，改完一定自相矛盾。
    """
    client = _client(config, runtime)
    with client:
        response = client.post(
            "/v1/decisions/evaluate",
            json=_decision_body(runtime, transaction_amt=2000.0),
        )
        assert response.status_code == 201
        with sqlite_connection(db_file) as conn:
            row = conn.execute(
                "SELECT flagged, score, threshold FROM decisions WHERE decision_id = ?",
                (DECISION_ID_A,),
            ).fetchone()
            if column == "flagged":
                value: object = 0 if row["flagged"] else 1
            elif column == "score":
                value = row["threshold"] - 0.05 if row["flagged"] else row["threshold"] + 0.05
            else:
                value = row["score"] + 0.05 if row["flagged"] else row["score"] - 0.05
            conn.execute(
                f"UPDATE decisions SET {column} = ? WHERE decision_id = ?",
                (value, DECISION_ID_A),
            )
        fetched = client.get(f"/v1/decisions/{DECISION_ID_A}")
    assert fetched.status_code == 500
    body = fetched.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "storage_contract_violation"
    # 统一 envelope：只有稳定码与**固定文案**，因此结构上不可能回显被篡改的取值或行内容。
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["message"] == ERROR_MESSAGES["storage_contract_violation"]


# --- 批量持久化 ---------------------------------------------------------


def _batch_body(runtime, ids: list[str]) -> dict:
    return {
        "items": [
            {"client_item_id": item_id, "features": risk_features(runtime)}
            for item_id in ids
        ]
    }


def test_batch_decision_returns_201_in_request_order(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.9, 0.05])
    client = _client(config, runtime, ids=(DECISION_ID_C, DECISION_ID_A, DECISION_ID_B))
    with client:
        response = client.post(
            "/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["c", "a"])
        )
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"request_id", "decisions"}
    decisions = body["decisions"]
    assert [item["client_item_id"] for item in decisions] == ["c", "a"]
    assert [item["decision_id"] for item in decisions] == [DECISION_ID_C, DECISION_ID_A]
    assert all(item["evaluation_request_id"] == body["request_id"] for item in decisions)
    assert [item["flagged"] for item in decisions] == [True, False]
    assert runtime.call_count == 1, "整批只调用一次模型"


def test_batch_decision_all_records_are_readable_afterwards(config, runtime) -> None:
    client = _client(config, runtime, ids=(DECISION_ID_A, DECISION_ID_B))
    with client:
        client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["a", "b"]))
        first = client.get(f"/v1/decisions/{DECISION_ID_A}")
        second = client.get(f"/v1/decisions/{DECISION_ID_B}")
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["decision"]["client_item_id"] == "a"
    assert second.json()["decision"]["client_item_id"] == "b"


def test_batch_decision_accepts_the_full_hundred(config) -> None:
    """100 笔（上限）必须一次评分、一次事务、全部落库。"""
    runtime = FakeRuntime(features=FEATURES, scores=[0.05])
    client = _client(config, runtime, ids=(DECISION_ID_A,))
    spec = [f"i{index}" for index in range(100)]
    with client:
        response = client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime, spec))
    assert response.status_code == 201
    decisions = response.json()["decisions"]
    assert len(decisions) == 100
    assert [item["client_item_id"] for item in decisions] == spec
    assert len({item["decision_id"] for item in decisions}) == 100
    assert runtime.call_count == 1


def test_single_decision_accepts_one_item_batch_boundary(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.05])
    client = _client(config, runtime, ids=(DECISION_ID_A,))
    with client:
        response = client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["only"]))
    assert response.status_code == 201
    assert len(response.json()["decisions"]) == 1


def test_batch_decision_101_items_is_rejected(config, runtime, db_file) -> None:
    client = _client(config, runtime, ids=tuple(f"id{i}" for i in range(101)))
    with client:
        response = client.post(
            "/v1/decisions/evaluate/batch",
            json=_batch_body(runtime, [f"i{index}" for index in range(101)]),
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "batch_contract_violation"
    assert not db_file.exists() or _row_count(db_file) == 0


def test_batch_decision_contract_error_writes_nothing(config, runtime, db_file) -> None:
    client = _client(config, runtime, ids=(DECISION_ID_A, DECISION_ID_B))
    payload = _batch_body(runtime, ["a", "b"])
    payload["items"][1]["features"] = {"TransactionAmt": 1.0}
    with client:
        response = client.post("/v1/decisions/evaluate/batch", json=payload)
    assert response.status_code == 422
    assert "decisions" not in response.json()
    assert not db_file.exists() or _row_count(db_file) == 0


def test_batch_decision_rolls_back_when_the_insert_fails(config, runtime, db_file) -> None:
    """插入失败时全批回滚，且**不返回**未持久化结果。"""
    repo = FakeDecisionRepository(
        insert_error=StorageLoadError(StorageErrorCode.UNAVAILABLE, "写入失败")
    )
    client = _client(config, runtime, storage_loader=RecordingStorageLoader(repo))
    with client:
        response = client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["a"]))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "storage_unavailable"
    assert "decisions" not in response.json()
    assert repo.insert_count == 1


def test_batch_decision_duplicate_generated_ids_roll_back(config, runtime, db_file) -> None:
    """ID 工厂若产生重复 ID，主键冲突必须让整批回滚（不留部分记录）。"""
    client = _client(config, runtime, ids=(DECISION_ID_A, DECISION_ID_A))
    with client:
        response = client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["a", "b"]))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "storage_unavailable"
    assert _row_count(db_file) == 0


# --- GET 查询 -----------------------------------------------------------


def test_get_does_not_touch_model_or_rules(config, runtime) -> None:
    """GET 只读存储：模型与政策都坏掉时，已有记录仍可读。"""
    seeded = _client(config, runtime)
    with seeded:
        seeded.post(
            "/v1/decisions/evaluate",
            json=_decision_body(runtime, transaction_amt=2000.0, c13=900.0),
        )

    # 加载器在 lifespan 里当然会被调用一次（那是"启动加载"，不是"请求期调用"）；
    # 这里要断言的是**评分调用**与**规则求值**都为零。
    loaded_runtime = FakeRuntime(features=FEATURES)
    engine = FakePolicyEngine(results=[evaluation_result(matched=2)])
    app = create_app(
        config,
        runtime_loader=RecordingLoader(loaded_runtime),
        policy_loader=RecordingPolicyLoader(engine),
    )
    engine.evaluate_calls.clear()
    with TestClient(app) as client:
        client.get("/health/ready")
        response = client.get(f"/v1/decisions/{DECISION_ID_A}")
    assert response.status_code == 200
    assert loaded_runtime.call_count == 0, "GET 不得调用模型评分"
    assert engine.call_count == 0, "GET 不得运行规则"


def test_get_works_when_model_and_policy_are_both_broken(config, runtime) -> None:
    """模型与政策**双双**故障时，GET 仍按存储快照返回 200。"""
    seeded = _client(config, runtime)
    with seeded:
        seeded.post("/v1/decisions/evaluate", json=_decision_body(runtime))

    app = create_app(
        config,
        runtime_loader=RecordingLoader(error=BundleLoadError(BundleErrorCode.MISSING, "缺失")),
        policy_loader=RecordingPolicyLoader(
            error=PolicyLoadError(PolicyErrorCode.MISSING, "缺失")
        ),
    )
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        response = client.get(f"/v1/decisions/{DECISION_ID_A}")
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "model_not_ready"
    assert response.status_code == 200
    assert response.json()["decision"]["decision_id"] == DECISION_ID_A


def test_decisions_are_readable_after_the_app_is_rebuilt(config, runtime) -> None:
    """跨 app 生命周期持久：重建 app 后仍能按 ID 读回。"""
    first = _client(config, runtime)
    with first:
        created = first.post("/v1/decisions/evaluate", json=_decision_body(runtime)).json()

    second = _client(config, runtime, ids=(DECISION_ID_B,))
    with second:
        response = second.get(f"/v1/decisions/{created['decision']['decision_id']}")
    assert response.status_code == 200
    assert response.json()["decision"] == created["decision"]


def test_get_unknown_id_returns_404(config, runtime) -> None:
    client = _client(config, runtime)
    with client:
        response = client.get(f"/v1/decisions/{DECISION_ID_C}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "decision_not_found"
    assert response.json()["error"]["message"] == ERROR_MESSAGES["decision_not_found"]


@pytest.mark.parametrize(
    "hostile",
    [
        "not-a-uuid",
        "' OR 1=1 --",
        "'; DROP TABLE decisions; --",
        "11111111-1111-4111-8111-111111111111' --",
        "a" * 300,
        "00000000-0000-0000-0000-000000000000",
        "AAAAAAAA-1111-4111-8111-111111111111",
        "11111111-1111-3111-8111-111111111111",
    ],
)
def test_get_hostile_or_invalid_id_returns_404_without_leaking(
    config, runtime, db_file, hostile: str
) -> None:
    client = _client(config, runtime)
    with client:
        client.post("/v1/decisions/evaluate", json=_decision_body(runtime))
        response = client.get(f"/v1/decisions/{hostile}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "decision_not_found"
    blob = json.dumps(response.json(), ensure_ascii=False)
    assert hostile not in blob, "不得回显调用方传入的 path 参数"
    assert "DROP TABLE" not in blob
    # 表仍然存在且记录完好
    assert _row_count(db_file) == 1


def test_get_rejects_non_uuid_shaped_ids_before_hitting_sqlite(config, runtime) -> None:
    """形状非法的 ID 在触达 SQLite 之前就被拒（统一 404）。"""
    repo = FakeDecisionRepository()
    client = _client(config, runtime, storage_loader=RecordingStorageLoader(repo))
    with client:
        response = client.get("/v1/decisions/not-a-uuid")
    assert response.status_code == 404
    assert repo.get_calls == [], "非法形状不得触达存储层"


def test_corrupt_row_returns_500_contract_violation(config, runtime, db_file) -> None:
    client = _client(config, runtime)
    with client:
        client.post("/v1/decisions/evaluate", json=_decision_body(runtime))
    with sqlite_connection(db_file) as conn:
        conn.execute("UPDATE decisions SET risk_level = 'tampered'")

    reader = _client(config, runtime, ids=(DECISION_ID_B,))
    with reader:
        response = reader.get(f"/v1/decisions/{DECISION_ID_A}")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "storage_contract_violation"
    blob = json.dumps(response.json(), ensure_ascii=False)
    assert "tampered" not in blob, "不得回显损坏行内容"


# --- 非幂等 -------------------------------------------------------------


def test_repeated_submission_creates_a_new_record(config, runtime, db_file) -> None:
    """相同输入、相同 ``X-Request-ID`` 也必须得到不同 UUID4 与两条记录。"""
    client = _client(
        config, runtime, ids=(DECISION_ID_A, DECISION_ID_B)
    )
    with client:
        first = client.post(
            "/v1/decisions/evaluate",
            json=_decision_body(runtime),
            headers={REQUEST_ID_HEADER: "same-request-id"},
        )
        second = client.post(
            "/v1/decisions/evaluate",
            json=_decision_body(runtime),
            headers={REQUEST_ID_HEADER: "same-request-id"},
        )
    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["decision"]["decision_id"] != second.json()["decision"]["decision_id"]
    assert first.json()["decision"]["evaluation_request_id"] == "same-request-id"
    assert second.json()["decision"]["evaluation_request_id"] == "same-request-id"
    assert _row_count(db_file) == 2


def test_repeated_batch_submission_is_not_deduplicated(config, runtime, db_file) -> None:
    client = _client(
        config,
        runtime,
        ids=(DECISION_ID_A, DECISION_ID_B, DECISION_ID_C, DECISION_ID_D),
    )
    payload = _batch_body(runtime, ["a", "b"])
    with client:
        client.post("/v1/decisions/evaluate/batch", json=payload)
        client.post("/v1/decisions/evaluate/batch", json=payload)
    assert _row_count(db_file) == 4, "重复批量不得去重"


# --- storage 故障隔离 ---------------------------------------------------


def _degraded_config(tmp_path: Path, case: str):
    """构造"存储不可用"的配置。

    - ``not_configured``：完全没有 storage 段；
    - ``corrupt_database``：路径形状合法，但该文件不是合法 SQLite（打不开/schema 不符）。
    """
    from tests.cw007_fixtures import storage_section, staged_db_relative

    if case == "not_configured":
        return _make_config(tmp_path, storage=None)
    if case == "corrupt_database":
        path = staged_db_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"definitely not a sqlite database" * 8)
        return _make_config(tmp_path, storage=storage_section(staged_db_relative(tmp_path)))
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["not_configured", "corrupt_database"])
def test_storage_failure_keeps_the_old_endpoints_working(
    tmp_path: Path, case: str, runtime
) -> None:
    config = _degraded_config(tmp_path, case)
    client = _client(config, runtime)
    with client:
        assert client.get("/health/live").status_code == 200
        assert client.post("/v1/score", json=_decision_body(runtime)).status_code == 200
        assert client.post("/v1/risk/evaluate", json=_decision_body(runtime)).status_code == 200
        assert client.get("/v1/model").status_code == 200
        assert client.get("/v1/policy").status_code == 200


@pytest.mark.parametrize("case", ["not_configured", "corrupt_database"])
def test_storage_failure_makes_decisions_503(tmp_path: Path, case: str, runtime) -> None:
    config = _degraded_config(tmp_path, case)
    client = _client(config, runtime)
    with client:
        responses = [
            client.post("/v1/decisions/evaluate", json=_decision_body(runtime)),
            client.post("/v1/decisions/evaluate/batch", json=_batch_body(runtime, ["a"])),
            client.get(f"/v1/decisions/{DECISION_ID_A}"),
        ]
    for response in responses:
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "storage_not_ready"
        assert response.json()["error"]["message"] == ERROR_MESSAGES["storage_not_ready"]


def test_storage_not_configured_does_not_affect_readiness(tmp_path: Path, runtime) -> None:
    """未配置 storage 时 readiness 只由 model + policy 决定（保持 CW-006 语义）。"""
    config = _degraded_config(tmp_path, "not_configured")
    client = _client(config, runtime)
    with client:
        assert client.get("/health/ready").status_code == 200


@pytest.mark.parametrize("case", ["corrupt_database"])
def test_configured_but_broken_storage_makes_readiness_fail(
    tmp_path: Path, case: str, runtime
) -> None:
    """仓库配置启用 storage 后，readiness 必须反映它未就绪。"""
    config = _degraded_config(tmp_path, case)
    client = _client(config, runtime)
    with client:
        ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "storage_not_ready"


def test_broken_storage_file_is_not_overwritten_or_deleted(tmp_path: Path, runtime) -> None:
    """schema 不符时失败关闭：**不删除、不覆盖、不迁移**那个文件。"""
    config = _degraded_config(tmp_path, "corrupt_database")
    path = staged_db_path(tmp_path)
    before = path.read_bytes()
    client = _client(config, runtime)
    with client:
        assert client.post("/v1/decisions/evaluate", json=_decision_body(runtime)).status_code == 503
    assert path.exists(), "损坏的数据库不得被删除"
    assert path.read_bytes() == before, "损坏的数据库不得被覆盖或迁移"


def test_storage_failure_never_falls_back_to_a_memory_database(
    tmp_path: Path, runtime
) -> None:
    """未配置存储时：decisions 接口 503，且**没有**任何仓储被装配或创建。"""
    config = _make_config(tmp_path, storage=None)
    app = create_app(config, runtime_loader=RecordingLoader(runtime))
    with TestClient(app) as client:
        assert client.post("/v1/decisions/evaluate", json=_decision_body(runtime)).status_code == 503
        assert client.get(f"/v1/decisions/{DECISION_ID_A}").status_code == 503
    assert getattr(app.state, "decision_repository", None) is None
    assert not staged_db_path(tmp_path).exists(), "未配置时不得创建或改用任何数据库文件"


def test_runtime_read_failure_returns_503(config, runtime) -> None:
    repo = FakeDecisionRepository(
        get_error=StorageLoadError(StorageErrorCode.UNAVAILABLE, "读取失败")
    )
    client = _client(config, runtime, storage_loader=RecordingStorageLoader(repo))
    with client:
        response = client.get(f"/v1/decisions/{DECISION_ID_A}")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "storage_unavailable"


def test_model_failure_still_takes_precedence_for_readiness(config, runtime) -> None:
    app = create_app(
        config,
        runtime_loader=RecordingLoader(error=BundleLoadError(BundleErrorCode.MISSING, "缺失")),
    )
    with TestClient(app) as client:
        ready = client.get("/health/ready")
        decisions = client.post("/v1/decisions/evaluate", json=_decision_body(runtime))
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "model_not_ready"
    assert decisions.status_code == 503
    assert decisions.json()["error"]["code"] == "model_not_ready"


# --- 旧八接口回归与 OpenAPI --------------------------------------------


def test_old_endpoints_keep_their_exact_success_contract(config, runtime) -> None:
    client = _client(config, runtime)
    with client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        model = client.get("/v1/model")
        score = client.post("/v1/score", json=_decision_body(runtime))
        score_batch = client.post(
            "/v1/score/batch",
            json={
                "items": [
                    {"client_item_id": "a", "features": risk_features(runtime)},
                    {"client_item_id": "b", "features": risk_features(runtime)},
                ]
            },
        )
        policy = client.get("/v1/policy")
        risk = client.post("/v1/risk/evaluate", json=_decision_body(runtime))
        risk_batch = client.post("/v1/risk/evaluate/batch", json=_batch_body(runtime, ["a"]))
    assert set(live.json()) == {"status", "service", "api_version"}
    assert set(ready.json()) == {"status", "model_version", "bundle_sha256"}
    assert set(model.json()) == {
        "model_version",
        "bundle_sha256",
        "schema_version",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "required_features",
        "batch_limit",
    }
    assert set(score.json()) == {
        "request_id",
        "model_version",
        "score",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "flagged",
    }
    assert set(score_batch.json()) == {
        "request_id",
        "model_version",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "batch_size",
        "results",
    }
    assert set(policy.json()) == {
        "policy_version",
        "policy_schema_version",
        "policy_sha256",
        "policy_semantics",
        "threshold_source",
        "rules",
        "fusion",
        "risk_levels",
        "signal_point_range",
        "disclaimer",
        "batch_limit",
    }
    assert set(risk.json()) == {
        "request_id",
        "model_version",
        "score",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "flagged",
        "policy_version",
        "policy_sha256",
        "policy_semantics",
        "rule_score",
        "matched_rule_ids",
        "unevaluable_rule_ids",
        "rule_evaluation_complete",
        "signal_points",
        "risk_level",
    }
    assert set(risk_batch.json()) == {
        "request_id",
        "model_version",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "policy_version",
        "policy_sha256",
        "policy_semantics",
        "batch_size",
        "results",
    }


def test_openapi_exposes_eleven_paths(config, runtime) -> None:
    app = create_app(config, runtime_loader=RecordingLoader(runtime))
    paths = app.openapi()["paths"]
    assert set(paths) == {
        *OLD_ENDPOINTS,
        "/v1/decisions/evaluate",
        "/v1/decisions/evaluate/batch",
        "/v1/decisions/{decision_id}",
    }
    assert len(paths) == 11
    assert set(paths["/v1/decisions/evaluate"]) == {"post"}
    assert set(paths["/v1/decisions/evaluate/batch"]) == {"post"}
    assert set(paths["/v1/decisions/{decision_id}"]) == {"get"}


# --- 日志 ---------------------------------------------------------------


def _completion_records(caplog) -> list[dict]:
    records = []
    for record in caplog.records:
        if record.name != "ai_detector.service":
            continue
        try:
            payload = json.loads(record.getMessage())
        except json.JSONDecodeError:  # pragma: no cover
            continue
        if payload.get("event") == "request_completed":
            records.append(payload)
    return records


def test_decisions_endpoints_emit_one_log_with_only_the_new_fields(
    config, runtime, caplog
) -> None:
    client = _client(config, runtime, ids=(DECISION_ID_A,))
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.post("/v1/decisions/evaluate", json=_decision_body(runtime))
            client.get(f"/v1/decisions/{DECISION_ID_A}")
            client.get(f"/v1/decisions/{DECISION_ID_B}")
    records = _completion_records(caplog)
    assert len(records) == 3, "每个请求恰好一条 completion log"
    write = next(record for record in records if record["route"] == "/v1/decisions/evaluate")
    assert write["storage_operation"] == "write"
    assert write["record_count"] == 1
    read = [record for record in records if record["route"] == "/v1/decisions/{decision_id}"]
    assert [record["storage_operation"] for record in read] == ["read", "read"]
    assert read[0]["record_count"] == 1
    assert read[1]["record_count"] == 0


def test_decisions_logs_never_contain_records_ids_paths_or_scores(config, runtime, caplog) -> None:
    client = _client(config, runtime, ids=(DECISION_ID_A,))
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.post(
                "/v1/decisions/evaluate",
                json=_decision_body(runtime, transaction_amt=2000.0, c13=900.0),
                headers={REQUEST_ID_HEADER: SENTINEL},
            )
            client.get(f"/v1/decisions/{DECISION_ID_A}")
            client.get(f"/v1/decisions/{SENTINEL}")
    blob = "\n".join(record.getMessage() for record in caplog.records)
    for banned in (
        DECISION_ID_A,
        "decisions.sqlite3",
        ".sqlite3",
        "sqlite",
        "INSERT",
        "SELECT",
        "transaction_amt_train_q99_tail",
        "c13_train_q99_tail",
        "rule_score",
        "matched_rule_ids",
        "risk_record",
        "TransactionAmt",
        "features",
        "0.05",
        "sqlite3",
    ):
        assert banned not in blob, banned


def test_old_endpoint_log_fields_are_unchanged(config, runtime, caplog) -> None:
    client = _client(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.post("/v1/score", json=_decision_body(runtime))
    record = _completion_records(caplog)[0]
    assert set(record) <= {
        "event",
        "request_id",
        "method",
        "route",
        "status_code",
        "duration_ms",
        "model_version",
        "batch_size",
        "outcome",
        "policy_version",
        "risk_level",
        "rule_match_count",
    }
