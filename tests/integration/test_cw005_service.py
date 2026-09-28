"""CW-005 服务集成测试：TestClient + fake runtime，覆盖五个接口与全部安全行为。

刻意**不依赖正式 bundle**（它被 Git 忽略，且属于 smoke 命令的职责）：这里注入 fake runtime，
因此测试既可重复、又可精确统计"评分被调用了几次"——批量原子性靠"无效批次零次调用"证明。

被反复验证的几条不变量：

- 导入包无副作用、``create_app`` 本身不加载模型（加载只发生在 lifespan）；
- 每个 app 生命周期最多加载一次运行时（loader 调用计数 == 1）；
- 校验失败时**零次**评分调用（原子性）；
- 错误响应统一结构，且不回显 input / 载荷 / 凭据 / 路径 / 堆栈；
- 每个请求恰好一条 JSON completion log，且不含载荷、分数、client_item_id。
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from pathlib import Path

import pytest

# 服务专项依赖来自 `service` extra：未安装时清晰跳过本模块，安装后必须正常执行（不得跳过）。
pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_detector.service.app import create_app
from ai_detector.service.errors import BundleErrorCode, BundleLoadError
from tests.cw005_fixtures import FakeRuntime, RecordingLoader, features_payload, make_config

SENTINEL = "SENTINEL_LEAK_PROBE_9f3a"
REQUEST_ID_HEADER = "X-Request-ID"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


@pytest.fixture
def config(tmp_path: Path):
    return make_config(tmp_path)


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(scores=[0.05, 0.9, 0.11])


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _make(config, runtime: FakeRuntime, **kwargs) -> tuple[TestClient, RecordingLoader]:
    loader = RecordingLoader(runtime, **kwargs)
    app = create_app(config, runtime_loader=loader)
    return _client(app), loader


# --- 导入与工厂语义 -----------------------------------------------------


def test_package_modules_define_no_module_level_app() -> None:
    """导入不得构造应用，更不得加载模型或启动服务器。"""
    for name in ("app", "config", "schemas", "runtime", "errors", "logging"):
        module = importlib.import_module(f"ai_detector.service.{name}")
        for attribute, value in vars(module).items():
            assert not isinstance(value, FastAPI), f"{module.__name__}.{attribute} 是模块级应用"


def test_create_app_does_not_load_runtime_until_startup(config) -> None:
    """``create_app`` 本身不得碰 bundle；加载只发生在 lifespan 启动阶段。"""
    loader = RecordingLoader(error=AssertionError("loader 不应在 create_app 阶段被调用"))
    app = create_app(config, runtime_loader=loader)
    assert loader.calls == 0
    assert getattr(app.state, "runtime", None) is None


def test_runtime_is_loaded_once_per_app_lifecycle(config, runtime) -> None:
    client, loader = _make(config, runtime)
    with client:
        for _ in range(5):
            client.get("/health/live")
        client.get("/health/ready")
        client.post("/v1/score", json={"features": features_payload(runtime)})
    assert loader.calls == 1, "每个 app 生命周期只能加载一次模型"


# --- 健康检查与降级 -----------------------------------------------------


def test_live_is_200_and_does_not_need_the_model(config) -> None:
    loader = RecordingLoader(error=BundleLoadError(BundleErrorCode.MISSING, "缺失"))
    client = _client(create_app(config, runtime_loader=loader))
    with client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "live"


def test_ready_is_200_when_the_model_loaded(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_version"] == runtime.model_version
    assert body["bundle_sha256"] == runtime.bundle_sha256


@pytest.mark.parametrize(
    "error",
    [
        BundleLoadError(BundleErrorCode.MISSING, "bundle 文件不存在"),
        BundleLoadError(BundleErrorCode.HASH_MISMATCH, "指纹不符"),
        BundleLoadError(BundleErrorCode.DESERIALIZE_FAILED, "反序列化失败"),
        RuntimeError("依赖缺失"),
    ],
)
def test_degraded_startup_keeps_live_but_fails_ready_and_scoring(config, error) -> None:
    """bundle 出问题时：live 仍 200，ready 与评分 503，且不泄漏根因。"""
    loader = RecordingLoader(error=error)
    client = _client(create_app(config, runtime_loader=loader))
    with client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        score = client.post("/v1/score", json={"features": {"feat_a": 1.0}})
        batch = client.post(
            "/v1/score/batch",
            json={"items": [{"client_item_id": "a", "features": {"feat_a": 1.0}}]},
        )
    from ai_detector.service.errors import ERROR_MESSAGES

    for response in (ready, score, batch):
        assert response.status_code == 503
        body = json.dumps(response.json(), ensure_ascii=False)
        assert response.json()["error"]["code"] == "model_not_ready"
        # 只给固定的通用文案，不含根因分类、路径或异常文本。
        assert response.json()["error"]["message"] == ERROR_MESSAGES["model_not_ready"]
        for banned in ("C:\\", "D:\\", "/home/", "/Users/", "Traceback"):
            assert banned not in body, banned
        if isinstance(error, BundleLoadError):
            assert error.code not in body, "根因分类不得出现在对外响应里"


# --- 模型元数据 ---------------------------------------------------------


def test_model_endpoint_reports_schema_level_metadata(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.get("/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == runtime.model_version
    assert body["bundle_sha256"] == runtime.bundle_sha256
    assert body["score_semantics"] == "uncalibrated_model_score"
    assert body["threshold"] == runtime.threshold
    assert body["threshold_policy"] == "validation_f1"
    assert body["required_features"] == runtime.source_features
    assert body["batch_limit"] == config.batch_limit


def test_model_endpoint_leaks_no_paths_or_probability_fields(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        blob = json.dumps(client.get("/v1/model").json(), ensure_ascii=False)
    for banned in ("C:\\", "D:\\", "/Users/", "/home/", "probability", "fraud_probability"):
        assert banned not in blob, banned


# --- 单笔评分 -----------------------------------------------------------


def test_single_score_returns_the_full_contract(config) -> None:
    runtime = FakeRuntime(scores=[0.05])  # 低于阈值 0.11 -> flagged False
    client, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/score", json={"features": features_payload(runtime)})
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == pytest.approx(0.05)
    assert body["score_semantics"] == "uncalibrated_model_score"
    assert body["threshold"] == runtime.threshold
    assert body["threshold_policy"] == "validation_f1"
    assert body["model_version"] == runtime.model_version
    assert body["flagged"] is False
    assert body["request_id"]
    assert set(body) == {
        "request_id",
        "model_version",
        "score",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "flagged",
    }


@pytest.mark.parametrize(
    "score,expected_flagged",
    [(0.0, False), (0.109, False), (0.11, True), (0.5, True), (1.0, True)],
)
def test_flagged_uses_greater_or_equal(config, score: float, expected_flagged: bool) -> None:
    """``flagged = score >= threshold``：**等于阈值也算命中**。"""
    runtime = FakeRuntime(scores=[score], threshold=0.11)
    client, _ = _make(config, runtime)
    with client:
        body = client.post("/v1/score", json={"features": features_payload(runtime)}).json()
    assert body["flagged"] is expected_flagged


def test_single_score_rejects_missing_and_extra_columns(config, runtime) -> None:
    client, _ = _make(config, runtime)
    features = features_payload(runtime)
    missing = dict(features)
    missing.pop(runtime.source_features[0])
    extra = {**features, "unexpected_column": 1.0}

    with client:
        for payload in (missing, extra):
            response = client.post("/v1/score", json={"features": payload})
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "feature_contract_violation"
    assert runtime.call_count == 0, "契约不满足时不得调用模型"


def test_single_score_rejects_extra_envelope_field(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.post(
            "/v1/score", json={"features": features_payload(runtime), "extra": 1}
        )
    assert response.status_code == 422
    assert runtime.call_count == 0


def test_single_score_rejects_bool_value(config, runtime) -> None:
    client, _ = _make(config, runtime)
    features = features_payload(runtime, {runtime.source_features[0]: True})
    with client:
        response = client.post("/v1/score", json={"features": features})
    assert response.status_code == 422
    assert runtime.call_count == 0


def test_single_score_rejects_non_finite_value(config, runtime) -> None:
    """NaN 不是合法 JSON，但调用方可能硬塞进来；必须 422 而不是让它进模型。"""
    client, _ = _make(config, runtime)
    raw = json.dumps({"features": features_payload(runtime, {runtime.source_features[0]: None})})
    raw = raw.replace("null", "NaN", 1)
    with client:
        response = client.post(
            "/v1/score", content=raw, headers={"content-type": "application/json"}
        )
    assert response.status_code == 422
    assert runtime.call_count == 0


# --- 批量评分 -----------------------------------------------------------


def _batch(runtime: FakeRuntime, ids: list[str]) -> dict:
    return {
        "items": [
            {"client_item_id": item_id, "features": features_payload(runtime)}
            for item_id in ids
        ]
    }


def test_batch_preserves_request_order_and_echoes_ids(config) -> None:
    runtime = FakeRuntime(scores=[0.9, 0.01, 0.5])
    client, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/score/batch", json=_batch(runtime, ["c", "a", "b"]))
    assert response.status_code == 200
    body = response.json()
    assert [item["client_item_id"] for item in body["results"]] == ["c", "a", "b"]
    assert [item["score"] for item in body["results"]] == pytest.approx([0.9, 0.01, 0.5])
    assert body["batch_size"] == 3
    assert body["threshold"] == runtime.threshold
    assert runtime.call_count == 1, "整批应只调用一次模型"


def test_batch_flags_each_item_independently(config) -> None:
    runtime = FakeRuntime(scores=[0.11, 0.109], threshold=0.11)
    client, _ = _make(config, runtime)
    with client:
        body = client.post("/v1/score/batch", json=_batch(runtime, ["a", "b"])).json()
    assert [item["flagged"] for item in body["results"]] == [True, False]


def test_batch_limit_is_enforced_with_zero_scoring(config, tmp_path) -> None:
    limited = make_config(tmp_path, batch_limit=3)
    runtime = FakeRuntime(scores=[0.5])
    client, _ = _make(limited, runtime)
    with client:
        ok = client.post("/v1/score/batch", json=_batch(runtime, ["a", "b", "c"]))
        too_many = client.post("/v1/score/batch", json=_batch(runtime, ["a", "b", "c", "d"]))
    assert ok.status_code == 200
    assert too_many.status_code == 422
    assert too_many.json()["error"]["code"] == "batch_contract_violation"
    assert runtime.call_count == 1, "超限批次不得产生任何评分调用"


def test_batch_rejects_empty_items(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/score/batch", json={"items": []})
    assert response.status_code == 422
    assert runtime.call_count == 0


def test_batch_rejects_duplicate_client_item_ids(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/score/batch", json=_batch(runtime, ["dup", "dup"]))
    assert response.status_code == 422
    assert runtime.call_count == 0


def test_batch_is_atomic_when_one_item_violates_the_contract(config, runtime) -> None:
    """任一无效则**零次**评分——不允许部分成功。"""
    client, _ = _make(config, runtime)
    payload = _batch(runtime, ["ok-1", "ok-2", "bad"])
    payload["items"][2]["features"] = {"feat_a": 1.0}  # 缺列

    with client:
        response = client.post("/v1/score/batch", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "feature_contract_violation"
    assert runtime.call_count == 0, "整个批次必须原子地零评分"


def test_batch_item_with_bool_value_makes_the_whole_batch_fail(config, runtime) -> None:
    client, _ = _make(config, runtime)
    payload = _batch(runtime, ["ok", "bad"])
    payload["items"][1]["features"] = features_payload(runtime, {runtime.source_features[0]: True})
    with client:
        response = client.post("/v1/score/batch", json=payload)
    assert response.status_code == 422
    assert runtime.call_count == 0


def test_batch_error_does_not_echo_client_item_ids(config, runtime) -> None:
    client, _ = _make(config, runtime)
    payload = _batch(runtime, [SENTINEL, "other"])
    payload["items"][1]["features"] = {"wrong": 1.0}
    with client:
        response = client.post("/v1/score/batch", json=payload)
    assert SENTINEL not in response.text


# --- 请求 ID ------------------------------------------------------------


def test_request_id_is_generated_when_absent(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.get("/health/live")
    generated = response.headers[REQUEST_ID_HEADER]
    assert UUID_RE.match(generated)


def test_valid_request_id_is_echoed_unchanged(config, runtime) -> None:
    client, _ = _make(config, runtime)
    supplied = "trace-42:abc_1.2"
    with client:
        response = client.get("/health/live", headers={REQUEST_ID_HEADER: supplied})
    assert response.headers[REQUEST_ID_HEADER] == supplied


@pytest.mark.parametrize(
    "supplied",
    ["has space", "with/slash", "semi;colon", "quote\"mark", "x" * 65, "a" * 64],
)
def test_request_id_validation(config, runtime, supplied: str) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.get("/health/live", headers={REQUEST_ID_HEADER: supplied})
    if re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", supplied):
        assert response.status_code == 200
        assert response.headers[REQUEST_ID_HEADER] == supplied
    else:
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request_id"
        # 非法值一律不回显，避免把任意字符串写进响应头。
        assert response.headers[REQUEST_ID_HEADER] != supplied
        assert UUID_RE.match(response.headers[REQUEST_ID_HEADER])


def test_request_id_is_present_on_every_error_response(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        responses = [
            client.get("/v1/nope"),
            client.post("/v1/score", json={"features": {"wrong": 1}}),
            client.get("/health/ready", headers={REQUEST_ID_HEADER: "bad id"}),
        ]
    for response in responses:
        assert REQUEST_ID_HEADER in response.headers
        assert response.json()["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_request_id_in_body_matches_header_for_success(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/score", json={"features": features_payload(runtime)})
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]


# --- 错误结构与防泄漏 ---------------------------------------------------


def test_not_found_uses_the_unified_error_shape(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.get("/does/not/exist")
    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == "not_found"


def test_validation_error_never_echoes_the_input(config, runtime) -> None:
    client, _ = _make(config, runtime)
    payload = {"features": features_payload(runtime, {runtime.source_features[0]: SENTINEL})}
    payload["features"][runtime.source_features[0]] = [SENTINEL]  # 数组 -> 非法
    with client:
        response = client.post("/v1/score", json=payload)
    assert response.status_code == 422
    assert SENTINEL not in response.text
    assert "input" not in response.text
    assert "detail" not in response.text


def test_validation_error_hides_internal_details(config, runtime) -> None:
    client, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/score", json={"features": "not-an-object"})
    assert response.status_code == 422
    blob = response.text.lower()
    for banned in ("traceback", "pydantic", "value_error", "src\\", "src/"):
        assert banned not in blob, banned


def test_unexpected_model_error_becomes_a_safe_500(config) -> None:
    runtime = FakeRuntime(error=RuntimeError("internal detail " + SENTINEL))
    client, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/score", json={"features": features_payload(runtime)})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert SENTINEL not in response.text


def test_non_finite_model_output_becomes_a_safe_500(config) -> None:
    runtime = FakeRuntime(scores=[float("nan")])
    client, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/score", json={"features": features_payload(runtime)})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


# --- completion log -----------------------------------------------------


def _completion_records(caplog) -> list[dict]:
    records = []
    for record in caplog.records:
        if record.name != "ai_detector.service":
            continue
        try:
            payload = json.loads(record.getMessage())
        except json.JSONDecodeError:  # pragma: no cover - message 必须是 JSON
            continue
        if payload.get("event") == "request_completed":
            records.append(payload)
    return records


def test_exactly_one_completion_log_per_request(config, runtime, caplog) -> None:
    client, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.get("/health/live")
            client.get("/health/ready")
            client.post("/v1/score", json={"features": features_payload(runtime)})
            client.post("/v1/score/batch", json=_batch(runtime, ["a", "b"]))
            client.get("/does/not/exist")
    records = _completion_records(caplog)
    assert len(records) == 5, "每个请求恰好一条 completion log"


def test_completion_log_carries_the_required_fields(config, runtime, caplog) -> None:
    client, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.post("/v1/score/batch", json=_batch(runtime, ["a", "b"]))
    payload = _completion_records(caplog)[0]
    assert payload["request_id"]
    assert payload["method"] == "POST"
    assert payload["route"] == "/v1/score/batch"
    assert payload["status_code"] == 200
    assert isinstance(payload["duration_ms"], (int, float))
    assert payload["model_version"] == runtime.model_version
    assert payload["batch_size"] == 2
    assert set(payload) <= {
        "event",
        "request_id",
        "method",
        "route",
        "status_code",
        "duration_ms",
        "model_version",
        "batch_size",
        "outcome",
    }


def test_completion_log_does_not_contain_payloads_scores_or_ids(config, runtime, caplog) -> None:
    client, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.post("/v1/score/batch", json=_batch(runtime, [SENTINEL, "other"]))
            client.post(
                "/v1/score",
                json={"features": features_payload(runtime, {runtime.source_features[0]: 0.987654})},
            )
    blob = "\n".join(record.getMessage() for record in caplog.records)
    for banned in (SENTINEL, "0.987654", "feat_a", "features", "Authorization", "Cookie"):
        assert banned not in blob, banned


def test_completion_log_is_emitted_for_error_responses(config, runtime, caplog) -> None:
    client, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.post("/v1/score", json={"features": {"wrong": 1}})
            client.get("/nope")
    records = _completion_records(caplog)
    assert len(records) == 2
    assert all(record["outcome"] == "error" for record in records)
    assert [record["status_code"] for record in records] == [422, 404]


def test_log_route_never_contains_the_raw_url(config, runtime, caplog) -> None:
    """未匹配到路由时用占位名，不把调用方的原始路径写进日志。"""
    client, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.get(f"/{SENTINEL}")
    payload = _completion_records(caplog)[0]
    assert payload["route"] == "<unmatched>"
    assert SENTINEL not in json.dumps(payload)


def test_scrub_drops_non_whitelisted_fields() -> None:
    from ai_detector.service.logging import scrub

    cleaned = scrub({"request_id": "r", "body": "SECRET", "features": {"a": 1}, "status_code": 200})
    assert cleaned == {"request_id": "r", "status_code": 200}


# --- OpenAPI ------------------------------------------------------------


def test_openapi_exposes_the_five_fixed_endpoints(config, runtime) -> None:
    """CW-005 五个接口各自的方法必须保留。

    CW-006 新增了 ``/v1/policy``、``/v1/risk/evaluate``、``/v1/risk/evaluate/batch``，
    CW-007 又新增了 ``/v1/decisions/evaluate``、``/v1/decisions/evaluate/batch``、
    ``/v1/decisions/{decision_id}``，因此这里的期望集合相应扩展——断言仍是**完全相等**
    （而不是"包含"），所以任何多出来的、未申报的接口依旧会让本测试失败。
    CW-005 五个路径的方法约束逐条不变。
    """
    app = create_app(config, runtime_loader=RecordingLoader(runtime))
    paths = app.openapi()["paths"]
    assert set(paths) == {
        "/health/live",
        "/health/ready",
        "/v1/model",
        "/v1/score",
        "/v1/score/batch",
        # CW-006 新增
        "/v1/policy",
        "/v1/risk/evaluate",
        "/v1/risk/evaluate/batch",
        # CW-007 新增
        "/v1/decisions/evaluate",
        "/v1/decisions/evaluate/batch",
        "/v1/decisions/{decision_id}",
    }
    assert set(paths["/health/live"]) == {"get"}
    assert set(paths["/health/ready"]) == {"get"}
    assert set(paths["/v1/model"]) == {"get"}
    assert set(paths["/v1/score"]) == {"post"}
    assert set(paths["/v1/score/batch"]) == {"post"}
