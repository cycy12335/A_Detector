"""CW-006 服务集成测试：三个新接口、降级、原子性、日志与 CW-005 回归。

刻意**不依赖正式 bundle**（它被 Git 忽略，属于 smoke 命令的职责）：这里注入 fake runtime，
政策则用临时目录里那份规范政策 + 真实 policy engine，因此规则求值走的是**产品代码**，
不是替身。只有在需要精确控制求值结果（例如"坏引擎输出"）时才注入 fake engine。

被反复验证的不变量：

- 政策每个 app 生命周期最多加载一次，单次请求**零次**读文件；
- policy 故障时应用仍启动：旧评分接口照常 200，新接口与 ``/health/ready`` 503 ``policy_not_ready``，
  且**不回退**到默认或旧政策；
- risk 批量先完成全部契约与规则求值，再**恰好一次** ``score_many``；任一契约错误时零次评分
  且不返回部分结果；
- 每个请求恰好一条 completion log，且不含规则 ID 列表、分数、特征值或载荷。
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="需要 service extra：pip install -e \".[test,tree,service]\"")
pytest.importorskip("httpx", reason="需要 service extra：pip install -e \".[test,tree,service]\"")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from ai_detector.rules.config import PolicyErrorCode  # noqa: E402
from ai_detector.service.app import create_app  # noqa: E402
from ai_detector.service.errors import (  # noqa: E402
    ERROR_MESSAGES,
    BundleErrorCode,
    BundleLoadError,
)
from tests.cw005_fixtures import FakeRuntime, RecordingLoader, make_config  # noqa: E402
from tests.cw006_fixtures import (  # noqa: E402
    ALL_RULE_IDS,
    FakePolicyEngine,
    RecordingPolicyLoader,
    matched_result,
    policy_mapping,
    risk_features,
    stage_relative,
)

SENTINEL = "SENTINEL_LEAK_PROBE_6c1d"
REQUEST_ID_HEADER = "X-Request-ID"

#: 含两个规则字段的合成源特征（仅列名，取值全部由测试给出）。
FEATURES = ["TransactionAmt", "C13", "feat_c"]

AMT_RULE, C13_RULE = ALL_RULE_IDS


@pytest.fixture
def config(tmp_path: Path):
    """合法政策 + 指向临时目录的服务配置。"""
    return make_config(tmp_path, features=len(FEATURES))


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(features=FEATURES, scores=[0.05, 0.9, 0.11])


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _make(config, runtime: FakeRuntime, **kwargs):
    """构造 app；默认用**真实** policy loader（读取临时政策文件）。"""
    kwargs.setdefault("policy_loader", None)
    policy_loader = kwargs.pop("policy_loader")
    loader = RecordingLoader(runtime, **kwargs)
    if policy_loader is None:
        app = create_app(config, runtime_loader=loader)
        return _client(app), loader, None
    app = create_app(config, runtime_loader=loader, policy_loader=policy_loader)
    return _client(app), loader, policy_loader


# --- 工厂语义与生命周期 -------------------------------------------------


def test_package_modules_define_no_module_level_app() -> None:
    for name in ("app", "config", "schemas", "runtime", "errors", "logging"):
        module = importlib.import_module(f"ai_detector.service.{name}")
        for attribute, value in vars(module).items():
            assert not isinstance(value, FastAPI), f"{module.__name__}.{attribute} 是模块级应用"


def test_create_app_loads_neither_model_nor_policy(config) -> None:
    runtime_loader = RecordingLoader(error=AssertionError("不应在 create_app 阶段被调用"))
    policy_loader = RecordingPolicyLoader(error=AssertionError("不应在 create_app 阶段被调用"))
    app = create_app(config, runtime_loader=runtime_loader, policy_loader=policy_loader)
    assert runtime_loader.calls == 0
    assert policy_loader.calls == 0
    assert getattr(app.state, "runtime", None) is None
    assert getattr(app.state, "policy_engine", None) is None


def test_policy_is_loaded_once_per_app_lifecycle(config, runtime) -> None:
    """多次请求不得重复读取政策文件（否则等于每请求一次磁盘 I/O + 哈希）。"""
    loaded: list[int] = []

    from ai_detector.service import runtime as runtime_module

    real_loader = runtime_module.load_policy_engine

    def _counting(config_):  # noqa: ANN001, ANN202
        loaded.append(1)
        return real_loader(config_)

    app = create_app(config, runtime_loader=RecordingLoader(runtime), policy_loader=_counting)
    client = _client(app)
    with client:
        for _ in range(3):
            client.get("/v1/policy")
            client.post(
                "/v1/risk/evaluate",
                json={"features": risk_features(runtime, transaction_amt=2000.0, c13=900.0)},
            )
        client.post("/v1/risk/evaluate/batch", json=_risk_batch(runtime, [("a", 1.0, 1.0)]))
    assert len(loaded) == 1, "政策只能加载一次"


def test_policy_files_are_read_only_during_startup(config, runtime, monkeypatch) -> None:
    """请求期间不得再读政策文件：启动后把文件删掉，服务照常工作。"""
    client, _, _ = _make(config, runtime)
    with client:
        assert client.get("/v1/policy").status_code == 200
        config.policy_path.unlink()
        assert client.get("/v1/policy").status_code == 200
        response = client.post(
            "/v1/risk/evaluate",
            json={"features": risk_features(runtime, transaction_amt=2000.0, c13=900.0)},
        )
    assert response.status_code == 200


def test_fake_engine_can_be_injected(config, runtime) -> None:
    engine = FakePolicyEngine(results=[matched_result(AMT_RULE)])
    client, _, loader = _make(config, runtime, policy_loader=RecordingPolicyLoader(engine))
    with client:
        body = client.post(
            "/v1/risk/evaluate",
            json={"features": risk_features(runtime)},
        ).json()
    assert loader.calls == 1
    assert body["matched_rule_ids"] == [AMT_RULE]
    assert body["policy_version"] == engine.policy_version


# --- GET /v1/policy -----------------------------------------------------


def test_policy_endpoint_exposes_the_frozen_contract(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    with client:
        response = client.get("/v1/policy")
    assert response.status_code == 200
    body = response.json()
    assert body["policy_version"] == "cw006-risk-policy-v1"
    assert body["policy_schema_version"] == "cw006_policy_v1"
    assert body["policy_semantics"] == "research_signal_band"
    assert body["threshold_source"] == "cw003_train_only_q0.99"
    assert len(body["policy_sha256"]) == 64
    assert body["rules"] == [
        {
            "rule_id": AMT_RULE,
            "feature": "TransactionAmt",
            "operator": "gt",
            "threshold": 1092.0,
        },
        {"rule_id": C13_RULE, "feature": "C13", "operator": "gt", "threshold": 580.0},
    ]
    assert body["fusion"]["flagged_signal_points"] == 2
    assert body["fusion"]["matched_rule_signal_points"] == 1
    assert body["fusion"]["bands"] == {
        "low_signal": [0, 0],
        "elevated_signal": [1, 2],
        "high_signal": [3, 4],
    }
    assert body["risk_levels"] == [
        "low_signal",
        "elevated_signal",
        "high_signal",
        "indeterminate",
    ]
    assert "不是" in body["disclaimer"]


def test_policy_endpoint_leaks_no_paths_or_payloads(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    with client:
        blob = json.dumps(client.get("/v1/policy").json(), ensure_ascii=False)
    for banned in ("C:\\", "D:\\", "/Users/", "/home/", ".yaml", "configs/", "artifacts", "risk_policy"):
        assert banned not in blob, banned


def test_policy_endpoint_rejects_non_get(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/policy", json={})
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


# --- POST /v1/risk/evaluate --------------------------------------------


def test_single_risk_returns_the_full_contract(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.05])  # 低于阈值 0.11 -> 未标记
    client, _, _ = _make(config, runtime)
    with client:
        response = client.post(
            "/v1/risk/evaluate",
            json={"features": risk_features(runtime, transaction_amt=2000.0, c13=1.0)},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == pytest.approx(0.05)
    assert body["score_semantics"] == "uncalibrated_model_score"
    assert body["threshold"] == runtime.threshold
    assert body["threshold_policy"] == "validation_f1"
    assert body["model_version"] == runtime.model_version
    assert body["flagged"] is False
    assert body["policy_version"] == "cw006-risk-policy-v1"
    assert body["policy_semantics"] == "research_signal_band"
    assert len(body["policy_sha256"]) == 64
    assert body["rule_score"] == pytest.approx(0.5)
    assert body["matched_rule_ids"] == [AMT_RULE]
    assert body["unevaluable_rule_ids"] == []
    assert body["rule_evaluation_complete"] is True
    assert body["signal_points"] == 1
    assert body["risk_level"] == "elevated_signal"
    assert body["request_id"]
    assert set(body) == {
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


@pytest.mark.parametrize(
    "amt,c13,score,points,level,matched",
    [
        (1.0, 1.0, 0.05, 0, "low_signal", []),
        (2000.0, 1.0, 0.05, 1, "elevated_signal", [AMT_RULE]),
        (1.0, 900.0, 0.05, 1, "elevated_signal", [C13_RULE]),
        (2000.0, 900.0, 0.05, 2, "elevated_signal", [AMT_RULE, C13_RULE]),
        (1.0, 1.0, 0.9, 2, "elevated_signal", []),
        (2000.0, 1.0, 0.9, 3, "high_signal", [AMT_RULE]),
        (2000.0, 900.0, 0.9, 4, "high_signal", [AMT_RULE, C13_RULE]),
    ],
)
def test_single_risk_fusion_over_http(
    config, amt: float, c13: float, score: float, points: int, level: str, matched: list[str]
) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[score])
    client, _, _ = _make(config, runtime)
    with client:
        body = client.post(
            "/v1/risk/evaluate",
            json={"features": risk_features(runtime, transaction_amt=amt, c13=c13)},
        ).json()
    assert body["signal_points"] == points
    assert body["risk_level"] == level
    assert body["matched_rule_ids"] == matched
    assert body["rule_evaluation_complete"] is True


@pytest.mark.parametrize("value", [None, "2000.0", "1092.0"])
def test_single_risk_is_indeterminate_for_unevaluable_rule_input(
    config, value: object
) -> None:
    """null/字符串是**合法但不可评估**的输入：200 + indeterminate，不是 422。"""
    runtime = FakeRuntime(features=FEATURES, scores=[0.9])
    client, _, _ = _make(config, runtime)
    with client:
        response = client.post(
            "/v1/risk/evaluate",
            json={"features": risk_features(runtime, transaction_amt=value, c13=900.0)},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] == "indeterminate"
    assert body["signal_points"] is None
    assert body["rule_evaluation_complete"] is False
    assert body["unevaluable_rule_ids"] == [AMT_RULE]
    assert body["matched_rule_ids"] == [C13_RULE]
    assert body["rule_score"] == pytest.approx(0.5)


def test_single_risk_never_returns_production_decision_fields(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    with client:
        blob = json.dumps(
            client.post(
                "/v1/risk/evaluate", json={"features": risk_features(runtime)}
            ).json(),
            ensure_ascii=False,
        ).lower()
    for banned in (
        "approve",
        "decline",
        "block",
        "allow",
        "fraud_probability",
        "probability",
        "recommendation",
    ):
        assert banned not in blob, banned


def test_single_risk_rejects_missing_and_extra_columns_with_zero_scoring(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    features = risk_features(runtime)
    missing = dict(features)
    missing.pop("TransactionAmt")
    extra = {**features, "unexpected_column": 1.0}
    with client:
        for payload in (missing, extra):
            response = client.post("/v1/risk/evaluate", json={"features": payload})
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "feature_contract_violation"
    assert runtime.call_count == 0


def test_single_risk_rejects_bool_value(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    with client:
        response = client.post(
            "/v1/risk/evaluate",
            json={"features": risk_features(runtime, transaction_amt=True)},
        )
    assert response.status_code == 422
    assert runtime.call_count == 0


# --- POST /v1/risk/evaluate/batch --------------------------------------


def _risk_batch(runtime: FakeRuntime, spec: list[tuple[str, float, float]]) -> dict:
    return {
        "items": [
            {
                "client_item_id": item_id,
                "features": risk_features(runtime, transaction_amt=amt, c13=c13),
            }
            for item_id, amt, c13 in spec
        ]
    }


def test_risk_batch_preserves_order_and_calls_the_model_once(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.9, 0.01, 0.9])
    client, _, _ = _make(config, runtime)
    spec = [("c", 2000.0, 900.0), ("a", 1.0, 1.0), ("b", 2000.0, 1.0)]
    with client:
        response = client.post("/v1/risk/evaluate/batch", json=_risk_batch(runtime, spec))
    assert response.status_code == 200
    body = response.json()
    assert [item["client_item_id"] for item in body["results"]] == ["c", "a", "b"]
    assert [item["score"] for item in body["results"]] == pytest.approx([0.9, 0.01, 0.9])
    assert [item["risk_level"] for item in body["results"]] == [
        "high_signal",  # 4 点：标记 + 两条规则
        "low_signal",  # 0 点
        "high_signal",  # 3 点：标记 + 一条规则
    ]
    assert [item["signal_points"] for item in body["results"]] == [4, 0, 3]
    assert body["batch_size"] == 3
    assert body["policy_version"] == "cw006-risk-policy-v1"
    assert runtime.call_count == 1, "整批应只调用一次模型"


def test_risk_batch_top_level_carries_model_and_policy_metadata(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.5])
    client, _, _ = _make(config, runtime)
    with client:
        body = client.post(
            "/v1/risk/evaluate/batch", json=_risk_batch(runtime, [("a", 1.0, 1.0)])
        ).json()
    assert set(body) == {
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
    assert body["model_version"] == runtime.model_version
    assert body["policy_semantics"] == "research_signal_band"


def test_risk_batch_item_does_not_repeat_payload_or_feature_values(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.5])
    client, _, _ = _make(config, runtime)
    with client:
        body = client.post(
            "/v1/risk/evaluate/batch", json=_risk_batch(runtime, [("a", 2000.0, 900.0)])
        ).json()
    item = body["results"][0]
    assert set(item) == {
        "client_item_id",
        "score",
        "flagged",
        "rule_score",
        "matched_rule_ids",
        "unevaluable_rule_ids",
        "rule_evaluation_complete",
        "signal_points",
        "risk_level",
    }
    blob = json.dumps(item, ensure_ascii=False)
    assert "features" not in blob
    assert "2000" not in blob and "900" not in blob


def test_risk_batch_allows_one_item(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.5])
    client, _, _ = _make(config, runtime)
    with client:
        response = client.post(
            "/v1/risk/evaluate/batch", json=_risk_batch(runtime, [("only", 1.0, 1.0)])
        )
    assert response.status_code == 200
    assert response.json()["batch_size"] == 1


def test_risk_batch_at_the_limit_is_accepted(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.5])
    client, _, _ = _make(config, runtime)
    spec = [(f"i{index}", 1.0, 1.0) for index in range(config.batch_limit)]
    with client:
        response = client.post("/v1/risk/evaluate/batch", json=_risk_batch(runtime, spec))
    assert response.status_code == 200
    assert response.json()["batch_size"] == 100
    assert runtime.call_count == 1


def test_risk_batch_above_the_limit_is_rejected_with_zero_scoring(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.5])
    client, _, _ = _make(config, runtime)
    spec = [(f"i{index}", 1.0, 1.0) for index in range(config.batch_limit + 1)]
    with client:
        response = client.post("/v1/risk/evaluate/batch", json=_risk_batch(runtime, spec))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "batch_contract_violation"
    assert runtime.call_count == 0


def test_risk_batch_rejects_empty_items(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    with client:
        response = client.post("/v1/risk/evaluate/batch", json={"items": []})
    assert response.status_code == 422
    assert runtime.call_count == 0


def test_risk_batch_rejects_duplicate_client_item_ids(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    payload = _risk_batch(runtime, [("dup", 1.0, 1.0), ("dup", 2.0, 2.0)])
    with client:
        response = client.post("/v1/risk/evaluate/batch", json=payload)
    assert response.status_code == 422
    assert runtime.call_count == 0


def test_risk_batch_is_atomic_when_one_item_violates_the_feature_contract(config, runtime) -> None:
    """任一无效则**零次**评分、零次规则求值，且不返回部分结果。"""
    client, _, _ = _make(config, runtime)
    payload = _risk_batch(runtime, [("ok-1", 1.0, 1.0), ("ok-2", 2.0, 2.0), ("bad", 3.0, 3.0)])
    payload["items"][2]["features"] = {"TransactionAmt": 1.0}
    with client:
        response = client.post("/v1/risk/evaluate/batch", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "feature_contract_violation"
    assert "results" not in response.json()
    assert runtime.call_count == 0, "整个批次必须原子地零评分"


def test_risk_batch_is_atomic_when_one_item_has_an_invalid_value_type(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    payload = _risk_batch(runtime, [("ok", 1.0, 1.0), ("bad", 1.0, 1.0)])
    payload["items"][1]["features"]["C13"] = [1, 2]
    with client:
        response = client.post("/v1/risk/evaluate/batch", json=payload)
    assert response.status_code == 422
    assert runtime.call_count == 0


def test_risk_batch_error_does_not_echo_client_item_ids(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    payload = _risk_batch(runtime, [(SENTINEL, 1.0, 1.0), ("other", 1.0, 1.0)])
    payload["items"][1]["features"] = {"wrong": 1.0}
    with client:
        response = client.post("/v1/risk/evaluate/batch", json=payload)
    assert SENTINEL not in response.text


def test_risk_batch_indeterminate_items_do_not_fail_the_batch(config) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.9, 0.05])
    client, _, _ = _make(config, runtime)
    payload = {
        "items": [
            {
                "client_item_id": "a",
                "features": risk_features(runtime, transaction_amt=None, c13=900.0),
            },
            {
                "client_item_id": "b",
                "features": risk_features(runtime, transaction_amt=1.0, c13=1.0),
            },
        ]
    }
    with client:
        response = client.post("/v1/risk/evaluate/batch", json=payload)
    assert response.status_code == 200
    results = response.json()["results"]
    # a：amt 为 null -> 不可评估；b：两条规则都不命中且未标记 -> 0 点。
    assert [item["risk_level"] for item in results] == ["indeterminate", "low_signal"]
    assert results[0]["signal_points"] is None
    assert results[0]["unevaluable_rule_ids"] == [AMT_RULE]
    assert results[1]["signal_points"] == 0
    assert results[1]["rule_evaluation_complete"] is True


# --- 政策故障时的安全降级 ----------------------------------------------


def _degraded_config(tmp_path: Path, case: str):
    """构造一个"政策不可用"的服务配置。

    每个 case 用独立子目录，避免多个 case 共享同一个政策文件而互相覆盖。
    刻意**只构造被请求的那一个**：``wrong_schema`` 这类配置错误应在构造阶段就抛出
    （与 CW-005 拒绝非规范 ``threshold_policy`` 的做法一致），不该混进运行期降级用例。

    注意"schema 错误"指的是**政策文件**的 schema 违规（额外字段、错误版本等），
    它必须在运行期被安全降级；而"服务配置声明的 schema 版本不对"是**配置**错误，
    由 ``ServiceConfig`` 直接拒绝（见 ``test_cw006_policy_config.py``）。
    """
    if case == "missing":
        # 声明的路径必须仍是**项目相对**形状（生产加载器的硬契约），只是文件不存在。
        return make_config(
            tmp_path, features=len(FEATURES), policy_path=stage_relative(tmp_path, "nope.yaml")
        )
    if case == "hash_mismatch":
        return make_config(tmp_path, features=len(FEATURES), policy_sha256="f" * 64)
    if case == "tampered_thresholds":
        tampered = policy_mapping()
        tampered["policy"]["rules"][0]["threshold"] = 100.0  # 与 CW-003 阈值不符
        return make_config(tmp_path, features=len(FEATURES), policy=tampered)
    if case == "bad_schema":
        extra = policy_mapping()
        extra["policy"]["relaxed_mode"] = True  # 政策文件里的额外字段
        return make_config(tmp_path, features=len(FEATURES), policy=extra)
    if case == "not_configured":
        return make_config(tmp_path, features=len(FEATURES), policy=None)
    raise AssertionError(f"未知用例：{case}")


DEGRADED_CASES = ["missing", "hash_mismatch", "tampered_thresholds", "bad_schema", "not_configured"]


@pytest.mark.parametrize("case", DEGRADED_CASES)
def test_policy_failure_keeps_the_old_score_endpoint_working(
    tmp_path: Path, case: str, runtime
) -> None:
    config = _degraded_config(tmp_path, case)
    client, _, _ = _make(config, runtime)
    with client:
        assert client.get("/health/live").status_code == 200
        score = client.post("/v1/score", json={"features": risk_features(runtime)})
        batch = client.post(
            "/v1/score/batch",
            json={
                "items": [
                    {"client_item_id": "a", "features": risk_features(runtime)},
                    {"client_item_id": "b", "features": risk_features(runtime)},
                ]
            },
        )
        model = client.get("/v1/model")
    assert score.status_code == 200
    assert batch.status_code == 200
    assert model.status_code == 200
    assert set(score.json()) == {
        "request_id",
        "model_version",
        "score",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "flagged",
    }, "CW-005 评分响应字段不得变化"


@pytest.mark.parametrize("case", DEGRADED_CASES)
def test_policy_failure_makes_the_new_endpoints_503(tmp_path: Path, case: str, runtime) -> None:
    config = _degraded_config(tmp_path, case)
    client, _, _ = _make(config, runtime)
    with client:
        responses = [
            client.get("/health/ready"),
            client.get("/v1/policy"),
            client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)}),
            client.post(
                "/v1/risk/evaluate/batch",
                json={
                    "items": [{"client_item_id": "a", "features": risk_features(runtime)}]
                },
            ),
        ]
    for response in responses:
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "policy_not_ready"
        assert response.json()["error"]["message"] == ERROR_MESSAGES["policy_not_ready"]


@pytest.mark.parametrize("case", DEGRADED_CASES)
def test_policy_failure_does_not_leak_or_fall_back(tmp_path: Path, case: str, runtime) -> None:
    """503 只给固定文案；且绝不因为政策坏了就改用内置默认阈值或旧政策。"""
    config = _degraded_config(tmp_path, case)
    client, _, _ = _make(config, runtime)
    with client:
        policy = client.get("/v1/policy")
        blob = json.dumps(policy.json(), ensure_ascii=False)
        risk = client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
    assert risk.status_code == 503
    assert "1092" not in blob and "580" not in blob, "不得回退到默认阈值"
    for banned in ("C:\\", "D:\\", "/home/", "/Users/", "Traceback", ".yaml"):
        assert banned not in blob, banned
    assert "threshold" not in blob
    assert "policy_version" not in blob


def test_policy_failure_keeps_ready_false_but_model_info_intact(tmp_path: Path, runtime) -> None:
    config = _degraded_config(tmp_path, "hash_mismatch")
    client, _, _ = _make(config, runtime)
    with client:
        ready = client.get("/health/ready")
        model = client.get("/v1/model")
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "policy_not_ready"
    assert model.status_code == 200
    assert model.json()["threshold"] == runtime.threshold


def test_policy_metadata_is_independent_of_the_model(config) -> None:
    """政策元数据不依赖模型：模型坏掉时 ``/v1/policy`` 仍 200。

    这是刻意的：政策版本与阈值来源是**静态声明**，把它挂在模型可用性上会让
    "模型为什么没起来"和"用的是哪份政策"两件事互相遮蔽。风险求值仍然要求模型就绪。
    """
    runtime = FakeRuntime(features=FEATURES)
    loader = RecordingLoader(error=BundleLoadError(BundleErrorCode.MISSING, "缺失"))
    client = _client(create_app(config, runtime_loader=loader))
    with client:
        policy = client.get("/v1/policy")
        ready = client.get("/health/ready")
        risk = client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
    assert policy.status_code == 200
    assert policy.json()["policy_version"] == "cw006-risk-policy-v1"
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "model_not_ready"
    assert risk.status_code == 503
    assert risk.json()["error"]["code"] == "model_not_ready"


def test_model_failure_still_reports_model_not_ready(config) -> None:
    """模型故障的原有降级语义不变：先报 model_not_ready，而不是 policy_not_ready。"""
    loader = RecordingLoader(error=BundleLoadError(BundleErrorCode.HASH_MISMATCH, "指纹不符"))
    client = _client(create_app(config, runtime_loader=loader))
    with client:
        ready = client.get("/health/ready")
        risk = client.post("/v1/risk/evaluate", json={"features": {"TransactionAmt": 1.0}})
    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "model_not_ready"
    assert risk.status_code == 503
    assert risk.json()["error"]["code"] == "model_not_ready"


def test_startup_diagnostic_reports_policy_state_without_leaking(tmp_path: Path, caplog) -> None:
    from ai_detector.service.logging import STARTUP_EVENT

    runtime = FakeRuntime(features=FEATURES)
    expected_codes = {
        "hash_mismatch": PolicyErrorCode.HASH_MISMATCH,
        "missing": PolicyErrorCode.MISSING,
        "tampered_thresholds": PolicyErrorCode.SCHEMA_VIOLATION,
        "bad_schema": PolicyErrorCode.SCHEMA_VIOLATION,
        "not_configured": PolicyErrorCode.NOT_CONFIGURED,
    }
    for case, expected_code in expected_codes.items():
        config = _degraded_config(tmp_path, case)
        app = create_app(config, runtime_loader=RecordingLoader(runtime))
        with caplog.at_level(logging.INFO, logger="ai_detector.service"):
            caplog.clear()
            with TestClient(app):
                pass
        records = [
            json.loads(record.getMessage())
            for record in caplog.records
            if record.name == "ai_detector.service"
            and json.loads(record.getMessage()).get("event") == STARTUP_EVENT
        ]
        assert len(records) == 1
        payload = records[0]
        assert payload["outcome"] == "not_ready", case
        assert payload["policy_error_code"] == expected_code, case
        # 模型是好的，因此模型侧信息照常报出，便于定位"坏的是哪一个"。
        assert payload["model_version"] == runtime.model_version, case
        blob = json.dumps(payload, ensure_ascii=False)
        assert "policy_version" not in payload, case
        for banned in ("C:\\", "D:\\", "/home/", "/Users/", "Traceback", ".yaml"):
            assert banned not in blob, banned


def test_startup_diagnostic_reports_the_policy_version_on_success(tmp_path: Path, caplog) -> None:
    from ai_detector.service.logging import STARTUP_EVENT

    config = make_config(tmp_path, features=len(FEATURES))
    runtime = FakeRuntime(features=FEATURES)
    app = create_app(config, runtime_loader=RecordingLoader(runtime))
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        caplog.clear()
        with TestClient(app):
            pass
    payload = next(
        json.loads(record.getMessage())
        for record in caplog.records
        if json.loads(record.getMessage()).get("event") == STARTUP_EVENT
    )
    assert payload["outcome"] == "ready"
    assert payload["policy_version"] == "cw006-risk-policy-v1"
    assert "policy_error_code" not in payload


# --- 坏引擎输出 ---------------------------------------------------------


def test_structurally_invalid_engine_output_becomes_a_safe_500(config, runtime) -> None:
    """引擎给出"既命中又不可评估"的规则时，服务端政策已坏：500 且不泄漏内部细节。"""
    from ai_detector.rules.engine import RuleEvaluation, RuleEvaluationResult

    broken = RuleEvaluationResult(
        evaluations=(
            RuleEvaluation(rule_id=AMT_RULE, matched=True, evaluable=False),
            RuleEvaluation(rule_id=C13_RULE, matched=False, evaluable=True),
        )
    )
    engine = FakePolicyEngine(results=[broken])
    client, _, _ = _make(config, runtime, policy_loader=RecordingPolicyLoader(engine))
    with client:
        response = client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "policy_contract_violation"
    assert SENTINEL not in response.text


def test_engine_exception_is_folded_into_a_safe_500(config, runtime) -> None:
    engine = FakePolicyEngine(error=RuntimeError("internal detail " + SENTINEL))
    client, _, _ = _make(config, runtime, policy_loader=RecordingPolicyLoader(engine))
    with client:
        response = client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert SENTINEL not in response.text


def _non_boolean_result(*, matched: Any = False, evaluable: Any = True):
    from ai_detector.rules.engine import RuleEvaluation, RuleEvaluationResult

    return RuleEvaluationResult(
        evaluations=(
            RuleEvaluation(rule_id=AMT_RULE, matched=matched, evaluable=evaluable),
            RuleEvaluation(rule_id=C13_RULE, matched=False, evaluable=True),
        )
    )


@pytest.mark.parametrize("value", [1, 0, 1.0, 0.0, "true", "", None])
def test_non_boolean_matched_from_an_injected_engine_is_rejected(
    config, runtime, value: Any
) -> None:
    """验收阻塞项：注入引擎返回整数/字符串/null 时必须是 500，而不是 200 参与分级。

    整型 ``1`` 最危险——它是 ``int``，真值判断会把它当成"命中"，于是结构异常被静默
    当成合法的分级依据，并一路走到 200 响应里。
    """
    engine = FakePolicyEngine(results=[_non_boolean_result(matched=value)])
    client, _, _ = _make(config, runtime, policy_loader=RecordingPolicyLoader(engine))
    with client:
        response = client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "policy_contract_violation"
    assert response.json()["error"]["message"] == ERROR_MESSAGES["policy_contract_violation"]
    blob = json.dumps(response.json(), ensure_ascii=False)
    assert "signal_points" not in blob and "risk_level" not in blob
    # 不泄漏被注入的取值（空字符串没有可检查的内容，跳过）。
    if isinstance(value, str) and value:
        assert value not in blob
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        assert f'"{value}"' not in blob


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_non_boolean_evaluable_from_an_injected_engine_is_rejected(
    config, runtime, value: Any
) -> None:
    engine = FakePolicyEngine(results=[_non_boolean_result(evaluable=value)])
    client, _, _ = _make(config, runtime, policy_loader=RecordingPolicyLoader(engine))
    with client:
        response = client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "policy_contract_violation"


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_non_boolean_engine_output_fails_the_batch_before_scoring(
    config, runtime, value: Any
) -> None:
    """批量也必须：统一 500、**零次**模型调用、不返回部分结果。"""
    engine = FakePolicyEngine(results=[_non_boolean_result(matched=value)])
    client, _, _ = _make(config, runtime, policy_loader=RecordingPolicyLoader(engine))
    with client:
        response = client.post(
            "/v1/risk/evaluate/batch",
            json=_risk_batch(runtime, [("a", 2000.0, 900.0), ("b", 1.0, 1.0)]),
        )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "policy_contract_violation"
    assert "results" not in response.json()
    assert runtime.call_count == 0, "规则输出结构异常时必须在模型调用之前失败"


def test_genuine_boolean_engine_output_is_unaffected(config, runtime) -> None:
    """收紧类型不得误伤合法实现：真布尔结果照常 200 并正确分级。"""
    engine = FakePolicyEngine(results=[matched_result(AMT_RULE, C13_RULE)])
    client, _, _ = _make(config, runtime, policy_loader=RecordingPolicyLoader(engine))
    with client:
        body = client.post(
            "/v1/risk/evaluate", json={"features": risk_features(runtime)}
        ).json()
    assert body["matched_rule_ids"] == [AMT_RULE, C13_RULE]
    assert body["rule_evaluation_complete"] is True
    assert body["risk_level"] in {"low_signal", "elevated_signal", "high_signal"}


def test_nested_non_rule_evaluation_entry_is_rejected(config, runtime) -> None:
    """列表里混进非 RuleEvaluation 条目同样按契约破坏处理。"""
    from ai_detector.rules.engine import RuleEvaluation, RuleEvaluationResult

    broken = RuleEvaluationResult(
        evaluations=(
            RuleEvaluation(rule_id=AMT_RULE, matched=True, evaluable=True),
            {"rule_id": C13_RULE, "matched": False, "evaluable": True},  # type: ignore[arg-type]
        )
    )
    engine = FakePolicyEngine(results=[broken])
    client, _, _ = _make(config, runtime, policy_loader=RecordingPolicyLoader(engine))
    with client:
        response = client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "policy_contract_violation"


# --- 请求 ID、错误结构与日志 -------------------------------------------


def test_request_id_is_present_on_new_endpoints(config, runtime) -> None:
    """每个响应都带请求 ID；风险接口的响应体也回显它（与 CW-005 评分接口一致）。"""
    client, _, _ = _make(config, runtime)
    supplied = "risk-42:abc_1.2"
    with client:
        policy = client.get("/v1/policy", headers={REQUEST_ID_HEADER: supplied})
        single = client.post(
            "/v1/risk/evaluate",
            json={"features": risk_features(runtime)},
            headers={REQUEST_ID_HEADER: supplied},
        )
        batch = client.post(
            "/v1/risk/evaluate/batch",
            json=_risk_batch(runtime, [("a", 1.0, 1.0)]),
            headers={REQUEST_ID_HEADER: supplied},
        )
    for response in (policy, single, batch):
        assert response.headers[REQUEST_ID_HEADER] == supplied
    # /v1/policy 与 CW-005 的 /v1/model 一样是**声明级**元数据响应，不回显请求 ID；
    # 两个风险求值接口与评分接口一致，在响应体里回显。
    assert "request_id" not in policy.json()
    assert single.json()["request_id"] == supplied
    assert batch.json()["request_id"] == supplied


def test_new_endpoints_validate_the_request_id_header(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    with client:
        responses = [
            client.get("/v1/policy", headers={REQUEST_ID_HEADER: "bad id"}),
            client.post(
                "/v1/risk/evaluate",
                json={"features": risk_features(runtime)},
                headers={REQUEST_ID_HEADER: "bad/id"},
            ),
        ]
    for response in responses:
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request_id"


def test_error_responses_on_new_endpoints_use_the_unified_shape(tmp_path: Path, runtime) -> None:
    """政策故障时新接口的错误体与 CW-005 完全同构，且请求 ID 一致。"""
    config = _degraded_config(tmp_path, "hash_mismatch")
    client, _, _ = _make(config, runtime)
    with client:
        response = client.get("/v1/policy")
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_risk_endpoints_reject_unknown_envelope_fields(config, runtime) -> None:
    client, _, _ = _make(config, runtime)
    with client:
        response = client.post(
            "/v1/risk/evaluate",
            json={"features": risk_features(runtime), "extra": 1},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request_body"
    assert runtime.call_count == 0


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


def test_exactly_one_completion_log_per_risk_request(config, runtime, caplog) -> None:
    client, _, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.get("/v1/policy")
            client.get("/health/ready")
            client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
            client.post("/v1/risk/evaluate/batch", json=_risk_batch(runtime, [("a", 1.0, 1.0)]))
            client.get("/v1/does-not-exist")
    records = _completion_records(caplog)
    assert len(records) == 5, "每个请求恰好一条 completion log"
    assert [record["status_code"] for record in records] == [200, 200, 200, 200, 404]


def test_completion_log_on_a_policy_failure_still_has_one_record(tmp_path, runtime, caplog) -> None:
    config = _degraded_config(tmp_path, "hash_mismatch")
    client, _, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.get("/v1/policy")
            client.post("/v1/risk/evaluate", json={"features": risk_features(runtime)})
    records = _completion_records(caplog)
    assert len(records) == 2
    assert all(record["status_code"] == 503 for record in records)
    assert all(record["outcome"] == "error" for record in records)
    # 政策未就绪时不得写出政策版本（那会暗示它已被加载）。
    assert all("policy_version" not in record for record in records)


def test_completion_log_carries_policy_and_risk_fields(config, runtime, caplog) -> None:
    """fixture 运行时首笔分数为 0.05（低于阈值），两条规则都命中 -> 2 点 -> elevated。"""
    client, _, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.get("/v1/policy")
            client.post(
                "/v1/risk/evaluate",
                json={"features": risk_features(runtime, transaction_amt=2000.0, c13=900.0)},
            )
            client.post("/v1/risk/evaluate/batch", json=_risk_batch(runtime, [("a", 2000.0, 1.0)]))
    records = _completion_records(caplog)
    assert len(records) == 3, "每个请求恰好一条 completion log"
    risk_records = [record for record in records if record["route"] == "/v1/risk/evaluate"]
    assert len(risk_records) == 1
    assert risk_records[0]["policy_version"] == "cw006-risk-policy-v1"
    assert risk_records[0]["risk_level"] == "elevated_signal"
    assert risk_records[0]["rule_match_count"] == 2
    # 批量没有"一个"风险等级，因此不记 risk_level，只记命中总数。
    batch_records = [
        record for record in records if record["route"] == "/v1/risk/evaluate/batch"
    ]
    assert "risk_level" not in batch_records[0]
    assert batch_records[0]["rule_match_count"] == 1
    assert batch_records[0]["policy_version"] == "cw006-risk-policy-v1"
    assert set(risk_records[0]) <= {
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


def test_completion_log_never_contains_rule_ids_scores_or_payloads(config, caplog) -> None:
    runtime = FakeRuntime(features=FEATURES, scores=[0.987654])
    client, _, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.post(
                "/v1/risk/evaluate",
                json={"features": risk_features(runtime, transaction_amt=2000.0, c13=900.0)},
            )
            client.post("/v1/risk/evaluate/batch", json=_risk_batch(runtime, [(SENTINEL, 1.0, 1.0)]))
    blob = "\n".join(record.getMessage() for record in caplog.records)
    for banned in (
        AMT_RULE,
        C13_RULE,
        "0.987654",
        "2000",
        "TransactionAmt",
        "C13",
        "features",
        SENTINEL,
        "matched_rule_ids",
        "unevaluable_rule_ids",
        "rule_score",
    ):
        assert banned not in blob, banned


def test_log_route_never_contains_the_raw_url_on_risk_paths(config, runtime, caplog) -> None:
    client, _, _ = _make(config, runtime)
    with caplog.at_level(logging.INFO, logger="ai_detector.service"):
        with client:
            caplog.clear()
            client.get(f"/v1/risk/{SENTINEL}")
    payload = _completion_records(caplog)[0]
    assert payload["route"] == "<unmatched>"
    assert SENTINEL not in json.dumps(payload)


# --- CW-005 回归与 OpenAPI ---------------------------------------------


def test_cw005_endpoints_keep_their_exact_success_contract(config, runtime) -> None:
    """政策正常时，CW-005 五个接口的路径与成功响应字段必须逐字不变。"""
    client, _, _ = _make(config, runtime)
    with client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        model = client.get("/v1/model")
        score = client.post("/v1/score", json={"features": risk_features(runtime)})
        batch = client.post(
            "/v1/score/batch",
            json={
                "items": [
                    {"client_item_id": "a", "features": risk_features(runtime)},
                    {"client_item_id": "b", "features": risk_features(runtime)},
                ]
            },
        )
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
    assert set(batch.json()) == {
        "request_id",
        "model_version",
        "score_semantics",
        "threshold",
        "threshold_policy",
        "batch_size",
        "results",
    }
    assert set(batch.json()["results"][0]) == {"client_item_id", "score", "flagged"}


def test_openapi_exposes_the_five_cw005_paths_plus_the_three_new_ones(config, runtime) -> None:
    """CW-007 之后期望集合扩展为 11 个（新增三个 ``/v1/decisions/*``）；断言仍是完全相等。"""
    app = create_app(config, runtime_loader=RecordingLoader(runtime))
    paths = app.openapi()["paths"]
    assert set(paths) == {
        "/health/live",
        "/health/ready",
        "/v1/model",
        "/v1/score",
        "/v1/score/batch",
        "/v1/policy",
        "/v1/risk/evaluate",
        "/v1/risk/evaluate/batch",
        "/v1/decisions/evaluate",
        "/v1/decisions/evaluate/batch",
        "/v1/decisions/{decision_id}",
    }
    assert set(paths["/v1/policy"]) == {"get"}
    assert set(paths["/v1/risk/evaluate"]) == {"post"}
    assert set(paths["/v1/risk/evaluate/batch"]) == {"post"}
