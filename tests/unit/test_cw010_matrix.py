"""CW-010 单元测试：故障矩阵的完整性与夹具自身的正确性。

这一层刻意**不发 HTTP 请求**。它守的是两件事：

1. **矩阵不退化**：验收标准要求覆盖的注入点（三类启动失败、三种组合故障、模型/政策/
   存储的请求期未预期异常、三种 ``TimeoutError``）必须都在矩阵里，且每行给出的接口、
   状态码与错误码必须自洽——readiness 的"模型 → 政策 → 存储"优先级由行数据**推导**出来
   并与行里声明的取值比对，而不是靠人肉核对。
2. **夹具不撒谎**：故障注入层确实拦住了调用、确实没有把调用透传给真实实现、故障解除后
   确实恢复；泄漏断言确实能抓到哨兵（否则它会变成一个永远通过的假保护）。

行为本身（真实状态码、原子性、脱敏）由 ``tests/integration/test_cw010_*.py`` 在 HTTP
层验证。
"""

from __future__ import annotations

import pytest

from ai_detector.service.errors import ERROR_MESSAGES, ServiceErrorCode
from ai_detector.service.logging import ALLOWED_LOG_FIELDS
from ai_detector.storage.sqlite import StorageUnavailableError
from tests.cw010_fixtures import (
    FAULT_MATRIX,
    SENTINEL_FEATURE_VALUE,
    SENTINEL_ITEM_ID,
    SENTINEL_KEY,
    SENTINEL_MESSAGE,
    FaultInjectingRepository,
    FlakyPolicyEngine,
    FlakyRuntime,
    NthCallFault,
    OperationFault,
    assert_error_body_is_safe,
    assert_no_leak,
    case,
)
from tests.cw005_fixtures import FakeRuntime
from tests.cw006_fixtures import FakePolicyEngine

#: 全部对外路径（含路径参数占位形式），矩阵只允许引用这些。
CANONICAL_PATHS = frozenset(
    {
        "/health/live",
        "/health/ready",
        "/v1/model",
        "/v1/policy",
        "/v1/score",
        "/v1/score/batch",
        "/v1/risk/evaluate",
        "/v1/risk/evaluate/batch",
        "/v1/decisions/evaluate",
        "/v1/decisions/evaluate/batch",
        "/v1/decisions/{id}",
    }
)

#: 组件 → 未就绪时对外开放的错误码。
COMPONENT_CODE = {
    "model": ServiceErrorCode.MODEL_NOT_READY,
    "policy": ServiceErrorCode.POLICY_NOT_READY,
    "storage": ServiceErrorCode.STORAGE_NOT_READY,
}

#: 允许出现在 500 行的错误码：未预期异常折叠为 internal_error；
#: 结构/契约损坏折叠为对应的 *_contract_violation。
ALLOWED_500_CODES = frozenset(
    {
        ServiceErrorCode.INTERNAL_ERROR,
        ServiceErrorCode.POLICY_CONTRACT_VIOLATION,
        ServiceErrorCode.STORAGE_CONTRACT_VIOLATION,
        ServiceErrorCode.IDEMPOTENCY_CONTRACT_VIOLATION,
    }
)

#: 允许出现在 503 行的错误码。
ALLOWED_503_CODES = frozenset(
    {
        ServiceErrorCode.MODEL_NOT_READY,
        ServiceErrorCode.POLICY_NOT_READY,
        ServiceErrorCode.STORAGE_NOT_READY,
        ServiceErrorCode.STORAGE_UNAVAILABLE,
    }
)

#: readiness 的固定优先级：越靠前越先阻塞。
READINESS_ORDER = ("model", "policy", "storage")


# --- 覆盖完整性：验收要求过的注入点一个都不能少 --------------------------


def test_matrix_keys_are_unique() -> None:
    keys = [row.key for row in FAULT_MATRIX]
    assert len(keys) == len(set(keys)), "矩阵 key 必须唯一，否则用例会互相覆盖"


def test_matrix_covers_every_required_injection_point() -> None:
    """三类启动失败 + 三种组合故障，缺一不可。"""
    required = {
        "startup.model",
        "startup.policy",
        "startup.storage",
        "startup.model+policy",
        "startup.policy+storage",
        "startup.model+storage",
    }
    assert required <= {row.key for row in FAULT_MATRIX}


def test_matrix_covers_request_time_faults_per_component() -> None:
    """模型、政策、存储各至少一次"未预期异常"，且各至少一次 ``TimeoutError``。"""
    request_rows = [row for row in FAULT_MATRIX if row.stage == "request"]
    for component in ("model", "policy", "storage"):
        rows = [row for row in request_rows if row.component == component]
        assert rows, f"{component} 缺少请求期故障行"
        assert any(row.operation.endswith("RuntimeError") or "未预期" in row.operation for row in rows), (
            f"{component} 缺少请求期未预期异常行"
        )
        assert any("TimeoutError" in row.operation for row in rows), (
            f"{component} 缺少请求期 TimeoutError 行"
        )


def test_matrix_has_a_timeout_row_for_each_component() -> None:
    timeouts = {row.component for row in FAULT_MATRIX if "TimeoutError" in row.operation}
    assert timeouts == {"model", "policy", "storage"}, "三个组件都必须有 TimeoutError 覆盖"


def test_matrix_distinguishes_classified_storage_failure_from_a_bare_exception() -> None:
    """存储的裸异常与**已分类**失败必须都覆盖：两者的对外码不同（500 vs 503）。"""
    assert set(case("request.storage.commit.timeout").codes) == {ServiceErrorCode.INTERNAL_ERROR}
    assert set(case("request.storage.commit.rollback").codes) == {
        ServiceErrorCode.STORAGE_UNAVAILABLE
    }
    assert set(case("request.storage.lookup.timeout").codes) == {ServiceErrorCode.INTERNAL_ERROR}
    assert set(case("request.storage.lookup.unavailable").codes) == {
        ServiceErrorCode.STORAGE_UNAVAILABLE
    }
    assert set(case("request.storage.insert.unexpected").codes) == {
        ServiceErrorCode.INTERNAL_ERROR
    }
    assert set(case("request.storage.insert.unavailable").codes) == {
        ServiceErrorCode.STORAGE_UNAVAILABLE
    }


# --- 行内自洽性 ----------------------------------------------------------


def test_every_row_references_real_endpoints() -> None:
    for row in FAULT_MATRIX:
        referenced = {path for path, _, _ in row.affected} | set(row.unaffected)
        assert referenced <= CANONICAL_PATHS, f"{row.key} 引用了不存在的路径"


def test_affected_and_unaffected_are_disjoint() -> None:
    for row in FAULT_MATRIX:
        affected = {path for path, _, _ in row.affected}
        overlap = affected & set(row.unaffected)
        assert not overlap, f"{row.key} 把 {overlap} 同时列为受影响与未受影响"


def test_live_endpoint_is_never_degraded() -> None:
    """``/health/live`` 只表示进程存活，任何故障都不得让它降级。"""
    for row in FAULT_MATRIX:
        assert "/health/live" in row.unaffected, f"{row.key} 错误地影响了 /health/live"


def test_status_and_code_pairs_are_from_the_frozen_vocabulary() -> None:
    for row in FAULT_MATRIX:
        for path, status, code in row.affected:
            assert code in ERROR_MESSAGES, f"{row.key} 的 {code} 不在冻结错误码表里"
            if status == 500:
                assert code in ALLOWED_500_CODES, f"{row.key} 的 500 码 {code} 不合契约"
            elif status == 503:
                assert code in ALLOWED_503_CODES, f"{row.key} 的 503 码 {code} 不合契约"
            else:
                pytest.fail(f"{row.key} 的受影响接口 {path} 状态码 {status} 不是错误状态")


def test_log_fields_stay_inside_the_whitelist() -> None:
    """日志分类字段必须白名单内——矩阵不能承诺记录任何未授权字段。"""
    for row in FAULT_MATRIX:
        unknown = set(row.log_fields) - ALLOWED_LOG_FIELDS
        assert not unknown, f"{row.key} 声明的日志字段 {unknown} 不在白名单内"


def test_every_row_states_a_write_effect() -> None:
    for row in FAULT_MATRIX:
        assert row.write_effect.strip(), f"{row.key} 未说明预期写入效果"


# --- readiness 优先级由矩阵推导 ------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["startup.model+policy", "startup.policy+storage", "startup.model+storage"],
)
def test_readiness_reports_the_first_blocked_component(key: str) -> None:
    """组合故障下 readiness 的码必须是"模型 → 政策 → 存储"里最靠前的那个组件。"""
    row = case(key)
    components = key.removeprefix("startup.").split("+")
    expected = COMPONENT_CODE[components[0]]
    ready = dict((path, code) for path, _, code in row.affected)
    assert ready["/health/ready"] == expected


def test_readiness_order_is_covered_in_full() -> None:
    """优先级链条的三段（模型压政策、政策压存储、模型压存储）都必须有用例。

    单组件行同样参与：它给出"只有该组件坏掉"这一段的基准取值。
    """
    seen = set()
    for row in (item for item in FAULT_MATRIX if item.stage == "startup"):
        components = row.component.split("+")
        ready = {path: code for path, _, code in row.affected}.get("/health/ready")
        assert ready is not None, f"{row.key} 缺少 /health/ready"
        winner = next(name for name in READINESS_ORDER if name in components)
        assert COMPONENT_CODE[winner] == ready, f"{row.key} 的 readiness 码与优先级不符"
        seen.add(winner)
    assert seen == {"model", "policy", "storage"}


def test_case_lookup_fails_loudly_for_an_unknown_key() -> None:
    with pytest.raises(KeyError):
        case("startup.does-not-exist")


# --- 夹具自证：故障注入确实生效，且确实能恢复 ----------------------------


def _stub_delegate() -> tuple[FaultInjectingRepository, list[str]]:
    seen: list[str] = []

    class Delegate:
        def insert_many(self, records):  # pragma: no cover - 计数器由外层断言
            seen.append("insert_many")

        def get(self, decision_id):
            seen.append("get")
            return None

        def lookup_idempotent(self, *args):
            seen.append("lookup_idempotent")
            return None

        def commit_idempotent(self, *args):
            seen.append("commit_idempotent")
            return None

        def close(self):
            seen.append("close")

    return FaultInjectingRepository(Delegate()), seen


def test_operation_fault_defaults_to_failing_exactly_once() -> None:
    fault = OperationFault(TimeoutError(SENTINEL_MESSAGE))
    with pytest.raises(TimeoutError):
        fault.maybe_raise()
    assert not fault.active
    fault.maybe_raise()  # 故障已解除：不再抛出
    assert fault.raised == 1


def test_operation_fault_can_stay_active_forever() -> None:
    fault = OperationFault(TimeoutError(SENTINEL_MESSAGE), times=OperationFault.ALWAYS)
    for _ in range(3):
        with pytest.raises(TimeoutError):
            fault.maybe_raise()
    assert fault.active and fault.raised == 3


def test_injected_fault_blocks_the_call_and_never_reaches_the_delegate() -> None:
    """注入故障时真实实现**一次都不该被调用**——否则"零残留"就无从谈起。"""
    repository, seen = _stub_delegate()
    repository._faults["insert_many"] = OperationFault(
        StorageUnavailableError(SENTINEL_MESSAGE), times=OperationFault.ALWAYS
    )
    with pytest.raises(StorageUnavailableError):
        repository.insert_many([])
    assert seen == [], "故障被注入时不得透传到真实实现"
    assert repository.calls["insert_many"] == 1
    assert repository.delivered["insert_many"] == 0


def test_release_restores_the_delegate_immediately() -> None:
    repository, seen = _stub_delegate()
    repository._faults["get"] = OperationFault(RuntimeError("boom"), times=OperationFault.ALWAYS)
    with pytest.raises(RuntimeError):
        repository.get("x")
    repository.release("get")
    repository.get("x")
    assert seen == ["get"]
    assert repository.delivered["get"] == 1


def test_nth_call_fault_only_trips_exactly_once_at_the_given_index() -> None:
    fault = NthCallFault(2, TimeoutError(SENTINEL_MESSAGE))
    fault.check()  # 第 1 次：透传
    with pytest.raises(TimeoutError):
        fault.check()  # 第 2 次：抛出
    fault.check()  # 第 3 次：透传


def test_flaky_runtime_delegates_attributes_and_counts_real_calls() -> None:
    inner = FakeRuntime(features=["a", "b"], scores=[0.25, 0.75])
    fault = NthCallFault(2, TimeoutError(SENTINEL_MESSAGE))
    flaky = FlakyRuntime(inner, fault)
    assert flaky.source_features == ["a", "b"], "未拦截的属性必须透传"
    assert flaky.score_many([{}, {}]) == [0.25, 0.75]
    with pytest.raises(TimeoutError):
        flaky.score_many([{}])
    assert flaky.calls == 2, "被拦下的那一次也算收到过调用"
    assert flaky.call_count == 1, "底层 fake 只应看到真正执行的那一次"
    assert fault.calls == 2


def test_flaky_policy_engine_delegates_attributes_and_counts_real_calls() -> None:
    inner = FakePolicyEngine()
    fault = NthCallFault(1, TimeoutError(SENTINEL_MESSAGE))
    flaky = FlakyPolicyEngine(inner, fault)
    assert flaky.policy_version == inner.policy_version
    with pytest.raises(TimeoutError):
        flaky.evaluate_rules({"a": 1.0})
    assert flaky.calls == 1
    assert flaky.call_count == 0, "被拦下的调用不得到达底层引擎"


# --- 泄漏断言自身有效 ----------------------------------------------------


def test_assert_no_leak_accepts_a_clean_payload() -> None:
    assert_no_leak('{"error":{"code":"internal_error","message":"服务内部错误，请稍后重试。"}}')


@pytest.mark.parametrize(
    "leaky",
    [
        f"score was {SENTINEL_FEATURE_VALUE}",
        f"key {SENTINEL_KEY} rejected",
        f"TimeoutError: {SENTINEL_MESSAGE}",
        "artifacts/runtime/cw007_decisions.sqlite3",
        "Traceback (most recent call last):",
    ],
)
def test_assert_no_leak_rejects_each_leak_class(leaky: str) -> None:
    with pytest.raises(AssertionError):
        assert_no_leak(leaky)


def test_success_body_may_echo_client_item_id_but_error_body_may_not() -> None:
    """``client_item_id`` 在成功响应里是契约要求回显的；错误响应回显它才是泄漏。"""
    echoed = f'{{"results":[{{"client_item_id":"{SENTINEL_ITEM_ID}"}}]}}'
    assert_no_leak(echoed)  # 成功形状：允许
    with pytest.raises(AssertionError):
        assert_error_body_is_safe(echoed)  # 错误形状：禁止
