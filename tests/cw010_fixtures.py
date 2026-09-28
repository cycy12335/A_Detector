"""CW-010 测试夹具：跨组件故障注入、故障矩阵与泄漏哨兵。

设计要点：

- **真实实现始终在底层**：:class:`FaultInjectingRepository` 包住一个由生产入口
  :func:`~ai_detector.service.runtime.load_storage_repository` 打开的真实 SQLite，
  因此"回滚后零残留"是在真库上验证的，而不是靠 fake 的承诺。故障注入只发生在
  调用边界，不替换业务语义。
- **故障可自动解除**：:class:`OperationFault` 默认只失败一次，之后透传。这让"故障中"
  与"故障解除后重试"能在同一个 app 生命周期里对照，而不是靠两个互不相干的用例。
- **中途故障**：:class:`NthCallFault` 在第 N 次调用抛错，用于构造"批量走到一半才失败"
  这种最容易被写成部分结果的场景。
- **故障矩阵是数据**：:data:`FAULT_MATRIX` 把注入点、受影响/未受影响接口、状态码、
  稳定错误码、日志分类与预期写入效果写成可被测试遍历的结构；`test_cw010_matrix.py`
  对它做一致性校验，因此矩阵不会与实现悄悄漂移。
- 全部数据都是项目自建合成值，不访问网络，不读取真实数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_detector.config import PROJECT_ROOT
from ai_detector.service.runtime import load_storage_repository

# --------------------------------------------------------------------------
# 泄漏哨兵：任何一条都不允许出现在对外响应、日志或结果文档里
# --------------------------------------------------------------------------

#: 特征取值哨兵（故意是可识别的大数，回显时一眼能看出）。
SENTINEL_FEATURE_VALUE = 987654.321

#: 幂等键哨兵（合法形状，因此能走到幂等路径）。
SENTINEL_KEY = "cw010-sentinel-key-0001"

#: 异常原文哨兵：注入的异常消息里带上它，用来证明原文没有被折叠进响应或日志。
SENTINEL_MESSAGE = "SENTINEL_CW010_EXC_4b2e"

#: 客户端条目标识哨兵。
#:
#: 注意：``client_item_id`` 是**契约要求回显**的调用方关联 ID（批量成功响应与决策快照里
#: 都会原样返回），因此它不属于"泄漏"。只有**错误响应**里出现它才是泄漏——用
#: :func:`assert_error_body_is_safe` 断言。
SENTINEL_ITEM_ID = "SENTINEL-CW010-ITEM"

#: 结果文档与响应里都不允许出现的"内部结构"片段。
FORBIDDEN_FRAGMENTS = (
    "artifacts/",
    "artifacts\\",
    ".sqlite3",
    ".joblib",
    ".py",
    "Traceback",
    "BEGIN IMMEDIATE",
    "SELECT ",
    "INSERT INTO",
)


def build_leaky_error(kind: type[BaseException] = RuntimeError) -> BaseException:
    """构造一个**消息里带哨兵**的异常，用于验证原文不会被外泄。"""
    return kind(f"{SENTINEL_MESSAGE}: 绝对路径 {PROJECT_ROOT} 与内部细节")


def assert_no_leak(text: str, *, extra: Sequence[str] = ()) -> None:
    """断言一段对外可见文本不含任何哨兵、绝对路径或内部结构片段。

    只检查"是否泄漏"，不假设具体措辞，因此对错误折叠方式的改动是稳健的。
    刻意**不**包含 :data:`SENTINEL_ITEM_ID`：它在成功响应里是契约要求回显的字段。
    """
    normalized = text.replace("\\\\", "\\")
    for sentinel in (
        str(SENTINEL_FEATURE_VALUE),
        SENTINEL_KEY,
        SENTINEL_MESSAGE,
        *extra,
    ):
        assert sentinel not in normalized, f"对外文本泄漏了哨兵：{sentinel!r}"
    assert str(PROJECT_ROOT) not in normalized, "对外文本泄漏了项目绝对路径"
    assert PROJECT_ROOT.name not in normalized, "对外文本泄漏了项目目录名"
    for fragment in FORBIDDEN_FRAGMENTS:
        assert fragment not in normalized, f"对外文本泄漏了内部结构片段：{fragment!r}"


def assert_error_body_is_safe(text: str) -> None:
    """错误响应的脱敏断言：在 :func:`assert_no_leak` 之上，额外禁止回显请求体内容。

    成功响应回显 ``client_item_id`` 是契约；错误响应回显它就把调用方数据写进了错误面。
    """
    assert_no_leak(text, extra=(SENTINEL_ITEM_ID,))


# --------------------------------------------------------------------------
# 故障注入：按操作名
# --------------------------------------------------------------------------


class OperationFault:
    """在某个操作上注入故障；默认只失败一次，之后自动透传。

    Args:
        error: 注入的异常实例（每次抛出**同一个**实例，便于断言它是被同一个异常触发的）。
        times: 失败次数；``OperationFault.ALWAYS`` 表示一直失败，``1``（默认）表示
            "故障中一次、随后解除"，用于区分故障中与故障解除后的行为。
    """

    #: ``times`` 取该值表示永不自动恢复。
    ALWAYS = -1

    def __init__(self, error: BaseException, *, times: int = 1) -> None:
        self.error = error
        self.remaining = int(times)
        self.raised = 0

    @property
    def active(self) -> bool:
        return self.remaining != 0

    def maybe_raise(self) -> None:
        """按剩余次数决定是否抛出；调用方保证每个操作只调用一次。"""
        if self.remaining == 0:
            return
        if self.remaining > 0:
            self.remaining -= 1
        self.raised += 1
        raise self.error


class FaultInjectingRepository:
    """包装**真实** repository，在指定操作上注入故障并统计调用。

    包装层只做两件事：数调用、按计划抛异常；其余一律透传给真实实现。因此
    "故障解除后能正常重试""失败后零残留"验证的都是真实存储行为。
    """

    OPERATIONS = ("insert_many", "get", "lookup_idempotent", "commit_idempotent")

    def __init__(
        self,
        delegate: Any,
        faults: Mapping[str, OperationFault] | None = None,
    ) -> None:
        self._delegate = delegate
        self._faults: dict[str, OperationFault] = dict(faults or {})
        #: 每个操作被**调用**的次数（含被故障拦下的那次）。
        self.calls: dict[str, int] = {name: 0 for name in self.OPERATIONS}
        #: 每个操作**真正到达**真实实现的次数。
        self.delivered: dict[str, int] = {name: 0 for name in self.OPERATIONS}
        self.close_calls = 0

    @property
    def delegate(self) -> Any:
        """底层真实 repository（供需要直接查库/改库的用例使用）。"""
        return self._delegate

    def fault(self, operation: str) -> OperationFault | None:
        return self._faults.get(operation)

    def release(self, operation: str | None = None) -> None:
        """解除故障（``None`` 表示全部解除）。用于显式构造"故障解除后"的阶段。"""
        if operation is None:
            self._faults.clear()
        else:
            self._faults.pop(operation, None)

    def _guard(self, operation: str) -> None:
        self.calls[operation] += 1
        fault = self._faults.get(operation)
        if fault is not None:
            fault.maybe_raise()
        self.delivered[operation] += 1

    # --- 逐操作透传 -----------------------------------------------------

    def insert_many(self, records: Sequence[Any]) -> None:
        self._guard("insert_many")
        self._delegate.insert_many(records)

    def get(self, decision_id: str) -> Any:
        self._guard("get")
        return self._delegate.get(decision_id)

    def lookup_idempotent(self, *args: Any, **kwargs: Any) -> Any:
        self._guard("lookup_idempotent")
        return self._delegate.lookup_idempotent(*args, **kwargs)

    def commit_idempotent(self, *args: Any, **kwargs: Any) -> Any:
        self._guard("commit_idempotent")
        return self._delegate.commit_idempotent(*args, **kwargs)

    def close(self) -> None:
        self.close_calls += 1
        self._delegate.close()


class FaultingStorageLoader:
    """用**生产加载器**打开真实 SQLite，再包上故障注入层。

    刻意不自己拼路径：数据库位置策略（项目相对 + 限定 ``artifacts/runtime/*.sqlite3``）
    仍然由 :func:`load_storage_repository` 决定，因此测试跑的是与生产同一条链路。
    """

    def __init__(
        self,
        *,
        faults: Mapping[str, OperationFault] | None = None,
        cache: bool = True,
    ) -> None:
        self._faults = dict(faults or {})
        self._cache = cache
        self.calls = 0
        self.repository: FaultInjectingRepository | None = None

    def fault(self, operation: str) -> OperationFault:
        """取（或创建）某个操作的故障计划，供用例在启动后再安排注入。"""
        if operation not in self._faults:
            self._faults[operation] = OperationFault(RuntimeError("未指定故障"))
        return self._faults[operation]

    def __call__(self, config: Any) -> FaultInjectingRepository:
        self.calls += 1
        if self.repository is not None and self._cache:
            return self.repository
        inner = load_storage_repository(config)
        self.repository = FaultInjectingRepository(inner, self._faults)
        return self.repository


# --------------------------------------------------------------------------
# 故障注入：按调用次数（用于"批量中途失败"）
# --------------------------------------------------------------------------


class NthCallFault:
    """在第 ``nth`` 次调用（1-based）抛错，其余调用透传。"""

    def __init__(self, nth: int, error: BaseException) -> None:
        self.nth = int(nth)
        self.error = error
        self.calls = 0

    def check(self) -> None:
        self.calls += 1
        if self.calls == self.nth:
            raise self.error


class FlakyRuntime:
    """透传一个 fake runtime 的全部属性，只在第 N 次 ``score_many`` 抛错。

    用于构造"批量评分走到一半才失败"——这正是部分结果最容易泄漏出来的位置。

    ``calls`` 是**收到的调用次数**（含被拦下的那次）；``call_count`` 经 ``__getattr__``
    透传给底层 fake，因此仍是**真正执行了评分**的次数。两者的差值就是被故障吃掉的那些。
    """

    def __init__(self, runtime: Any, fault: NthCallFault | None = None) -> None:
        self._runtime = runtime
        self._fault = fault
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def score_many(self, rows: Sequence[dict[str, Any]]) -> list[float]:
        self.calls += 1
        if self._fault is not None:
            self._fault.check()
        return self._runtime.score_many(rows)


class FlakyPolicyEngine:
    """透传一个 fake engine 的全部属性，只在第 N 次 ``evaluate_rules`` 抛错。

    ``calls`` / ``call_count`` 的语义同 :class:`FlakyRuntime`。
    """

    def __init__(self, engine: Any, fault: NthCallFault | None = None) -> None:
        self._engine = engine
        self._fault = fault
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def evaluate_rules(self, features: Mapping[str, Any]) -> Any:
        self.calls += 1
        if self._fault is not None:
            self._fault.check()
        return self._engine.evaluate_rules(features)


# --------------------------------------------------------------------------
# 接口集合：用于"未受影响的接口必须照常工作"的逐项断言
# --------------------------------------------------------------------------

#: CW-005 五个只读接口 + CW-006 三个风险接口 + CW-007 三个 decisions 接口。
MODEL_ONLY_ENDPOINTS = ("/v1/model",)
POLICY_ENDPOINTS = ("/v1/policy",)
SCORING_ENDPOINTS = ("/v1/score", "/v1/score/batch")
RISK_ENDPOINTS = ("/v1/risk/evaluate", "/v1/risk/evaluate/batch")
DECISION_WRITE_ENDPOINTS = ("/v1/decisions/evaluate", "/v1/decisions/evaluate/batch")


# --------------------------------------------------------------------------
# 故障矩阵（可被测试遍历的数据）
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FaultCase:
    """故障矩阵的一行。

    Args:
        key: 注入点标识（测试按它交叉引用）。
        stage: ``"startup"``（生命周期加载阶段）或 ``"request"``（请求期）。
        component: ``"model"`` / ``"policy"`` / ``"storage"``。
        operation: 具体被注入的操作名。
        affected: 受影响接口及其 ``(状态码, 稳定错误码)``。
        unaffected: 必须照常工作的接口（隔离不变量）。
        write_effect: 预期写入效果。
        log_fields: 该场景下日志里允许出现的**分类**字段。
        preexisting: 旧测试已有覆盖的位置；``None`` 表示本包新增覆盖。
    """

    key: str
    stage: str
    component: str
    operation: str
    affected: tuple[tuple[str, int, str], ...]
    unaffected: tuple[str, ...]
    write_effect: str
    log_fields: tuple[str, ...]
    preexisting: str | None

    @property
    def statuses(self) -> tuple[int, ...]:
        return tuple(status for _, status, _ in self.affected)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(code for _, _, code in self.affected)


#: 启动阶段：单个组件失败。
_STARTUP_MODEL = FaultCase(
    key="startup.model",
    stage="startup",
    component="model",
    operation="load_runtime",
    affected=(
        ("/health/ready", 503, "model_not_ready"),
        ("/v1/model", 503, "model_not_ready"),
        ("/v1/score", 503, "model_not_ready"),
        ("/v1/score/batch", 503, "model_not_ready"),
        ("/v1/risk/evaluate", 503, "model_not_ready"),
        ("/v1/risk/evaluate/batch", 503, "model_not_ready"),
        ("/v1/decisions/evaluate", 503, "model_not_ready"),
        ("/v1/decisions/evaluate/batch", 503, "model_not_ready"),
    ),
    unaffected=("/health/live", "/v1/policy", "/v1/decisions/{id}"),
    write_effect="零写入：求值阶段就失败，触不到存储",
    log_fields=("event", "outcome", "error_code"),
    preexisting="test_cw005_service.py::test_degraded_startup_keeps_live_but_fails_ready_and_scoring",
)

_STARTUP_POLICY = FaultCase(
    key="startup.policy",
    stage="startup",
    component="policy",
    operation="load_policy_engine",
    affected=(
        ("/health/ready", 503, "policy_not_ready"),
        ("/v1/policy", 503, "policy_not_ready"),
        ("/v1/risk/evaluate", 503, "policy_not_ready"),
        ("/v1/risk/evaluate/batch", 503, "policy_not_ready"),
        ("/v1/decisions/evaluate", 503, "policy_not_ready"),
        ("/v1/decisions/evaluate/batch", 503, "policy_not_ready"),
    ),
    unaffected=("/health/live", "/v1/model", "/v1/score", "/v1/score/batch", "/v1/decisions/{id}"),
    write_effect="零写入：政策不可用时 decisions 接口在求值前失败",
    log_fields=("event", "outcome", "policy_error_code"),
    preexisting="test_cw006_risk_service.py::test_policy_failure_makes_the_new_endpoints_503",
)

_STARTUP_STORAGE = FaultCase(
    key="startup.storage",
    stage="startup",
    component="storage",
    operation="load_storage_repository",
    affected=(
        ("/health/ready", 503, "storage_not_ready"),
        ("/v1/decisions/evaluate", 503, "storage_not_ready"),
        ("/v1/decisions/evaluate/batch", 503, "storage_not_ready"),
        ("/v1/decisions/{id}", 503, "storage_not_ready"),
    ),
    unaffected=(
        "/health/live",
        "/v1/model",
        "/v1/policy",
        "/v1/score",
        "/v1/score/batch",
        "/v1/risk/evaluate",
        "/v1/risk/evaluate/batch",
    ),
    write_effect="零写入：不创建、不覆盖、不回退到内存库",
    log_fields=("event", "outcome", "storage_error_code"),
    preexisting="test_cw007_decisions_service.py::test_storage_failure_makes_decisions_503",
)

#: 启动阶段：组合故障（本包新增覆盖）。readiness 优先级固定为 模型 → 政策 → 存储。
_STARTUP_COMBINED = (
    FaultCase(
        key="startup.model+policy",
        stage="startup",
        component="model+policy",
        operation="load_runtime + load_policy_engine",
        affected=(
            ("/health/ready", 503, "model_not_ready"),
            ("/v1/policy", 503, "policy_not_ready"),
            ("/v1/model", 503, "model_not_ready"),
        ),
        unaffected=("/health/live",),
        write_effect="零写入",
        log_fields=("event", "outcome", "error_code", "policy_error_code"),
        preexisting=None,
    ),
    FaultCase(
        key="startup.policy+storage",
        stage="startup",
        component="policy+storage",
        operation="load_policy_engine + load_storage_repository",
        affected=(
            ("/health/ready", 503, "policy_not_ready"),
            ("/v1/policy", 503, "policy_not_ready"),
            ("/v1/risk/evaluate", 503, "policy_not_ready"),
            ("/v1/decisions/evaluate", 503, "policy_not_ready"),
            ("/v1/decisions/evaluate/batch", 503, "storage_not_ready"),
        ),
        unaffected=("/health/live", "/v1/model", "/v1/score", "/v1/score/batch"),
        write_effect="零写入",
        log_fields=("event", "outcome", "error_code", "policy_error_code", "storage_error_code"),
        preexisting=None,
    ),
    FaultCase(
        key="startup.model+storage",
        stage="startup",
        component="model+storage",
        operation="load_runtime + load_storage_repository",
        affected=(
            ("/health/ready", 503, "model_not_ready"),
            ("/v1/model", 503, "model_not_ready"),
            ("/v1/score", 503, "model_not_ready"),
            ("/v1/decisions/evaluate", 503, "model_not_ready"),
            ("/v1/decisions/evaluate/batch", 503, "storage_not_ready"),
            ("/v1/decisions/{id}", 503, "storage_not_ready"),
        ),
        unaffected=("/health/live", "/v1/policy"),
        write_effect="零写入",
        log_fields=("event", "outcome", "error_code", "storage_error_code"),
        preexisting=None,
    ),
)

#: 请求期：模型。
_REQUEST_MODEL = (
    FaultCase(
        key="request.model.timeout",
        stage="request",
        component="model",
        operation="score_many -> TimeoutError",
        affected=tuple(
            (path, 500, "internal_error")
            for path in (*SCORING_ENDPOINTS, *RISK_ENDPOINTS, *DECISION_WRITE_ENDPOINTS)
        ),
        unaffected=(
            "/health/live",
            "/health/ready",
            "/v1/model",
            "/v1/policy",
            "/v1/decisions/{id}",
        ),
        write_effect="零写入：评分失败发生在记录构造与写入之前",
        log_fields=("event", "outcome", "status_code"),
        preexisting="test_cw005_service.py::test_unexpected_model_error_becomes_a_safe_500（非超时）",
    ),
    FaultCase(
        key="request.model.unexpected",
        stage="request",
        component="model",
        operation="score_many -> RuntimeError",
        affected=tuple(
            (path, 500, "internal_error")
            for path in (*SCORING_ENDPOINTS, *RISK_ENDPOINTS, *DECISION_WRITE_ENDPOINTS)
        ),
        unaffected=("/health/live", "/health/ready", "/v1/model", "/v1/policy"),
        write_effect="零写入",
        log_fields=("event", "outcome", "status_code"),
        preexisting="test_cw005_service.py::test_unexpected_model_error_becomes_a_safe_500",
    ),
)

#: 请求期：政策。
_REQUEST_POLICY = (
    FaultCase(
        key="request.policy.timeout",
        stage="request",
        component="policy",
        operation="evaluate_rules -> TimeoutError",
        affected=tuple(
            (path, 500, "internal_error") for path in (*RISK_ENDPOINTS, *DECISION_WRITE_ENDPOINTS)
        ),
        unaffected=(
            "/health/live",
            "/health/ready",
            "/v1/model",
            "/v1/policy",
            *SCORING_ENDPOINTS,
        ),
        write_effect="零写入：规则求值在模型调用与写入之前",
        log_fields=("event", "outcome", "status_code"),
        preexisting="test_cw006_risk_service.py::test_engine_exception_is_folded_into_a_safe_500（非超时）",
    ),
    FaultCase(
        key="request.policy.unexpected",
        stage="request",
        component="policy",
        operation="evaluate_rules -> RuntimeError",
        affected=tuple(
            (path, 500, "internal_error") for path in (*RISK_ENDPOINTS, *DECISION_WRITE_ENDPOINTS)
        ),
        unaffected=("/health/live", "/health/ready", "/v1/model", "/v1/policy", *SCORING_ENDPOINTS),
        write_effect="零写入",
        log_fields=("event", "outcome", "status_code"),
        preexisting="test_cw006_risk_service.py::test_engine_exception_is_folded_into_a_safe_500",
    ),
)

#: 请求期：存储。未分类的裸异常折叠为 500；存储层**已分类**的失败折叠为 503。
_REQUEST_STORAGE = (
    FaultCase(
        key="request.storage.insert.timeout",
        stage="request",
        component="storage",
        operation="insert_many -> TimeoutError",
        affected=tuple(
            (path, 500, "internal_error") for path in DECISION_WRITE_ENDPOINTS
        ),
        unaffected=(
            "/health/live",
            "/health/ready",
            "/v1/model",
            "/v1/policy",
            *SCORING_ENDPOINTS,
            *RISK_ENDPOINTS,
        ),
        write_effect="零写入：仓储层回滚，无部分结果",
        log_fields=("event", "outcome", "status_code"),
        preexisting="test_cw007_decisions_service.py::test_batch_decision_rolls_back_when_the_insert_fails（已分类）",
    ),
    FaultCase(
        key="request.storage.insert.unexpected",
        stage="request",
        component="storage",
        operation="insert_many -> RuntimeError",
        affected=tuple((path, 500, "internal_error") for path in DECISION_WRITE_ENDPOINTS),
        unaffected=(
            "/health/live",
            "/health/ready",
            "/v1/model",
            "/v1/policy",
            *SCORING_ENDPOINTS,
            *RISK_ENDPOINTS,
        ),
        write_effect="零写入：仓储层回滚，无部分结果",
        log_fields=("event", "outcome", "status_code"),
        preexisting=None,
    ),
    FaultCase(
        key="request.storage.insert.unavailable",
        stage="request",
        component="storage",
        operation="insert_many -> StorageUnavailableError",
        affected=tuple(
            (path, 503, "storage_unavailable") for path in DECISION_WRITE_ENDPOINTS
        ),
        unaffected=(
            "/health/live",
            "/health/ready",
            "/v1/model",
            "/v1/policy",
            *SCORING_ENDPOINTS,
            *RISK_ENDPOINTS,
        ),
        write_effect="零写入：已回滚",
        log_fields=("event", "outcome", "status_code"),
        preexisting="test_cw007_decisions_service.py::test_batch_decision_rolls_back_when_the_insert_fails",
    ),
    FaultCase(
        key="request.storage.get.timeout",
        stage="request",
        component="storage",
        operation="get -> TimeoutError",
        affected=(("/v1/decisions/{id}", 500, "internal_error"),),
        unaffected=(
            "/health/live",
            "/health/ready",
            "/v1/model",
            "/v1/policy",
            *SCORING_ENDPOINTS,
            *RISK_ENDPOINTS,
            *DECISION_WRITE_ENDPOINTS,
        ),
        write_effect="零写入：读路径不写任何东西",
        log_fields=("event", "outcome", "storage_operation"),
        preexisting="test_cw007_decisions_service.py::test_runtime_read_failure_returns_503（已分类）",
    ),
    FaultCase(
        key="request.storage.lookup.timeout",
        stage="request",
        component="storage",
        operation="lookup_idempotent -> TimeoutError",
        affected=tuple(
            (path, 500, "internal_error") for path in DECISION_WRITE_ENDPOINTS
        ),
        unaffected=("/health/live", "/health/ready", *SCORING_ENDPOINTS, *RISK_ENDPOINTS),
        write_effect="零写入：查表发生在求值之前，模型与规则一次都不运行",
        log_fields=("event", "outcome", "status_code"),
        preexisting=None,
    ),
    FaultCase(
        key="request.storage.lookup.unavailable",
        stage="request",
        component="storage",
        operation="lookup_idempotent -> StorageUnavailableError",
        affected=tuple(
            (path, 503, "storage_unavailable") for path in DECISION_WRITE_ENDPOINTS
        ),
        unaffected=("/health/live", "/health/ready", *SCORING_ENDPOINTS, *RISK_ENDPOINTS),
        write_effect="零写入",
        log_fields=("event", "outcome", "status_code"),
        preexisting=None,
    ),
    FaultCase(
        key="request.storage.commit.timeout",
        stage="request",
        component="storage",
        operation="commit_idempotent -> TimeoutError",
        affected=tuple(
            (path, 500, "internal_error") for path in DECISION_WRITE_ENDPOINTS
        ),
        unaffected=("/health/live", "/health/ready", *SCORING_ENDPOINTS, *RISK_ENDPOINTS),
        write_effect="零写入：无部分 decision、无部分幂等映射",
        log_fields=("event", "outcome", "status_code"),
        preexisting=None,
    ),
    FaultCase(
        key="request.storage.commit.rollback",
        stage="request",
        component="storage",
        operation="commit_idempotent -> 真实 SQLite 回滚",
        affected=tuple(
            (path, 503, "storage_unavailable") for path in DECISION_WRITE_ENDPOINTS
        ),
        unaffected=("/health/live", "/health/ready", *SCORING_ENDPOINTS, *RISK_ENDPOINTS),
        write_effect="零写入：一个事务内失败，decisions 与幂等映射同时回滚",
        log_fields=("event", "outcome", "status_code"),
        preexisting=None,
    ),
)

#: 完整故障矩阵。测试按 ``key`` 交叉引用，因此新增一行就必须同时给出证据。
FAULT_MATRIX: tuple[FaultCase, ...] = (
    _STARTUP_MODEL,
    _STARTUP_POLICY,
    _STARTUP_STORAGE,
    *_STARTUP_COMBINED,
    *_REQUEST_MODEL,
    *_REQUEST_POLICY,
    *_REQUEST_STORAGE,
)


def case(key: str) -> FaultCase:
    """按 ``key`` 取矩阵行；找不到就抛错（避免测试静默跳过某个场景）。"""
    for row in FAULT_MATRIX:
        if row.key == key:
            return row
    raise KeyError(key)


def staged_relative(tmp_path: Path) -> str:
    """项目相对的测试数据库路径（与 CW-007/CW-008 夹具同源，位于被忽略的目录下）。"""
    from tests.cw007_fixtures import staged_db_relative

    return staged_db_relative(tmp_path)


__all__ = [
    "FAULT_MATRIX",
    "FORBIDDEN_FRAGMENTS",
    "MODEL_ONLY_ENDPOINTS",
    "POLICY_ENDPOINTS",
    "DECISION_WRITE_ENDPOINTS",
    "RISK_ENDPOINTS",
    "SCORING_ENDPOINTS",
    "SENTINEL_FEATURE_VALUE",
    "SENTINEL_ITEM_ID",
    "SENTINEL_KEY",
    "SENTINEL_MESSAGE",
    "FaultCase",
    "FaultInjectingRepository",
    "FaultingStorageLoader",
    "FlakyPolicyEngine",
    "FlakyRuntime",
    "NthCallFault",
    "OperationFault",
    "assert_error_body_is_safe",
    "assert_no_leak",
    "build_leaky_error",
    "case",
    "staged_relative",
]
