"""FastAPI 应用工厂：五个只读接口 + 请求 ID + 安全错误 + 单条结构化日志。

## 无副作用地导入

``import ai_detector.service`` 不会读取配置、不会加载模型、不会启动服务器：
应用只在 :func:`create_app` **被调用**时构造，bundle 只在 app 的 lifespan 启动阶段加载，
且**每个 app 生命周期最多加载一次**（有测试用调用计数守住）。

## 运行时可注入

:func:`create_app` 接受 ``runtime_loader``，默认是真实的
:func:`ai_detector.service.runtime.load_runtime`。测试注入 fake runtime 后，既能断言
"每请求不重复加载"，也能精确统计评分调用次数——批量原子性就是靠"无效批次零次调用"证明的。

## 降级是设计的一部分

bundle 缺失、指纹不符、依赖缺失或契约校验失败时，应用**仍然可以创建**：
``/health/live`` 依旧 200，``/health/ready`` 与两个评分接口返回 503。
服务不会自动下载、重训或回退到别的模型。

## 分数不是概率，flagged 不是处置

所有评分响应只给 ``score``（未校准的模型分数）、``threshold``（在 CW-004 的 validation 上
冻结的 F1 阈值）与 ``flagged = score >= threshold``。没有批准/拒绝语义。
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Sequence

from fastapi import FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_detector.rules.config import (
    ALL_RISK_LEVELS,
    FLAGGED_SIGNAL_POINTS,
    FROZEN_BANDS,
    MATCHED_RULE_SIGNAL_POINTS,
    MAX_SIGNAL_POINTS,
    PolicyLoadError,
)
from ai_detector.rules.engine import (
    PolicyContractError,
    RiskPolicyEngine,
    RiskAssessment,
    RuleEvaluationResult,
    fuse,
    validate_rule_result,
)
from ai_detector.service.config import ServiceConfig
from ai_detector.service.errors import (
    ApiError,
    BundleLoadError,
    ServiceErrorCode,
)
from ai_detector.service.logging import log_completion, log_startup
from ai_detector.service.runtime import (
    ScoringRuntime,
    load_policy_engine,
    load_runtime,
    load_storage_repository,
)
from ai_detector.service.schemas import (
    BatchScoreItemResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    DecisionBatchResponse,
    DecisionRecord,
    DecisionResponse,
    LiveResponse,
    ModelInfoResponse,
    PolicyFusionInfo,
    PolicyResponse,
    PolicyRuleInfo,
    ReadyResponse,
    RiskBatchItemResponse,
    RiskBatchResponse,
    RiskEvaluateResponse,
    ScoreRequest,
    ScoreResponse,
)
from ai_detector.storage.config import (
    STORAGE_SCHEMA_VERSION,
    StorageErrorCode,
    StorageLoadError,
)
from ai_detector.storage.idempotency import (
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENCY_KEY_MAX_LENGTH,
    IDEMPOTENCY_KEY_MIN_LENGTH,
    IDEMPOTENCY_KEY_PATTERN,
    OPERATION_BATCH,
    OPERATION_SINGLE,
    REPLAYED_HEADER,
    IdempotencyContractError,
    IdempotencyKeyError,
    batch_payload,
    key_digest,
    request_fingerprint,
    single_payload,
    validate_idempotency_key,
)
from ai_detector.storage.records import StoredDecision, is_canonical_decision_id
from ai_detector.storage.sqlite import (
    IDEMPOTENCY_CONFLICT,
    IDEMPOTENCY_CREATED,
    IDEMPOTENCY_REPLAY,
    IDEMPOTENCY_REPLAYED,
    DecisionRepository,
    IdempotencyHit,
    StorageUnavailableError,
)

#: 未匹配到路由时的占位 route 名。刻意不使用原始 URL——
#: 原始路径可能带调用方数据，写进日志等于把输入泄漏出去。
UNMATCHED_ROUTE = "<unmatched>"

RuntimeLoader = Callable[[ServiceConfig], ScoringRuntime]
PolicyLoader = Callable[[ServiceConfig], RiskPolicyEngine]
StorageLoader = Callable[[ServiceConfig], DecisionRepository]
DecisionIdFactory = Callable[[], str]
UtcClock = Callable[[], str]

#: 政策引擎内部异常折叠成的启动诊断分类码（与 ``PolicyErrorCode`` 区分开：
#: 那些是**可预期**的加载失败，这个是"没想到的异常"，只报类型名不报文本）。
POLICY_UNEXPECTED_ERROR = "policy_unexpected_error"

#: 存储内部异常折叠成的启动诊断分类码（同上）。
STORAGE_UNEXPECTED_ERROR = "storage_unexpected_error"


def _default_decision_id() -> str:
    """服务端生成的规范小写 UUID4。"""
    return str(uuid.uuid4())


def _default_utc_now() -> str:
    """UTC RFC 3339（秒级精度，后缀 ``Z``）。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _runtime_of(request: Request) -> ScoringRuntime | None:
    return getattr(request.app.state, "runtime", None)


def _policy_of(request: Request) -> RiskPolicyEngine | None:
    return getattr(request.app.state, "policy_engine", None)


def _storage_of(request: Request) -> DecisionRepository | None:
    return getattr(request.app.state, "decision_repository", None)


def _require_runtime(request: Request) -> ScoringRuntime:
    """取运行时；未就绪时抛 503。"""
    runtime = _runtime_of(request)
    if runtime is None:
        raise ApiError(503, ServiceErrorCode.MODEL_NOT_READY)
    return runtime


def _require_policy(request: Request) -> RiskPolicyEngine:
    """取政策引擎；未就绪时抛 503。

    政策故障**绝不**回退到内置默认政策或旧政策：宁可让政策相关接口明确不可用，
    也不要让调用方拿到一份来源不明的判定依据。
    """
    engine = _policy_of(request)
    if engine is None:
        raise ApiError(503, ServiceErrorCode.POLICY_NOT_READY)
    return engine


def _validate_feature_contract(features: dict[str, Any], runtime: ScoringRuntime) -> None:
    """features 的键集合必须与运行时记录的源特征集合**完全相等**。

    Raises:
        ApiError: 缺列或多列（422）。**只报数量，不回显列名与取值**——
            回显会让错误响应变成"请告诉我你的 schema 对不对"的探测器。
    """
    expected = set(runtime.source_features)
    actual = set(features)
    if actual == expected:
        return
    missing = len(expected - actual)
    unexpected = len(actual - expected)
    raise ApiError(
        422,
        ServiceErrorCode.FEATURE_CONTRACT_VIOLATION,
        f"features 必须与模型要求的 {len(expected)} 个源特征完全一致："
        f"缺少 {missing} 个、多出 {unexpected} 个。",
    )


def _score_one(runtime: ScoringRuntime, features: dict[str, Any]) -> float:
    """调用运行时并做输出侧健全性检查（分数必须是有限数）。"""
    scores = runtime.score_many([features])
    if len(scores) != 1:
        raise ApiError(500, ServiceErrorCode.INTERNAL_ERROR)
    return _finite_score(scores[0])


def _finite_score(value: Any) -> float:
    import math

    number = float(value)
    if not math.isfinite(number):
        raise ApiError(500, ServiceErrorCode.INTERNAL_ERROR)
    return number


def _evaluate_rules(engine: RiskPolicyEngine, features: dict[str, Any]) -> RuleEvaluationResult:
    """求值并校验引擎输出。

    引擎内部异常折叠为通用 500；引擎给出**结构不一致**的结果（未知/重复规则 ID、
    既命中又不可评估等）折叠为 500 ``policy_contract_violation``——那表示服务端政策逻辑
    已坏，宁可明确失败，也不把可疑结果当成合法事实返回给调用方。
    """
    try:
        result = engine.evaluate_rules(features)
    except ApiError:
        raise
    except Exception:  # noqa: BLE001 - 引擎异常细节不外泄
        raise ApiError(500, ServiceErrorCode.INTERNAL_ERROR) from None
    try:
        validate_rule_result(engine, result)
    except PolicyContractError:
        raise ApiError(500, ServiceErrorCode.POLICY_CONTRACT_VIOLATION) from None
    return result


def _require_storage(request: Request) -> DecisionRepository:
    """取存储仓储；未就绪时抛 503。

    存储故障**绝不**回退到内存数据库或临时文件：宁可让 decisions 接口明确不可用，
    也不要让调用方以为结果已经持久化。
    """
    repository = _storage_of(request)
    if repository is None:
        raise ApiError(503, ServiceErrorCode.STORAGE_NOT_READY)
    return repository


def _storage_api_error(error: StorageLoadError) -> ApiError:
    """把存储层分类码映射为对外错误（只给稳定码与固定文案，不含路径/SQL/记录值）。"""
    if isinstance(error, StorageUnavailableError) or error.code == StorageErrorCode.UNAVAILABLE:
        return ApiError(503, ServiceErrorCode.STORAGE_UNAVAILABLE)
    if error.code == StorageErrorCode.CONTRACT_VIOLATION:
        return ApiError(500, ServiceErrorCode.STORAGE_CONTRACT_VIOLATION)
    return ApiError(503, ServiceErrorCode.STORAGE_NOT_READY)


def _persist(repository: DecisionRepository, records: list[StoredDecision]) -> None:
    """一个事务写入全部记录；失败时统一折叠为安全错误。

    任一 insert/commit 失败都已在仓储层回滚，因此这里**不会**返回任何部分结果，
    也**不会**退化成"只返回未持久化结果"。
    """
    try:
        repository.insert_many(records)
    except StorageLoadError as error:
        raise _storage_api_error(error) from None
    except Exception:  # noqa: BLE001 - 存储异常文本不外泄
        raise ApiError(500, ServiceErrorCode.INTERNAL_ERROR) from None


#: 幂等日志结果枚举之一：请求没带 key，因此**没有**任何幂等查询或映射写入。
IDEMPOTENCY_NOT_REQUESTED = "not_requested"


#: 两个 decisions POST 上**真实声明**的可选 ``Idempotency-Key`` 参数的文案与 schema 片段。
#:
#: 它是路由签名里的一等参数，因此会出现在 OpenAPI 的 ``parameters`` 里（name / in / required /
#: schema 一应俱全），而不是只写在 docstring 里等着被全文搜索"搜到"。
#:
#: 形状约束刻意只放进 ``json_schema_extra``（**只进 schema、不做框架级校验**）：非法取值必须由
#: 服务自己按冻结契约回 400 ``invalid_idempotency_key``，而不是被框架拦成 422 —— 冻结的错误码
#: 不能因为"用了框架的声明方式"而漂移。
_IDEMPOTENCY_KEY_DESCRIPTION = (
    "可选的请求幂等键：16-128 个 ASCII 字符，以字母或数字开头，只允许字母、数字与 . _ : - 。"
    "同键同请求返回首次提交的快照（Idempotency-Replayed: true），同键异请求返回 409。"
)
_IDEMPOTENCY_KEY_SCHEMA: dict[str, Any] = {
    "minLength": IDEMPOTENCY_KEY_MIN_LENGTH,
    "maxLength": IDEMPOTENCY_KEY_MAX_LENGTH,
    "pattern": IDEMPOTENCY_KEY_PATTERN.pattern,
}


def _idempotency_key(request: Request, declared: str | None) -> str | None:
    """读取并校验可选的 ``Idempotency-Key``。

    - 缺省 → ``None``，调用方走原 CW-007 路径，**不触发**任何幂等查询或映射写入；
    - 同一请求出现多次 → 400（"用哪一个"没有合理默认，猜测等于替调用方做决定）；
    - 形状非法 → 400，且此时任何模型/规则/数据库动作都还没发生。

    取值以路由签名声明的那一个为准（重复 header 由下面的 ``getlist`` 拦下，不会静默取其一）。
    错误只给稳定码与固定文案，**绝不**回显原始取值。
    """
    values = request.headers.getlist(IDEMPOTENCY_KEY_HEADER)
    if not values:
        return None
    if len(values) > 1:
        raise ApiError(400, ServiceErrorCode.INVALID_IDEMPOTENCY_KEY)
    try:
        return validate_idempotency_key(values[0] if declared is None else declared)
    except IdempotencyKeyError:
        raise ApiError(400, ServiceErrorCode.INVALID_IDEMPOTENCY_KEY) from None


def _idempotency_error(error: Exception) -> ApiError:
    """把幂等层的异常折叠成稳定错误：损坏是 500，存储故障沿用既有映射。"""
    if isinstance(error, IdempotencyContractError):
        return ApiError(500, ServiceErrorCode.IDEMPOTENCY_CONTRACT_VIOLATION)
    if isinstance(error, StorageLoadError):
        return _storage_api_error(error)
    return ApiError(500, ServiceErrorCode.INTERNAL_ERROR)


def _replay_header(response: Response, replayed: bool) -> None:
    """只有**带 key** 的请求才返回这个 header；无 key 时字段集合逐字不变。"""
    response.headers[REPLAYED_HEADER] = "true" if replayed else "false"


def create_app(
    config: ServiceConfig,
    *,
    runtime_loader: RuntimeLoader = load_runtime,
    policy_loader: PolicyLoader = load_policy_engine,
    storage_loader: StorageLoader = load_storage_repository,
    decision_id_factory: DecisionIdFactory = _default_decision_id,
    utc_clock: UtcClock = _default_utc_now,
) -> FastAPI:
    """构造应用。

    Args:
        config: 已解析的服务配置。
        runtime_loader: 运行时加载器；默认加载真实 bundle，测试可注入 fake。
        policy_loader: 政策加载器；默认加载真实政策文件，测试可注入 fake engine。
        storage_loader: 存储加载器；默认打开真实 SQLite，测试可注入 fake repository。
        decision_id_factory: decision ID 生成器（默认 UUID4）；测试可注入确定性序列。
        utc_clock: 创建时间生成器（默认 UTC RFC 3339 ``Z``）；测试可注入固定时钟。

    Returns:
        配置完毕的 :class:`fastapi.FastAPI`。**此时尚未加载模型、政策或存储**——三者都发生在
        lifespan 启动阶段，且各**最多加载一次**（有测试用调用计数守住）。工厂本身**不做任何
        文件 I/O**：目录与数据库只在存储初始化阶段创建。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """启动时各加载一次模型、政策与存储；任一失败都降级为"未就绪"，但仍允许应用运行。

        降级是设计的一部分，且是**按组件**的：bundle 坏掉时评分接口不可用；政策坏掉时
        评分接口照常；存储坏掉时旧八个接口全部照常，只有 ``/v1/decisions/*`` 与 readiness 降级。
        """
        app.state.runtime = None
        app.state.runtime_error_code = None
        app.state.policy_engine = None
        app.state.policy_error_code = None
        app.state.decision_repository = None
        app.state.storage_error_code = None

        try:
            app.state.runtime = runtime_loader(config)
        except BundleLoadError as error:
            # 只记录分类码与安全措辞：不含绝对路径、文件内容或堆栈。
            app.state.runtime_error_code = error.code
            app.state.runtime_error_detail = error.detail
        except Exception as error:  # noqa: BLE001 - 任何加载失败都降级，不得让进程起不来
            app.state.runtime_error_code = "bundle_unexpected_error"
            app.state.runtime_error_detail = f"加载运行时出现未预期错误：{type(error).__name__}"

        try:
            app.state.policy_engine = policy_loader(config)
        except PolicyLoadError as error:
            app.state.policy_error_code = error.code
            app.state.policy_error_detail = error.detail
        except Exception as error:  # noqa: BLE001 - 政策坏掉同样不得让进程起不来
            app.state.policy_error_code = POLICY_UNEXPECTED_ERROR
            app.state.policy_error_detail = f"加载政策出现未预期错误：{type(error).__name__}"

        try:
            # **未配置**存储时不尝试加载：那不是失败，而是"这项能力不适用于本次部署"。
            # 因此 storage_ready 保持 True（不参与 readiness 判定），也不会写出任何文件。
            if config.storage_path is None:
                app.state.decision_repository = None
            else:
                app.state.decision_repository = storage_loader(config)
        except StorageLoadError as error:
            app.state.storage_error_code = error.code
            app.state.storage_error_detail = error.detail
        except Exception as error:  # noqa: BLE001 - 存储坏掉同样不得让进程起不来
            app.state.storage_error_code = STORAGE_UNEXPECTED_ERROR
            app.state.storage_error_detail = f"加载存储出现未预期错误：{type(error).__name__}"

        # 启动诊断：成功与失败各一条，只含白名单字段，且区分模型、政策与存储三个组件。
        # 对外响应保持通用文案；这条日志是运维侧定位"为什么没就绪"的唯一入口，
        # 因此它必须是结构化且安全的——失败时不写路径、配置载荷、数据库内容、SQL 或堆栈。
        runtime = app.state.runtime
        engine = app.state.policy_engine
        repository = app.state.decision_repository
        # 只有"配置了存储"才有"存储是否就绪"这回事：未配置时该项不参与判定。
        storage_applicable = config.storage_path is not None
        log_startup(
            service=config.name,
            api_version=config.api_version,
            model_ready=runtime is not None,
            policy_ready=engine is not None,
            storage_ready=(not storage_applicable) or repository is not None,
            model_version=getattr(runtime, "model_version", None),
            n_source_features=len(getattr(runtime, "source_features", []) or []) or None,
            model_error_code=app.state.runtime_error_code,
            policy_version=getattr(engine, "policy_version", None),
            policy_error_code=app.state.policy_error_code,
            storage_schema_version=STORAGE_SCHEMA_VERSION if repository is not None else None,
            storage_error_code=app.state.storage_error_code,
        )

        yield
        if app.state.decision_repository is not None:
            app.state.decision_repository.close()
        app.state.runtime = None
        app.state.policy_engine = None
        app.state.decision_repository = None

    app = FastAPI(
        title=config.name,
        version=f"{config.api_version}",
        description=(
            "只读的交易风险评分服务（研究原型）。返回未校准的模型分数与基于冻结阈值的标记，"
            "不是欺诈概率，也不是批准/拒绝决定。"
        ),
        lifespan=lifespan,
    )
    app.state.config = config

    # --- 请求 ID + 单条 completion log ---------------------------------

    request_id_regex = config.request_id_regex()
    header_name = config.request_id_header

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        supplied = request.headers.get(header_name)
        invalid_supplied = supplied is not None and not request_id_regex.fullmatch(supplied)
        # 非法 ID 一律**不回显**：换一个新的，避免把任意字符串写进响应头。
        request_id = (
            supplied
            if supplied is not None and not invalid_supplied
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id

        if invalid_supplied:
            response: Any = JSONResponse(
                status_code=400,
                content=ApiError(400, ServiceErrorCode.INVALID_REQUEST_ID).to_body(request_id),
            )
        else:
            try:
                response = await call_next(request)
            except ApiError as error:
                response = JSONResponse(
                    status_code=error.status_code, content=error.to_body(request_id)
                )
            except Exception:  # noqa: BLE001 - 未预期异常折叠为 500，细节不外泄
                response = JSONResponse(
                    status_code=500,
                    content=ApiError(500, ServiceErrorCode.INTERNAL_ERROR).to_body(request_id),
                )

        response.headers[header_name] = request_id
        route = getattr(request.scope.get("route"), "path", None) or UNMATCHED_ROUTE
        log_completion(
            {
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "route": route,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "model_version": _runtime_of(request).model_version
                if _runtime_of(request) is not None
                else None,
                "batch_size": getattr(request.state, "batch_size", None),
                # CW-006：只记标识、计数与枚举等级。规则 ID 列表、rule score、特征值与
                # 匹配明细都不记——排障靠 request_id，不靠把判定依据抄进日志。
                # 这些字段由**政策/风险路由**显式写入 request.state，因此 CW-005 五个
                # 接口的日志字段集合逐字不变。
                "policy_version": getattr(request.state, "policy_version", None),
                "risk_level": getattr(request.state, "risk_level", None),
                "rule_match_count": getattr(request.state, "rule_match_count", None),
                # CW-007：只记**操作类别**与**记录条数**。decision_id、evaluation_request_id、
                # client_item_id、数据库路径与 SQL 都不记。
                "storage_operation": getattr(request.state, "storage_operation", None),
                "record_count": getattr(request.state, "record_count", None),
                # CW-008：只记**结果枚举**（not_requested / created / replayed / conflict）。
                # 原始 key、key digest 与 request fingerprint 都不记。
                "idempotency_outcome": getattr(request.state, "idempotency_outcome", None),
                "outcome": "ok" if response.status_code < 400 else "error",
            }
        )
        return response

    # --- 统一异常处理：绝不回显 input / 堆栈 / 内部文本 ------------------

    def _error_response(request: Request, status_code: int, code: str) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status_code, content=ApiError(status_code, code).to_body(request_id)
        )

    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(status_code=error.status_code, content=error.to_body(request_id))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        # 刻意不返回 error.errors()：它含有原始 input。
        return _error_response(request, 422, ServiceErrorCode.INVALID_REQUEST_BODY)

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        if error.status_code == 404:
            code = ServiceErrorCode.NOT_FOUND
        elif error.status_code == 405:
            code = ServiceErrorCode.METHOD_NOT_ALLOWED
        elif error.status_code == 503:
            code = ServiceErrorCode.MODEL_NOT_READY
        else:
            code = ServiceErrorCode.INVALID_REQUEST_BODY
        return _error_response(request, error.status_code, code)

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, error: Exception) -> JSONResponse:
        # 内部异常文本与堆栈只留在进程里，不进响应。
        return _error_response(request, 500, ServiceErrorCode.INTERNAL_ERROR)

    # --- 接口 -----------------------------------------------------------

    @app.get("/health/live", response_model=LiveResponse, tags=["health"])
    async def live() -> LiveResponse:
        """进程存活即 200，**不依赖模型**。"""
        return LiveResponse(service=config.name, api_version=config.api_version)

    @app.get("/health/ready", response_model=ReadyResponse, tags=["health"])
    async def ready(request: Request) -> ReadyResponse:
        """**模型、政策与存储都**已就绪则为 200，否则 503。

        顺序固定为模型 → 政策 → 存储：越靠前的组件越贴近实际阻塞原因（模型不可用时连评分
        都做不了）。**未配置**存储的部署不参与这一判定——仓库配置启用了存储，因此真实
        CLI 的 readiness 必须反映它。
        """
        runtime = _runtime_of(request)
        if runtime is None:
            raise ApiError(503, ServiceErrorCode.MODEL_NOT_READY)
        if _policy_of(request) is None:
            raise ApiError(503, ServiceErrorCode.POLICY_NOT_READY)
        if request.app.state.config.storage_path is not None and _storage_of(request) is None:
            raise ApiError(503, ServiceErrorCode.STORAGE_NOT_READY)
        return ReadyResponse(
            model_version=runtime.model_version, bundle_sha256=runtime.bundle_sha256
        )

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["model"])
    async def model_info(request: Request) -> ModelInfoResponse:
        """模型元数据：版本、指纹、分数语义、冻结阈值与必需的源特征名。"""
        runtime = _require_runtime(request)
        return ModelInfoResponse(
            model_version=runtime.model_version,
            bundle_sha256=runtime.bundle_sha256,
            schema_version=config.expected_schema_version,
            score_semantics=runtime.score_semantics,
            threshold=runtime.threshold,
            threshold_policy=runtime.threshold_policy,
            required_features=list(runtime.source_features),
            batch_limit=config.batch_limit,
        )

    @app.post("/v1/score", response_model=ScoreResponse, tags=["scoring"])
    async def score(request: Request, payload: ScoreRequest) -> ScoreResponse:
        """单笔评分。"""
        runtime = _require_runtime(request)
        _validate_feature_contract(payload.features, runtime)
        value = _score_one(runtime, payload.features)
        return ScoreResponse(
            request_id=request.state.request_id,
            model_version=runtime.model_version,
            score=value,
            score_semantics=runtime.score_semantics,
            threshold=runtime.threshold,
            threshold_policy=runtime.threshold_policy,
            flagged=bool(value >= runtime.threshold),
        )

    @app.post("/v1/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
    async def score_batch(request: Request, payload: BatchScoreRequest) -> BatchScoreResponse:
        """1–100 笔**原子**批量评分：任一项无效则整个批次零次评分。"""
        runtime = _require_runtime(request)
        request.state.batch_size = len(payload.items)

        if len(payload.items) > config.batch_limit:
            raise ApiError(
                422,
                ServiceErrorCode.BATCH_CONTRACT_VIOLATION,
                f"批量上限为 {config.batch_limit} 笔，实际 {len(payload.items)} 笔。",
            )

        # --- 先完成全部校验，再执行任何一次评分（原子性）---
        for item in payload.items:
            _validate_feature_contract(item.features, runtime)

        rows: Sequence[dict[str, Any]] = [item.features for item in payload.items]
        scores = runtime.score_many(rows)
        if len(scores) != len(payload.items):
            raise ApiError(500, ServiceErrorCode.INTERNAL_ERROR)

        results = [
            BatchScoreItemResponse(
                client_item_id=item.client_item_id,
                score=_finite_score(value),
                flagged=bool(_finite_score(value) >= runtime.threshold),
            )
            for item, value in zip(payload.items, scores)
        ]
        return BatchScoreResponse(
            request_id=request.state.request_id,
            model_version=runtime.model_version,
            score_semantics=runtime.score_semantics,
            threshold=runtime.threshold,
            threshold_policy=runtime.threshold_policy,
            batch_size=len(results),
            results=results,
        )

    # --- CW-006：政策元数据与风险求值 -----------------------------------

    @app.get("/v1/policy", response_model=PolicyResponse, tags=["policy"])
    async def policy_info(request: Request) -> PolicyResponse:
        """版本化政策的元数据：版本、指纹、冻结语义、规则、融合区间与免责声明。

        只暴露**声明级**信息（规则读哪个源特征、比哪个阈值），不含任何取值、载荷或
        文件系统路径。政策未就绪时返回 503，而不是给出一份"看起来像默认政策"的替身。
        """
        engine = _require_policy(request)
        request.state.policy_version = engine.policy_version
        return PolicyResponse(
            policy_version=engine.policy_version,
            policy_schema_version=engine.policy_schema_version,
            policy_sha256=engine.policy_sha256,
            policy_semantics=engine.policy_semantics,
            threshold_source=engine.threshold_source,
            rules=[
                PolicyRuleInfo(
                    rule_id=rule.rule_id,
                    feature=rule.feature,
                    operator=rule.operator,
                    threshold=float(rule.threshold),
                )
                for rule in engine.rules
            ],
            fusion=PolicyFusionInfo(
                flagged_signal_points=FLAGGED_SIGNAL_POINTS,
                matched_rule_signal_points=MATCHED_RULE_SIGNAL_POINTS,
                bands={band.name: [band.low, band.high] for band in FROZEN_BANDS},
            ),
            risk_levels=list(ALL_RISK_LEVELS),
            signal_point_range=[0, MAX_SIGNAL_POINTS],
            disclaimer=engine.disclaimer,
            batch_limit=config.batch_limit,
        )

    def _record_risk_log_fields(
        request: Request, engine: RiskPolicyEngine, *, risk_level: str | None, matched: int
    ) -> None:
        """把可安全记录的政策/风险字段写入 request.state（供 completion log 取用）。"""
        request.state.policy_version = engine.policy_version
        request.state.risk_level = risk_level
        request.state.rule_match_count = matched

    @app.post("/v1/risk/evaluate", response_model=RiskEvaluateResponse, tags=["risk"])
    async def risk_evaluate(request: Request, payload: ScoreRequest) -> RiskEvaluateResponse:
        """单笔：模型评分 + 规则求值 + 研究型风险等级。

        规则输入不可评估（``null`` / 字符串）时返回 **200 + indeterminate**，不是 422：
        那是"无法判断"，是合法且显式的结果，不是调用方的请求错误。
        """
        runtime = _require_runtime(request)
        engine = _require_policy(request)
        _validate_feature_contract(payload.features, runtime)

        result = _evaluate_rules(engine, payload.features)
        value = _score_one(runtime, payload.features)
        flagged = bool(value >= runtime.threshold)
        assessment: RiskAssessment = fuse(result, flagged=flagged)

        _record_risk_log_fields(
            request,
            engine,
            risk_level=assessment.risk_level,
            matched=len(assessment.matched_rule_ids),
        )
        return RiskEvaluateResponse(
            request_id=request.state.request_id,
            model_version=runtime.model_version,
            score=value,
            score_semantics=runtime.score_semantics,
            threshold=runtime.threshold,
            threshold_policy=runtime.threshold_policy,
            flagged=flagged,
            policy_version=engine.policy_version,
            policy_sha256=engine.policy_sha256,
            policy_semantics=engine.policy_semantics,
            rule_score=assessment.rule_score,
            matched_rule_ids=list(assessment.matched_rule_ids),
            unevaluable_rule_ids=list(assessment.unevaluable_rule_ids),
            rule_evaluation_complete=assessment.complete,
            signal_points=assessment.signal_points,
            risk_level=assessment.risk_level,
        )

    @app.post("/v1/risk/evaluate/batch", response_model=RiskBatchResponse, tags=["risk"])
    async def risk_evaluate_batch(
        request: Request, payload: BatchScoreRequest
    ) -> RiskBatchResponse:
        """1–100 笔**原子**批量风险求值：先全部校验、再全部规则求值、最后恰好一次评分。

        三段顺序是刻意的：任一契约错误时模型评分**零次**、规则求值也零次，且不返回
        部分结果；规则求值必须在模型调用**之前**全部完成，因此"批量"不会因为逐条
        交错调用而放大模型侧开销。
        """
        runtime = _require_runtime(request)
        engine = _require_policy(request)
        request.state.batch_size = len(payload.items)

        if len(payload.items) > config.batch_limit:
            raise ApiError(
                422,
                ServiceErrorCode.BATCH_CONTRACT_VIOLATION,
                f"批量上限为 {config.batch_limit} 笔，实际 {len(payload.items)} 笔。",
            )

        # --- 1) 先完成全部特征契约校验（零次模型调用、零次规则求值）---
        for item in payload.items:
            _validate_feature_contract(item.features, runtime)

        # --- 2) 再完成全部规则求值（仍在任何模型调用之前）---
        rule_results = [_evaluate_rules(engine, item.features) for item in payload.items]

        # --- 3) 恰好一次模型调用 ---
        rows: Sequence[dict[str, Any]] = [item.features for item in payload.items]
        scores = runtime.score_many(rows)
        if len(scores) != len(payload.items):
            raise ApiError(500, ServiceErrorCode.INTERNAL_ERROR)

        results: list[RiskBatchItemResponse] = []
        total_matched = 0
        for item, raw_score, rule_result in zip(payload.items, scores, rule_results):
            score = _finite_score(raw_score)
            assessment = fuse(rule_result, flagged=bool(score >= runtime.threshold))
            total_matched += len(assessment.matched_rule_ids)
            results.append(
                RiskBatchItemResponse(
                    client_item_id=item.client_item_id,
                    score=score,
                    flagged=bool(score >= runtime.threshold),
                    rule_score=assessment.rule_score,
                    matched_rule_ids=list(assessment.matched_rule_ids),
                    unevaluable_rule_ids=list(assessment.unevaluable_rule_ids),
                    rule_evaluation_complete=assessment.complete,
                    signal_points=assessment.signal_points,
                    risk_level=assessment.risk_level,
                )
            )

        # 批量没有"一个"风险等级，因此只记 policy version 与命中总数；
        # risk_level 留空（scrub 会丢弃 None），不发明"最高等级"这类语义。
        _record_risk_log_fields(request, engine, risk_level=None, matched=total_matched)
        return RiskBatchResponse(
            request_id=request.state.request_id,
            model_version=runtime.model_version,
            score_semantics=runtime.score_semantics,
            threshold=runtime.threshold,
            threshold_policy=runtime.threshold_policy,
            policy_version=engine.policy_version,
            policy_sha256=engine.policy_sha256,
            policy_semantics=engine.policy_semantics,
            batch_size=len(results),
            results=results,
        )

    # --- CW-007：决策结果持久化 -----------------------------------------

    def _build_record(
        *,
        runtime: ScoringRuntime,
        engine: RiskPolicyEngine,
        rule_result: RuleEvaluationResult,
        score: float,
        evaluation_request_id: str,
        client_item_id: str | None,
    ) -> StoredDecision:
        """把一次求值的结果**最小化**地构造成可持久化记录。

        派生字段全部来自真实的 :func:`fuse` 结果，而不是在路由里重新拼一遍——手写一次就等于
        把融合逻辑复制了一份，早晚与冻结真值表漂移。记录构造本身会做完整校验（含跨字段
        不变量），因此"构造失败"必然发生在写入之前。

        ``flagged`` 在这里**只算一次**，并把同一个布尔值同时交给 :func:`fuse` 与
        :class:`StoredDecision`：它是模型阈值判定，即使规则不可评估（``signal_points`` 为
        ``None``）也仍然已知，不能从可空的融合点数反推。
        """
        flagged = bool(score >= runtime.threshold)
        assessment = fuse(rule_result, flagged=flagged)
        return StoredDecision.from_assessment(
            assessment,
            flagged=flagged,
            decision_id=decision_id_factory(),
            created_at=utc_clock(),
            evaluation_request_id=evaluation_request_id,
            client_item_id=client_item_id,
            model_version=runtime.model_version,
            score=score,
            threshold=runtime.threshold,
            policy_version=engine.policy_version,
            policy_sha256=engine.policy_sha256,
        )

    def _evaluate_single_record(
        request: Request,
        payload: ScoreRequest,
        runtime: ScoringRuntime,
        engine: RiskPolicyEngine,
    ) -> tuple[StoredDecision, str]:
        """单笔求值（无 key 与 keyed-miss 共用）：契约 → 规则 → 评分 → 构造记录。"""
        _validate_feature_contract(payload.features, runtime)
        rule_result = _evaluate_rules(engine, payload.features)
        score = _score_one(runtime, payload.features)
        record = _build_record(
            runtime=runtime,
            engine=engine,
            rule_result=rule_result,
            score=score,
            evaluation_request_id=request.state.request_id,
            client_item_id=None,
        )
        return record, engine.policy_version

    def _evaluate_and_insert_single(
        request: Request, payload: ScoreRequest
    ) -> DecisionResponse:
        """无 key 的单笔路径：与 CW-007 逐字一致（含组件检查顺序）。"""
        runtime = _require_runtime(request)
        engine = _require_policy(request)
        repository = _require_storage(request)
        record, policy_version = _evaluate_single_record(request, payload, runtime, engine)
        # 只有 commit 成功之后才构造响应。
        _persist(repository, [record])
        request.state.storage_operation = "write"
        request.state.record_count = 1
        request.state.policy_version = policy_version
        return DecisionResponse(
            request_id=request.state.request_id,
            decision=DecisionRecord(**record.to_public_dict()),
        )

    def _replay_single(
        request: Request, response: Response, hit: IdempotencyHit
    ) -> DecisionResponse:
        """重放单笔：整条路径**只依赖存储**，不调用模型/规则，也不生成 UUID 或时钟。"""
        request.state.idempotency_outcome = IDEMPOTENCY_REPLAYED
        request.state.storage_operation = "read"
        request.state.record_count = len(hit.records)
        _replay_header(response, True)
        return DecisionResponse(
            request_id=request.state.request_id,
            decision=DecisionRecord(**hit.records[0].to_public_dict()),
        )

    def _replay_batch(
        request: Request, response: Response, hit: IdempotencyHit
    ) -> DecisionBatchResponse:
        """重放批量：顺序与首次一致，外层 ``request_id`` 用**当前**这次 HTTP 请求的 ID。"""
        request.state.idempotency_outcome = IDEMPOTENCY_REPLAYED
        request.state.storage_operation = "read"
        request.state.record_count = len(hit.records)
        _replay_header(response, True)
        return DecisionBatchResponse(
            request_id=request.state.request_id,
            decisions=[DecisionRecord(**record.to_public_dict()) for record in hit.records],
        )

    @app.post(
        "/v1/decisions/evaluate",
        response_model=DecisionResponse,
        status_code=201,
        tags=["decisions"],
    )
    async def decisions_evaluate(
        request: Request,
        payload: ScoreRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias=IDEMPOTENCY_KEY_HEADER,
                                             description=_IDEMPOTENCY_KEY_DESCRIPTION,
                                             json_schema_extra=_IDEMPOTENCY_KEY_SCHEMA),
    ) -> DecisionResponse:
        """单笔：求值一次 → **提交成功后**才返回刚写入的记录。

        无 key 时顺序与 CW-007 完全一致：envelope → 组件就绪检查 → 特征契约 → 规则求值 →
        评分 → 构造并校验记录 → 一个事务写入并 commit → 响应。任一环节失败时数据库写入为零。

        带合法 ``Idempotency-Key`` 时先**只**依赖存储做一次查表：已提交的同请求直接重放、
        异请求直接 409（两者都发生在模型与规则之前）；未命中才进入上面的求值链路，并在提交
        事务里再次解决竞态。响应 body 与无 key 时逐字段相同，只有 ``Idempotency-Replayed``
        头不同。
        """
        key = _idempotency_key(request, idempotency_key)
        if key is None:
            request.state.idempotency_outcome = IDEMPOTENCY_NOT_REQUESTED
            return _evaluate_and_insert_single(request, payload)

        repository = _require_storage(request)
        digest = key_digest(key)
        fingerprint = request_fingerprint(OPERATION_SINGLE, single_payload(payload.features))
        try:
            hit = repository.lookup_idempotent(digest, OPERATION_SINGLE, fingerprint)
        except Exception as error:  # noqa: BLE001 - 统一折叠为稳定错误
            raise _idempotency_error(error) from None
        if hit.status == IDEMPOTENCY_CONFLICT:
            request.state.idempotency_outcome = "conflict"
            request.state.storage_operation = "read"
            raise ApiError(409, ServiceErrorCode.IDEMPOTENCY_CONFLICT)
        if hit.status == IDEMPOTENCY_REPLAY:
            return _replay_single(request, response, hit)

        runtime = _require_runtime(request)
        engine = _require_policy(request)
        record, engine_version = _evaluate_single_record(request, payload, runtime, engine)
        try:
            committed = repository.commit_idempotent(
                digest, OPERATION_SINGLE, fingerprint, [record]
            )
        except Exception as error:  # noqa: BLE001
            raise _idempotency_error(error) from None
        if committed.status == IDEMPOTENCY_CONFLICT:
            request.state.idempotency_outcome = "conflict"
            request.state.storage_operation = "read"
            raise ApiError(409, ServiceErrorCode.IDEMPOTENCY_CONFLICT)
        if committed.status == IDEMPOTENCY_REPLAYED:
            # 竞态输家：候选记录已随事务回滚，返回赢家已提交的那一份。
            return _replay_single(request, response, committed)
        request.state.idempotency_outcome = IDEMPOTENCY_CREATED
        request.state.storage_operation = "write"
        request.state.record_count = 1
        request.state.policy_version = engine_version
        _replay_header(response, False)
        return DecisionResponse(
            request_id=request.state.request_id,
            decision=DecisionRecord(**record.to_public_dict()),
        )

    @app.post(
        "/v1/decisions/evaluate/batch",
        response_model=DecisionBatchResponse,
        status_code=201,
        tags=["decisions"],
    )
    async def decisions_evaluate_batch(
        request: Request,
        payload: BatchScoreRequest,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias=IDEMPOTENCY_KEY_HEADER,
                                             description=_IDEMPOTENCY_KEY_DESCRIPTION,
                                             json_schema_extra=_IDEMPOTENCY_KEY_SCHEMA),
    ) -> DecisionBatchResponse:
        """1–100 笔：一次模型调用、**一个事务**写入全部记录，全成全败。

        无 key 时顺序与 CW-007 完全一致：envelope → 组件就绪检查 → 批量上限 → 全部特征契约 →
        全部规则求值 → **恰好一次** ``score_many`` → 构造并校验全部记录 → 一个事务写入并
        commit → 响应。任一环节失败时数据库写入为零，且**不返回**未持久化的结果。

        带合法 ``Idempotency-Key`` 时先只依赖存储查表（重放/冲突都在模型与规则之前），未命中
        才求值，并在提交事务里再次解决竞态。批量顺序与 ``client_item_id`` 都进入请求指纹。
        """
        key = _idempotency_key(request, idempotency_key)
        if key is None:
            request.state.idempotency_outcome = IDEMPOTENCY_NOT_REQUESTED
            return _evaluate_and_insert_batch(request, payload, config)

        repository = _require_storage(request)
        digest = key_digest(key)
        fingerprint = request_fingerprint(OPERATION_BATCH, batch_payload(payload.items))
        try:
            hit = repository.lookup_idempotent(digest, OPERATION_BATCH, fingerprint)
        except Exception as error:  # noqa: BLE001
            raise _idempotency_error(error) from None
        if hit.status == IDEMPOTENCY_CONFLICT:
            request.state.idempotency_outcome = "conflict"
            request.state.storage_operation = "read"
            raise ApiError(409, ServiceErrorCode.IDEMPOTENCY_CONFLICT)
        if hit.status == IDEMPOTENCY_REPLAY:
            request.state.batch_size = len(payload.items)
            return _replay_batch(request, response, hit)

        records, policy_version = _evaluate_batch_records(request, payload, config)
        request.state.batch_size = len(payload.items)
        try:
            committed = repository.commit_idempotent(
                digest, OPERATION_BATCH, fingerprint, records
            )
        except Exception as error:  # noqa: BLE001
            raise _idempotency_error(error) from None
        if committed.status == IDEMPOTENCY_CONFLICT:
            request.state.idempotency_outcome = "conflict"
            request.state.storage_operation = "read"
            raise ApiError(409, ServiceErrorCode.IDEMPOTENCY_CONFLICT)
        if committed.status == IDEMPOTENCY_REPLAYED:
            return _replay_batch(request, response, committed)
        request.state.idempotency_outcome = IDEMPOTENCY_CREATED
        request.state.storage_operation = "write"
        request.state.record_count = len(records)
        request.state.policy_version = policy_version
        _replay_header(response, False)
        return DecisionBatchResponse(
            request_id=request.state.request_id,
            decisions=[DecisionRecord(**record.to_public_dict()) for record in records],
        )

    def _evaluate_batch_records(
        request: Request, payload: BatchScoreRequest, config: ServiceConfig
    ) -> tuple[list[StoredDecision], str]:
        """批量的求值阶段（无 key 与 keyed-miss 共用），**不写任何东西**。"""
        runtime = _require_runtime(request)
        engine = _require_policy(request)

        if len(payload.items) > config.batch_limit:
            raise ApiError(
                422,
                ServiceErrorCode.BATCH_CONTRACT_VIOLATION,
                f"批量上限为 {config.batch_limit} 笔，实际 {len(payload.items)} 笔。",
            )

        # --- 1) 全部特征契约（零次模型调用、零次规则求值）---
        for item in payload.items:
            _validate_feature_contract(item.features, runtime)

        # --- 2) 全部规则求值（仍在任何模型调用之前）---
        rule_results = [_evaluate_rules(engine, item.features) for item in payload.items]

        # --- 3) 恰好一次模型调用 ---
        rows: Sequence[dict[str, Any]] = [item.features for item in payload.items]
        scores = runtime.score_many(rows)
        if len(scores) != len(payload.items):
            raise ApiError(500, ServiceErrorCode.INTERNAL_ERROR)

        # --- 4) 构造并校验全部记录（此时仍未写入任何内容）---
        records: list[StoredDecision] = []
        for item, raw_score, rule_result in zip(payload.items, scores, rule_results):
            records.append(
                _build_record(
                    runtime=runtime,
                    engine=engine,
                    rule_result=rule_result,
                    score=_finite_score(raw_score),
                    evaluation_request_id=request.state.request_id,
                    client_item_id=item.client_item_id,
                )
            )

        return records, engine.policy_version

    def _evaluate_and_insert_batch(
        request: Request, payload: BatchScoreRequest, config: ServiceConfig
    ) -> DecisionBatchResponse:
        """无 key 的批量路径：与 CW-007 逐字一致（含组件检查顺序与单事务写入）。"""
        repository = _require_storage(request)
        request.state.batch_size = len(payload.items)
        records, policy_version = _evaluate_batch_records(request, payload, config)
        # --- 5) 一个事务写入并 commit ---
        _persist(repository, records)
        request.state.storage_operation = "write"
        request.state.record_count = len(records)
        request.state.policy_version = policy_version
        return DecisionBatchResponse(
            request_id=request.state.request_id,
            decisions=[DecisionRecord(**record.to_public_dict()) for record in records],
        )

    @app.get(
        "/v1/decisions/{decision_id}",
        response_model=DecisionResponse,
        tags=["decisions"],
    )
    async def decisions_get(request: Request, decision_id: str) -> DecisionResponse:
        """按 ID 读取**存储快照**：不调用模型、不运行规则，也不重新计算任何字段。

        形状非法的 ID 在触达存储层之前就统一返回 404（因此带引号、SQL 关键字或超长参数只会
        得到"查不到"，不会影响 schema，也不会回显原始 path 参数）。模型或政策未就绪时，
        已有记录**仍可读取**。
        """
        repository = _require_storage(request)
        request.state.storage_operation = "read"
        request.state.record_count = 0
        if not is_canonical_decision_id(decision_id):
            raise ApiError(404, ServiceErrorCode.DECISION_NOT_FOUND)
        try:
            record = repository.get(decision_id)
        except StorageLoadError as error:
            raise _storage_api_error(error) from None
        except Exception:  # noqa: BLE001 - 存储异常文本不外泄
            raise ApiError(500, ServiceErrorCode.INTERNAL_ERROR) from None
        if record is None:
            raise ApiError(404, ServiceErrorCode.DECISION_NOT_FOUND)
        request.state.record_count = 1
        return DecisionResponse(
            request_id=request.state.request_id,
            decision=DecisionRecord(**record.to_public_dict()),
        )

    return app
