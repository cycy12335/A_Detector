"""HTTP 请求/响应契约（Pydantic v2，全部 ``extra="forbid"``）。

## 三类严格性

1. **envelope 严格**：多一个字段就 422，避免"看起来被接受、实际被忽略"的字段悄悄进入
   调用方的心智模型。
2. **取值类型严格**：特征值只允许 ``string / integer / finite float / null``。
   显式拒绝 ``bool``——Python 里 ``bool`` 是 ``int`` 的子类，``True`` 会被静默当成 ``1``，
   那正是"类型契约看起来存在、实际不生效"的典型漏洞。数组、对象、``NaN``、``Infinity``
   同属拒绝之列。
3. **列集合严格**：features 必须与 bundle 的源特征集合**完全相等**（缺列、多列都失败）。
   该检查依赖运行时才知道的 199 个列名，因此放在路由层（见 ``app.py``），返回 422。

## 命名纪律

响应里**不出现** ``probability`` / ``fraud_probability`` / ``approve`` / ``decline``：
本服务给出的是未校准的模型分数与一个基于冻结阈值的标记，不是概率，也不是处置决定。
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: CW-006 政策语义。刻意**复用**政策模块里的常量而不是再写一遍字面量：
#: 两处各写一份，早晚会有一处被改而另一处没改。
from ai_detector.rules.config import POLICY_SEMANTICS

#: 分数语义的固定字符串（与 ``configs/service.yaml`` 一致）。
SCORE_SEMANTICS = "uncalibrated_model_score"

#: 阈值策略的固定字符串。
THRESHOLD_POLICY = "validation_f1"


class StrictModel(BaseModel):
    """所有请求/响应模型的基类：禁止额外字段。"""

    model_config = ConfigDict(extra="forbid")


def check_feature_value(value: Any, *, where: str) -> Any:
    """校验单个特征值并原样返回。

    Raises:
        ValueError: 类型不被允许（含 bool、嵌套容器、非有限浮点）。
    """
    if value is None or isinstance(value, str):
        return value
    # bool 必须排在 int 之前判断：bool 是 int 的子类。
    if isinstance(value, bool):
        raise ValueError(f"{where} 不允许布尔值：请显式使用字符串或整数")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} 不允许 NaN 或 Infinity")
        return value
    raise ValueError(f"{where} 只允许 string / integer / finite float / null")


class ScoreRequest(StrictModel):
    """单笔评分请求：``{"features": {...}}``。"""

    features: dict[str, Any] = Field(..., description="源特征名到取值的映射")

    @field_validator("features")
    @classmethod
    def _validate_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        for name, item in value.items():
            check_feature_value(item, where=f"features[{name}]")
        return value


class ScoreItem(StrictModel):
    """批量评分中的一项。"""

    client_item_id: str = Field(..., min_length=1, max_length=128)
    features: dict[str, Any]

    @field_validator("client_item_id")
    @classmethod
    def _validate_item_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("client_item_id 不能为空白")
        return value

    @field_validator("features")
    @classmethod
    def _validate_values(cls, value: dict[str, Any]) -> dict[str, Any]:
        for name, item in value.items():
            check_feature_value(item, where=f"features[{name}]")
        return value


class BatchScoreRequest(StrictModel):
    """批量评分请求：``{"items": [...]}``。

    数量上下限在模型层用 ``Field`` 约束，因此空批次与超限批次都会在进入路由前被拒。
    上限由路由层用配置值再校验一次（配置是权威来源）。
    """

    items: list[ScoreItem] = Field(..., min_length=1)

    @field_validator("items")
    @classmethod
    def _validate_unique_ids(cls, value: list[ScoreItem]) -> list[ScoreItem]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for item in value:
            if item.client_item_id in seen:
                duplicates.append(item.client_item_id)
            seen.add(item.client_item_id)
        if duplicates:
            # 不回显重复 ID 的具体取值，只给数量，避免把调用方数据写进响应。
            raise ValueError(f"client_item_id 必须唯一，检测到 {len(set(duplicates))} 个重复值")
        return value


class ScoreResponse(StrictModel):
    """单笔评分响应。"""

    request_id: str
    model_version: str
    score: float
    score_semantics: str = SCORE_SEMANTICS
    threshold: float
    threshold_policy: str = THRESHOLD_POLICY
    flagged: bool


class BatchScoreItemResponse(StrictModel):
    """批量评分中的单条结果（顺序与请求一致）。"""

    client_item_id: str
    score: float
    flagged: bool


class BatchScoreResponse(StrictModel):
    """批量评分响应。"""

    request_id: str
    model_version: str
    score_semantics: str = SCORE_SEMANTICS
    threshold: float
    threshold_policy: str = THRESHOLD_POLICY
    batch_size: int
    results: list[BatchScoreItemResponse]


class ModelInfoResponse(StrictModel):
    """模型元数据响应。

    只暴露**schema 级**信息：模型版本、bundle 指纹、分数语义、冻结阈值与必需的源特征名。
    不含任何类别取值、特征取值、逐行数据或文件系统路径。
    """

    model_version: str
    bundle_sha256: str
    schema_version: str
    score_semantics: str = SCORE_SEMANTICS
    threshold: float
    threshold_policy: str = THRESHOLD_POLICY
    required_features: list[str]
    batch_limit: int


class LiveResponse(StrictModel):
    """存活探针响应。"""

    status: str = "live"
    service: str
    api_version: str


class ReadyResponse(StrictModel):
    """就绪探针响应。

    CW-005 的字段集合**保持不变**：政策是否就绪通过状态码（503 ``policy_not_ready``）
    与启动诊断反映，而不是往这个响应里加字段——加字段会改动 CW-005 的成功响应契约。
    """

    status: str = "ready"
    model_version: str
    bundle_sha256: str


# --- CW-006 / 版本化规则政策与研究型风险分级 ----------------------------


class PolicyRuleInfo(StrictModel):
    """政策里一条规则的可公开元数据。

    只暴露**规则本身**（读哪个源特征、用什么运算符、比哪个阈值）——这正是"透明规则"
    的含义，也是本工作包要展示的可解释性；它不含任何取值、载荷或文件路径。
    """

    rule_id: str
    feature: str
    operator: str
    threshold: float


class PolicyFusionInfo(StrictModel):
    """融合配置：模型标志与规则命中的点值，以及信号点到等级的闭区间。"""

    flagged_signal_points: int
    matched_rule_signal_points: int
    bands: dict[str, list[int]]


class PolicyResponse(StrictModel):
    """政策元数据响应（版本、指纹、冻结语义、规则元数据、融合区间与免责声明）。"""

    policy_version: str
    policy_schema_version: str
    policy_sha256: str
    policy_semantics: str = POLICY_SEMANTICS
    threshold_source: str
    rules: list[PolicyRuleInfo]
    fusion: PolicyFusionInfo
    risk_levels: list[str]
    signal_point_range: list[int]
    disclaimer: str
    batch_limit: int


class RiskEvaluateResponse(StrictModel):
    """单笔风险求值响应：模型分数 + 透明规则结果 + 研究型风险等级。

    明确**不包含** ``approve`` / ``decline`` / ``block`` / ``allow`` / ``fraud_probability``：
    ``risk_level`` 是研究型信号分级，不是概率，也不是任何生产处置建议。
    """

    request_id: str
    model_version: str
    score: float
    score_semantics: str = SCORE_SEMANTICS
    threshold: float
    threshold_policy: str = THRESHOLD_POLICY
    flagged: bool
    policy_version: str
    policy_sha256: str
    policy_semantics: str = POLICY_SEMANTICS
    rule_score: float
    matched_rule_ids: list[str]
    unevaluable_rule_ids: list[str]
    rule_evaluation_complete: bool
    signal_points: int | None
    risk_level: str


class RiskBatchItemResponse(StrictModel):
    """批量风险求值中的单条结果（顺序与请求一致）。"""

    client_item_id: str
    score: float
    flagged: bool
    rule_score: float
    matched_rule_ids: list[str]
    unevaluable_rule_ids: list[str]
    rule_evaluation_complete: bool
    signal_points: int | None
    risk_level: str


class RiskBatchResponse(StrictModel):
    """批量风险求值响应：顶层只给公共模型/政策元数据，逐项给各自的等级结果。"""

    request_id: str
    model_version: str
    score_semantics: str = SCORE_SEMANTICS
    threshold: float
    threshold_policy: str = THRESHOLD_POLICY
    policy_version: str
    policy_sha256: str
    policy_semantics: str = POLICY_SEMANTICS
    batch_size: int
    results: list[RiskBatchItemResponse]


# --- CW-007 / 决策记录持久化 -------------------------------------------


class DecisionRecord(StrictModel):
    """已持久化的决策记录（与 ``ai_detector.storage.records.StoredDecision`` 逐字段对应）。

    只包含**求值产生的派生结果**：不含 ``features``、特征名/值、请求/响应体、header、
    原始 path/query 或数据库路径。字段顺序与记录 schema 一致。
    """

    record_schema_version: str
    decision_id: str
    created_at: str
    evaluation_request_id: str
    client_item_id: str | None
    model_version: str
    score: float
    score_semantics: str = SCORE_SEMANTICS
    threshold: float
    threshold_policy: str = THRESHOLD_POLICY
    flagged: bool
    policy_version: str
    policy_sha256: str
    policy_semantics: str = POLICY_SEMANTICS
    rule_score: float
    matched_rule_ids: list[str]
    unevaluable_rule_ids: list[str]
    rule_evaluation_complete: bool
    signal_points: int | None
    risk_level: str


class DecisionResponse(StrictModel):
    """单笔持久化请求与按 ID 查询的响应：``{"request_id": ..., "decision": ...}``。"""

    request_id: str
    decision: DecisionRecord


class DecisionBatchResponse(StrictModel):
    """批量持久化响应：逐项保留顺序与调用方关联 ID。"""

    request_id: str
    decisions: list[DecisionRecord]


class ErrorDetail(StrictModel):
    """统一错误体中的 detail 部分。"""

    code: str
    message: str
    request_id: str


class ErrorResponse(StrictModel):
    """统一安全错误响应。"""

    error: ErrorDetail
