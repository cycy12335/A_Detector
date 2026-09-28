"""CW-007 决策记录：不可变结构 + 严格编码/解码。

## 只存派生结果

每条记录只允许 :data:`~ai_detector.storage.sqlite.DECISION_COLUMNS` 那 20 个字段——全部是
**求值产生的派生结果**（版本、分数、规则求值事实、研究型等级）。**不得**出现 features、
特征名/值、请求或响应体、header 集合、原始 path/query、数据库路径、凭据、异常文本、
CSV 行、标签或批准/拒绝字段。

## 读回时要重新校验一切

数据库是**可变运行时产物**：它可能被外部工具改写、被中断的写入破坏，或来自别的版本。
因此 :meth:`StoredDecision.from_row` 重新校验每个字段的类型、取值与**跨字段不变量**
（布尔必须是 0/1、数字必须有限、UUID 必须是规范小写 v4、时间必须是 UTC ``Z``、
等级必须与信号点一致、不可评估 ID 必须与完整性一致）。损坏的行**不得**被真值转换或
静默修补——宁可明确报 ``storage_contract_violation``。

## 冻结字符串的来源

``policy_semantics`` 直接取自 :mod:`ai_detector.rules.config`；``score_semantics`` 与
``threshold_policy`` 取自 :mod:`ai_detector.storage.config`（那里解释了为什么不直接
依赖 ``service`` 包）。两者都由测试与 API 契约逐字比对，避免记录 schema 与接口语义漂移。
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Mapping, Sequence

from ai_detector.rules.config import (
    ALL_RISK_LEVELS,
    FLAGGED_SIGNAL_POINTS,
    FROZEN_BANDS,
    MATCHED_RULE_SIGNAL_POINTS,
    MAX_SIGNAL_POINTS,
    RISK_LEVEL_INDETERMINATE,
    RISK_LEVELS,
)
from ai_detector.rules.engine import RiskAssessment
from ai_detector.storage.config import (
    RECORD_SCHEMA_VERSION,
    SCORE_SEMANTICS,
    THRESHOLD_POLICY,
)
from ai_detector.rules.config import POLICY_SEMANTICS

#: 规范小写 UUID4。
_UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

#: UTC RFC 3339，秒级精度，后缀 ``Z``。
_CREATED_AT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CREATED_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class StorageContractError(ValueError):
    """记录结构或取值违反冻结契约。

    **不得**在消息里回显具体取值——损坏行可能来自任意来源，把它的内容抄进错误文本
    就等于把数据库内容泄漏到响应或日志里。因此消息只描述**违反了什么规则**。
    """


def _fail(rule: str) -> StorageContractError:
    return StorageContractError(rule)


# --- 列表编码：确定性 JSON ---------------------------------------------


def encode_rule_ids(rule_ids: Sequence[str]) -> str:
    """把规则 ID 序列编码成**确定性** JSON 字符串。

    刻意不用 pickle/joblib/CSV：JSON 是跨语言、可人工复核且顺序稳定的，
    读回时也能被严格校验。
    """
    return json.dumps(list(rule_ids), ensure_ascii=False, separators=(",", ":"))


def decode_rule_ids(raw: Any, *, label: str = "rule_ids") -> tuple[str, ...]:
    """把 JSON 字符串解码成规则 ID 元组，并严格校验。"""
    if not isinstance(raw, str):
        raise _fail(f"{label} 必须是 JSON 字符串")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise _fail(f"{label} 不是合法 JSON") from error
    if not isinstance(decoded, list):
        raise _fail(f"{label} 必须是 JSON 数组")
    result: list[str] = []
    for item in decoded:
        if not isinstance(item, str) or not item.strip():
            raise _fail(f"{label} 的元素必须是非空字符串")
        result.append(item)
    if len(set(result)) != len(result):
        raise _fail(f"{label} 不得含重复元素")
    return tuple(result)


# --- 逐字段校验工具 -----------------------------------------------------


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{label} 必须是非空字符串")
    return value


def _require_optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, label)


def _require_bool(value: Any, label: str) -> bool:
    """只接受**真正的布尔值**。

    ``RuleEvaluation`` 的教训（CW-006 整改）：``1``/``0`` 是 ``int``，在真值判断里会被静默
    当成布尔，于是结构异常一路通过。这里从源头拒绝非布尔。
    """
    if not isinstance(value, bool):
        raise _fail(f"{label} 必须是布尔值")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{label} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise _fail(f"{label} 必须是有限数")
    return number


def _require_sqlite_bool(value: Any, label: str) -> bool:
    """SQLite 里的布尔：只允许整数 0/1（**不接受** Python 的 ``True``/``False``）。"""
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise _fail(f"{label} 必须是 0 或 1")
    return bool(value)


@dataclass(frozen=True)
class StoredDecision:
    """一条已持久化的决策记录（20 个允许字段，顺序即 schema 顺序）。"""

    record_schema_version: str
    decision_id: str
    created_at: str
    evaluation_request_id: str
    client_item_id: str | None
    model_version: str
    score: float
    score_semantics: str
    threshold: float
    threshold_policy: str
    flagged: bool
    policy_version: str
    policy_sha256: str
    policy_semantics: str
    rule_score: float
    matched_rule_ids: tuple[str, ...]
    unevaluable_rule_ids: tuple[str, ...]
    rule_evaluation_complete: bool
    signal_points: int | None
    risk_level: str

    def __post_init__(self) -> None:
        self._validate()

    # --- 构造 -----------------------------------------------------------

    @classmethod
    def from_assessment(
        cls,
        assessment: RiskAssessment,
        *,
        flagged: bool,
        decision_id: str,
        created_at: str,
        evaluation_request_id: str,
        client_item_id: str | None,
        model_version: str,
        score: float,
        threshold: float,
        policy_version: str,
        policy_sha256: str,
        record_schema_version: str = RECORD_SCHEMA_VERSION,
    ) -> "StoredDecision":
        """由**真实的融合结果**构造记录。

        除 ``flagged`` 外的字段直接取自 ``RiskAssessment``，因此记录不可能与冻结真值表不一致
        ——这也是"不手写派生字段"的纪律：手写一次就等于把融合逻辑复制了一份。

        ``flagged`` 必须由调用方**显式传入**，且必须与 ``fuse()`` 收到的是**同一个**布尔值。
        它**不能**从 ``RiskAssessment`` 反推：规则不可评估时 ``signal_points`` 是 ``None``，
        而 ``flagged = score >= threshold`` 是模型阈值结论，与规则是否可评估无关。
        构造时会校验 ``flagged == (score >= threshold)``。
        """
        return cls(
            record_schema_version=record_schema_version,
            decision_id=decision_id,
            created_at=created_at,
            evaluation_request_id=evaluation_request_id,
            client_item_id=client_item_id,
            model_version=model_version,
            score=score,
            score_semantics=SCORE_SEMANTICS,
            threshold=threshold,
            threshold_policy=THRESHOLD_POLICY,
            flagged=flagged,
            policy_version=policy_version,
            policy_sha256=policy_sha256,
            policy_semantics=POLICY_SEMANTICS,
            rule_score=assessment.rule_score,
            matched_rule_ids=tuple(assessment.matched_rule_ids),
            unevaluable_rule_ids=tuple(assessment.unevaluable_rule_ids),
            rule_evaluation_complete=assessment.complete,
            signal_points=assessment.signal_points,
            risk_level=assessment.risk_level,
        )

    # --- 校验 -----------------------------------------------------------

    def _validate(self) -> None:
        if self.record_schema_version != RECORD_SCHEMA_VERSION:
            raise _fail(f"record_schema_version 必须是 {RECORD_SCHEMA_VERSION}")
        if not isinstance(self.decision_id, str) or not _UUID4_PATTERN.fullmatch(self.decision_id):
            raise _fail("decision_id 必须是规范小写 UUID4")
        try:
            parsed = uuid.UUID(self.decision_id)
        except (ValueError, AttributeError) as error:  # pragma: no cover - 正则已先拦下
            raise _fail("decision_id 必须是规范小写 UUID4") from error
        if parsed.version != 4 or str(parsed) != self.decision_id:
            raise _fail("decision_id 必须是规范小写 UUID4")

        if not isinstance(self.created_at, str) or not _CREATED_AT_PATTERN.fullmatch(
            self.created_at
        ):
            raise _fail("created_at 必须是 UTC RFC 3339 且以 Z 结尾")
        try:
            datetime.strptime(self.created_at, _CREATED_AT_FORMAT)
        except ValueError as error:
            raise _fail("created_at 必须是合法的 UTC 时间") from error

        _require_text(self.evaluation_request_id, "evaluation_request_id")
        _require_optional_text(self.client_item_id, "client_item_id")
        _require_text(self.model_version, "model_version")
        _require_text(self.policy_version, "policy_version")
        if not isinstance(self.policy_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.policy_sha256
        ):
            raise _fail("policy_sha256 必须是 64 位小写十六进制字符串")

        if self.score_semantics != SCORE_SEMANTICS:
            raise _fail(f"score_semantics 必须是 {SCORE_SEMANTICS}")
        if self.threshold_policy != THRESHOLD_POLICY:
            raise _fail(f"threshold_policy 必须是 {THRESHOLD_POLICY}")
        if self.policy_semantics != POLICY_SEMANTICS:
            raise _fail(f"policy_semantics 必须是 {POLICY_SEMANTICS}")

        _require_finite(self.score, "score")
        threshold = _require_finite(self.threshold, "threshold")
        if not 0.0 <= threshold <= 1.0:
            raise _fail("threshold 必须位于 [0, 1]")
        rule_score = _require_finite(self.rule_score, "rule_score")
        if not 0.0 <= rule_score <= 1.0:
            raise _fail("rule_score 必须位于 [0, 1]")

        flagged = self.flagged
        complete = self.rule_evaluation_complete
        if not isinstance(flagged, bool) or not isinstance(complete, bool):
            raise _fail("flagged 与 rule_evaluation_complete 必须是布尔值")

        # ``flagged`` 是**模型阈值判定**，与规则是否可评估无关，因此它的真值只能来自
        # ``score >= threshold``。这条不变量在写入与读回两侧都强制，矛盾行不会被静默修补。
        if flagged != (self.score >= threshold):
            raise _fail("flagged 必须等于 score >= threshold")

        matched = self._tuple_of_ids(self.matched_rule_ids, "matched_rule_ids")
        unevaluable = self._tuple_of_ids(self.unevaluable_rule_ids, "unevaluable_rule_ids")
        if set(matched) & set(unevaluable):
            raise _fail("matched_rule_ids 与 unevaluable_rule_ids 不得重叠")

        if self.risk_level not in ALL_RISK_LEVELS:
            raise _fail("risk_level 必须是冻结的研究型等级")

        points = self.signal_points
        if points is not None:
            if isinstance(points, bool) or not isinstance(points, int):
                raise _fail("signal_points 必须是整数或 null")
            if not 0 <= points <= MAX_SIGNAL_POINTS:
                raise _fail(f"signal_points 必须位于 0-{MAX_SIGNAL_POINTS}")

        # --- 跨字段不变量：等级、点数、完整性与命中数必须互相自洽 ---
        if complete:
            if unevaluable:
                raise _fail("完整求值的记录不得含不可评估规则")
            if points is None:
                raise _fail("完整求值的记录必须有 signal_points")
            expected_points = (
                FLAGGED_SIGNAL_POINTS if flagged else 0
            ) + len(matched) * MATCHED_RULE_SIGNAL_POINTS
            if points != expected_points:
                raise _fail("signal_points 必须等于 flagged 点数加命中规则点数")
            expected_level = next(
                (band.name for band in FROZEN_BANDS if band.low <= points <= band.high), None
            )
            if self.risk_level != expected_level:
                raise _fail("risk_level 必须与 signal_points 所在的冻结区间一致")
            if self.risk_level not in RISK_LEVELS:
                raise _fail("完整求值的等级不能是 indeterminate")
        else:
            if not unevaluable:
                raise _fail("不可评估的记录必须列出不可评估的规则 ID")
            if points is not None:
                raise _fail("不可评估的记录 signal_points 必须为 null")
            if self.risk_level != RISK_LEVEL_INDETERMINATE:
                raise _fail("不可评估的等级必须是 indeterminate")

    @staticmethod
    def _tuple_of_ids(value: Any, label: str) -> tuple[str, ...]:
        if not isinstance(value, tuple):
            raise _fail(f"{label} 必须是元组")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise _fail(f"{label} 的元素必须是非空字符串")
        if len(set(value)) != len(value):
            raise _fail(f"{label} 不得含重复元素")
        return value

    # --- 编码/解码 ------------------------------------------------------

    def to_row(self) -> dict[str, Any]:
        """编码为 SQLite 友好的行：布尔 → 0/1，列表 → 确定性 JSON 字符串。"""
        return {
            "record_schema_version": self.record_schema_version,
            "decision_id": self.decision_id,
            "created_at": self.created_at,
            "evaluation_request_id": self.evaluation_request_id,
            "client_item_id": self.client_item_id,
            "model_version": self.model_version,
            "score": float(self.score),
            "score_semantics": self.score_semantics,
            "threshold": float(self.threshold),
            "threshold_policy": self.threshold_policy,
            "flagged": 1 if self.flagged else 0,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "policy_semantics": self.policy_semantics,
            "rule_score": float(self.rule_score),
            "matched_rule_ids": encode_rule_ids(self.matched_rule_ids),
            "unevaluable_rule_ids": encode_rule_ids(self.unevaluable_rule_ids),
            "rule_evaluation_complete": 1 if self.rule_evaluation_complete else 0,
            "signal_points": self.signal_points,
            "risk_level": self.risk_level,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "StoredDecision":
        """从数据库行解码并**重新校验**全部字段与跨字段不变量。

        Raises:
            StorageContractError: 列集合不符、字段类型/取值非法或跨字段矛盾。
                消息只说明违反了什么规则，不回显取值。
        """
        if not isinstance(row, Mapping):
            raise _fail("记录必须是行映射")
        expected = {field.name for field in fields(cls)}
        actual = {str(key) for key in row}
        if actual != expected:
            raise _fail("列集合与冻结记录契约不一致")

        record = cls(
            record_schema_version=_require_text(row["record_schema_version"], "record_schema_version"),
            decision_id=row["decision_id"],
            created_at=row["created_at"],
            evaluation_request_id=row["evaluation_request_id"],
            client_item_id=_require_optional_text(row["client_item_id"], "client_item_id"),
            model_version=row["model_version"],
            score=_require_finite(row["score"], "score"),
            score_semantics=row["score_semantics"],
            threshold=_require_finite(row["threshold"], "threshold"),
            threshold_policy=row["threshold_policy"],
            flagged=_require_sqlite_bool(row["flagged"], "flagged"),
            policy_version=row["policy_version"],
            policy_sha256=row["policy_sha256"],
            policy_semantics=row["policy_semantics"],
            rule_score=_require_finite(row["rule_score"], "rule_score"),
            matched_rule_ids=decode_rule_ids(row["matched_rule_ids"], label="matched_rule_ids"),
            unevaluable_rule_ids=decode_rule_ids(
                row["unevaluable_rule_ids"], label="unevaluable_rule_ids"
            ),
            rule_evaluation_complete=_require_sqlite_bool(
                row["rule_evaluation_complete"], "rule_evaluation_complete"
            ),
            signal_points=_require_signal_points(row["signal_points"]),
            risk_level=row["risk_level"],
        )
        return record

    def to_public_dict(self) -> dict[str, Any]:
        """HTTP 响应形态：布尔还原为布尔、列表还原为列表（顺序保持不变）。"""
        return {
            "record_schema_version": self.record_schema_version,
            "decision_id": self.decision_id,
            "created_at": self.created_at,
            "evaluation_request_id": self.evaluation_request_id,
            "client_item_id": self.client_item_id,
            "model_version": self.model_version,
            "score": self.score,
            "score_semantics": self.score_semantics,
            "threshold": self.threshold,
            "threshold_policy": self.threshold_policy,
            "flagged": self.flagged,
            "policy_version": self.policy_version,
            "policy_sha256": self.policy_sha256,
            "policy_semantics": self.policy_semantics,
            "rule_score": self.rule_score,
            "matched_rule_ids": list(self.matched_rule_ids),
            "unevaluable_rule_ids": list(self.unevaluable_rule_ids),
            "rule_evaluation_complete": self.rule_evaluation_complete,
            "signal_points": self.signal_points,
            "risk_level": self.risk_level,
        }


def is_canonical_decision_id(value: Any) -> bool:
    """是否是**规范小写 UUID4**。

    路由用它把形状非法的 ID 在触达存储层之前就拦成统一 404，因此带引号、SQL 关键字或超长
    参数根本不会变成一次数据库查询。
    """
    return isinstance(value, str) and bool(_UUID4_PATTERN.fullmatch(value))


def _require_signal_points(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail("signal_points 必须是整数或 null")
    return int(value)
