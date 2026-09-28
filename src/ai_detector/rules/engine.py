"""CW-006 纯规则引擎与信号融合。

## 纯函数纪律

- 求值**无 I/O**：政策在构造时一次性注入，求值阶段不读文件（有测试在删除政策文件后
  仍然要求结果不变）；
- **无全局状态、无随机性**：同一输入逐字段给出同一结果；
- **不修改输入映射**，也不把任何载荷复制进结果——结果里只有规则 ID 与布尔事实；
- **不访问标签或数据**：规则只读政策声明的两个源特征。

## 不可评估（``indeterminate``）

规则字段为 ``null``、字符串、``bool``、非有限数或整体缺失时，该规则记为 ``evaluable=False``，
**绝不做字符串到数字的隐式转换**（``"2000.0"`` 不是"大于 1092"，而是"无法判断"）。
只要有一条规则不可评估，整体等级就是 ``indeterminate`` 且 ``signal_points`` 为 ``None``——
这是**合法且显式**的结果，不是错误，也不是 422。

## 融合真值表（冻结）

``signal_points = (flagged ? 2 : 0) + 命中规则数``，取值 0–4：

| 点数 | 等级 |
| --- | --- |
| 0 | ``low_signal`` |
| 1–2 | ``elevated_signal`` |
| 3–4 | ``high_signal`` |

``rule_score = 命中规则数 / 2``。等级是**研究型信号分级**，不是概率，也不是生产处置。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from ai_detector.rules.config import (
    FLAGGED_SIGNAL_POINTS,
    FROZEN_BANDS,
    MATCHED_RULE_SIGNAL_POINTS,
    RISK_LEVEL_INDETERMINATE,
    RULE_SCORE_DENOMINATOR,
    RiskPolicy,
    RuleSpec,
)


class PolicyContractError(RuntimeError):
    """引擎给出的求值结果与政策结构不一致。

    这表示**服务端政策逻辑已坏**（不是调用方的问题），因此对外折叠为 500
    ``policy_contract_violation``：宁可明确失败，也不把结构可疑的结果当成合法事实返回。
    """


@dataclass(frozen=True)
class RuleEvaluation:
    """单条规则的求值事实。``evaluable=False`` 时 ``matched`` 恒为 ``False``。"""

    rule_id: str
    matched: bool
    evaluable: bool


@dataclass(frozen=True)
class RuleEvaluationResult:
    """一次规则求值的完整结果，顺序与政策一致。"""

    evaluations: tuple[RuleEvaluation, ...]

    @property
    def matched_rule_ids(self) -> tuple[str, ...]:
        return tuple(item.rule_id for item in self.evaluations if item.matched)

    @property
    def unevaluable_rule_ids(self) -> tuple[str, ...]:
        return tuple(item.rule_id for item in self.evaluations if not item.evaluable)

    @property
    def matched_count(self) -> int:
        return sum(1 for item in self.evaluations if item.matched)

    @property
    def complete(self) -> bool:
        """所有规则都可评估时才为 ``True``。"""
        return all(item.evaluable for item in self.evaluations)


@dataclass(frozen=True)
class RiskAssessment:
    """规则结果与模型标志融合后的研究型风险分级。"""

    rule_score: float
    matched_rule_ids: tuple[str, ...]
    unevaluable_rule_ids: tuple[str, ...]
    complete: bool
    signal_points: int | None
    risk_level: str


@runtime_checkable
class RiskPolicyEngine(Protocol):
    """政策引擎接口：真实引擎与测试 fake 都实现它。

    应用工厂只依赖这个接口，因此测试可以注入一个不读任何文件的 fake，
    既能断言"每 app 生命周期只加载一次政策"，也能构造结构非法的结果验证防御性检查。
    """

    policy_version: str
    policy_schema_version: str
    policy_sha256: str
    policy_semantics: str
    threshold_source: str
    disclaimer: str
    rules: tuple[RuleSpec, ...]

    def evaluate_rules(self, features: Mapping[str, Any]) -> RuleEvaluationResult:
        """按政策顺序求值。"""
        ...


def _comparable(value: Any) -> float | None:
    """把规则字段取值转成可比较的有限浮点；不可比较时返回 ``None``。

    ``bool`` 必须排在 ``int`` 之前判断：``True`` 是 ``int`` 的子类，不显式拒绝就会被
    静默当成 ``1`` 参与比较。字符串**不做**数字转换——"看起来像数字"不等于"可以判断"。
    """
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


class PolicyEngine:
    """基于冻结政策的纯规则引擎。

    Args:
        policy: 已通过指纹与严格 schema 校验的政策。
        policy_sha256: 实际文件摘要（对外公布，供调用方核对用的是哪一份政策）。
    """

    def __init__(self, policy: RiskPolicy, *, policy_sha256: str) -> None:
        self._policy = policy
        self.policy_sha256 = str(policy_sha256)

    # --- 元数据 ---------------------------------------------------------

    @property
    def policy(self) -> RiskPolicy:
        return self._policy

    @property
    def policy_version(self) -> str:
        return self._policy.policy_version

    @property
    def policy_schema_version(self) -> str:
        return self._policy.schema_version

    @property
    def policy_semantics(self) -> str:
        return self._policy.semantics

    @property
    def threshold_source(self) -> str:
        return self._policy.threshold_source

    @property
    def disclaimer(self) -> str:
        return self._policy.disclaimer

    @property
    def rules(self) -> tuple[RuleSpec, ...]:
        return self._policy.rules

    # --- 求值 -----------------------------------------------------------

    def evaluate_rules(self, features: Mapping[str, Any]) -> RuleEvaluationResult:
        """逐条求值。

        比较运算符**写死**为严格 ``>`（政策加载阶段已拒绝其它运算符）：
        这里没有运算符分派、没有表达式求值、没有 ``eval``/``exec``，
        因此政策文件无法扩展成一个动态规则平台。
        """
        evaluations: list[RuleEvaluation] = []
        for rule in self._policy.rules:
            number = _comparable(features.get(rule.feature, _MISSING))
            if number is None:
                evaluations.append(RuleEvaluation(rule.rule_id, matched=False, evaluable=False))
                continue
            evaluations.append(
                RuleEvaluation(rule.rule_id, matched=number > rule.threshold, evaluable=True)
            )
        return RuleEvaluationResult(evaluations=tuple(evaluations))


class _Missing:
    """规则字段**缺列**时的哨兵。

    用它而不是 ``None``：``_comparable`` 对两者都判为不可评估，因此结果一致，
    但显式哨兵让"缺列"与"值为 null"在代码里保持可区分，避免将来某次改动
    把"缺列"误当成一个合法取值。
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - 仅供调试
        return "<missing>"


_MISSING = _Missing()


def validate_rule_result(engine: RiskPolicyEngine, result: RuleEvaluationResult) -> None:
    """校验引擎输出与政策结构一致。

    这里刻意**不信任**注入引擎的返回类型：

    - 整个结果必须是 :class:`RuleEvaluationResult`，条目必须是 :class:`RuleEvaluation`
      （鸭子类型的"看起来像"不算数——注入实现是外部输入，只按声明契约接收）；
    - ``matched`` 与 ``evaluable`` 必须是**真正的布尔值**：``1``/``0`` 是 ``int``，在真值
      判断里会被静默当成"命中/未命中"，从而让一个结构异常的结果一路参与风险分级并返回 200；
      字符串、``None`` 与其它真值对象同理。

    Raises:
        PolicyContractError: 未知/重复规则 ID、规则集合与政策不一致、条目或容器类型不符，
            或一条规则同时"命中"与"不可评估"、``matched``/``evaluable`` 不是布尔值。
    """
    if not isinstance(result, RuleEvaluationResult):
        raise PolicyContractError("求值结果不是 RuleEvaluationResult")
    if not isinstance(result.evaluations, tuple):
        raise PolicyContractError("求值结果的 evaluations 必须是元组")

    declared = [rule.rule_id for rule in engine.rules]
    seen: list[str] = []
    for item in result.evaluations:
        if not isinstance(item, RuleEvaluation):
            raise PolicyContractError("求值结果含非 RuleEvaluation 条目")
        if not isinstance(item.matched, bool) or not isinstance(item.evaluable, bool):
            # 报错误类别即可，不回显被注入的具体取值。
            raise PolicyContractError("matched 与 evaluable 必须是布尔值")
        if item.rule_id not in declared:
            raise PolicyContractError("求值结果含未知规则 ID")
        if item.rule_id in seen:
            raise PolicyContractError("求值结果含重复规则 ID")
        if item.matched and not item.evaluable:
            raise PolicyContractError("规则不能同时命中与不可评估")
        seen.append(item.rule_id)
    if seen != declared:
        raise PolicyContractError("求值结果的规则集合与政策不一致")


def _band_for(points: int) -> str:
    for band in FROZEN_BANDS:
        if band.low <= points <= band.high:
            return band.name
    raise PolicyContractError("信号点没有落在任何冻结区间内")


def fuse(result: RuleEvaluationResult, *, flagged: bool) -> RiskAssessment:
    """把规则求值结果与模型 ``flagged`` 融合成研究型风险等级（纯函数）。

    Args:
        result: 规则求值结果。
        flagged: ``score >= threshold`` 的模型标记；只是**信号**，不是处置。

    Returns:
        :class:`RiskAssessment`。任一规则不可评估时 ``risk_level`` 为
        ``indeterminate``、``signal_points`` 为 ``None``；``rule_score`` 仍按已知的
        命中数计算（命中与否是确定事实，被置空的只有等级与点数）。
    """
    matched = result.matched_rule_ids
    rule_score = len(matched) / RULE_SCORE_DENOMINATOR

    if not result.complete:
        return RiskAssessment(
            rule_score=rule_score,
            matched_rule_ids=matched,
            unevaluable_rule_ids=result.unevaluable_rule_ids,
            complete=False,
            signal_points=None,
            risk_level=RISK_LEVEL_INDETERMINATE,
        )

    points = (FLAGGED_SIGNAL_POINTS if flagged else 0) + len(matched) * MATCHED_RULE_SIGNAL_POINTS
    return RiskAssessment(
        rule_score=rule_score,
        matched_rule_ids=matched,
        unevaluable_rule_ids=(),
        complete=True,
        signal_points=points,
        risk_level=_band_for(points),
    )
