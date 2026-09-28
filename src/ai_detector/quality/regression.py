"""CW-009 固定回归：用**生产顺序**求实际结果并与冻结期望逐值比较。

## 取值顺序照抄生产链路

``/v1/risk/evaluate`` 的顺序是：特征契约 → 规则求值 → 规则结果结构校验 → 模型评分 →
``flagged = score >= threshold`` → 融合。本模块逐字照抄这一顺序，并且**只调用**生产函数：

- ``runtime.score_many``（内部走 ``LightGbmBundle.score_batch`` 与 bundle 自带的冻结阈值）；
- ``PolicyEngine.evaluate_rules`` + ``validate_rule_result``；
- ``fuse``。

这里**没有**复制模型阈值、两条规则或风险等级算法：阈值来自 ``runtime.threshold``，
规则来自政策引擎，等级来自 ``fuse``。任何一处改动都会直接反映到回归结果上。

## 比较规则

- ``score`` 用配置冻结的**绝对**容差（禁止相对容差、禁止四舍五入后比较）；
- 其余六个字段逐值**精确**相等；
- 同一进程内连续评分两次，两次的完整实际结果（含 score）必须**逐位相同**；
- 实际分布必须与"由 frozen expected 算出的"分布一致，且至少一个 false、一个 true，
  三种风险等级均非空。**不得**靠跳过失败 case 达到通过。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ai_detector.quality.corpus import (
    ROLE_BOUNDARY,
    RegressionCase,
    RegressionCorpus,
)
from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode
from ai_detector.rules.config import RISK_LEVELS
from ai_detector.rules.engine import fuse, validate_rule_result

#: 失败记录里允许出现的字段名（= ``expected`` 的字段集合，顺序固定）。
COMPARED_FIELDS: tuple[str, ...] = (
    "score",
    "model_flagged",
    "matched_rule_ids",
    "unevaluable_rule_ids",
    "rule_score",
    "signal_points",
    "risk_level",
)


@dataclass(frozen=True)
class CaseActual:
    """一个 case 的实际结果。字段集合与 :class:`RegressionExpectation` 一致。"""

    score: float
    model_flagged: bool
    matched_rule_ids: tuple[str, ...]
    unevaluable_rule_ids: tuple[str, ...]
    rule_score: float
    signal_points: int | None
    risk_level: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "model_flagged": self.model_flagged,
            "matched_rule_ids": list(self.matched_rule_ids),
            "unevaluable_rule_ids": list(self.unevaluable_rule_ids),
            "rule_score": self.rule_score,
            "signal_points": self.signal_points,
            "risk_level": self.risk_level,
        }


@dataclass(frozen=True)
class CaseFailure:
    """一个失败 case：**只记 ID 与字段名**，不记取值。"""

    case_id: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class RegressionOutcome:
    """回归阶段的完整结果（可直接摘要进报告）。"""

    cases_total: int
    failures: tuple[CaseFailure, ...]
    deterministic: bool
    max_score_absolute_deviation: float
    coverage: dict[str, Any]
    expected_distribution: dict[str, Any]
    actual_distribution: dict[str, Any]
    distribution_matches: bool

    @property
    def cases_failed(self) -> int:
        return len(self.failures)

    @property
    def passed(self) -> bool:
        return (
            not self.failures and self.deterministic and self.distribution_matches
        )


def score_case(runtime: Any, engine: Any, case: RegressionCase) -> CaseActual:
    """按生产顺序求一个 case 的实际结果。

    Raises:
        QualityGateError: 运行时抛错或引擎给出结构不一致的结果
            （``runtime_load_failed`` / ``agreement_violation``）。错误细节不外泄。
    """
    features = case.feature_mapping()
    # 1) 特征契约：生产接口是"集合相等"，语料加载阶段还额外锁死了顺序。
    if set(features) != set(runtime.source_features):
        raise QualityGateError(
            QualityGateErrorCode.AGREEMENT_VIOLATION, "case 特征集合与运行时不一致"
        )
    # 2) 规则求值 + 结构校验（与生产同序：规则先于模型调用）。
    try:
        result = engine.evaluate_rules(features)
        validate_rule_result(engine, result)
    except QualityGateError:
        raise
    except Exception as error:  # noqa: BLE001 - 引擎异常细节不外泄，只记类别
        raise QualityGateError(
            QualityGateErrorCode.RUNTIME_LOAD_FAILED,
            f"规则求值失败：{type(error).__name__}",
        ) from None
    # 3) 模型评分。阈值取运行时自带的冻结阈值，绝不硬编码。
    try:
        scores = runtime.score_many([features])
    except QualityGateError:
        raise
    except Exception as error:  # noqa: BLE001 - 同上
        raise QualityGateError(
            QualityGateErrorCode.RUNTIME_LOAD_FAILED,
            f"模型评分失败：{type(error).__name__}",
        ) from None
    if len(scores) != 1:
        raise QualityGateError(
            QualityGateErrorCode.RUNTIME_LOAD_FAILED, "模型返回的分数个数不为 1"
        )
    value = float(scores[0])
    if not math.isfinite(value):
        raise QualityGateError(
            QualityGateErrorCode.RUNTIME_LOAD_FAILED, "模型返回了非有限分数"
        )
    # 4) 融合：flagged 只算一次，同一个布尔值同时交给 fuse 与结果。
    flagged = bool(value >= runtime.threshold)
    assessment = fuse(result, flagged=flagged)
    return CaseActual(
        score=value,
        model_flagged=flagged,
        matched_rule_ids=tuple(assessment.matched_rule_ids),
        unevaluable_rule_ids=tuple(assessment.unevaluable_rule_ids),
        rule_score=float(assessment.rule_score),
        signal_points=assessment.signal_points,
        risk_level=str(assessment.risk_level),
    )


def _compare(case: RegressionCase, actual: CaseActual, tolerance: float) -> tuple[str, ...]:
    """返回不一致的字段名（空元组表示该 case 通过）。"""
    expected = case.expected
    failed: list[str] = []
    if not math.isclose(
        actual.score, expected.score, rel_tol=0.0, abs_tol=tolerance
    ):
        failed.append("score")
    if actual.model_flagged != expected.model_flagged:
        failed.append("model_flagged")
    if actual.matched_rule_ids != expected.matched_rule_ids:
        failed.append("matched_rule_ids")
    if actual.unevaluable_rule_ids != expected.unevaluable_rule_ids:
        failed.append("unevaluable_rule_ids")
    if actual.rule_score != expected.rule_score:
        failed.append("rule_score")
    if actual.signal_points != expected.signal_points:
        failed.append("signal_points")
    if actual.risk_level != expected.risk_level:
        failed.append("risk_level")
    return tuple(failed)


def _distribution(results: Mapping[str, CaseActual]) -> dict[str, Any]:
    """从一组结果统计分布：flagged、命中规则数、风险等级。"""
    match_counts = {0: 0, 1: 0, 2: 0}
    levels: dict[str, int] = {}
    flagged_true = 0
    for actual in results.values():
        if actual.model_flagged:
            flagged_true += 1
        matched = len(actual.matched_rule_ids)
        match_counts[matched] = match_counts.get(matched, 0) + 1
        levels[actual.risk_level] = levels.get(actual.risk_level, 0) + 1
    return {
        "model_flagged_true": flagged_true,
        "model_flagged_false": len(results) - flagged_true,
        "rule_match_counts": {str(key): value for key, value in sorted(match_counts.items())},
        "risk_level_counts": {key: levels[key] for key in sorted(levels)},
    }


def _expected_distribution(corpus: RegressionCorpus) -> dict[str, Any]:
    """由 **frozen expected** 算出基线分布（不是由实际结果反推）。"""
    match_counts = {0: 0, 1: 0, 2: 0}
    levels: dict[str, int] = {}
    flagged_true = 0
    for case in corpus.cases:
        if case.expected.model_flagged:
            flagged_true += 1
        matched = len(case.expected.matched_rule_ids)
        match_counts[matched] = match_counts.get(matched, 0) + 1
        levels[case.expected.risk_level] = levels.get(case.expected.risk_level, 0) + 1
    return {
        "model_flagged_true": flagged_true,
        "model_flagged_false": len(corpus.cases) - flagged_true,
        "rule_match_counts": {str(key): value for key, value in sorted(match_counts.items())},
        "risk_level_counts": {key: levels[key] for key in sorted(levels)},
    }


def check_coverage(corpus: RegressionCorpus, engine: Any) -> dict[str, Any]:
    """校验语料的**覆盖契约**（只读 frozen expected 与 frozen 规则阈值）。

    Raises:
        QualityGateError: 缺少任何一种必需组合（``corpus_contract_violation``）。
    """
    flagged_true = sum(1 for case in corpus.cases if case.expected.model_flagged)
    if flagged_true == 0 or flagged_true == len(corpus.cases):
        raise QualityGateError(
            QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
            "语料必须同时包含 model_flagged 为 true 与 false 的 case",
        )

    matched_kinds = {len(case.expected.matched_rule_ids) for case in corpus.cases}
    if not {0, 1, 2} <= matched_kinds:
        raise QualityGateError(
            QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
            "语料必须覆盖命中 0 / 1 / 2 条规则",
        )

    levels = {case.expected.risk_level for case in corpus.cases}
    missing_levels = [level for level in RISK_LEVELS if level not in levels]
    if missing_levels:
        raise QualityGateError(
            QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
            f"语料缺少风险等级：{'/'.join(missing_levels)}",
        )

    boundary_report: dict[str, Any] = {}
    for rule in engine.rules:
        probe_values: list[float] = []
        for case in corpus.cases:
            if case.role != ROLE_BOUNDARY or tuple(case.varied_features) != (rule.feature,):
                continue
            value = case.feature_mapping()[rule.feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            probe_values.append(float(value))
        below = any(value < rule.threshold for value in probe_values)
        equal = any(value == rule.threshold for value in probe_values)
        above = any(value > rule.threshold for value in probe_values)
        if not (below and equal and above):
            raise QualityGateError(
                QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
                f"规则特征 {rule.feature} 必须各有低于/等于/略高于冻结阈值的边界 case",
            )
        # 以 rule_id（而非 feature 名）作为键：报告里绝不出现源特征名。
        boundary_report[rule.rule_id] = {
            "probe_values": len(probe_values),
            "below": below,
            "equal": equal,
            "above": above,
        }

    for name in _categorical_names(corpus):
        distinct = {case.feature_mapping()[name] for case in corpus.cases}
        if len(distinct) < 2:
            raise QualityGateError(
                QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
                f"类别源特征 {name} 至少需要两种合成取值",
            )

    return {
        "cases_total": len(corpus.cases),
        "model_flagged_true": flagged_true,
        "model_flagged_false": len(corpus.cases) - flagged_true,
        "rule_match_kinds": sorted(matched_kinds),
        "risk_levels_present": [level for level in RISK_LEVELS if level in levels],
        "rule_feature_boundaries": boundary_report,
    }


def _categorical_names(corpus: RegressionCorpus) -> tuple[str, ...]:
    """语料里出现字符串取值的源特征名（即类别列）。"""
    names: list[str] = []
    for name in corpus.source_features:
        for case in corpus.cases:
            value = case.feature_mapping()[name]
            if isinstance(value, str):
                names.append(name)
                break
    return tuple(names)


def run_regression(runtime: Any, engine: Any, corpus: RegressionCorpus) -> RegressionOutcome:
    """跑完整回归：实际结果 → 双跑确定性 → 逐值比较 → 分布核对。

    Raises:
        QualityGateError: 运行时/引擎失败（``runtime_load_failed``）。
    """
    tolerance = float(corpus.score_absolute_tolerance)

    # 第一遍：实际结果。
    first = {case.case_id: score_case(runtime, engine, case) for case in corpus.cases}
    # 第二遍：同一进程内再评一次，完整结果必须逐位相同（含 score）。
    second = {case.case_id: score_case(runtime, engine, case) for case in corpus.cases}
    deterministic = all(first[key] == second[key] for key in first)

    failures: list[CaseFailure] = []
    max_deviation = 0.0
    for case in corpus.cases:
        actual = first[case.case_id]
        deviation = abs(actual.score - case.expected.score)
        max_deviation = max(max_deviation, deviation)
        fields = _compare(case, actual, tolerance)
        if fields:
            failures.append(CaseFailure(case_id=case.case_id, fields=fields))

    expected_distribution = _expected_distribution(corpus)
    actual_distribution = _distribution(first)
    distribution_matches = actual_distribution == expected_distribution

    return RegressionOutcome(
        cases_total=len(corpus.cases),
        failures=tuple(failures),
        deterministic=deterministic,
        max_score_absolute_deviation=max_deviation,
        coverage=check_coverage(corpus, engine),
        expected_distribution=expected_distribution,
        actual_distribution=actual_distribution,
        distribution_matches=distribution_matches,
    )


def aggregate_error_code(outcome: RegressionOutcome) -> str | None:
    """把回归结果折叠成**一个**稳定错误码（通过时返回 ``None``）。

    优先级：非确定性 > 逐值不一致 > 分布与 frozen expected 不一致。
    """
    if not outcome.deterministic:
        return QualityGateErrorCode.REGRESSION_NONDETERMINISTIC
    if outcome.failures:
        return QualityGateErrorCode.REGRESSION_MISMATCH
    if not outcome.distribution_matches:
        return QualityGateErrorCode.REGRESSION_BASELINE_MISMATCH
    return None


def failed_fields_by_case(outcome: RegressionOutcome) -> dict[str, list[str]]:
    """报告用：``case_id -> 失败字段名列表``（不含任何取值）。"""
    return {failure.case_id: list(failure.fields) for failure in outcome.failures}
