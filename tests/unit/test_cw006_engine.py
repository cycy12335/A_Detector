"""CW-006 纯规则引擎与融合真值表的单元测试。

本文件把三条冻结语义钉在**逐例**断言上，而不是抽查：

1. 比较运算符是严格 ``>`——恰好等于阈值必须**不命中**（``>=`` 的实现在此失败）；
2. 规则字段的不可评估语义——null、字符串、bool、非有限数、缺列都记为 unevaluable，
   且**绝不做字符串到数字的隐式转换**（``"2000"`` 虽然"看起来大于"阈值，也不得命中）；
3. 0–4 点融合真值表逐例正确，任一规则不可评估时整体为 ``indeterminate`` 且 signal points 为 null。
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ai_detector.rules.config import POLICY_SCHEMA_VERSION, load_policy
from ai_detector.rules.engine import (
    PolicyContractError,
    PolicyEngine,
    RuleEvaluation,
    RuleEvaluationResult,
    fuse,
    validate_rule_result,
)
from tests.cw006_fixtures import sha256_of, stage_dir, stage_policy

AMT_RULE = "transaction_amt_train_q99_tail"
C13_RULE = "c13_train_q99_tail"
AMT_THRESHOLD = 1092.0
C13_THRESHOLD = 580.0


@pytest.fixture
def engine(tmp_path: Path) -> PolicyEngine:
    """通过**生产加载函数**加载政策（项目相对路径的真实链路）。"""
    policy_file, relative = stage_policy(tmp_path)
    policy, digest = load_policy(
        relative,
        expected_sha256=sha256_of(policy_file),
        expected_schema_version=POLICY_SCHEMA_VERSION,
    )
    return PolicyEngine(policy, policy_sha256=digest)


def _evaluate(engine: PolicyEngine, *, amt: object = 0.0, c13: object = 0.0) -> RuleEvaluationResult:
    features: dict[str, object] = {"TransactionAmt": amt, "C13": c13}
    return engine.evaluate_rules(features)


def _verdict(result: RuleEvaluationResult) -> dict[str, bool]:
    return {item.rule_id: item.matched for item in result.evaluations}


# --- 严格 `>` 边界 ------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, False),
        (1091.999, False),
        (1092.0, False),  # 恰好等于阈值：**不命中**
        (1092.001, True),
        (2000.0, True),
        (1092, False),  # 整数形式的阈值同样不命中
        (1093, True),
    ],
)
def test_transaction_amt_uses_a_strict_greater_than(
    engine: PolicyEngine, value: object, expected: bool
) -> None:
    result = _evaluate(engine, amt=value)
    assert _verdict(result)[AMT_RULE] is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, False),
        (579.9, False),
        (580.0, False),  # 恰好等于阈值：**不命中**
        (580.1, True),
        (9999.0, True),
        (580, False),
        (581, True),
    ],
)
def test_c13_uses_a_strict_greater_than(
    engine: PolicyEngine, value: object, expected: bool
) -> None:
    result = _evaluate(engine, c13=value)
    assert _verdict(result)[C13_RULE] is expected


def test_both_rules_can_match_independently(engine: PolicyEngine) -> None:
    both = _evaluate(engine, amt=5000.0, c13=900.0)
    assert _verdict(both) == {AMT_RULE: True, C13_RULE: True}
    neither = _evaluate(engine, amt=1.0, c13=1.0)
    assert _verdict(neither) == {AMT_RULE: False, C13_RULE: False}
    only_c13 = _evaluate(engine, amt=1.0, c13=900.0)
    assert _verdict(only_c13) == {AMT_RULE: False, C13_RULE: True}


# --- 不可评估：null / 字符串 / bool / 非有限 / 缺列 -----------------------


@pytest.mark.parametrize("value", [None, "2000.0", "1092.0", "", "  ", True, False, [], {}])
def test_non_numeric_rule_values_are_unevaluable(engine: PolicyEngine, value: object) -> None:
    """字符串不做隐式数字转换：``"2000.0"`` 数值上大于阈值，但语义上是不可评估。"""
    result = _evaluate(engine, amt=value)
    assert result.unevaluable_rule_ids == (AMT_RULE,)
    assert AMT_RULE not in result.matched_rule_ids
    assert result.complete is False
    # 另一条规则仍然可评估：不可评估是**逐条**的。
    assert "c13_train_q99_tail" not in result.unevaluable_rule_ids


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rule_values_are_unevaluable(engine: PolicyEngine, value: float) -> None:
    result = _evaluate(engine, amt=value)
    assert result.unevaluable_rule_ids == (AMT_RULE,)
    assert result.complete is False


def test_both_rules_unevaluable_is_reported_together(engine: PolicyEngine) -> None:
    result = _evaluate(engine, amt=None, c13="not-a-number")
    assert result.unevaluable_rule_ids == (AMT_RULE, C13_RULE)
    assert result.matched_rule_ids == ()
    assert result.complete is False


def test_missing_rule_feature_is_unevaluable(engine: PolicyEngine) -> None:
    """规则字段整体缺列时记为不可评估，而不是抛异常（HTTP 层的列集合契约另行拦下）。"""
    result = engine.evaluate_rules({"something_else": 1.0})
    assert result.unevaluable_rule_ids == (AMT_RULE, C13_RULE)


def test_numeric_string_is_never_coerced_even_when_it_would_match(engine: PolicyEngine) -> None:
    """把这条单独钉死：任何"顺手转一下数字"的实现都会在这里失败。"""
    coerced = _evaluate(engine, amt="99999", c13="99999")
    assert coerced.matched_rule_ids == ()
    assert set(coerced.unevaluable_rule_ids) == {AMT_RULE, C13_RULE}


# --- 纯函数性质 ---------------------------------------------------------


def test_evaluation_does_not_modify_the_input(engine: PolicyEngine) -> None:
    features: dict[str, object] = {"TransactionAmt": 2000.0, "C13": 1.0, "extra": "keep me"}
    snapshot = dict(features)
    engine.evaluate_rules(features)
    assert features == snapshot


def test_rule_order_follows_the_policy(engine: PolicyEngine) -> None:
    result = engine.evaluate_rules({"TransactionAmt": 2000.0, "C13": 900.0})
    assert [item.rule_id for item in result.evaluations] == [AMT_RULE, C13_RULE]
    assert result.matched_rule_ids == (AMT_RULE, C13_RULE)


def test_evaluation_is_deterministic(engine: PolicyEngine) -> None:
    features = {"TransactionAmt": 2000.0, "C13": 100.0}
    first = engine.evaluate_rules(features)
    second = engine.evaluate_rules(features)
    assert first == second


def test_engine_needs_no_io_after_load(engine: PolicyEngine, tmp_path: Path) -> None:
    """求值阶段不读文件：删掉政策文件后仍然照常给出同样的结果。"""
    before = engine.evaluate_rules({"TransactionAmt": 2000.0, "C13": 900.0})
    removed = 0
    for path in stage_dir(tmp_path).glob("*.yaml"):
        path.unlink()
        removed += 1
    assert removed == 1, "应当确实删掉了那份政策文件"
    after = engine.evaluate_rules({"TransactionAmt": 2000.0, "C13": 900.0})
    assert before == after


def test_evaluation_results_are_immutable(engine: PolicyEngine) -> None:
    result = engine.evaluate_rules({"TransactionAmt": 2000.0, "C13": 1.0})
    with pytest.raises(FrozenInstanceError):
        result.evaluations[0].matched = False  # type: ignore[misc]


# --- 融合真值表（0–4 点逐例） ------------------------------------------


@pytest.mark.parametrize(
    "amt,c13,flagged,points,level",
    [
        # 无命中、未标记 -> 0 点
        (1.0, 1.0, False, 0, "low_signal"),
        # 命中一条规则 -> 1 点
        (2000.0, 1.0, False, 1, "elevated_signal"),
        (1.0, 900.0, False, 1, "elevated_signal"),
        # 命中两条规则，或仅模型标记 -> 2 点
        (2000.0, 900.0, False, 2, "elevated_signal"),
        (1.0, 1.0, True, 2, "elevated_signal"),
        # 模型标记 + 一条规则 -> 3 点
        (2000.0, 1.0, True, 3, "high_signal"),
        (1.0, 900.0, True, 3, "high_signal"),
        # 模型标记 + 两条规则 -> 4 点
        (2000.0, 900.0, True, 4, "high_signal"),
    ],
)
def test_fusion_truth_table(
    engine: PolicyEngine, amt: float, c13: float, flagged: bool, points: int, level: str
) -> None:
    result = engine.evaluate_rules({"TransactionAmt": amt, "C13": c13})
    assessment = fuse(result, flagged=flagged)
    assert assessment.signal_points == points
    assert assessment.risk_level == level
    assert assessment.complete is True
    assert assessment.rule_score == len(result.matched_rule_ids) / 2


@pytest.mark.parametrize(
    "amt,c13,flagged,expected_score",
    [
        (1.0, 1.0, False, 0.0),
        (2000.0, 1.0, False, 0.5),
        (2000.0, 900.0, False, 1.0),
        (2000.0, 900.0, True, 1.0),
    ],
)
def test_rule_score_is_matched_over_two(
    engine: PolicyEngine, amt: float, c13: float, flagged: bool, expected_score: float
) -> None:
    assessment = fuse(engine.evaluate_rules({"TransactionAmt": amt, "C13": c13}), flagged=flagged)
    assert assessment.rule_score == pytest.approx(expected_score)


@pytest.mark.parametrize("amt,c13", [(None, 900.0), (2000.0, None), ("x", 1.0), (1.0, float("nan"))])
def test_any_unevaluable_rule_makes_the_level_indeterminate(
    engine: PolicyEngine, amt: object, c13: object
) -> None:
    result = engine.evaluate_rules({"TransactionAmt": amt, "C13": c13})
    assessment = fuse(result, flagged=True)
    assert assessment.risk_level == "indeterminate"
    assert assessment.signal_points is None
    assert assessment.complete is False
    assert assessment.unevaluable_rule_ids == result.unevaluable_rule_ids
    assert assessment.unevaluable_rule_ids, "必须列出不可评估的规则 ID"


def test_indeterminate_still_reports_the_matched_rule_score(engine: PolicyEngine) -> None:
    """命中与否是已知事实，仍然给出 rule_score；被置空的是 signal points 与等级。"""
    assessment = fuse(
        engine.evaluate_rules({"TransactionAmt": 2000.0, "C13": None}), flagged=False
    )
    assert assessment.rule_score == pytest.approx(0.5)
    assert assessment.matched_rule_ids == (AMT_RULE,)
    assert assessment.unevaluable_rule_ids == (C13_RULE,)
    assert assessment.signal_points is None
    assert assessment.risk_level == "indeterminate"


def test_point_values_never_exceed_four(engine: PolicyEngine) -> None:
    """上界由"2 点模型 + 2 条规则"决定，不是任意区间。"""
    assessment = fuse(
        engine.evaluate_rules({"TransactionAmt": 1e9, "C13": 1e9}), flagged=True
    )
    assert assessment.signal_points == 4


# --- 结构校验：坏引擎输出不得被当成合法结果 -----------------------------


def test_matched_and_unevaluable_at_once_is_a_contract_violation(engine: PolicyEngine) -> None:
    broken = RuleEvaluationResult(
        evaluations=(
            RuleEvaluation(rule_id=AMT_RULE, matched=True, evaluable=False),
            RuleEvaluation(rule_id=C13_RULE, matched=False, evaluable=True),
        )
    )
    with pytest.raises(PolicyContractError):
        validate_rule_result(engine, broken)


def test_unknown_rule_id_is_a_contract_violation(engine: PolicyEngine) -> None:
    broken = RuleEvaluationResult(
        evaluations=(RuleEvaluation(rule_id="not_a_rule", matched=True, evaluable=True),)
    )
    with pytest.raises(PolicyContractError):
        validate_rule_result(engine, broken)


def test_duplicate_rule_id_is_a_contract_violation(engine: PolicyEngine) -> None:
    broken = RuleEvaluationResult(
        evaluations=(
            RuleEvaluation(rule_id=AMT_RULE, matched=True, evaluable=True),
            RuleEvaluation(rule_id=AMT_RULE, matched=False, evaluable=True),
        )
    )
    with pytest.raises(PolicyContractError):
        validate_rule_result(engine, broken)


def test_well_formed_result_passes_validation(engine: PolicyEngine) -> None:
    good = engine.evaluate_rules({"TransactionAmt": 2000.0, "C13": 1.0})
    validate_rule_result(engine, good)


@pytest.mark.parametrize("value", [1, 0, 1.0, 0.0, "true", "false", "", None, [], [0]])
def test_non_boolean_matched_is_a_contract_violation(engine: PolicyEngine, value: object) -> None:
    """验收阻塞项：``matched`` 必须是**真正的布尔值**。

    整型 ``1``/``0`` 是"看起来像布尔"的最危险情形——Python 里 ``bool`` 是 ``int`` 的子类，
    真值判断会把 ``1`` 当成命中、把 ``0`` 当成未命中，于是结构异常被静默当成合法分级依据。
    字符串、``None`` 与其它真值对象同样必须被拒。
    """
    broken = RuleEvaluationResult(
        evaluations=(
            RuleEvaluation(rule_id=AMT_RULE, matched=value, evaluable=True),
            RuleEvaluation(rule_id=C13_RULE, matched=False, evaluable=True),
        )
    )
    with pytest.raises(PolicyContractError):
        validate_rule_result(engine, broken)


@pytest.mark.parametrize("value", [1, 0, 1.0, 0.0, "true", "false", None, []])
def test_non_boolean_evaluable_is_a_contract_violation(
    engine: PolicyEngine, value: object
) -> None:
    """``evaluable`` 同样必须是真正的布尔值，否则"是否完整评估"就成了含糊的真值判断。"""
    broken = RuleEvaluationResult(
        evaluations=(
            RuleEvaluation(rule_id=AMT_RULE, matched=False, evaluable=value),
            RuleEvaluation(rule_id=C13_RULE, matched=False, evaluable=True),
        )
    )
    with pytest.raises(PolicyContractError):
        validate_rule_result(engine, broken)


def test_genuine_booleans_are_still_accepted(engine: PolicyEngine) -> None:
    """收紧类型不得误伤合法结果：真正的布尔值照常通过。"""
    for matched, evaluable in ((True, True), (False, True), (False, False)):
        good = RuleEvaluationResult(
            evaluations=(
                RuleEvaluation(rule_id=AMT_RULE, matched=matched, evaluable=evaluable),
                RuleEvaluation(rule_id=C13_RULE, matched=False, evaluable=True),
            )
        )
        validate_rule_result(engine, good)


def test_non_result_container_is_a_contract_violation(engine: PolicyEngine) -> None:
    """传进来的根本不是求值结果对象时，同样按政策契约破坏处理。"""
    for bogus in ({}, [], None, "not-a-result"):
        with pytest.raises(PolicyContractError):
            validate_rule_result(engine, bogus)  # type: ignore[arg-type]


def test_fusion_bands_are_exhaustive(engine: PolicyEngine) -> None:
    """0–4 每个可达点值都必须落到一个已知等级，不能有空洞。"""
    seen: dict[int, str] = {}
    for flagged in (False, True):
        for amt, c13 in ((1.0, 1.0), (2000.0, 1.0), (2000.0, 900.0)):
            assessment = fuse(
                engine.evaluate_rules({"TransactionAmt": amt, "C13": c13}), flagged=flagged
            )
            assert assessment.signal_points is not None
            seen[assessment.signal_points] = assessment.risk_level
    assert seen == {
        0: "low_signal",
        1: "elevated_signal",
        2: "elevated_signal",
        3: "high_signal",
        4: "high_signal",
    }
    assert all(math.isfinite(float(score)) for score in (0.0, 0.5, 1.0))
