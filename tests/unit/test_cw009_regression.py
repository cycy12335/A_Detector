"""CW-009 回归比较：生产顺序、绝对容差、确定性、失败只记字段名。

这里的断言刻意"逐字段"：任何一个冻结字段漂移都必须被点名，且失败记录里**只有**
case ID 与字段名——没有取值。
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from ai_detector.config import PROJECT_ROOT
from ai_detector.quality import regression as regression_module
from ai_detector.quality.corpus import parse_corpus
from ai_detector.quality.errors import QualityGateErrorCode
from ai_detector.quality.regression import (
    CaseActual,
    aggregate_error_code,
    failed_fields_by_case,
    run_regression,
    score_case,
)
from ai_detector.service.config import ServiceConfig
from ai_detector.service.runtime import load_policy_engine, load_runtime

CORPUS_REL = "tests/fixtures/cw009/model_regression_cases.json"


@pytest.fixture(scope="module")
def runtime():
    return load_runtime(ServiceConfig.from_yaml("configs/service.yaml"))


@pytest.fixture(scope="module")
def engine():
    return load_policy_engine(ServiceConfig.from_yaml("configs/service.yaml"))


@pytest.fixture(scope="module")
def corpus(runtime):
    raw = json.loads((PROJECT_ROOT / CORPUS_REL).read_text(encoding="utf-8"))
    return parse_corpus(
        raw,
        source_features=runtime.source_features,
        categorical_features=runtime.categorical_features,
    )


def with_expected(corpus, case_id: str, **changes):
    """在**内存副本**上篡改某个 case 的 expected（不改磁盘语料）。"""
    cases = tuple(
        dataclasses.replace(case, expected=dataclasses.replace(case.expected, **changes))
        if case.case_id == case_id
        else case
        for case in corpus.cases
    )
    return dataclasses.replace(corpus, cases=cases)


# --- 正常路径 -----------------------------------------------------------


def test_regression_passes_and_is_deterministic(runtime, engine, corpus) -> None:
    outcome = run_regression(runtime, engine, corpus)

    assert outcome.passed
    assert outcome.cases_total == 12
    assert outcome.failures == ()
    assert outcome.deterministic is True
    assert outcome.max_score_absolute_deviation == 0.0
    assert aggregate_error_code(outcome) is None


def test_frozen_distribution_matches_frozen_constants(runtime, engine, corpus) -> None:
    outcome = run_regression(runtime, engine, corpus)

    assert outcome.expected_distribution == {
        "model_flagged_true": 3,
        "model_flagged_false": 9,
        "rule_match_counts": {"0": 7, "1": 2 + 1, "2": 2},
        "risk_level_counts": {
            "elevated_signal": 4,
            "high_signal": 2,
            "low_signal": 6,
        },
    }
    assert outcome.actual_distribution == outcome.expected_distribution
    assert outcome.distribution_matches is True


def test_strict_greater_than_boundary_does_not_match(runtime, engine, corpus) -> None:
    """恰好等于冻结阈值不得命中——严格 ``>`` 是冻结语义。"""
    for case_id in ("case_amt_equal", "case_c13_equal"):
        actual = score_case(runtime, engine, corpus.case_by_id(case_id))
        assert actual.matched_rule_ids == ()

    for case_id in ("case_amt_above", "case_c13_above"):
        actual = score_case(runtime, engine, corpus.case_by_id(case_id))
        assert len(actual.matched_rule_ids) == 1


def test_two_runs_in_one_process_are_bit_identical(runtime, engine, corpus) -> None:
    first = {case.case_id: score_case(runtime, engine, case) for case in corpus.cases}
    second = {case.case_id: score_case(runtime, engine, case) for case in corpus.cases}
    assert first == second


def test_rule_fields_are_all_evaluable_in_the_frozen_corpus(runtime, engine, corpus) -> None:
    for case in corpus.cases:
        actual = score_case(runtime, engine, case)
        assert actual.unevaluable_rule_ids == ()


# --- 分数比较 -----------------------------------------------------------


def test_tampered_expected_score_fails(runtime, engine, corpus) -> None:
    tampered = with_expected(corpus, "case_stress_flagged", score=0.5)
    outcome = run_regression(runtime, engine, tampered)

    assert not outcome.passed
    assert failed_fields_by_case(outcome) == {"case_stress_flagged": ["score"]}
    assert aggregate_error_code(outcome) == QualityGateErrorCode.REGRESSION_MISMATCH


def test_comparison_uses_absolute_not_relative_tolerance(runtime, engine, corpus) -> None:
    """把分数挪动 1e-9（相对量级约 1e-8，绝对量级远超 1e-12）必须失败。

    如果实现用了相对容差或"四舍五入后比较"，这条会静默通过。
    """
    original = corpus.case_by_id("case_base_calm").expected.score
    tampered = with_expected(corpus, "case_base_calm", score=original + 1e-9)
    outcome = run_regression(runtime, engine, tampered)
    assert failed_fields_by_case(outcome) == {"case_base_calm": ["score"]}

    # 容差之内的小扰动则必须通过（证明容差确实生效，而不是"任何差异都失败"）。
    inside = with_expected(corpus, "case_base_calm", score=original + 1e-15)
    assert run_regression(runtime, engine, inside).passed


def test_max_deviation_is_reported_for_failing_case(runtime, engine, corpus) -> None:
    original = corpus.case_by_id("case_c13_above").expected.score
    tampered = with_expected(corpus, "case_c13_above", score=original + 0.25)
    outcome = run_regression(runtime, engine, tampered)
    assert outcome.max_score_absolute_deviation == pytest.approx(0.25, abs=1e-9)


# --- 派生字段逐值比较 ---------------------------------------------------


@pytest.mark.parametrize(
    "case_id, field, bad_value",
    [
        ("case_stress_flagged", "model_flagged", False),
        ("case_base_calm", "model_flagged", True),
        ("case_stress_amt_above", "risk_level", "low_signal"),
        ("case_base_calm", "risk_level", "high_signal"),
        ("case_amt_above", "matched_rule_ids", ()),
        ("case_base_calm", "matched_rule_ids", ("transaction_amt_train_q99_tail",)),
        ("case_amt_above", "matched_rule_ids", ("c13_train_q99_tail",)),
        ("case_amt_above", "rule_score", 1.0),
        ("case_stress_amt_above", "signal_points", 4),
        ("case_base_calm", "signal_points", None),
        ("case_base_calm", "unevaluable_rule_ids", ("c13_train_q99_tail",)),
    ],
)
def test_every_frozen_field_is_compared_exactly(
    runtime, engine, corpus, case_id: str, field: str, bad_value: object
) -> None:
    tampered = with_expected(corpus, case_id, **{field: bad_value})
    outcome = run_regression(runtime, engine, tampered)

    assert failed_fields_by_case(outcome) == {case_id: [field]}
    assert aggregate_error_code(outcome) == QualityGateErrorCode.REGRESSION_MISMATCH


def test_failed_fields_contain_no_values(corpus) -> None:
    """失败记录只允许 case ID 与字段名。"""
    from ai_detector.quality.regression import CaseFailure

    failure = CaseFailure(case_id="case_base_calm", fields=("score", "risk_level"))
    assert failure.fields == ("score", "risk_level")
    assert set(failure.fields) <= set(regression_module.COMPARED_FIELDS)


def test_compared_field_names_are_the_frozen_expected_fields() -> None:
    from ai_detector.quality.corpus import RegressionExpectation

    frozen = set(RegressionExpectation.__dataclass_fields__)
    assert set(regression_module.COMPARED_FIELDS) == frozen


# --- 确定性与分布 -------------------------------------------------------


def test_nondeterminism_is_detected_and_fails_separately(
    monkeypatch, runtime, engine, corpus
) -> None:
    """同一进程里两次结果只要有一位不同，就单独以"非确定性"失败。"""
    calls: dict[str, int] = {}
    original = regression_module.score_case

    def flaky(runtime_, engine_, case):
        calls[case.case_id] = calls.get(case.case_id, 0) + 1
        actual: CaseActual = original(runtime_, engine_, case)
        if calls[case.case_id] > 1:
            return dataclasses.replace(actual, score=actual.score + 1e-9)
        return actual

    monkeypatch.setattr(regression_module, "score_case", flaky)
    outcome = run_regression(runtime, engine, corpus)

    assert outcome.deterministic is False
    assert aggregate_error_code(outcome) == QualityGateErrorCode.REGRESSION_NONDETERMINISTIC


def test_nondeterminism_takes_priority_over_value_mismatch(runtime, engine, corpus) -> None:
    outcome = dataclasses.replace(
        run_regression(runtime, engine, corpus),
        deterministic=False,
        distribution_matches=False,
    )
    assert aggregate_error_code(outcome) == QualityGateErrorCode.REGRESSION_NONDETERMINISTIC


def test_distribution_mismatch_fails_separately(monkeypatch, runtime, engine, corpus) -> None:
    """逐 case 全对、但分布与 frozen expected 不一致时，必须单独以分布失败。"""
    original = regression_module._distribution

    def shifted(results):
        summary = dict(original(results))
        summary["model_flagged_true"] = summary["model_flagged_true"] + 1
        return summary

    monkeypatch.setattr(regression_module, "_distribution", shifted)
    outcome = run_regression(runtime, engine, corpus)

    assert outcome.failures == ()
    assert outcome.deterministic is True
    assert outcome.distribution_matches is False
    assert aggregate_error_code(outcome) == QualityGateErrorCode.REGRESSION_BASELINE_MISMATCH


def test_expected_distribution_is_derived_from_frozen_expected(corpus) -> None:
    """基线分布来自 frozen expected，而不是来自实际结果反推。"""
    expected = regression_module._expected_distribution(corpus)
    flagged = sum(1 for case in corpus.cases if case.expected.model_flagged)
    assert expected["model_flagged_true"] == flagged
    assert sum(expected["rule_match_counts"].values()) == len(corpus.cases)
    assert sum(expected["risk_level_counts"].values()) == len(corpus.cases)
