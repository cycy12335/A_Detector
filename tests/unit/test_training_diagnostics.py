"""常数预测器诊断文字的单元测试。

二次验收整改项：诊断文字曾把“常数不低于网格下界”概括为“选出最低阈值”，并把
“选中阈值下全部判为负”概括为“常数低于全部候选阈值”，两者都不是一般规律。这里直接
对 :func:`ai_detector.training._model_diagnostics` 断言三种情形下的文字，避免集成测试
中因固定配置只走得到第一种情形而让其余分支无人验证。
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_detector.training import _model_diagnostics

#: 与 configs/baseline.yaml 一致的网格上下界。
GRID_MIN = 0.05
GRID_MAX = 0.95


def make_proba(value: float, size: int = 100) -> np.ndarray:
    return np.full(size, value)


def test_constant_below_grid_minimum_reports_highest_threshold() -> None:
    """常数低于网格下界：选中阈值下无正例，说明所有候选并列，取网格最高阈值。"""
    diagnostics = _model_diagnostics(
        make_proba(0.047), make_proba(0.047), GRID_MAX
    )

    assert diagnostics["distinct_validation_probabilities"] == 1
    assert diagnostics["constant_probability"] == pytest.approx(0.047)
    assert diagnostics["validation_positive_rate"] == 0.0
    assert "网格最高阈值" in diagnostics["note"]
    assert "最大候选阈值" not in diagnostics["note"]


def test_constant_within_grid_reports_largest_candidate_not_grid_minimum() -> None:
    """常数不低于网格下界且判正率大于零：取不超过常数的最大候选阈值。"""
    diagnostics = _model_diagnostics(
        make_proba(0.12), make_proba(0.12), 0.10
    )

    assert diagnostics["validation_positive_rate"] == 1.0
    assert "最大候选阈值" in diagnostics["note"]
    assert "网格最高阈值" not in diagnostics["note"]
    # 选中 0.10 而不是网格下界 0.05，文字必须体现这一点。
    assert diagnostics["note"].count("0.10") == 1


def test_note_never_claims_constant_is_below_grid_without_evidence() -> None:
    """验证集无正例时，即使常数不低于网格下界也会落进判正率为零的分支。

    此时文字不得断言“常数低于全部候选阈值”，因为该断言在此情形下不成立。
    """
    diagnostics = _model_diagnostics(
        make_proba(0.12), make_proba(0.12), GRID_MAX
    )

    assert diagnostics["constant_probability"] == pytest.approx(0.12)
    assert diagnostics["validation_positive_rate"] == 0.0
    assert "低于全部候选阈值" not in diagnostics["note"]
    assert "网格最高阈值" in diagnostics["note"]


def test_non_constant_predictor_gets_no_note() -> None:
    """非常数预测器不产生常数说明，避免把一般模型误标成无区分能力。"""
    diagnostics = _model_diagnostics(
        np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100), 0.5
    )

    assert diagnostics["distinct_validation_probabilities"] == 100
    assert "note" not in diagnostics
    assert "constant_probability" not in diagnostics


def test_diagnostics_record_positive_rates_for_both_splits() -> None:
    """诊断信息必须同时记录验证集与测试集的判正率。"""
    diagnostics = _model_diagnostics(make_proba(0.9), make_proba(0.1), 0.5)

    assert diagnostics["validation_positive_rate"] == 1.0
    assert diagnostics["test_positive_rate"] == 0.0
