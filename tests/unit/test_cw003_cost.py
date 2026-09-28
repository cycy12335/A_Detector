"""CW-003 阈值曲线与相对成本的单元测试。

守住四件事：指标公式、成本公式、两套 tie-break 的确定性、以及错误路径明确失败。
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_detector.evaluation.cost import (
    ThresholdError,
    cost_matrix_note,
    metrics_at_threshold,
    select_cost_threshold,
    select_f1_threshold,
    threshold_curve,
)

F1_TIE_BREAK = ("f1_desc", "recall_desc", "threshold_desc")
COST_TIE_BREAK = ("cost_asc", "threshold_desc")


def make_labels_and_scores() -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([0] * 80 + [1] * 20)
    scores = np.concatenate([np.linspace(0.0, 0.6, 80), np.linspace(0.3, 0.95, 20)])
    return labels, scores


# --- 指标公式 -----------------------------------------------------------


def test_confusion_matrix_and_metrics_are_exact() -> None:
    labels = np.array([1, 1, 0, 0, 0, 1])
    scores = np.array([0.9, 0.4, 0.8, 0.2, 0.1, 0.7])

    metrics = metrics_at_threshold(labels, scores, threshold=0.5)

    assert (metrics.true_positive, metrics.false_positive, metrics.true_negative, metrics.false_negative) == (2, 1, 2, 1)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.false_positive_rate == pytest.approx(1 / 3)
    assert metrics.predicted_positive_rate == pytest.approx(0.5)
    assert metrics.n_samples == 6
    assert metrics.n_positive == 3


def test_threshold_comparison_is_inclusive() -> None:
    """``score >= threshold`` 记为正例：等于阈值必须算命中。"""
    labels = np.array([0, 1])
    scores = np.array([0.5, 0.5])

    metrics = metrics_at_threshold(labels, scores, threshold=0.5)

    assert metrics.true_positive == 1
    assert metrics.false_positive == 1


def test_zero_denominator_metrics_are_zero() -> None:
    """分母为 0 时指标定义为 0，而不是 NaN 或异常。"""
    labels = np.array([0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3])

    metrics = metrics_at_threshold(labels, scores, threshold=0.9)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


@pytest.mark.parametrize("bad_threshold", [0.0, 1.0, -0.5, 1.5])
def test_invalid_threshold_raises(bad_threshold: float) -> None:
    labels, scores = make_labels_and_scores()
    with pytest.raises(ThresholdError, match="阈值"):
        metrics_at_threshold(labels, scores, threshold=bad_threshold)


@pytest.mark.parametrize("bad", [-0.1, 1.2, np.nan, np.inf])
def test_invalid_scores_raise(bad: float) -> None:
    labels, scores = make_labels_and_scores()
    scores = scores.copy()
    scores[0] = bad
    with pytest.raises(ThresholdError):
        metrics_at_threshold(labels, scores, threshold=0.5)


def test_continuous_labels_raise() -> None:
    with pytest.raises(ThresholdError, match=r"\{0, 1\}"):
        metrics_at_threshold([0.0, 0.3, 1.0], [0.1, 0.5, 0.9], threshold=0.5)


def test_empty_grid_raises() -> None:
    labels, scores = make_labels_and_scores()
    with pytest.raises(ThresholdError, match="网格不能为空"):
        threshold_curve(labels, scores, [])


# --- 成本公式 -----------------------------------------------------------


def test_cost_formula_and_average() -> None:
    metrics = metrics_at_threshold(
        np.array([0, 0, 1, 1]), np.array([0.9, 0.1, 0.9, 0.1]), threshold=0.5
    )
    # tp=1, fp=1, tn=1, fn=1
    assert metrics.total_cost(fp_cost=1.0, fn_cost=1.0) == pytest.approx(2.0)
    assert metrics.total_cost(fp_cost=1.0, fn_cost=20.0) == pytest.approx(21.0)
    assert metrics.average_cost(fp_cost=1.0, fn_cost=1.0) == pytest.approx(0.5)


def test_cost_selects_minimum_and_breaks_ties_conservatively() -> None:
    """成本并列时取阈值更高者（更保守）。"""
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.full(100, 0.5)
    curve = threshold_curve(labels, scores, [0.2, 0.3, 0.4, 0.5, 0.6])

    point, total = select_cost_threshold(
        curve, fp_cost=1.0, fn_cost=1.0, tie_break=COST_TIE_BREAK
    )

    # 常数分数 0.5：阈值 <= 0.5 时全预测为正（fp=50, fn=0, 成本 50）；
    # 阈值 > 0.5 时全预测为负（fp=0, fn=50, 成本 50）。全部并列 -> 取最高阈值。
    assert total == pytest.approx(50.0)
    assert point.threshold == 0.6


def test_cost_scenario_changes_selected_threshold() -> None:
    """提高 FN 成本应使阈值不升（更倾向多报警）。"""
    labels, scores = make_labels_and_scores()
    curve = threshold_curve(labels, scores, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    cheap_fn, _ = select_cost_threshold(curve, fp_cost=1.0, fn_cost=1.0, tie_break=COST_TIE_BREAK)
    dear_fn, _ = select_cost_threshold(curve, fp_cost=1.0, fn_cost=20.0, tie_break=COST_TIE_BREAK)

    assert dear_fn.threshold <= cheap_fn.threshold


@pytest.mark.parametrize(("fp_cost", "fn_cost"), [(-1.0, 1.0), (1.0, -5.0)])
def test_negative_cost_raises(fp_cost: float, fn_cost: float) -> None:
    labels, scores = make_labels_and_scores()
    curve = threshold_curve(labels, scores, [0.5])
    with pytest.raises(ThresholdError, match="非负"):
        select_cost_threshold(curve, fp_cost=fp_cost, fn_cost=fn_cost, tie_break=COST_TIE_BREAK)


# --- tie-break ----------------------------------------------------------


def test_f1_selection_maximises_f1() -> None:
    labels, scores = make_labels_and_scores()
    grid = [round(0.05 * step, 2) for step in range(1, 20)]
    curve = threshold_curve(labels, scores, grid)

    chosen = select_f1_threshold(curve, tie_break=F1_TIE_BREAK)

    best = max(point.f1 for point in curve)
    assert chosen.f1 == pytest.approx(best)
    assert chosen.threshold in grid


def test_f1_tie_break_prefers_higher_recall_then_higher_threshold() -> None:
    """F1 并列时先取 Recall 高者，再取阈值高者。"""
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.full(100, 0.7)
    curve = threshold_curve(labels, scores, [0.3, 0.5, 0.7, 0.9])

    chosen = select_f1_threshold(curve, tie_break=F1_TIE_BREAK)

    # 阈值 <= 0.7 全预测为正：recall=1, f1=2/3；阈值 0.9 全为负：f1=0。
    assert chosen.recall == pytest.approx(1.0)
    assert chosen.threshold == 0.7


@pytest.mark.parametrize(
    ("select", "bad"),
    [
        (select_f1_threshold, ("threshold_desc",)),
        (select_cost_threshold, ("cost_desc",)),
    ],
)
def test_unsupported_tie_break_raises(select, bad) -> None:
    labels, scores = make_labels_and_scores()
    curve = threshold_curve(labels, scores, [0.5])
    with pytest.raises(ThresholdError, match="tie-break"):
        if select is select_f1_threshold:
            select(curve, tie_break=bad)
        else:
            select(curve, fp_cost=1.0, fn_cost=1.0, tie_break=bad)


def test_selection_is_deterministic() -> None:
    labels, scores = make_labels_and_scores()
    grid = [round(0.05 * step, 2) for step in range(1, 20)]

    first = select_f1_threshold(threshold_curve(labels, scores, grid), tie_break=F1_TIE_BREAK)
    second = select_f1_threshold(threshold_curve(labels, scores, grid), tie_break=F1_TIE_BREAK)

    assert first == second


# --- 成本口径 -----------------------------------------------------------


def test_cost_note_declares_relative_units() -> None:
    """成本口径必须声明为相对单位，避免被误读成货币。"""
    note = cost_matrix_note()

    assert note["units"] == "研究用相对成本单位"
    assert "false_positive" in note["formula"]
    assert "不是真实货币损失" in note["disclaimer"]
    for currency in ("人民币", "美元", "元"):
        assert currency not in note["formula"]
