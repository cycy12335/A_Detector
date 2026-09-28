"""基线模型构造与阈值选择测试。"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from ai_detector.models.baseline import (
    DUMMY_MODEL_KEY,
    LOGISTIC_MODEL_KEY,
    ModelSpecError,
    build_estimator,
    build_model_specs,
)
from ai_detector.models.threshold import (
    DEFAULT_THRESHOLD_GRID,
    select_threshold,
)

VALID_CONFIG = {
    "dummy": {"type": "dummy_classifier", "params": {"strategy": "prior"}},
    "logistic_regression": {"type": "logistic_regression", "params": {"C": 1.0}},
}


def test_model_specs_require_both_baselines() -> None:
    """缺少任一必需基线都必须报错，避免只跑一个模型就宣称完成。"""
    with pytest.raises(ModelSpecError, match="缺少必需基线"):
        build_model_specs({"dummy": {"type": "dummy_classifier"}})


def test_unsupported_model_type_is_rejected() -> None:
    """P0 只允许两个基线；引入 LightGBM 等应被显式拒绝。"""
    config = {**VALID_CONFIG, "lightgbm": {"type": "lightgbm"}}
    with pytest.raises(ModelSpecError, match="不受支持"):
        build_model_specs(config)


def test_model_spec_requires_type_field() -> None:
    with pytest.raises(ModelSpecError, match="type"):
        build_model_specs({**VALID_CONFIG, "broken": {"params": {}}})


def test_build_estimator_returns_expected_classes() -> None:
    specs = {spec.key: spec for spec in build_model_specs(VALID_CONFIG)}

    dummy = build_estimator(specs[DUMMY_MODEL_KEY], seed=42)
    logistic = build_estimator(specs[LOGISTIC_MODEL_KEY], seed=42)

    assert isinstance(dummy, DummyClassifier)
    assert isinstance(logistic, LogisticRegression)
    assert logistic.random_state == 42
    assert logistic.class_weight == "balanced"
    assert logistic.max_iter == 1000


def test_logistic_params_can_be_overridden() -> None:
    config = {
        **VALID_CONFIG,
        "logistic_regression": {"type": "logistic_regression", "params": {"C": 0.25}},
    }
    specs = {spec.key: spec for spec in build_model_specs(config)}
    logistic = build_estimator(specs[LOGISTIC_MODEL_KEY], seed=1)

    assert logistic.C == 0.25


def test_threshold_grid_is_fixed_and_data_independent() -> None:
    """固定网格必须只由代码决定，不受任何数据影响。"""
    assert len(DEFAULT_THRESHOLD_GRID) == 19
    assert DEFAULT_THRESHOLD_GRID[0] == 0.05
    assert DEFAULT_THRESHOLD_GRID[-1] == 0.95
    assert list(DEFAULT_THRESHOLD_GRID) == sorted(DEFAULT_THRESHOLD_GRID)
    assert len(set(DEFAULT_THRESHOLD_GRID)) == len(DEFAULT_THRESHOLD_GRID)


def test_threshold_selection_maximises_validation_f1() -> None:
    """选出的阈值必须真的是网格内验证集 F1 最大的那个。"""
    labels = np.array([0] * 80 + [1] * 20)
    probabilities = np.concatenate(
        [np.linspace(0.0, 0.6, 80), np.linspace(0.3, 0.95, 20)]
    )

    choice = select_threshold(labels, probabilities, model_key="logistic_regression")

    def f1_at(threshold: float) -> float:
        predictions = (probabilities >= threshold).astype(int)
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    best_possible = max(f1_at(threshold) for threshold in DEFAULT_THRESHOLD_GRID)
    assert choice.threshold in DEFAULT_THRESHOLD_GRID
    assert choice.validation_f1 == pytest.approx(best_possible)
    assert choice.n_candidates == len(DEFAULT_THRESHOLD_GRID)
    assert choice.rule


def test_threshold_selection_is_deterministic() -> None:
    labels = np.array([0] * 50 + [1] * 10)
    probabilities = np.linspace(0.0, 1.0, 60)

    first = select_threshold(labels, probabilities, model_key="m")
    second = select_threshold(labels, probabilities, model_key="m")

    assert first == second


def test_constant_below_grid_selects_highest_threshold() -> None:
    """常数概率低于网格下界时，所有候选都只预测负例，应返回网格最高阈值。

    本仓库当前 Dummy 基线的实际情形：常数约 0.047 < 网格下界 0.05。
    """
    labels = np.array([0] * 90 + [1] * 10)
    probabilities = np.full(labels.shape, 0.047)

    choice = select_threshold(labels, probabilities, model_key="dummy")

    assert choice.threshold == max(DEFAULT_THRESHOLD_GRID)
    assert choice.validation_f1 == 0.0
    assert choice.validation_recall == 0.0


def test_constant_within_grid_selects_largest_candidate_not_grid_minimum() -> None:
    """常数不低于网格下界时，选出的是不超过该常数的**最大**候选阈值，而非网格下界。

    二次验收给出的反例：常数 0.12、默认网格 ``0.05, 0.10, 0.15, ...`` 时，
    0.05 与 0.10 都把全部样本判为正例、Recall 与 F1 完全相同，按 tie-break
    取其中最高者，因此实际选中 0.10 而不是 0.05。
    """
    labels = np.array([0] * 80 + [1] * 20)
    constant = 0.12
    probabilities = np.full(labels.shape, constant)

    choice = select_threshold(labels, probabilities, model_key="dummy")

    within_constant = [value for value in DEFAULT_THRESHOLD_GRID if value <= constant]
    assert within_constant == [0.05, 0.10]

    assert choice.threshold == 0.10
    assert choice.threshold != min(DEFAULT_THRESHOLD_GRID)
    assert choice.threshold == max(within_constant)
    # 全部样本被判为正例：Recall 为 1，Precision 等于验证集正例比例。
    assert choice.validation_recall == 1.0
    assert choice.validation_precision == pytest.approx(20 / 100)
    assert choice.validation_f1 == pytest.approx(2 * (20 / 100) * 1.0 / ((20 / 100) + 1.0))


@pytest.mark.parametrize("constant", [0.05, 0.08, 0.10, 0.12, 0.30, 0.93, 0.99])
def test_constant_within_grid_always_selects_largest_candidate_at_or_below_it(
    constant: float,
) -> None:
    """对落在网格区间内的常数，「选中不超过它的最大候选阈值」应普遍成立。"""
    labels = np.array([0] * 80 + [1] * 20)
    probabilities = np.full(labels.shape, constant)

    choice = select_threshold(labels, probabilities, model_key="dummy")

    assert choice.threshold == max(
        value for value in DEFAULT_THRESHOLD_GRID if value <= constant
    )


def test_constant_with_no_positive_labels_selects_highest_threshold() -> None:
    """常数不低于网格下界但验证集无正例时，F1 全为 0，同样返回网格最高阈值。"""
    labels = np.zeros(100, dtype=int)
    probabilities = np.full(labels.shape, 0.12)

    choice = select_threshold(labels, probabilities, model_key="dummy")

    assert choice.threshold == max(DEFAULT_THRESHOLD_GRID)
    assert choice.validation_f1 == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="长度不一致"):
        select_threshold([0, 1, 0], [0.1, 0.2], model_key="m")


def test_empty_grid_raises() -> None:
    with pytest.raises(ValueError, match="网格不能为空"):
        select_threshold([0, 1], [0.1, 0.9], model_key="m", grid=())
