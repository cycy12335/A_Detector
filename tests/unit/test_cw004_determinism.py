"""CW-004 候选运行 A/B 确定性比较的单元测试。

CW-004 任务验收要求"候选 A/B 的确定性比较能发现 best iteration、分数、阈值或
重要性差异"。这里用**构造出来的比较载荷**逐项扰动，确认每一类差异都会让比较失败，
并且失败时 ``test_opened`` 恒为 ``False``。
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from ai_detector.lightgbm_training import compare_determinism

ATOL = 1e-12


def _payload() -> dict:
    return {
        "kept_numeric": ["N1", "N2", "N3"],
        "kept_categorical": ["C1", "C2"],
        "transformed_dimension": 9,
        "output_format": "sparse_csr",
        "scale_pos_weight": 12.5,
        "best_iteration": 137,
        "validation_scores": np.linspace(0.0, 1.0, 64),
        "frozen_thresholds": {
            "selected_f1_threshold": 0.35,
            "validation_f1": 0.42,
            "validation_precision": 0.5,
            "validation_recall": 0.36,
            "cost_scenarios": {
                "fp_fn_1_1": {
                    "fp_cost": 1.0,
                    "fn_cost": 1.0,
                    "selected_threshold": 0.4,
                    "validation_total_cost": 120.0,
                },
                "fp_fn_1_20": {
                    "fp_cost": 1.0,
                    "fn_cost": 20.0,
                    "selected_threshold": 0.2,
                    "validation_total_cost": 980.0,
                },
            },
        },
        "validation_confusion_matrix": {
            "threshold": 0.35,
            "true_positive": 10,
            "false_positive": 3,
            "true_negative": 40,
            "false_negative": 11,
            "precision": 0.5,
            "recall": 0.36,
            "f1": 0.42,
        },
        "aggregated_gain_importance": {"N1": 100.0, "N2": 55.5, "C1": 12.0},
        "aggregated_split_importance": {"N1": 30, "N2": 21, "C1": 9},
    }


def _compare(left: dict, right: dict) -> dict:
    return compare_determinism(left, right, absolute_tolerance=ATOL, relative_tolerance=0.0)


def test_identical_payloads_pass() -> None:
    result = _compare(_payload(), _payload())
    assert result["passed"] is True
    assert result["failed_items"] == []
    assert result["test_opened"] is False
    assert result["tolerance"] == {"absolute": ATOL, "relative": 0.0}
    assert "best_iteration" in result["compared_items"]
    assert "validation_scores" in result["compared_items"]
    assert "aggregated_gain_importance" in result["compared_items"]


def test_compared_items_cover_every_required_dimension() -> None:
    result = _compare(_payload(), _payload())
    assert set(result["compared_items"]) == {
        "kept_numeric",
        "kept_categorical",
        "output_format",
        "transformed_dimension",
        "scale_pos_weight",
        "best_iteration",
        "validation_scores",
        "frozen_thresholds",
        "validation_confusion_matrix",
        "aggregated_gain_importance",
        "aggregated_split_importance",
    }


def test_best_iteration_difference_is_detected() -> None:
    changed = _payload()
    changed["best_iteration"] = 138
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "best_iteration" in result["failed_items"]
    assert result["items"]["best_iteration"] == {"match": False, "a": 137, "b": 138}


def test_score_difference_above_tolerance_is_detected() -> None:
    changed = _payload()
    scores = np.array(changed["validation_scores"])
    scores[7] += 1e-6
    changed["validation_scores"] = scores
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "validation_scores" in result["failed_items"]
    assert result["items"]["validation_scores"]["max_absolute_difference"] == pytest.approx(1e-6)
    assert result["items"]["validation_scores"]["n_beyond_tolerance"] == 1


def test_score_difference_within_tolerance_passes() -> None:
    changed = _payload()
    scores = np.array(changed["validation_scores"])
    scores[7] += ATOL / 10
    changed["validation_scores"] = scores
    assert _compare(_payload(), changed)["passed"] is True


def test_score_length_difference_is_detected() -> None:
    changed = _payload()
    changed["validation_scores"] = np.linspace(0.0, 1.0, 63)
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert result["items"]["validation_scores"]["max_absolute_difference"] is None


def test_frozen_f1_threshold_difference_is_detected() -> None:
    changed = _payload()
    changed["frozen_thresholds"]["selected_f1_threshold"] = 0.36
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "frozen_thresholds" in result["failed_items"]


def test_cost_scenario_threshold_difference_is_detected() -> None:
    changed = _payload()
    changed["frozen_thresholds"]["cost_scenarios"]["fp_fn_1_20"]["selected_threshold"] = 0.25
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    detail = result["items"]["frozen_thresholds"]["detail"]["cost_scenarios"]
    assert detail["fp_fn_1_20"]["match"] is False
    assert detail["fp_fn_1_1"]["match"] is True


def test_validation_total_cost_difference_is_detected() -> None:
    changed = _payload()
    changed["frozen_thresholds"]["cost_scenarios"]["fp_fn_1_1"]["validation_total_cost"] = 121.0
    assert _compare(_payload(), changed)["passed"] is False


def test_confusion_matrix_difference_is_detected() -> None:
    changed = _payload()
    changed["validation_confusion_matrix"]["false_positive"] = 4
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "validation_confusion_matrix" in result["failed_items"]
    assert result["items"]["validation_confusion_matrix"]["detail"]["false_positive"]["match"] is False


def test_aggregated_gain_difference_is_detected() -> None:
    changed = _payload()
    changed["aggregated_gain_importance"]["N2"] += 1e-5
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "aggregated_gain_importance" in result["failed_items"]
    assert result["items"]["aggregated_gain_importance"]["n_keys"] == 3


def test_aggregated_split_difference_is_detected() -> None:
    changed = _payload()
    changed["aggregated_split_importance"]["C1"] = 10
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "aggregated_split_importance" in result["failed_items"]


def test_importance_key_set_difference_is_detected() -> None:
    changed = _payload()
    changed["aggregated_gain_importance"].pop("C1")
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert result["items"]["aggregated_gain_importance"]["reason"] == "键集合不一致"


def test_kept_column_difference_is_detected() -> None:
    changed = _payload()
    changed["kept_numeric"] = ["N1", "N2", "N9"]
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "kept_numeric" in result["failed_items"]
    assert result["items"]["kept_numeric"]["first_difference"] == {
        "index": 2,
        "a": "N3",
        "b": "N9",
    }


def test_kept_column_count_difference_is_detected() -> None:
    changed = _payload()
    changed["kept_categorical"] = ["C1"]
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert result["items"]["kept_categorical"]["n_a"] == 2
    assert result["items"]["kept_categorical"]["n_b"] == 1


def test_transformed_dimension_difference_is_detected() -> None:
    changed = _payload()
    changed["transformed_dimension"] = 10
    assert _compare(_payload(), changed)["passed"] is False


def test_output_format_difference_is_detected() -> None:
    changed = _payload()
    changed["output_format"] = "dense_ndarray"
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "output_format" in result["failed_items"]


def test_scale_pos_weight_difference_is_detected() -> None:
    changed = _payload()
    changed["scale_pos_weight"] = 12.5 + 1e-6
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert "scale_pos_weight" in result["failed_items"]


def test_inputs_are_not_mutated() -> None:
    left, right = _payload(), _payload()
    snapshot = copy.deepcopy(left)
    _compare(left, right)
    assert left["best_iteration"] == snapshot["best_iteration"]
    assert np.array_equal(left["validation_scores"], snapshot["validation_scores"])


def test_failed_comparison_never_reports_test_opened() -> None:
    changed = _payload()
    changed["best_iteration"] = 0
    result = _compare(_payload(), changed)
    assert result["passed"] is False
    assert result["test_opened"] is False
