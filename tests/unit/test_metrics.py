"""指标计算与报告输出测试。

关键约束：无法计算的指标必须抛错，绝不能静默写入 0 或 NaN 之类的占位值。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai_detector.evaluation.metrics import (
    MetricComputationError,
    check_probabilities,
    compute_binary_metrics,
)
from ai_detector.evaluation.report import (
    environment_versions,
    jsonable,
    write_json_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_labels(n_positive: int = 20, n_negative: int = 180) -> np.ndarray:
    return np.array([1] * n_positive + [0] * n_negative)


def test_metrics_have_expected_fields_and_ranges() -> None:
    labels = make_labels()
    probabilities = np.concatenate(
        [np.linspace(0.55, 0.95, 20), np.linspace(0.02, 0.45, 180)]
    )

    metrics = compute_binary_metrics(labels, probabilities, threshold=0.5)

    for key in (
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "false_positive_rate",
    ):
        assert 0.0 <= metrics[key] <= 1.0, key
    assert metrics["n_samples"] == len(labels)
    assert metrics["n_positive"] == 20
    assert metrics["n_negative"] == 180
    assert metrics["threshold"] == 0.5
    assert set(metrics["confusion_matrix"]) == {
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }


def test_confusion_matrix_is_consistent_with_counts() -> None:
    labels = make_labels(n_positive=10, n_negative=90)
    probabilities = np.concatenate([np.full(10, 0.9), np.full(90, 0.1)])

    metrics = compute_binary_metrics(labels, probabilities, threshold=0.5)
    matrix = metrics["confusion_matrix"]

    assert matrix == {
        "true_negative": 90,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 10,
    }
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_single_class_labels_raise_instead_of_faking_values() -> None:
    """只有单一类别时不得返回占位指标。"""
    labels = np.zeros(50, dtype=int)
    probabilities = np.linspace(0.0, 1.0, 50)

    with pytest.raises(MetricComputationError, match="单一类别"):
        compute_binary_metrics(labels, probabilities, threshold=0.5)


def test_continuous_labels_raise() -> None:
    with pytest.raises(MetricComputationError, match=r"\{0, 1\}"):
        compute_binary_metrics([0.0, 1.0, 0.3], [0.1, 0.9, 0.5], threshold=0.5)


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_probabilities_raise(bad_value: float) -> None:
    probabilities = np.array([0.1, 0.2, 0.9, 1.0], dtype=float)
    probabilities[1] = bad_value
    with pytest.raises(MetricComputationError):
        check_probabilities(probabilities)


def test_probabilities_at_bounds_are_accepted() -> None:
    assert check_probabilities([0.0, 0.5, 1.0]).tolist() == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("bad_threshold", [0.0, 1.0, -0.2, 1.7])
def test_out_of_range_threshold_raises(bad_threshold: float) -> None:
    with pytest.raises(MetricComputationError, match="阈值"):
        compute_binary_metrics(make_labels(), np.linspace(0.0, 1.0, 200), threshold=bad_threshold)


def test_length_mismatch_raises() -> None:
    with pytest.raises(MetricComputationError, match="长度不一致"):
        compute_binary_metrics([0, 1, 0], [0.1, 0.9], threshold=0.5)


def test_environment_versions_include_key_packages() -> None:
    versions = environment_versions()

    assert set(versions) >= {
        "python",
        "platform",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "PyYAML",
    }
    for key, value in versions.items():
        assert isinstance(value, str) and value, key


def test_every_declared_runtime_dependency_is_recorded() -> None:
    """报告必须记录 ``pyproject.toml`` 声明的全部运行时依赖，否则环境无法完整复现。

    直接以报告实际内容为基准比对，防止将来新增依赖（或漏记 PyYAML 这类依赖）时
    报告悄悄少一项而无人发现。
    """
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared_block = pyproject.split("dependencies = [", 1)[1].split("]", 1)[0]
    declared = {
        line.strip().strip('",').split(">=")[0].split("[")[0].strip()
        for line in declared_block.splitlines()
        if line.strip().startswith('"')
    }

    recorded_fields = {field.lower() for field in environment_versions()}
    missing = sorted(name for name in declared if name.lower() not in recorded_fields)

    assert declared, "未能从 pyproject.toml 解析出依赖，本测试需要更新"
    assert not missing, f"以下运行时依赖未记入报告：{missing}"


def test_pyyaml_version_is_recorded_as_a_real_version() -> None:
    """PyYAML 是实际运行时依赖，不能只记成 unavailable。"""
    version = environment_versions()["PyYAML"]

    assert version != "unavailable", "PyYAML 是运行时依赖，必须记录真实版本"
    assert version != "unknown"
    assert version[0].isdigit(), version


def test_jsonable_handles_numpy_and_dataclasses() -> None:
    payload = {
        "int": np.int64(3),
        "float": np.float64(0.5),
        "bool": np.bool_(True),
        "array": [np.float32(1.5), np.int32(2)],
    }

    converted = jsonable(payload)
    assert json.loads(json.dumps(converted)) == {
        "int": 3,
        "float": 0.5,
        "bool": True,
        "array": [1.5, 2],
    }


def test_write_json_report_creates_parent_directories(tmp_path) -> None:
    path = write_json_report({"a": 1, "中文": "值"}, tmp_path / "deeper" / "report.json")

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "中文": "值"}
