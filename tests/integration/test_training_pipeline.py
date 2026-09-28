"""端到端训练流程集成测试。

在小型模拟数据上跑完整闭环，检查报告字段、指标合法性、模型产物可加载，
以及“测试集指标确实来自时间外测试集”。
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ai_detector.config import BaselineConfig
from ai_detector.data.simulate import generate_transactions, write_transactions_csv
from ai_detector.training import TrainingError, load_transactions, run_training

SMALL_ROWS = 600

#: 与本模块配置中使用的阈值候选网格保持一致。
THRESHOLD_GRID = [round(0.05 * step, 2) for step in range(1, 20)]

REQUIRED_REPORT_KEYS = {
    "work_package",
    "generated_at",
    "package_version",
    "random_seed",
    "data",
    "split",
    "features",
    "models",
    "test_metrics",
    "environment",
}

REQUIRED_METRIC_KEYS = {
    "n_samples",
    "n_positive",
    "n_negative",
    "threshold",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "false_positive_rate",
    "confusion_matrix",
}


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path, dict[str, Path]]:
    """在临时目录中跑一次完整训练，供本模块多个测试复用。"""
    workspace = tmp_path_factory.mktemp("cw001")
    data_path = workspace / "transactions.csv"
    write_transactions_csv(
        generate_transactions(seed=42, rows=SMALL_ROWS), data_path
    )

    config = BaselineConfig.from_mapping(
        {
            "seed": 42,
            "input_path": str(data_path),
            "report_path": str(workspace / "reports" / "report.json"),
            "model_dir": str(workspace / "models"),
            "split": {"train_ratio": 0.6, "validation_ratio": 0.2},
            "threshold_grid": [
                round(0.05 * step, 2) for step in range(1, 20)
            ],
            "models": {
                "dummy": {"type": "dummy_classifier", "params": {"strategy": "prior"}},
                "logistic_regression": {
                    "type": "logistic_regression",
                    "params": {"class_weight": "balanced", "max_iter": 1000},
                },
            },
        }
    )
    result = run_training(config)
    return result.report, result.report_path, result.model_paths


def test_report_has_all_required_sections(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    report, _, _ = trained_run

    assert REQUIRED_REPORT_KEYS <= set(report)
    assert report["work_package"] == "CW-001"
    assert report["random_seed"] == 42
    assert report["data"]["n_rows"] == SMALL_ROWS
    assert report["split"]["strategy"] == "time_based_chronological"


def test_report_is_written_to_disk_and_is_valid_json(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    report, report_path, _ = trained_run

    assert report_path.is_file()
    reloaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert reloaded["work_package"] == report["work_package"]
    assert reloaded["split"]["sizes"] == report["split"]["sizes"]


def test_split_sizes_sum_to_total_and_never_overlap(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    report, _, _ = trained_run
    sizes = report["split"]["sizes"]
    boundaries = report["split"]["boundaries"]

    assert sum(sizes.values()) == SMALL_ROWS
    assert sizes["train"] > 0 and sizes["validation"] > 0 and sizes["test"] > 0
    assert boundaries["train_end"] <= boundaries["validation_start"]
    assert boundaries["validation_end"] <= boundaries["test_start"]


def test_both_baselines_are_reported(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    """两个基线都必须有结果。

    刻意不断言“LogisticRegression 必须优于 Dummy”：本工作包的目的是建立可复现
    基线，不是承诺提升。任何此类断言都会变成用测试去倒逼指标。
    """
    report, _, _ = trained_run

    assert {"dummy", "logistic_regression"} <= set(report["models"])
    assert {"dummy", "logistic_regression"} <= set(report["test_metrics"])
    for key in ("dummy", "logistic_regression"):
        assert report["models"][key]["params"]
        assert report["models"][key]["threshold"]["rule"]


def test_test_metrics_are_valid_and_complete(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    report, _, _ = trained_run

    for key, metrics in report["test_metrics"].items():
        assert REQUIRED_METRIC_KEYS <= set(metrics), key
        assert metrics["n_samples"] == report["split"]["sizes"]["test"], key
        assert metrics["n_positive"] + metrics["n_negative"] == metrics["n_samples"]
        for name in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate"):
            value = metrics[name]
            assert isinstance(value, float), name
            assert np.isfinite(value), f"{key}.{name} 不是有限值"
            assert 0.0 <= value <= 1.0, f"{key}.{name}={value} 超出 [0, 1]"


def test_threshold_comes_from_validation_not_test(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    """测试集指标使用的阈值必须等于验证集选出的阈值。

    若二者不一致，说明测试集参与了阈值选择。
    """
    report, _, _ = trained_run
    grid = [round(0.05 * step, 2) for step in range(1, 20)]

    for key, metrics in report["test_metrics"].items():
        selected = report["models"][key]["threshold"]["threshold"]
        assert selected in grid
        assert metrics["threshold"] == selected


def test_models_are_saved_as_loadable_pipelines(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    report, _, model_paths = trained_run

    assert set(model_paths) == {"dummy", "logistic_regression"}
    for key, path in model_paths.items():
        assert path.is_file(), key
        pipeline = joblib.load(path)
        assert set(pipeline.named_steps) == {"preprocessor", "model"}
        assert list(pipeline.named_steps["preprocessor"].feature_names_in_) == report["features"][
            "feature_columns"
        ]


def test_saved_pipeline_predicts_probabilities_in_range(
    trained_run: tuple[dict, Path, dict[str, Path]],
    tmp_path: Path,
) -> None:
    """加载保存的管线后，对任意新数据的预测概率必须位于 [0, 1]。"""
    _, _, model_paths = trained_run
    fresh = load_transactions_from(tmp_path)
    features = fresh[["amount", "transaction_count_1h", "is_new_device"]]

    for key, path in model_paths.items():
        probabilities = joblib.load(path).predict_proba(features)[:, 1]
        assert probabilities.shape == (len(features),), key
        assert np.isfinite(probabilities).all(), key
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all(), key


def load_transactions_from(tmp_path: Path) -> pd.DataFrame:
    """生成一份小数据并读出，用于验证已保存管线的推理路径。"""
    path = tmp_path / "fresh.csv"
    write_transactions_csv(generate_transactions(seed=777, rows=120), path)
    return load_transactions(path)


def test_constant_predictor_note_matches_observed_behaviour(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    """常数预测器的说明文字必须与实测判正率一致。

    二次验收整改项：原说明把“常数不低于网格下界”概括为“选出最低阈值”，这不是一般
    规律。真正的一般规律是：不超过常数的候选阈值都预测全正而彼此并列，tie-break 取
    其中最高者，即不超过常数的最大候选阈值；只有该常数落在
    ``[网格下界, 第二个候选阈值)`` 时结果才等于网格下界。本测试锁定“说明与观测一致”
    这一不变量，并覆盖三种情形。
    """
    report, _, _ = trained_run
    dummy = report["models"]["dummy"]
    diagnostics = dummy["diagnostics"]

    assert diagnostics["distinct_validation_probabilities"] == 1
    constant = diagnostics["constant_probability"]
    threshold = dummy["threshold"]["threshold"]

    grid_min, grid_max = min(THRESHOLD_GRID), max(THRESHOLD_GRID)
    within_constant = [value for value in THRESHOLD_GRID if value <= constant]

    if not within_constant:
        # 常数低于网格下界：所有候选阈值都只预测负例，按 tie-break 选网格最高阈值。
        assert diagnostics["validation_positive_rate"] == 0.0
        assert threshold == grid_max
        assert "网格最高阈值" in diagnostics["note"]
    elif diagnostics["validation_positive_rate"] > 0.0:
        # 不超过常数的候选全部预测全正而并列，取其中最高者，而不是网格下界。
        assert threshold == max(within_constant)
        assert "最大候选阈值" in diagnostics["note"]
    else:
        # 常数不低于网格下界，但验证集无正例：F1 全为 0，同样取网格最高阈值。
        assert threshold == grid_max
        assert "网格最高阈值" in diagnostics["note"]


def test_features_section_excludes_target_and_ids(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    report, _, _ = trained_run

    feature_columns = report["features"]["feature_columns"]
    assert "is_fraud" not in feature_columns
    assert "timestamp" not in feature_columns
    assert not {"transaction_id", "user_id", "device_id"} & set(feature_columns)
    for column in ("is_fraud", "timestamp", "transaction_id"):
        assert column in report["features"]["excluded_columns"]


def test_environment_section_records_versions(
    trained_run: tuple[dict, Path, dict[str, Path]],
) -> None:
    report, _, _ = trained_run

    assert set(report["environment"]) >= {
        "python",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "PyYAML",
    }
    # PyYAML 是读取 configs/*.yaml 的实际运行时依赖，必须记录真实版本。
    assert report["environment"]["PyYAML"] not in {"unavailable", "unknown"}


def test_training_is_reproducible_across_runs(tmp_path: Path) -> None:
    """同配置两次训练的指标必须完全一致（除时间戳外）。"""
    data_path = write_transactions_csv(
        generate_transactions(seed=42, rows=SMALL_ROWS), tmp_path / "transactions.csv"
    )

    def build_config(suffix: str) -> BaselineConfig:
        return BaselineConfig.from_mapping(
            {
                "seed": 42,
                "input_path": str(data_path),
                "report_path": str(tmp_path / f"report_{suffix}.json"),
                "model_dir": str(tmp_path / f"models_{suffix}"),
                "split": {"train_ratio": 0.6, "validation_ratio": 0.2},
                "threshold_grid": [round(0.05 * step, 2) for step in range(1, 20)],
                "models": {
                    "dummy": {"type": "dummy_classifier", "params": {"strategy": "prior"}},
                    "logistic_regression": {
                        "type": "logistic_regression",
                        "params": {"class_weight": "balanced"},
                    },
                },
            }
        )

    first = run_training(build_config("a")).report
    second = run_training(build_config("b")).report

    assert first["test_metrics"] == second["test_metrics"]
    assert first["split"]["boundaries"] == second["split"]["boundaries"]
    assert first["models"] == second["models"]


def test_missing_input_file_raises_clear_error(tmp_path: Path) -> None:
    config = BaselineConfig.from_mapping(
        {
            "seed": 42,
            "input_path": str(tmp_path / "does_not_exist.csv"),
            "report_path": str(tmp_path / "report.json"),
            "model_dir": str(tmp_path / "models"),
            "split": {"train_ratio": 0.6, "validation_ratio": 0.2},
            "threshold_grid": [0.5],
            "models": {
                "dummy": {"type": "dummy_classifier"},
                "logistic_regression": {"type": "logistic_regression"},
            },
        }
    )

    with pytest.raises(TrainingError, match="不存在"):
        run_training(config)
