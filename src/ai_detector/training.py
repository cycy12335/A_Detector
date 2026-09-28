"""训练编排：读数据 -> 时间切分 -> 训练基线 -> 验证集选阈值 -> 测试集评估 -> 写报告。

数据集使用顺序是硬约束：

- 训练集：拟合预处理参数与模型参数。
- 验证集：选择分类阈值。
- 测试集：只调用一次 :func:`compute_binary_metrics`，之后不再回头修改任何设置。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ai_detector import __version__
from ai_detector.config import BaselineConfig
from ai_detector.data.schema import (
    ALL_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    validate_transactions,
)
from ai_detector.evaluation.metrics import compute_binary_metrics
from ai_detector.evaluation.report import environment_versions, jsonable, write_json_report
from ai_detector.features.pipeline import build_model_pipeline, split_features_target
from ai_detector.features.split import TimeSplit, time_based_split
from ai_detector.models.baseline import ModelSpec, build_estimator, build_model_specs
from ai_detector.models.threshold import ThresholdChoice, select_threshold

WORK_PACKAGE = "CW-001"


class TrainingError(RuntimeError):
    """训练前置条件不满足时抛出。"""


@dataclass(frozen=True)
class TrainingResult:
    """一次训练运行的产物位置与报告内容。"""

    report: dict[str, Any]
    report_path: Path
    model_paths: dict[str, Path]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """读取交易 CSV 并做契约校验。

    时间列在这里解析为 datetime，避免下游把时间当字符串比较。
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise TrainingError(
            f"输入数据不存在：{csv_path}。请先运行 scripts/generate_data.py 生成模拟数据。"
        )

    frame = pd.read_csv(csv_path)
    missing = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing:
        raise TrainingError(f"输入数据缺少列：{', '.join(missing)}")

    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="raise")
    frame = frame.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    validate_transactions(frame, min_rows=3, require_sorted=True)
    return frame


def _model_diagnostics(
    validation_proba: Any, test_proba: Any, threshold: float
) -> dict[str, Any]:
    """记录模型输出的“形状”诊断信息。

    最重要的用途是让常数预测器显式可见：``DummyClassifier`` 的验证集概率只有一个
    取值，不具备区分能力。该常数最终落在哪个阈值区间取决于它与候选网格的相对位置，
    说明文字因此按实际观测到的判正率分支，而不是写死某一种情形，避免读者把“本次
    恰好如此”当成普遍规律。这属于基线的固有性质，必须写进报告，而不是让使用者
    误以为代码出错。
    """
    distinct = int(np.unique(np.asarray(validation_proba)).size)
    validation_positive_rate = float((np.asarray(validation_proba) >= threshold).mean())
    diagnostics: dict[str, Any] = {
        "distinct_validation_probabilities": distinct,
        "validation_positive_rate": validation_positive_rate,
        "test_positive_rate": float((np.asarray(test_proba) >= threshold).mean()),
    }
    if distinct == 1:
        constant = float(np.unique(np.asarray(validation_proba))[0])
        diagnostics["constant_probability"] = constant
        if validation_positive_rate == 0.0:
            # 选中阈值下全部判为负例，说明所有候选阈值的 F1 都为 0，属完全并列。
            # 注意此处不能断言“常数低于全部候选阈值”：当验证集不含正例时，即使常数
            # 不低于网格下界，也会落进这一分支。
            regime = (
                "本次选中的阈值下所有样本都被判为正常，说明全部候选阈值的 F1 都为 0、"
                "彼此完全并列，按 tie-break 规则最终选中了网格最高阈值"
            )
        else:
            regime = (
                "所有不超过该常数的候选阈值都会把全部样本判为正例，它们的 Recall 与 F1 "
                "完全相同而彼此并列，按 tie-break 规则取其中最高者，"
                "即不超过该常数的最大候选阈值"
            )
        diagnostics["note"] = (
            f"该模型在验证集上输出常数概率 {constant:.4f}，不具备区分能力；"
            f"{regime}（本次选中阈值 {threshold:.2f}）。"
            "这是基线应有的表现，不是流程错误。"
        )
    return diagnostics


def _build_report(
    *,
    frame: pd.DataFrame,
    split: TimeSplit,
    input_path: Path,
    seed: int,
    model_params: dict[str, dict[str, Any]],
    threshold_choices: dict[str, ThresholdChoice],
    test_metrics: dict[str, dict[str, Any]],
    model_diagnostics: dict[str, dict[str, Any]],
    feature_columns: list[str],
) -> dict[str, Any]:
    """组装 JSON 报告。字段顺序即人读顺序。"""
    return {
        "work_package": WORK_PACKAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "random_seed": seed,
        "data": {
            "input_path": str(input_path),
            "n_rows": int(len(frame)),
            "n_columns": int(frame.shape[1]),
            "columns": list(frame.columns),
            "fraud_rate": float(frame[TARGET_COLUMN].mean()),
            "n_fraud": int(frame[TARGET_COLUMN].sum()),
            "time_range": {
                "start": frame[TIMESTAMP_COLUMN].min().isoformat(),
                "end": frame[TIMESTAMP_COLUMN].max().isoformat(),
            },
        },
        "split": {
            "strategy": "time_based_chronological",
            "ratios": {
                "train": round(len(split.train) / len(frame), 6),
                "validation": round(len(split.validation) / len(frame), 6),
                "test": round(len(split.test) / len(frame), 6),
            },
            "sizes": split.sizes(),
            "boundaries": split.boundaries(),
            "fraud_rates": split.fraud_rates(),
        },
        "features": {
            "feature_columns": feature_columns,
            "excluded_columns": [
                column for column in frame.columns if column not in feature_columns
            ],
        },
        "models": {
            key: {
                "params": jsonable(params),
                "threshold": threshold_choices[key].to_dict(),
                "diagnostics": model_diagnostics[key],
            }
            for key, params in model_params.items()
        },
        "test_metrics": test_metrics,
        "environment": environment_versions(),
    }


def run_training(config: BaselineConfig) -> TrainingResult:
    """执行完整训练闭环并写出报告与模型产物。

    Raises:
        TrainingError: 输入缺失，或验证集/测试集不具备计算指标所需的类别分布。
    """
    frame = load_transactions(config.input_path)
    split = time_based_split(
        frame,
        train_ratio=config.split.train_ratio,
        validation_ratio=config.split.validation_ratio,
    )

    x_train, y_train = split_features_target(split.train)
    x_validation, y_validation = split_features_target(split.validation)
    x_test, y_test = split_features_target(split.test)
    feature_columns = list(x_train.columns)

    specs: list[ModelSpec] = build_model_specs(config.models)
    config.model_dir.mkdir(parents=True, exist_ok=True)

    model_params: dict[str, dict[str, Any]] = {}
    threshold_choices: dict[str, ThresholdChoice] = {}
    test_metrics: dict[str, dict[str, Any]] = {}
    model_diagnostics: dict[str, dict[str, Any]] = {}
    model_paths: dict[str, Path] = {}

    for spec in specs:
        estimator = build_estimator(spec, seed=config.seed)
        pipeline = build_model_pipeline(estimator)

        # 只有这里会 fit：预处理统计量与模型参数全部来自训练集。
        pipeline.fit(x_train, y_train)

        validation_proba = pipeline.predict_proba(x_validation)[:, 1]
        choice = select_threshold(
            y_validation,
            validation_proba,
            model_key=spec.key,
            grid=config.threshold_grid,
        )
        threshold_choices[spec.key] = choice

        test_proba = pipeline.predict_proba(x_test)[:, 1]
        test_metrics[spec.key] = compute_binary_metrics(
            y_test, test_proba, threshold=choice.threshold
        )
        model_diagnostics[spec.key] = _model_diagnostics(
            validation_proba, test_proba, choice.threshold
        )

        model_params[spec.key] = {
            "type": spec.model_type,
            "estimator_class": type(estimator).__name__,
            **pipeline.named_steps["model"].get_params(),
        }

        model_path = config.model_dir / f"{spec.key}_pipeline.joblib"
        joblib.dump(pipeline, model_path)
        model_paths[spec.key] = model_path

    report = _build_report(
        frame=frame,
        split=split,
        input_path=config.input_path,
        seed=config.seed,
        model_params=model_params,
        threshold_choices=threshold_choices,
        test_metrics=test_metrics,
        model_diagnostics=model_diagnostics,
        feature_columns=feature_columns,
    )
    report_path = write_json_report(report, config.report_path)
    return TrainingResult(report=report, report_path=report_path, model_paths=model_paths)
