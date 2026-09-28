"""二分类评估指标。

设计原则：**无法计算的指标必须报错，而不是返回 0 或 NaN 之类的占位值。**
在只有单一类别的评估集上，ROC-AUC、PR-AUC 和 FPR 在数学上没有定义；此时静默
写入 0.0 会伪造出一个看起来正常的结果，因此这里统一抛出
:class:`MetricComputationError`。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricComputationError(RuntimeError):
    """指标在给定数据上无定义时抛出（例如评估集只有单一类别）。"""


def check_probabilities(y_proba: np.ndarray | list[float]) -> np.ndarray:
    """校验概率向量：一维、有限、且全部位于 ``[0, 1]``。"""
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.ndim != 1:
        raise MetricComputationError(f"概率必须是一维数组，实际维度为 {probabilities.ndim}")
    if probabilities.size == 0:
        raise MetricComputationError("概率数组为空")
    if not np.isfinite(probabilities).all():
        raise MetricComputationError("概率数组包含非有限值（NaN 或 inf）")
    if probabilities.min() < 0.0 or probabilities.max() > 1.0:
        raise MetricComputationError(
            f"概率超出 [0, 1]：min={probabilities.min()}, max={probabilities.max()}"
        )
    return probabilities


def coerce_binary_labels(y_true: np.ndarray | list[int]) -> np.ndarray:
    """把标签转换为 int 数组，并拒绝任何非 ``{0, 1}`` 的取值。

    刻意不做 ``astype(int)`` 这种静默截断：标签 ``0.3`` 会被截成 ``0``，把上游的
    数据错误伪装成合法输入，因此这里必须显式报错。
    """
    raw = np.asarray(y_true)
    if raw.size == 0:
        raise MetricComputationError("标签数组为空")
    try:
        as_float = raw.astype(float)
    except (TypeError, ValueError) as error:
        raise MetricComputationError(f"标签无法转换为数值：{raw.dtype}") from error
    if not np.isfinite(as_float).all():
        raise MetricComputationError("标签包含非有限值（NaN 或 inf）")
    if not np.isin(as_float, (0.0, 1.0)).all():
        raise MetricComputationError(
            f"标签必须是 {{0, 1}}，实际为 {np.unique(as_float).tolist()}"
        )
    return as_float.astype(int)


def _check_both_classes(labels: np.ndarray, metric_name: str) -> None:
    unique = np.unique(labels)
    if unique.size < 2:
        raise MetricComputationError(
            f"无法计算 {metric_name}：评估集只包含单一类别 {unique.tolist()}。"
            "请检查数据切分与标签生成，不要写入占位数值。"
        )


def compute_binary_metrics(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray | list[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    """计算阈值化的分类指标与不依赖阈值的排序指标。

    Args:
        y_true: 真实标签，取值必须严格为 ``{0, 1}``（不接受可截断为 0/1 的浮点值）。
        y_proba: 正类概率，必须位于 ``[0, 1]``。
        threshold: 判定阈值，``y_proba >= threshold`` 判为正例。

    Returns:
        含 Precision、Recall、F1、ROC-AUC、PR-AUC、FPR、混淆矩阵与样本数的字典。

    Raises:
        MetricComputationError: 概率非法、标签不是二分类，或评估集只有单一类别。
    """
    labels = coerce_binary_labels(y_true)
    probabilities = check_probabilities(y_proba)

    if labels.shape != probabilities.shape:
        raise MetricComputationError(
            f"标签与概率长度不一致：{labels.shape} vs {probabilities.shape}"
        )
    if not 0.0 < threshold < 1.0:
        raise MetricComputationError(f"阈值必须位于开区间 (0, 1)，实际为 {threshold}")

    _check_both_classes(labels, "ROC-AUC / PR-AUC / FPR")

    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    negatives = tn + fp
    if negatives == 0:
        raise MetricComputationError("评估集没有负类样本，FPR 无定义")

    return {
        "n_samples": int(labels.size),
        "n_positive": int(tp + fn),
        "n_negative": int(negatives),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "false_positive_rate": float(fp / negatives),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }
