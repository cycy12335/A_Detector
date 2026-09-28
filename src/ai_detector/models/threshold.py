"""分类阈值选择。

阈值**只在验证集上选择**，测试集仅用于最终一次评估。这是本工作包的核心
纪律之一：用测试集调阈值等同于把测试集变成训练数据。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

#: 阈值候选网格固定为 0.05 到 0.95 的步长 0.05，不依赖任何数据分布，
#: 因此换一份数据或换一个种子不会改变候选集合本身。
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(
    round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05)
)

THRESHOLD_RULE_DESCRIPTION = (
    "在验证集上遍历固定阈值网格，取 F1 最大的阈值；"
    "并列时先取 Recall 更高者，再取阈值更高者（更保守）。"
    "测试集不参与阈值选择。"
)


@dataclass(frozen=True)
class ThresholdChoice:
    """验证集上的阈值选择结果。"""

    model_key: str
    threshold: float
    validation_f1: float
    validation_precision: float
    validation_recall: float
    n_candidates: int
    rule: str = THRESHOLD_RULE_DESCRIPTION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_threshold(
    y_true: np.ndarray | list[int],
    y_proba: np.ndarray | list[float],
    *,
    model_key: str,
    grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> ThresholdChoice:
    """在验证集上按 F1 最大化选择阈值。

    关于常数概率模型（例如 ``DummyClassifier``）：其输出在验证集上只有一个取值 ``c``，
    因此最终选出哪个阈值完全取决于 ``c`` 与候选网格的相对位置。记
    ``A = {t ∈ grid : t <= c}``（即不超过 ``c`` 的候选阈值集合），有三种情形：

    - ``A`` 为空（``c`` 低于网格下界）：所有候选阈值都只预测负例，F1 全为 0 而形成
      完全并列，按 tie-break 规则选出**网格最高阈值**。
    - ``A`` 非空且验证集含正例：``A`` 中每个候选阈值都把全部样本判为正例，因此它们的
      Recall 全是 1、F1 也完全相同；这些候选彼此并列，按 tie-break 规则选出其中
      **最高**者，即 ``max(A)``。注意结果一般**不是**网格下界，而是不超过 ``c`` 的
      最大候选阈值——只有 ``c`` 恰好落在 ``[网格下界, 第二个候选阈值)`` 区间内时
      ``max(A)`` 才等于网格下界。例如 ``c = 0.12`` 且网格为
      ``0.05, 0.10, 0.15, ...`` 时，``A = {0.05, 0.10}``，选中 ``0.10``。
    - ``A`` 非空但验证集无正例：F1 与 Recall 全为 0，同样完全并列，选出**网格最高阈值**。

    本仓库当前的数据属于第一种情形（常数约 0.047，网格下界 0.05，``A`` 为空），
    因此选中网格最高阈值 0.95。三种结果都由同一套规则产生，不需要为常数预测器写特殊
    分支；这不是流程缺陷，而是基线不具备区分能力的体现，报告中会连同该行为一起说明。

    Args:
        y_true: 验证集真实标签。
        y_proba: 验证集正类概率。
        model_key: 模型标识，仅用于结果记录。
        grid: 候选阈值。

    Raises:
        ValueError: 网格为空，或标签/概率长度不一致。
    """
    labels = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_proba, dtype=float)
    if labels.shape != probabilities.shape:
        raise ValueError(
            f"y_true 与 y_proba 长度不一致：{labels.shape} vs {probabilities.shape}"
        )
    if not grid:
        raise ValueError("阈值网格不能为空")

    best: ThresholdChoice | None = None
    best_key: tuple[float, float, float] | None = None

    for threshold in grid:
        predictions = (probabilities >= threshold).astype(int)
        f1 = float(f1_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        precision = float(precision_score(labels, predictions, zero_division=0))
        # 排序键：F1 优先，其次 Recall，最后阈值（越大越保守）。
        key = (f1, recall, float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = ThresholdChoice(
                model_key=model_key,
                threshold=float(threshold),
                validation_f1=f1,
                validation_precision=precision,
                validation_recall=recall,
                n_candidates=len(grid),
            )

    assert best is not None  # grid 非空时循环必然至少执行一次
    return best
