"""阈值曲线与相对成本选择。

## 边界

- 本模块只在给定标签与分数上做**纯计算**，不关心数据来自哪个分区。调用方负责保证：
  阈值只在 validation 上选择，test 只在阈值冻结后评估一次。
- 成本一律是**研究用相对单位**：``total = FP × fp_cost + FN × fn_cost``。它不是货币，
  不代表任何机构的真实损失。

## 为什么 tie-break 必须确定

网格上常出现多个阈值给出同样的 F1 或同样的成本（例如常数预测器、或只有少数几个不同分数
的规则基线）。此时若顺序不确定，重复运行会给出不同阈值，可复现性就没了。因此两个选择器
都使用显式的、可测试的排序键。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

#: 成本与指标的固定计算口径。
COST_UNITS = "研究用相对成本单位"


class ThresholdError(ValueError):
    """阈值或输入不合法。"""


@dataclass(frozen=True)
class ThresholdMetrics:
    """单个阈值下的混淆矩阵与派生指标。"""

    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    predicted_positive_rate: float
    n_samples: int
    n_positive: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def total_cost(self, *, fp_cost: float, fn_cost: float) -> float:
        """相对成本总和。单位为研究用相对单位，非货币。"""
        return self.false_positive * fp_cost + self.false_negative * fn_cost

    def average_cost(self, *, fp_cost: float, fn_cost: float) -> float:
        if self.n_samples <= 0:
            raise ThresholdError("样本数为 0，无法计算平均成本")
        return self.total_cost(fp_cost=fp_cost, fn_cost=fn_cost) / self.n_samples


def _validate_scores(y_score: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(y_score, dtype=float)
    labels = np.asarray(y_true)
    if scores.ndim != 1:
        raise ThresholdError(f"分数必须是一维，实际维度 {scores.ndim}")
    if scores.size == 0:
        raise ThresholdError("分数为空")
    if not np.isfinite(scores).all():
        raise ThresholdError("分数包含非有限值（NaN 或 inf）")
    if scores.min() < 0.0 or scores.max() > 1.0:
        raise ThresholdError(f"分数必须位于 [0, 1]，实际 [{scores.min()}, {scores.max()}]")
    if labels.shape != scores.shape:
        raise ThresholdError(f"标签与分数长度不一致：{labels.shape} vs {scores.shape}")
    try:
        as_float = labels.astype(float)
    except (TypeError, ValueError) as error:  # pragma: no cover - 防御性
        raise ThresholdError(f"标签无法转换为数值：{labels.dtype}") from error
    if not np.isfinite(as_float).all():
        raise ThresholdError("标签包含非有限值")
    if not np.isin(as_float, (0.0, 1.0)).all():
        raise ThresholdError(f"标签必须严格属于 {{0, 1}}，实际为 {np.unique(as_float).tolist()}")
    return scores, as_float.astype(int)


def metrics_at_threshold(
    y_true: np.ndarray | Sequence[int],
    y_score: np.ndarray | Sequence[float],
    *,
    threshold: float,
) -> ThresholdMetrics:
    """在固定阈值上计算混淆矩阵与派生指标。

    判定规则：``score >= threshold`` 记为正例。指标在分母为 0 时定义为 0（并保持
    不变的确定性），但混淆矩阵本身始终是真实计数。
    """
    if not 0.0 < threshold < 1.0:
        raise ThresholdError(f"阈值必须位于开区间 (0, 1)，实际为 {threshold}")
    scores, labels = _validate_scores(y_score, y_true)

    predicted_positive = scores >= threshold
    actual_positive = labels == 1
    tp = int(np.sum(predicted_positive & actual_positive))
    fp = int(np.sum(predicted_positive & ~actual_positive))
    tn = int(np.sum(~predicted_positive & ~actual_positive))
    fn = int(np.sum(~predicted_positive & actual_positive))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return ThresholdMetrics(
        threshold=float(threshold),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        predicted_positive_rate=float(predicted_positive.mean()),
        n_samples=int(labels.size),
        n_positive=int(actual_positive.sum()),
    )


def threshold_curve(
    y_true: np.ndarray | Sequence[int],
    y_score: np.ndarray | Sequence[float],
    grid: Sequence[float],
) -> list[ThresholdMetrics]:
    """在固定网格上计算完整阈值曲线。

    Raises:
        ThresholdError: 网格为空或含非法阈值。
    """
    if not grid:
        raise ThresholdError("阈值网格不能为空")
    return [metrics_at_threshold(y_true, y_score, threshold=float(t)) for t in grid]


def curve_to_frame(curve: Sequence[ThresholdMetrics]) -> pd.DataFrame:
    """把阈值曲线转为 CSV 友好的数据框。"""
    return pd.DataFrame([point.to_dict() for point in curve])


def select_f1_threshold(
    curve: Sequence[ThresholdMetrics], *, tie_break: Sequence[str]
) -> ThresholdMetrics:
    """按声明的 tie-break 选择 F1 最大的阈值。

    默认排序键 ``(f1, recall, threshold)`` 全部降序：F1 高者优先，其次 Recall 高者，
    最后阈值更高者（更保守）。三个键都参与比较，因此结果唯一且可复现。

    Raises:
        ThresholdError: 曲线为空或 tie-break 不受支持。
    """
    if not curve:
        raise ThresholdError("阈值曲线为空，无法选择 F1 阈值")
    if list(tie_break) != ["f1_desc", "recall_desc", "threshold_desc"]:
        raise ThresholdError(f"不支持的 F1 tie-break：{list(tie_break)}")
    return max(curve, key=lambda p: (p.f1, p.recall, p.threshold))


def select_cost_threshold(
    curve: Sequence[ThresholdMetrics],
    *,
    fp_cost: float,
    fn_cost: float,
    tie_break: Sequence[str],
) -> tuple[ThresholdMetrics, float]:
    """按声明的 tie-break 选择相对成本最低的阈值。

    排序键 ``(total_cost 升序, threshold 降序)``：成本相同者取阈值更高者（更保守）。

    Returns:
        ``(选中的阈值点, 该点的成本总和)``。

    Raises:
        ThresholdError: 曲线为空、成本为负或 tie-break 不受支持。
    """
    if not curve:
        raise ThresholdError("阈值曲线为空，无法选择成本阈值")
    if fp_cost < 0 or fn_cost < 0:
        raise ThresholdError(f"成本必须非负，实际 fp_cost={fp_cost}, fn_cost={fn_cost}")
    if list(tie_break) != ["cost_asc", "threshold_desc"]:
        raise ThresholdError(f"不支持的成本 tie-break：{list(tie_break)}")

    scored = [(point, point.total_cost(fp_cost=fp_cost, fn_cost=fn_cost)) for point in curve]
    best = min(scored, key=lambda item: (item[1], -item[0].threshold))
    return best[0], best[1]


def cost_matrix_note() -> dict[str, str]:
    """成本口径说明，直接写入报告，避免读者误把相对单位当成货币。"""
    return {
        "units": COST_UNITS,
        "formula": "total = false_positive * fp_cost + false_negative * fn_cost",
        "average": "average = total / n_samples",
        "disclaimer": (
            "成本比例是研究假设，不是真实货币损失，也不代表任何机构的风险偏好；"
            "报告中的成本一律为研究用相对单位，不得换算为人民币、美元或真实损失金额。"
        ),
    }
