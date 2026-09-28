"""透明尾部异常规则基线。

**它是什么**：一个不使用标签的启发式分数。对预先声明的若干数值字段，若取值超过**训练分区**
的高分位数则记 1 分，否则记 0，命中数除以字段数得到 ``[0, 1]`` 分数。

**它不是什么**：不是业务规则，不是任何机构的真实反欺诈规则，也不对应已确认的业务含义。
它只提供一个完全透明、可以人工逐条复核的下界参照。

分数只有 ``0, 0.5, 1.0`` 三种取值，因此阈值曲线只有有限的几个操作点。这是透明性的直接
代价，报告会如实展示，而不是用连续分数把它包装成更强的模型。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class RuleBaselineError(ValueError):
    """规则基线配置或输入不合法。"""


@dataclass(frozen=True)
class RuleBaselineState:
    """规则基线的冻结状态：字段、分位数与由此得到的阈值。"""

    name: str
    kind: str
    fields: tuple[str, ...]
    quantile: float
    thresholds: dict[str, float]
    scoring: str
    train_rows: int
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """计算 ``[0, 1]`` 分数。

        缺失值记 0 分（不视为异常）——这是声明过的确定性规则，而非隐式填充：
        两个声明字段在训练分区均无缺失，val/test 上若出现缺失，按"不触发"处理。
        """
        missing = [name for name in self.fields if name not in frame.columns]
        if missing:
            raise RuleBaselineError(f"规则字段缺失：{missing}")

        hits = np.zeros(len(frame), dtype=float)
        for name in self.fields:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            triggered = np.isfinite(values) & (values > self.thresholds[name])
            hits += triggered.astype(float)
        return hits / len(self.fields)

    def write(self, path: Path) -> Path:
        """把冻结状态写成 JSON（模型产物，默认被 Git 忽略）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return path


def fit_rule_baseline(config_section: dict[str, Any], train_frame: pd.DataFrame) -> RuleBaselineState:
    """只在训练分区上拟合规则基线的分位数阈值。

    Args:
        config_section: 配置中的 ``rule_baseline`` 段。
        train_frame: **仅训练分区**的数据。

    Raises:
        RuleBaselineError: 字段缺失、分位数非法或训练分区为空。
    """
    fields = tuple(str(item["name"]) for item in config_section["fields"])
    if len(fields) < 2:
        raise RuleBaselineError("规则基线至少需要两个数值字段")
    quantile = float(config_section["quantile"])
    if not 0.0 < quantile < 1.0:
        raise RuleBaselineError(f"分位数必须位于开区间 (0, 1)，实际为 {quantile}")
    if train_frame.empty:
        raise RuleBaselineError("训练分区为空，无法拟合规则阈值")

    thresholds: dict[str, float] = {}
    for name in fields:
        if name not in train_frame.columns:
            raise RuleBaselineError(f"训练分区缺少规则字段：{name}")
        values = pd.to_numeric(train_frame[name], errors="coerce").to_numpy(dtype=float)
        usable = values[np.isfinite(values)]
        if usable.size == 0:
            raise RuleBaselineError(f"规则字段 {name} 在训练分区没有有限取值")
        thresholds[name] = float(np.quantile(usable, quantile))

    return RuleBaselineState(
        name=str(config_section["name"]),
        kind=str(config_section["kind"]),
        fields=fields,
        quantile=quantile,
        thresholds=thresholds,
        scoring=str(config_section["scoring"]),
        train_rows=int(len(train_frame)),
        disclaimer=str(config_section["disclaimer"]),
    )
