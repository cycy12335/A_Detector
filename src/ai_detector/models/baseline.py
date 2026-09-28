"""两个基线分类器的构造。

1. ``DummyClassifier``：最低基线，用于说明“任何真实模型都必须优于它”。
2. ``LogisticRegression``：线性基线，使用 ``class_weight="balanced"`` 应对
   欺诈样本比例很低的情况，所有参数显式记录到报告中。

P0 阶段刻意不引入梯度提升或深度学习：先建立可信的基线，再谈模型复杂度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

DUMMY_MODEL_KEY = "dummy"
LOGISTIC_MODEL_KEY = "logistic_regression"

_SUPPORTED_TYPES = {
    "dummy_classifier": DUMMY_MODEL_KEY,
    "logistic_regression": LOGISTIC_MODEL_KEY,
}

#: LogisticRegression 的固定参数。``class_weight="balanced"`` 按类别频率反向加权，
#: 避免模型在 5% 正例下退化为“全部预测为正常”。
DEFAULT_LOGISTIC_PARAMS: dict[str, Any] = {
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 1000,
}


class ModelSpecError(ValueError):
    """模型配置非法或不受支持时抛出。"""


@dataclass(frozen=True)
class ModelSpec:
    """一个待训练的模型配置。"""

    key: str
    model_type: str
    params: dict[str, Any]


def build_model_specs(models_config: dict[str, dict[str, Any]]) -> list[ModelSpec]:
    """把 YAML 中的 ``models`` 段解析为有序的 :class:`ModelSpec` 列表。"""
    specs: list[ModelSpec] = []
    for key, raw in models_config.items():
        if not isinstance(raw, dict) or "type" not in raw:
            raise ModelSpecError(f"模型 {key} 的配置必须包含 type 字段")
        model_type = str(raw["type"])
        if model_type not in _SUPPORTED_TYPES:
            raise ModelSpecError(
                f"模型 {key} 的类型 {model_type} 不受支持；"
                f"P0 仅允许 {sorted(_SUPPORTED_TYPES)}"
            )
        params = dict(raw.get("params") or {})
        specs.append(ModelSpec(key=str(key), model_type=model_type, params=params))

    expected = {DUMMY_MODEL_KEY, LOGISTIC_MODEL_KEY}
    found = {_SUPPORTED_TYPES[spec.model_type] for spec in specs}
    if not expected <= found:
        raise ModelSpecError(f"缺少必需基线：{sorted(expected - found)}")
    return specs


def build_estimator(spec: ModelSpec, *, seed: int) -> Any:
    """按配置实例化估计器，随机种子统一由参数注入。"""
    if spec.model_type == "dummy_classifier":
        return DummyClassifier(
            strategy=spec.params.get("strategy", "prior"), random_state=seed
        )
    if spec.model_type == "logistic_regression":
        params = {**DEFAULT_LOGISTIC_PARAMS, **spec.params}
        return LogisticRegression(random_state=seed, **params)
    raise ModelSpecError(f"不受支持的模型类型：{spec.model_type}")
