"""CW-003 的两个 scikit-learn 基线：Dummy 与 Logistic Regression。

参数完全固定、由配置声明，**不做任何超参数搜索**。随机种子统一由配置注入。

关于收敛：``LogisticRegression`` 未收敛时 scikit-learn 会发出 ``ConvergenceWarning``。
本模块**不吞掉**该警告，而是把它记录进报告的收敛诊断；允许与收敛直接相关的有限调整
（例如迭代上限），但必须在报告中写明理由，且**不得在查看 test 指标之后调整**。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

DUMMY_KEY = "dummy"
LOGISTIC_KEY = "logistic_regression"

#: 支持的模型类型 -> 报告中的稳定键名。
SUPPORTED_TYPES = {
    "dummy_classifier": DUMMY_KEY,
    "logistic_regression": LOGISTIC_KEY,
}


class BaselineSpecError(ValueError):
    """模型配置非法或不受支持。"""


@dataclass
class ConvergenceDiagnostics:
    """收敛诊断。未收敛时必须如实记录，不得隐藏。"""

    estimator: str
    n_iter: list[int] = field(default_factory=list)
    converged: bool | None = None
    warnings: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "n_iter": list(self.n_iter),
            "converged": self.converged,
            "warnings": list(self.warnings),
            "adjustments": list(self.adjustments),
        }


def build_estimator(model_type: str, params: dict[str, Any], *, seed: int) -> Any:
    """按配置实例化估计器。

    Raises:
        BaselineSpecError: 类型不受支持。
    """
    if model_type not in SUPPORTED_TYPES:
        raise BaselineSpecError(
            f"不支持的模型类型 {model_type}；CW-003 只允许 {sorted(SUPPORTED_TYPES)}"
        )
    if model_type == "dummy_classifier":
        return DummyClassifier(strategy=str(params.get("strategy", "prior")), random_state=seed)
    return LogisticRegression(random_state=seed, **params)


def fit_with_diagnostics(
    pipeline: Any,
    features: Any,
    target: Any,
    *,
    diagnostics_estimator: Any | None = None,
    adjustment_note: str | None = None,
) -> ConvergenceDiagnostics:
    """拟合管线并捕获收敛诊断。

    ``ConvergenceWarning`` 被**捕获后记录**而不是忽略：异常与警告都不会被吞掉，
    而是转换成报告里的结构化字段，便于验收者判断模型是否真的收敛。

    Args:
        pipeline: 待拟合对象（通常是完整 Pipeline）。
        diagnostics_estimator: 用于读取 ``n_iter_`` / ``max_iter`` 的**最终估计器**。
            必须显式传入：``n_iter_`` 只存在于最终估计器上，而 ``Pipeline`` 没有该属性，
            从管线读取会得到"永远无法判断是否收敛"的空诊断——那等于把未收敛藏起来。
        adjustment_note: 如果为了收敛做过有限调整（例如提高 ``max_iter``），在此写明理由。
    """
    diagnostics = ConvergenceDiagnostics(
        estimator=type(diagnostics_estimator or pipeline).__name__
    )
    if adjustment_note:
        diagnostics.adjustments.append(adjustment_note)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(features, target)

    for warning in caught:
        diagnostics.warnings.append(f"{warning.category.__name__}: {warning.message}")
        if issubclass(warning.category, ConvergenceWarning):
            diagnostics.converged = False

    source = diagnostics_estimator
    n_iter = getattr(source, "n_iter_", None)
    if n_iter is not None:
        diagnostics.n_iter = np_atleast_1d_int(n_iter)
        if diagnostics.converged is None:
            # lbfgs 在达到 max_iter 前停止即视为收敛。
            max_iter = getattr(source, "max_iter", None)
            diagnostics.converged = bool(
                max_iter is None or all(value < max_iter for value in diagnostics.n_iter)
            )
    return diagnostics


def np_atleast_1d_int(values: Any) -> list[int]:
    """把 ``n_iter_`` 规整为 int 列表（二分类下可能是 0 维或 1 维数组）。"""
    try:
        import numpy as np

        array = np.atleast_1d(np.asarray(values))
        return [int(value) for value in array.tolist()]
    except Exception:  # pragma: no cover - 防御性
        return []


def describe_estimator(estimator: Any) -> dict[str, Any]:
    """把估计器参数整理为报告字段（可 JSON 序列化）。"""
    described: dict[str, Any] = {"estimator_class": type(estimator).__name__}
    for key, value in estimator.get_params().items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            described[key] = value
        else:
            described[key] = str(value)
    return described
