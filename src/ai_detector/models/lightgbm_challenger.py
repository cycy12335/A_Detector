"""CW-004 LightGBM 挑战者：固定参数、无搜索、延迟导入。

## 参数是冻结的，不是"调出来的"

本模块**不接受任何形式的超参数搜索**：参数由配置逐项声明，并由
:func:`build_lgbm_estimator` 逐项核对后原样传给 ``LGBMClassifier``。任何键的增删都会
直接报错，因此"实现时顺手加点参数"这条路是堵死的。

## 延迟导入

``lightgbm`` 只在真正需要拟合或加载产物时才导入。未安装 ``tree`` extra 时，CW-001 至
CW-003 的功能仍可正常导入，只有本工作包的入口会抛出带安装命令的
:class:`LightGbmUnavailableError`。

## 类别不平衡

``scale_pos_weight`` **只由 train 标签计算**（``train_negative_count / train_positive_count``）。
它由配置声明公式，由 :func:`compute_scale_pos_weight` 实现，不接受调用方传入外部统计量。

## 分数不是校准概率

``scale_pos_weight`` 会系统性抬高正类分数，因此模型输出在本工作包中一律称为
**模型分数（score）**，不宣称为已校准的欺诈概率；本轮不做概率校准。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

LGBM_KEY = "lightgbm"

#: 冻结参数的**完整**键集合。多一个或少一个都视为配置被改写。
FROZEN_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "boosting_type",
        "objective",
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "random_state",
        "n_jobs",
        "deterministic",
        "force_col_wise",
        "verbosity",
    }
)

#: 冻结参数的**取值**。除 ``random_state`` 外逐项写死：``random_state`` 必须等于配置的
#: ``seed``（由配置层校验），因此夹具可以用不同的种子走同一条代码路径，而其余 15 个值
#: 无论加载哪份配置都不允许偏离 —— 参数冻结由代码与配置两层共同守住。
FROZEN_PARAM_VALUES: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}

#: 只允许 CPU 单线程确定性训练：出现这些键即视为试图改变计算资源或引入搜索。
FORBIDDEN_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "device",
        "device_type",
        "gpu_device_id",
        "gpu_platform_id",
        "gpu_use_dp",
        "num_gpu",
        "n_jobs_boost",
        "bagging_freq",
        "feature_fraction",
        "bagging_fraction",
    }
)

INSTALL_HINT = (
    "CW-004 需要 LightGBM 二进制 wheel。请执行：\n"
    '  python -m pip install --only-binary=:all: "lightgbm>=4.5,<5"\n'
    '  python -m pip install -e ".[test,tree]"\n'
    "不得从源码编译、降级 Python、新建替代环境或改用其他模型框架。"
)


class LightGbmUnavailableError(RuntimeError):
    """LightGBM 未安装或无法导入。出现即应据协议把任务置为 blocked。"""


class LightGbmSpecError(ValueError):
    """模型规格违反冻结参数契约。"""


def lightgbm_available() -> bool:
    """LightGBM 是否可导入（不抛错，供报告与测试使用）。"""
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def require_lightgbm() -> Any:
    """导入并返回 ``lightgbm`` 模块。

    Raises:
        LightGbmUnavailableError: 未安装 ``tree`` extra 或导入失败（附安装命令）。
    """
    try:
        import lightgbm
    except ImportError as error:  # pragma: no cover - 依赖缺失属环境问题
        raise LightGbmUnavailableError(
            f"无法导入 lightgbm：{error}\n{INSTALL_HINT}"
        ) from error
    return lightgbm


def lightgbm_version() -> str:
    """实际安装的 LightGBM 版本；未安装时返回明确标记而不是崩溃。"""
    try:
        module = require_lightgbm()
    except LightGbmUnavailableError:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def validate_frozen_params(params: dict[str, Any]) -> dict[str, Any]:
    """逐项核对冻结参数，返回原样副本。

    Raises:
        LightGbmSpecError: 键集合不符、出现被禁参数、或取值类型不合法。
    """
    keys = set(params)
    missing = sorted(FROZEN_PARAM_KEYS - keys)
    extra = sorted(keys - FROZEN_PARAM_KEYS)
    if missing or extra:
        raise LightGbmSpecError(
            "LightGBM 参数与冻结集合不一致："
            f"缺少 {missing}，多出 {extra}。CW-004 不允许增删或搜索参数。"
        )
    forbidden = sorted(keys & FORBIDDEN_PARAM_KEYS)
    if forbidden:
        raise LightGbmSpecError(f"禁止的参数（GPU / 采样 / 搜索相关）：{forbidden}")
    for key, expected in FROZEN_PARAM_VALUES.items():
        actual = params[key]
        if isinstance(expected, bool):
            matches = bool(actual) is expected
        elif isinstance(expected, float):
            matches = float(actual) == expected
        else:
            matches = actual == expected
        if not matches:
            raise LightGbmSpecError(
                f"参数 {key} 必须是冻结值 {expected!r}，实际为 {actual!r}；"
                "CW-004 不允许依据任何指标调整参数。"
            )
    return dict(params)


def compute_scale_pos_weight(train_labels: Sequence[int] | np.ndarray) -> float:
    """由 **train** 标签计算 ``scale_pos_weight = 负例数 / 正例数``。

    公式来自配置的 ``model.imbalance.formula``。本函数只接受一个标签向量，因此调用方
    无法"顺手"把 validation 或 test 标签混进来；测试用扰动 held-out 分区的方式守住该边界。

    Raises:
        LightGbmSpecError: 标签不是二值，或没有正例。
    """
    labels = np.asarray(train_labels)
    values = set(np.unique(labels).tolist())
    if not values <= {0, 1}:
        raise LightGbmSpecError(f"训练标签必须严格属于 {{0, 1}}，实际为 {sorted(values)}")
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0:
        raise LightGbmSpecError("训练分区没有正例，scale_pos_weight 无定义")
    return float(negative) / float(positive)


def build_lgbm_estimator(params: dict[str, Any], *, scale_pos_weight: float) -> Any:
    """按冻结参数实例化 ``LGBMClassifier``。

    ``scale_pos_weight`` 由调用方从 **train** 标签算出（见 :func:`compute_scale_pos_weight`），
    但**不是**冻结参数集合的一部分——它属于数据派生的不平衡权重，需要单独记录。
    """
    require_lightgbm()
    from lightgbm import LGBMClassifier

    validated = validate_frozen_params(params)
    return LGBMClassifier(**validated, scale_pos_weight=float(scale_pos_weight))


@dataclass
class EarlyStoppingDiagnostics:
    """早停与拟合诊断。未触发早停时必须如实记录，不得隐藏。"""

    metric: str
    stopping_rounds: int
    first_metric_only: bool
    #: ``n_estimators`` 是**上限**（冻结参数）；``rounds_run`` 是实际跑掉的提升轮数；
    #: ``trees_kept`` 是早停后保留、真正用于预测的树数。三者含义不同，报告里分别列示，
    #: 避免读者把"只保留了 1 棵树"误读成"只训练了 1 轮"。
    n_estimators: int
    best_iteration: int = 0
    best_score: float | None = None
    early_stopping_fired: bool = False
    rounds_run: int = 0
    trees_kept: int = 0
    eval_history: dict[str, list[float]] = field(default_factory=dict)
    fit_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    api_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        history = {name: list(values) for name, values in self.eval_history.items()}
        return {
            "metric": self.metric,
            "stopping_rounds": self.stopping_rounds,
            "first_metric_only": self.first_metric_only,
            "n_estimators_cap": self.n_estimators,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "early_stopping_fired": self.early_stopping_fired,
            "rounds_run": self.rounds_run,
            "trees_kept": self.trees_kept,
            "history_length": len(next(iter(history.values()), [])),
            "history_head": _head_tail(history, 5),
            "fit_seconds": round(self.fit_seconds, 3),
            "warnings": list(self.warnings),
            "api_notes": list(self.api_notes),
        }


def _head_tail(history: dict[str, list[float]], n: int) -> dict[str, dict[str, list[float]]]:
    """评估历史只保留首尾若干点，避免报告被上千行曲线淹没（完整曲线不写入报告）。"""
    summary: dict[str, dict[str, list[float]]] = {}
    for name, values in history.items():
        if not values:
            continue
        summary[name] = {
            "first": [round(float(value), 10) for value in values[:n]],
            "last": [round(float(value), 10) for value in values[-n:]],
        }
    return summary


def fit_lgbm_with_early_stopping(
    estimator: Any,
    train_features: Any,
    train_labels: np.ndarray,
    validation_features: Any,
    validation_labels: np.ndarray,
    *,
    metric: str,
    stopping_rounds: int,
    first_metric_only: bool = True,
    api_notes: Sequence[str] = (),
) -> EarlyStoppingDiagnostics:
    """在 train 上拟合，用 **validation** 做早停。

    早停通过 LightGBM callback API 实现（``lightgbm.early_stopping``），不使用已废弃的
    ``early_stopping_rounds`` 构造参数。评估集只有 validation 一个，调用方无法传入 test：
    本函数不接收分区名，只接收两个已经装配好的矩阵。

    ``lightgbm 4.7.0`` 已弃用 ``eval_set`` 关键字，因此评估集经由 ``eval_X`` / ``eval_y``
    传入——这是 API 名称的等价调整，不改动任何冻结参数。
    """
    import warnings

    lightgbm = require_lightgbm()
    diagnostics = EarlyStoppingDiagnostics(
        metric=str(metric),
        stopping_rounds=int(stopping_rounds),
        first_metric_only=bool(first_metric_only),
        n_estimators=int(estimator.get_params().get("n_estimators", 0)),
        api_notes=list(api_notes),
    )
    callbacks = [
        lightgbm.early_stopping(
            int(stopping_rounds), first_metric_only=bool(first_metric_only), verbose=False
        ),
        lightgbm.log_evaluation(period=0),
    ]

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(
            train_features,
            train_labels,
            eval_X=validation_features,
            eval_y=validation_labels,
            eval_metric=str(metric),
            callbacks=callbacks,
        )
    diagnostics.fit_seconds = time.perf_counter() - started
    for warning in caught:
        diagnostics.warnings.append(f"{warning.category.__name__}: {warning.message}")

    diagnostics.best_iteration = int(getattr(estimator, "best_iteration_", 0) or 0)
    # ``n_estimators_`` 在早停生效时等于保留的树数（不是跑过的轮数）。
    diagnostics.trees_kept = int(getattr(estimator, "n_estimators_", 0) or 0)
    diagnostics.eval_history = _flatten_evals_result(
        getattr(estimator, "evals_result_", None) or {}
    )
    history = diagnostics.eval_history
    primary = next(iter(history.values()), [])
    # 实际跑掉的提升轮数 = 评估历史的长度。
    diagnostics.rounds_run = len(primary)
    diagnostics.best_score = _best_score_from_lightgbm(estimator, primary, diagnostics.best_iteration)
    # 早停"生效"的判据：训练在上限之前停止。用实际轮数而不是保留树数判定，
    # 否则"只保留了 1 棵树、但跑了 51 轮"会被误判成"没跑满"以外的东西。
    diagnostics.early_stopping_fired = bool(
        diagnostics.rounds_run and diagnostics.rounds_run < diagnostics.n_estimators
    )
    return diagnostics


def _best_score_from_lightgbm(
    estimator: Any, history: list[float], best_iteration: int
) -> float | None:
    """取早停处的最优分数。

    优先使用 LightGBM 自己记录的 ``best_score_``（权威值）；不可用时回退到评估历史。
    注意索引口径：``evals_result_`` 的第 ``i`` 项是**第 i+1 棵树之后**的分数，而
    ``best_iteration_`` 是 1 基的树序号，因此回退索引是 ``best_iteration - 1``。
    """
    recorded = getattr(estimator, "best_score_", None)
    if isinstance(recorded, dict) and recorded:
        dataset = next(iter(recorded.values()))
        if isinstance(dataset, dict) and dataset:
            try:
                return float(next(iter(dataset.values())))
            except (TypeError, ValueError):  # pragma: no cover - 防御性
                pass
    if history:
        index = min(max(best_iteration - 1, 0), len(history) - 1)
        return float(history[index])
    return None


def _flatten_evals_result(raw: dict[str, Any]) -> dict[str, list[float]]:
    """把 ``evals_result_`` 规整为 ``{"<数据集>.<指标>": [每轮取值]}``。

    lightgbm 4.x 的结构是 ``{数据集名: {指标名: [取值]}}``；更早的版本是
    ``{指标名: [取值]}``。两种形状都在这里归一化，避免把 metric 名字符串当成浮点数。
    """
    flattened: dict[str, list[float]] = {}
    for name, values in raw.items():
        if isinstance(values, dict):
            for metric_name, series in values.items():
                flattened[f"{name}.{metric_name}"] = [float(value) for value in series]
        else:
            flattened[str(name)] = [float(value) for value in values]
    return flattened


def booster_importance(estimator: Any, importance_type: str) -> np.ndarray:
    """取底层 booster 的逐维重要性。

    Raises:
        LightGbmSpecError: 重要性类型不受支持（只允许 gain / split）。
    """
    if importance_type not in {"gain", "split"}:
        raise LightGbmSpecError(
            f"只允许 gain / split 重要性（不做 SHAP / Permutation / PDP），实际为 {importance_type}"
        )
    booster = getattr(estimator, "booster_", None)
    if booster is None:
        raise LightGbmSpecError("估计器尚未拟合，无法读取特征重要性")
    values = np.asarray(booster.feature_importance(importance_type=importance_type), dtype=float)
    return values


def describe_lightgbm(estimator: Any) -> dict[str, Any]:
    """把估计器的**公开参数**整理为可 JSON 序列化的报告字段。"""
    described: dict[str, Any] = {
        "estimator_class": type(estimator).__name__,
        "lightgbm_version": lightgbm_version(),
    }
    for key, value in estimator.get_params().items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            described[key] = value
        else:
            described[key] = str(value)
    return described


class LightGbmBundle:
    """可加载的推理 bundle：同语义预处理器 + LightGBM 模型 + 冻结阈值 + 指纹。

    **不包含任何原始训练/验证/测试数据**，也不包含逐行分数。批量打分接口对列做严格
    集合校验：缺列与多列都会立即抛错，不会静默按位置对齐或忽略多余列。
    """

    SCHEMA_VERSION = "cw004_bundle_v1"

    def __init__(
        self,
        *,
        preprocessor: Any,
        model: Any,
        best_iteration: int,
        source_features: Sequence[str],
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        transformed_feature_names: Sequence[str],
        scale_pos_weight: float,
        frozen_thresholds: dict[str, Any],
        fingerprints: dict[str, Any],
        model_params: dict[str, Any],
        config_sha256: str,
        notes: Sequence[str] = (),
    ) -> None:
        self.schema_version = self.SCHEMA_VERSION
        self.preprocessor = preprocessor
        self.model = model
        self.best_iteration = int(best_iteration)
        self.source_features = [str(name) for name in source_features]
        self.numeric_features = [str(name) for name in numeric_features]
        self.categorical_features = [str(name) for name in categorical_features]
        self.transformed_feature_names = [str(name) for name in transformed_feature_names]
        self.scale_pos_weight = float(scale_pos_weight)
        self.frozen_thresholds = dict(frozen_thresholds)
        self.fingerprints = dict(fingerprints)
        self.model_params = dict(model_params)
        self.config_sha256 = str(config_sha256)
        self.notes = [str(item) for item in notes]

    # --- 校验 -----------------------------------------------------------

    def validate(self) -> None:
        """结构自检。

        Raises:
            LightGbmSpecError: schema 版本、特征清单或预测器缺失/不一致。
        """
        if self.schema_version != self.SCHEMA_VERSION:
            raise LightGbmSpecError(
                f"bundle schema 版本不匹配：{self.schema_version} != {self.SCHEMA_VERSION}"
            )
        if not self.source_features:
            raise LightGbmSpecError("bundle 未记录源特征清单")
        if len(set(self.source_features)) != len(self.source_features):
            raise LightGbmSpecError("bundle 的源特征清单存在重复")
        if self.preprocessor is None or self.model is None:
            raise LightGbmSpecError("bundle 缺少已拟合的预处理器或模型")
        if len(self.transformed_feature_names) == 0:
            raise LightGbmSpecError("bundle 未记录转换后特征名")

    def check_columns(self, columns: Sequence[str]) -> list[str]:
        """校验输入列集合与源特征清单**完全一致**，返回按 bundle 顺序排列的列名。

        Raises:
            LightGbmSpecError: 缺列或多列。
        """
        provided = list(columns)
        duplicated = sorted({name for name in provided if provided.count(name) > 1})
        if duplicated:
            raise LightGbmSpecError(f"输入包含重复列：{duplicated}")
        expected = set(self.source_features)
        actual = set(provided)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise LightGbmSpecError(
                f"输入列与推理契约不一致：缺少 {missing}，多出 {extra}。"
                "批量打分接口不会静默接受缺列或多列。"
            )
        return list(self.source_features)

    # --- 推理 -----------------------------------------------------------

    def score_batch(self, frame: Any) -> np.ndarray:
        """对一个 DataFrame 批量打分，返回**模型分数**（不是校准概率）。

        Raises:
            LightGbmSpecError: 缺列、多列、重复列或输入不是 DataFrame。
        """
        if not hasattr(frame, "columns"):
            raise LightGbmSpecError("批量打分只接受带列名的 DataFrame")
        ordered = self.check_columns(list(frame.columns))
        import scipy.sparse as sp

        matrix = self.preprocessor.transform(frame.loc[:, ordered])
        if not sp.issparse(matrix):
            raise LightGbmSpecError("预处理器输出不是稀疏矩阵：预处理语义已被改变")
        return np.asarray(self.model.predict_proba(matrix))[:, 1]

    def describe(self) -> dict[str, Any]:
        """bundle 的自描述（写入报告，不含任何数据内容）。"""
        return {
            "schema_version": self.schema_version,
            "best_iteration": self.best_iteration,
            "n_source_features": len(self.source_features),
            "n_transformed_features": len(self.transformed_feature_names),
            "scale_pos_weight": self.scale_pos_weight,
            "frozen_thresholds": {
                "selected_f1_threshold": self.frozen_thresholds.get("selected_f1_threshold"),
                "cost_scenario_thresholds": {
                    name: payload.get("selected_threshold")
                    for name, payload in (self.frozen_thresholds.get("cost_scenarios") or {}).items()
                },
            },
            "config_sha256": self.config_sha256,
            "fingerprints": dict(self.fingerprints),
            "model_params": dict(self.model_params),
            "contains_raw_data": False,
            "notes": list(self.notes),
        }


def save_bundle(bundle: LightGbmBundle, path: Any) -> Any:
    """把 bundle 写入 joblib 文件（只含预处理器、模型与元数据，不含原始数据）。"""
    import joblib

    bundle.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Any) -> LightGbmBundle:
    """加载并自检 bundle。

    Raises:
        LightGbmSpecError: 文件内容不是本工作包的 bundle。
    """
    import joblib

    payload = joblib.load(path)
    if not isinstance(payload, LightGbmBundle):
        raise LightGbmSpecError(
            f"bundle 类型不符：{type(payload).__name__}，期望 LightGbmBundle"
        )
    payload.validate()
    return payload
