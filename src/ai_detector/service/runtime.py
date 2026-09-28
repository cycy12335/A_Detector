"""推理运行时：**先校验 SHA-256，再反序列化**，然后做完整契约校验。

## 顺序本身就是一条安全边界

``joblib.load`` 等价于 ``pickle.load``，对不可信文件可以执行任意代码。因此本模块把顺序
写死成：

1. 文件存在性检查；
2. **流式** SHA-256 校验（分块读取，不把 55 KB 之外的东西读进内存，也不反序列化）；
3. 只有哈希完全等于配置里那个固定值，才调用 ``joblib.load``；
4. 反序列化后再校验类型、schema 版本、源特征集合、阈值取值与"不含原始数据"标记。

哈希不符时 ``joblib.load`` **一次都不会被调用**（有测试用 monkeypatch 守住这一点）。

## 分数不是概率

bundle 里记录的 ``score_semantics`` 是 ``uncalibrated_model_score``；服务对外一律使用
"模型分数"这一说法，不提供 ``probability`` / ``fraud_probability`` 之类的字段，
也不输出批准/拒绝等生产处置结论。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from ai_detector.rules.config import PolicyErrorCode, PolicyLoadError, load_policy
from ai_detector.rules.engine import PolicyEngine, RiskPolicyEngine  # noqa: F401 - 供类型标注
from ai_detector.service.config import ServiceConfig
from ai_detector.service.errors import BundleErrorCode, BundleLoadError
from ai_detector.storage.config import (
    STORAGE_SCHEMA_VERSION,
    StorageErrorCode,
    StorageLoadError,
    resolve_storage_path,
)
from ai_detector.storage.sqlite import DecisionRepository, open_repository

#: 流式校验的分块大小。
HASH_CHUNK_SIZE = 1 << 20

#: 分数的固定语义标签（与配置一致才接受）。
REQUIRED_SCORE_SEMANTICS = "uncalibrated_model_score"

#: 阈值策略的固定值：冻结阈值来自 validation 上选出的 F1 阈值。
REQUIRED_THRESHOLD_POLICY = "validation_f1"


class ScoringRuntime(Protocol):
    """评分运行时接口：真实 bundle 与测试 fake 都实现它。

    应用工厂只依赖这个接口，因此测试可以注入一个不加载任何文件的 fake，
    既能断言"每请求不重复加载"，也能精确统计评分调用次数（批量原子性）。
    """

    model_version: str
    bundle_sha256: str
    threshold: float
    threshold_policy: str
    score_semantics: str
    source_features: list[str]

    def score_many(self, rows: Sequence[dict[str, Any]]) -> list[float]:
        """按**输入顺序**返回模型分数。"""
        ...


def sha256_of_file(path: Path, *, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """流式计算文件 SHA-256。不反序列化、不解析内容。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle_hash(path: Path, expected_sha256: str) -> str:
    """在反序列化之前校验 bundle 指纹。

    Raises:
        BundleLoadError: 文件缺失或指纹不符。**不得**在此之后才做校验。
    """
    if not path.is_file():
        raise BundleLoadError(BundleErrorCode.MISSING, "bundle 文件不存在")
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise BundleLoadError(
            BundleErrorCode.HASH_MISMATCH,
            f"bundle 指纹不符：实际 {actual[:12]}… 期望 {expected_sha256[:12]}…",
        )
    return actual


def _check_threshold(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleLoadError(BundleErrorCode.THRESHOLD_INVALID, f"{label} 不是数值")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise BundleLoadError(
            BundleErrorCode.THRESHOLD_INVALID, f"{label} 必须是 [0,1] 内的有限数"
        )
    return number


@dataclass
class BundleRuntime:
    """已通过指纹与契约校验的真实运行时。"""

    bundle: Any
    model_version: str
    bundle_sha256: str
    threshold: float
    threshold_policy: str
    score_semantics: str
    source_features: list[str] = field(default_factory=list)
    #: 源特征里的数值列 / 类别列划分。只用于生成合成 smoke 输入与自描述，
    #: 不含任何取值或类别集合。
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)

    def score_many(self, rows: Sequence[dict[str, Any]]) -> list[float]:
        """按输入顺序批量打分。

        列顺序在调用前显式重排成 bundle 记录的源特征顺序，因此调用方传入的字典顺序
        不影响结果；列集合的校验由上层完成（缺列/多列都是 422，不会走到这里）。
        """
        import pandas as pd

        frame = pd.DataFrame([dict(row) for row in rows], columns=list(self.source_features))
        scores = self.bundle.score_batch(frame)
        return [float(value) for value in scores]


def load_runtime(config: ServiceConfig) -> BundleRuntime:
    """按"先验哈希、后反序列化"的顺序加载 bundle 并完成契约校验。

    Args:
        config: 已解析的服务配置（提供 bundle 路径、固定指纹与期望特征数）。

    Returns:
        可直接用于评分的 :class:`BundleRuntime`。

    Raises:
        BundleLoadError: 文件缺失、指纹不符、依赖缺失、反序列化失败或契约校验不通过。
            调用方必须把应用置为"未就绪"，**不得**下载、重训或回退到其它模型。
    """
    # 1) 指纹校验必须在反序列化之前。
    verify_bundle_hash(config.bundle_path, config.bundle_sha256)

    # 2) 只有校验通过后才会执行到这里。
    try:
        import joblib
    except ImportError as error:  # pragma: no cover - joblib 是既有依赖
        raise BundleLoadError(BundleErrorCode.DEPENDENCY_MISSING, f"joblib 不可用：{error}") from error

    try:
        from ai_detector.models.lightgbm_challenger import LightGbmBundle

        payload = joblib.load(config.bundle_path)
    except ImportError as error:
        raise BundleLoadError(
            BundleErrorCode.DEPENDENCY_MISSING, f"bundle 类定义不可导入：{error}"
        ) from error
    except BundleLoadError:
        raise
    except Exception as error:  # noqa: BLE001 - 反序列化失败原因很多，统一折叠
        raise BundleLoadError(
            BundleErrorCode.DESERIALIZE_FAILED, f"bundle 反序列化失败：{type(error).__name__}"
        ) from error

    if not isinstance(payload, LightGbmBundle):
        raise BundleLoadError(
            BundleErrorCode.WRONG_TYPE, f"bundle 类型为 {type(payload).__name__}"
        )

    # 3) 契约校验。
    if str(payload.schema_version) != config.expected_schema_version:
        raise BundleLoadError(
            BundleErrorCode.SCHEMA_MISMATCH,
            f"schema 版本为 {payload.schema_version}，期望 {config.expected_schema_version}",
        )

    features = [str(name) for name in payload.source_features]
    if len(features) != config.expected_source_features:
        raise BundleLoadError(
            BundleErrorCode.FEATURE_CONTRACT,
            f"源特征数 {len(features)} != 期望 {config.expected_source_features}",
        )
    if len(set(features)) != len(features):
        raise BundleLoadError(BundleErrorCode.FEATURE_CONTRACT, "源特征清单存在重复")
    if not features:
        raise BundleLoadError(BundleErrorCode.FEATURE_CONTRACT, "源特征清单为空")

    thresholds = getattr(payload, "frozen_thresholds", None) or {}
    threshold = _check_threshold(thresholds.get("selected_f1_threshold"), "selected_f1_threshold")
    for name, payload_threshold in (thresholds.get("cost_scenarios") or {}).items():
        _check_threshold(payload_threshold.get("selected_threshold"), f"cost_scenarios.{name}")

    # 分数语义与阈值策略：配置与 bundle 自描述都必须是固定值。
    # 配置层（ServiceConfig.__post_init__）已经拦过一道；这里再查一次是**纵深防御**：
    # 只要有人塞进一个不是由 ServiceConfig 正规构造的对象，运行时仍会拒绝，
    # 而不会把一个任意策略名当作"已冻结事实"对外声明。
    described = payload.describe()
    if config.score_semantics != REQUIRED_SCORE_SEMANTICS:
        raise BundleLoadError(
            BundleErrorCode.SCHEMA_MISMATCH,
            f"配置的 score_semantics 必须是 {REQUIRED_SCORE_SEMANTICS}",
        )
    if config.threshold_policy != REQUIRED_THRESHOLD_POLICY:
        raise BundleLoadError(
            BundleErrorCode.SCHEMA_MISMATCH,
            f"配置的 threshold_policy 必须是 {REQUIRED_THRESHOLD_POLICY}",
        )
    declared = {str(key).lower() for key in described}
    if any("probab" in key or "calibrat" in key for key in declared):
        raise BundleLoadError(
            BundleErrorCode.SCHEMA_MISMATCH, "bundle 自描述里出现概率/校准字段"
        )

    if bool(described.get("contains_raw_data", False)):
        raise BundleLoadError(BundleErrorCode.CONTAINS_RAW_DATA, "bundle 标记为包含原始数据")

    return BundleRuntime(
        bundle=payload,
        model_version=config.model_version,
        bundle_sha256=config.bundle_sha256,
        threshold=threshold,
        threshold_policy=str(config.threshold_policy),
        score_semantics=REQUIRED_SCORE_SEMANTICS,
        source_features=features,
        numeric_features=[str(name) for name in getattr(payload, "numeric_features", [])],
        categorical_features=[str(name) for name in getattr(payload, "categorical_features", [])],
    )


def load_policy_engine(config: ServiceConfig) -> PolicyEngine:
    """按"先验哈希、后解析"的顺序加载政策并构造规则引擎。

    与 :func:`load_runtime` 同构：指纹校验失败时 ``yaml.safe_load`` 一次都不会被调用
    （由 ``ai_detector.rules.config`` 保证，有测试守住）。

    这里传的是**项目相对路径** ``policy_relative_path``，而不是已经解析过的绝对
    ``policy_path``：生产链路上，路径安全校验必须由底层加载函数自己做一遍，
    而不是依赖"上层已经查过了"。``load_policy()`` 也**没有**任何放宽该检查的开关。

    Args:
        config: 已解析的服务配置（提供政策相对路径、固定指纹与 schema 版本）。

    Returns:
        可直接用于规则求值的 :class:`~ai_detector.rules.engine.PolicyEngine`。

    Raises:
        PolicyLoadError: 未配置政策、路径不安全、文件缺失、指纹不符或 schema 违规。
            调用方必须把政策相关能力置为"未就绪"，**不得**下载政策、不得回退到内置默认
            政策，也不得静默改用其它阈值。
    """
    if config.policy_relative_path is None or config.policy_sha256 is None:
        raise PolicyLoadError(PolicyErrorCode.NOT_CONFIGURED, "服务配置未声明 policy 段")
    policy, digest = load_policy(
        config.policy_relative_path,
        expected_sha256=config.policy_sha256,
        expected_schema_version=config.expected_policy_schema_version or "",
    )
    return PolicyEngine(policy, policy_sha256=digest)


def load_storage_repository(config: ServiceConfig) -> DecisionRepository:
    """解析安全的数据库路径并初始化（必要时创建）SQLite 存储。

    这是存储的**生产入口**，也是唯一做位置策略的地方：路径先经
    :func:`~ai_detector.storage.config.resolve_storage_path` 校验（项目相对 + 限定在
    ``artifacts/runtime/*.sqlite3`` + 无符号链接逃逸），**通过之后**才创建父目录与 schema。
    因此"把库写到别处"不是配置问题，而是加载失败。

    文件 I/O 只发生在这里——配置解析与 :func:`create_app` 都不碰磁盘。

    Args:
        config: 已解析的服务配置（提供项目相对路径与 schema 版本）。

    Returns:
        已就绪的 :class:`~ai_detector.storage.sqlite.DecisionRepository`。

    Raises:
        StorageLoadError: 未配置存储、路径不安全、目录不可创建、数据库打不开、
            结构未知/损坏或版本不符。调用方必须把 decisions 接口与 readiness 置为未就绪，
            **不得**改用内存数据库或临时文件。
    """
    if config.storage_relative_path is None or config.storage_schema_version is None:
        raise StorageLoadError(StorageErrorCode.NOT_CONFIGURED, "服务配置未声明 storage 段")
    if config.storage_schema_version != STORAGE_SCHEMA_VERSION:
        raise StorageLoadError(
            StorageErrorCode.SCHEMA_MISMATCH,
            f"storage.schema_version 必须是 {STORAGE_SCHEMA_VERSION}",
        )
    path = resolve_storage_path(config.storage_relative_path)
    return open_repository(path)
