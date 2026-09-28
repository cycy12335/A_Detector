"""CW-005 测试夹具：fake runtime、合成 bundle 与配置构造器。

设计要点：

- **fake runtime** 实现与真实 bundle 相同的接口，因此路由、契约校验、错误处理与日志
  都能在没有模型、没有 LightGBM、没有正式 bundle 的情况下被精确测试；
  它还记录每次 ``score_many`` 的入参，批量原子性就是靠"无效批次零次调用"证明的。
- **合成 bundle** 用真实的 :class:`LightGbmBundle` 类构造并写入临时文件，
  因此"先验哈希、后反序列化"的成功路径也能在单元测试里跑通，不依赖被 Git 忽略的正式产物。
- 全部数据为项目自建，不访问网络，不读取真实 CSV 或 test。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import yaml

from ai_detector.models.lightgbm_challenger import LightGbmBundle
from ai_detector.service.config import ServiceConfig
from tests.cw006_fixtures import stage_policy


class RawDataReportingBundle(LightGbmBundle):
    """测试用子类：``describe()`` 谎报包含原始数据，用于验证加载器会拒绝。

    必须定义在**模块层**：``joblib.dump`` 需要能按限定名找到它，局部函数无法被 pickle。
    """

    def describe(self) -> dict[str, Any]:
        described = super().describe()
        described["contains_raw_data"] = True
        return described

#: 默认的合成源特征：足以覆盖"数值 + 类别"两种取值，且数量与正式契约无关。
DEFAULT_FEATURES = ["feat_a", "feat_b", "feat_c"]
DEFAULT_SCORE_SEMANTICS = "uncalibrated_model_score"
DEFAULT_THRESHOLD = 0.11


class FakeRuntime:
    """实现 ``ScoringRuntime`` 的测试替身。"""

    def __init__(
        self,
        *,
        features: Sequence[str] | None = None,
        numeric_features: Sequence[str] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        model_version: str = "fake-model-v1",
        bundle_sha256: str = "0" * 64,
        score_semantics: str = DEFAULT_SCORE_SEMANTICS,
        threshold_policy: str = "validation_f1",
        scores: Sequence[float] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.source_features = list(features if features is not None else DEFAULT_FEATURES)
        self.numeric_features = list(
            numeric_features if numeric_features is not None else self.source_features
        )
        self.categorical_features = [
            name for name in self.source_features if name not in set(self.numeric_features)
        ]
        self.threshold = float(threshold)
        self.model_version = model_version
        self.bundle_sha256 = bundle_sha256
        self.score_semantics = score_semantics
        self.threshold_policy = threshold_policy
        self._scores = list(scores) if scores is not None else None
        self._error = error
        #: 每次调用的入参快照，用于断言"零次评分"。
        self.score_calls: list[list[dict[str, Any]]] = []

    @property
    def call_count(self) -> int:
        return len(self.score_calls)

    def score_many(self, rows: Sequence[dict[str, Any]]) -> list[float]:
        self.score_calls.append([dict(row) for row in rows])
        if self._error is not None:
            raise self._error
        count = len(rows)
        if self._scores is None:
            return [0.5] * count
        values = list(self._scores)
        while len(values) < count:
            values.append(values[-1] if values else 0.5)
        return values[:count]


class RecordingLoader:
    """记录调用次数的运行时加载器；可选地抛出预设异常。"""

    def __init__(self, runtime: Any = None, error: Exception | None = None) -> None:
        self.runtime = runtime if runtime is not None else FakeRuntime()
        self.error = error
        self.calls = 0

    def __call__(self, config: ServiceConfig) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.runtime


def make_config(
    tmp_path: Path,
    *,
    features: int = 3,
    bundle_path: str = "artifacts/models/cw004/lightgbm_bundle.joblib",
    bundle_sha256: str = "a" * 64,
    batch_limit: int = 100,
    host: str = "127.0.0.1",
    request_id_header: str = "X-Request-ID",
    schema_version: str = "cw004_bundle_v1",
    policy: dict | None | bool = True,
    policy_path: str | None = None,
    policy_sha256: str | None = None,
    policy_schema_version: str = "cw006_policy_v1",
    storage: dict | None = None,
) -> ServiceConfig:
    """构造一个指向临时目录的合法服务配置（不读取仓库配置）。

    Args:
        policy: ``True``（默认）写一份**规范政策**并把它声明进配置——这与 CW-006 之后
            仓库 ``configs/service.yaml`` 的真实形态一致，因此 CW-005 的接口测试继续在
            "政策正常"的前提下断言原有成功响应；``None`` 表示配置**不含** policy 段
            （用于验证"未配置政策"的降级行为）；传入映射则写入该自定义政策。
        policy_path: 只覆盖配置里**声明**的政策路径，且必须是**项目相对**形状
            （生产加载器只接受项目相对路径）；因此可以用它构造"声明的路径上根本没有文件"
            这类场景。通常配合 :func:`tests.cw006_fixtures.stage_relative` 使用。
        policy_sha256: 覆盖固定指纹（用于"指纹不符"等场景）。
        storage: CW-007 的 storage 段映射；``None``（默认）表示**不含** storage 段，
            因此 CW-005/CW-006 的既有断言在"未配置存储"的前提下原样成立。
    """
    outputs = tmp_path / "service.yaml"
    raw = {
        "service": {
            "name": "ai-detector-risk-scoring",
            "api_version": "v1",
            "model_version": "cw004-lightgbm-challenger-v1",
            "host": host,
            "port": 8000,
            "batch_limit": batch_limit,
            "request_id_header": request_id_header,
            "score_semantics": DEFAULT_SCORE_SEMANTICS,
            "threshold_policy": "validation_f1",
            "expected_source_features": features,
        },
        "bundle": {
            "path": bundle_path,
            "sha256": bundle_sha256,
            "expected_schema_version": schema_version,
        },
        "limits": {
            "max_request_id_length": 64,
            "request_id_pattern": "^[A-Za-z0-9._:-]{1,64}$",
        },
    }
    if policy is not None:
        # 政策文件写到**项目内**、被 Git 忽略的暂存目录，并声明其**项目相对**路径：
        # 生产的政策加载器只接受项目相对路径（硬安全契约），夹具因此走同一条真实链路，
        # 而不是给加载器开一个"测试专用"的放宽开关。
        policy_file, policy_relative = stage_policy(tmp_path, None if policy is True else policy)
        raw["policy"] = {
            "path": policy_path if policy_path is not None else policy_relative,
            "sha256": policy_sha256 if policy_sha256 is not None else sha256_of(policy_file),
            "expected_schema_version": policy_schema_version,
        }
    if storage is not None:
        raw["storage"] = dict(storage)
    outputs.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    # 夹具的 bundle 位于临时目录，因此显式打开 allow_external_paths；
    # 仓库配置与 CLI 入口始终使用严格的默认值。
    return ServiceConfig.from_mapping(
        raw, config_path=outputs, config_sha256="0" * 64, allow_external_paths=True
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_synthetic_bundle(
    path: Path,
    *,
    features: Sequence[str] | None = None,
    schema_version: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    numeric_features: Sequence[str] | None = None,
    bundle_class: type = LightGbmBundle,
) -> Path:
    """用真实的 ``LightGbmBundle`` 类写一个合成 bundle（不含任何真实数据）。

    预处理器与模型用占位对象即可：本夹具只用于验证"加载 + 契约校验"这条路，
    不执行真实推理（推理路径由 fake runtime 覆盖）。
    """
    import joblib

    names = list(features if features is not None else DEFAULT_FEATURES)
    numeric = list(numeric_features if numeric_features is not None else names)
    bundle = bundle_class(
        preprocessor=object(),
        model=object(),
        best_iteration=1,
        source_features=names,
        numeric_features=numeric,
        categorical_features=[name for name in names if name not in set(numeric)],
        transformed_feature_names=[f"numeric__{name}" for name in names],
        scale_pos_weight=1.0,
        frozen_thresholds={
            "selected_f1_threshold": threshold,
            "cost_scenarios": {"fp_fn_1_1": {"selected_threshold": threshold}},
        },
        fingerprints={"cw004_config_sha256": "0" * 64},
        model_params={"n_estimators": 1000},
        config_sha256="0" * 64,
    )
    if schema_version is not None:
        bundle.schema_version = schema_version
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def features_payload(runtime: Any, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """按运行时的源特征清单生成一份合法 features（常量取值）。"""
    payload = {name: 0.0 for name in runtime.source_features}
    if overrides:
        payload.update(overrides)
    return payload
