"""CW-004 训练编排：固定参数 LightGBM 挑战者 + 两阶段确定性 + 受控 test 评估。

## 执行顺序是硬约束

```text
前置门禁(含 CW-003 数据门禁与指纹)
  -> 复用 CW-002 切分 -> train-only 列决策(仅 train) -> 校验 199 列契约
  -> 装配三个分区(test 标签在装配阶段即被封存)
  -> 候选运行 A：fit 预处理(仅 train) + fit LightGBM + validation 早停 + 冻结阈值
  -> 候选运行 B：用相同数据、配置、种子重跑同一流程
  -> 确定性比较(atol 1e-12)；不一致立即失败且**不开放 test**
  -> [冻结] 模型、best_iteration 与全部阈值冻结
  -> 一次受控 test：恰好一次 features 读取 + 恰好一次 labels 读取
  -> 四模型来源化对比 + 源特征聚合重要性 + 推理 bundle + 报告
```

## 与 CW-003 的关系

- 数据路径、CW-002 切分、431 列允许清单、train-only 列决策规则、预处理语义、99 个阈值网格
  与 3 个成本场景**全部来自已验收的 CW-003 配置**（按 SHA-256 硬校验），本模块不重新定义。
- 三基线（Rule / Dummy / Logistic Regression）只从已验收 ``report.json`` **读取**并记录来源
  指纹，**不重新评估**它们的 test。
- 本模块不修改 CW-003 的任何代码路径、配置、报告、模型或指标。

## test 隔离是结构性的

``assemble_partitions`` 在装配阶段就把 test 目标列剥离并封存；此后 test 标签只能经
:class:`HoldoutAccessTracker` 取出，且必须已 ``freeze``。候选 A/B 全程不接触 test，
确定性比较通过后才调用 ``freeze``。
"""

from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

from ai_detector import __version__
from ai_detector.config import PROJECT_ROOT, ConfigError, resolve_path
from ai_detector.data.modeling import (
    ASSEMBLY_CHUNK_SIZE,
    PARTITION_TEST,
    PARTITION_TRAIN,
    PARTITION_VALIDATION,
    HoldoutAccessTracker,
    ModelingConfig,
    ModelingContractError,
    assemble_partitions,
    assert_partition_invariants,
    load_partition_lookup,
    verify_data_gate,
)
from ai_detector.evaluation.cost import curve_to_frame, metrics_at_threshold
from ai_detector.evaluation.feature_importance import (
    CONSERVATION_ATOL,
    CONSERVATION_RTOL,
    aggregate_by_source,
    assert_conserved,
    build_importance_frame,
    transformed_dim_sources,
)
from ai_detector.evaluation.report import environment_versions
from ai_detector.features.ieee_cis import (
    FilterDecision,
    accumulate_train_stats,
    assert_no_forbidden_features,
    assert_no_forbidden_transformed_names,
    build_preprocessor,
    decide_columns,
)
from ai_detector.ieee_training import freeze_thresholds, json_dumps
from ai_detector.models.lightgbm_challenger import (
    LGBM_KEY,
    LightGbmBundle,
    booster_importance,
    build_lgbm_estimator,
    compute_scale_pos_weight,
    describe_lightgbm,
    fit_lgbm_with_early_stopping,
    lightgbm_version,
    save_bundle,
    validate_frozen_params,
)

WORK_PACKAGE = "CW-004"

#: 正式配置中声明的 CW-003 契约（194 数值 + 5 类别 = 199 列 → 348 维）。
#: **代码不硬编码这两个数**：它们由配置声明、由门禁核对，因此合成夹具可以走同一条代码路径。
REAL_SOURCE_FEATURES = 199
REAL_TRANSFORMED_DIMENSION = 348

#: 聚合重要性的报告上限（完整源特征表写入 CSV，报告只放 Top N）。
IMPORTANCE_TOP_N = 30

#: 明确禁止的分析，写入报告以免读者误读。
FORBIDDEN_ANALYSES = (
    "SHAP",
    "Permutation Importance",
    "Partial Dependence (PDP)",
    "概率校准",
    "匿名字段的业务语义解释",
    "因果解释",
)


class PrerequisiteGateError(RuntimeError):
    """CW-004 开工前置门禁不通过。出现即应把任务置为 blocked 并停止。"""


class DeterminismError(RuntimeError):
    """候选运行 A/B 未通过确定性比较。出现即**不得**开放 test。"""


class FeatureContractError(RuntimeError):
    """特征契约与 CW-003 已验收的 feature manifest 不一致。"""


# --- 峰值内存（只有可靠测量才报告）--------------------------------------


def peak_memory_mib() -> float | None:
    """返回当前进程的峰值 RSS（MiB）；无法可靠测量时返回 ``None``。

    只做测量，不影响任何结论。测量不到时报告写 ``null``，不编造数值。
    """
    try:  # POSIX
        import platform
        import resource

        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux 以 KiB 计，macOS 以字节计。
        divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
        return round(usage / divisor, 1)
    except (ImportError, ValueError, OSError):
        pass
    try:  # Windows
        import ctypes
        from ctypes import wintypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # 必须显式声明 argtypes/restype：默认的 c_int 返回类型会把伪句柄截断，
        # 导致 GetProcessMemoryInfo 静默失败并返回 0（实测过）。
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
        if ok:
            return round(counters.PeakWorkingSetSize / (1024.0 * 1024.0), 1)
    except Exception:  # noqa: BLE001 - 测量失败不应影响实验
        pass
    return None


# --- 配置 ---------------------------------------------------------------


def _assert_project_relative(value: Any, label: str) -> None:
    """硬校验：配置里的路径必须是项目相对路径，不得写绝对路径或凭据。"""
    text = str(value)
    lowered = text.lower()
    if Path(text).is_absolute() or (len(text) > 1 and text[1] == ":"):
        raise ConfigError(f"{label} 必须是项目相对路径，不得写绝对路径：{text}")
    if text.startswith(("/", "\\")) or ".." in Path(text).parts:
        raise ConfigError(f"{label} 不得逃出项目根目录：{text}")
    for credential in ("password", "token", "cookie", "api_key", "secret"):
        if credential in lowered:
            raise ConfigError(f"{label} 含疑似凭据字段 {credential}：{text}")


@dataclass(frozen=True)
class LightGbmConfig:
    """``configs/ieee_cis_lightgbm.yaml`` 的类型化表示。"""

    work_package: str
    seed: int
    cw003: dict[str, Any]
    model: dict[str, Any]
    determinism: dict[str, Any]
    importance: dict[str, Any]
    outputs: dict[str, Path]
    limits: dict[str, Any]
    modeling: ModelingConfig
    config_path: Path
    config_sha256: str = ""

    # --- 便捷访问 -------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model["params"])

    @property
    def early_stopping(self) -> dict[str, Any]:
        return dict(self.model["early_stopping"])

    @property
    def imbalance(self) -> dict[str, Any]:
        return dict(self.model["imbalance"])

    @property
    def baseline_keys(self) -> tuple[str, ...]:
        return tuple(str(key) for key in self.cw003["baseline_keys"])

    @property
    def expected_source_features(self) -> int:
        """配置声明的源特征数（正式配置为 194 + 5 = 199）。"""
        return int(self.cw003["expected_kept_numeric"]) + int(
            self.cw003["expected_kept_categorical"]
        )

    @property
    def expected_transformed_dimension(self) -> int:
        """配置声明的转换维度（正式配置为 348）。"""
        return int(self.cw003["expected_transformed_dimension"])

    @property
    def absolute_tolerance(self) -> float:
        return float(self.determinism["absolute_tolerance"])

    @property
    def relative_tolerance(self) -> float:
        return float(self.determinism.get("relative_tolerance", 0.0))

    def relative(self, path: str | Path) -> str:
        from ai_detector.data.fingerprint import report_path

        return report_path(path, PROJECT_ROOT)[0]

    def input_path(self, key: str) -> Path:
        return resolve_path(str(self.cw003[key]))

    # --- 加载 -----------------------------------------------------------

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        config_path: Path,
        config_sha256: str = "",
        allow_external_paths: bool = False,
    ) -> LightGbmConfig:
        """从已解析的 YAML 映射构造配置。

        Args:
            allow_external_paths: 仅测试夹具使用。为 ``False``（仓库配置与命令行入口的默认值）时，
                配置里的每一个路径都必须是**项目相对路径**；测试夹具位于临时目录，必须显式
                打开该开关，因此"仓库配置里不得出现绝对路径"这条纪律无法被悄悄绕过。
        """
        required = (
            "work_package",
            "seed",
            "cw003",
            "model",
            "determinism",
            "importance",
            "paths",
            "limits",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ConfigError("CW-004 配置缺少字段：" + ", ".join(missing))
        if str(raw["work_package"]) != WORK_PACKAGE:
            raise ConfigError(f"work_package 必须是 {WORK_PACKAGE}，实际为 {raw['work_package']}")

        cw003 = dict(raw["cw003"])
        for key in (
            "config_path",
            "expected_config_sha256",
            "feature_manifest_path",
            "expected_feature_manifest_sha256",
            "expected_kept_numeric",
            "expected_kept_categorical",
            "expected_transformed_dimension",
            "expected_output_format",
            "report_path",
            "baseline_keys",
            "required_artifacts",
            "prerequisite_task_status",
            "prerequisite_required_status",
        ):
            if key not in cw003:
                raise ConfigError(f"cw003 段缺少字段 {key}")
        if not allow_external_paths:
            for key in (
                "config_path",
                "feature_manifest_path",
                "report_path",
                "prerequisite_task_status",
            ):
                _assert_project_relative(cw003[key], f"cw003.{key}")
            for path in cw003["required_artifacts"]:
                _assert_project_relative(path, "cw003.required_artifacts")
        if int(cw003["expected_kept_numeric"]) <= 0 or int(cw003["expected_kept_categorical"]) < 0:
            raise ConfigError("cw003.expected_kept_numeric / expected_kept_categorical 必须为非负数")
        if int(cw003["expected_transformed_dimension"]) <= 0:
            raise ConfigError("cw003.expected_transformed_dimension 必须为正整数")
        if str(cw003["expected_output_format"]) != "sparse_csr":
            raise ConfigError("cw003.expected_output_format 必须是 sparse_csr")

        # 冻结参数逐项核对：多一个、少一个、或出现 GPU/搜索类参数都直接失败。
        model = dict(raw["model"])
        if str(model.get("type")) != "LGBMClassifier":
            raise ConfigError("model.type 必须是冻结值 LGBMClassifier")
        validate_frozen_params(dict(model.get("params") or {}))
        if int(model["params"]["random_state"]) != int(raw["seed"]):
            raise ConfigError("model.params.random_state 必须与顶层 seed 一致")
        early = dict(model.get("early_stopping") or {})
        if str(early.get("validation_partition")) != PARTITION_VALIDATION:
            raise ConfigError("early_stopping.validation_partition 必须是 validation")
        if int(early.get("stopping_rounds", 0)) <= 0:
            raise ConfigError("early_stopping.stopping_rounds 必须为正整数")
        if str(early.get("metric")) != "binary_logloss":
            raise ConfigError("early_stopping.metric 必须是冻结值 binary_logloss")
        imbalance = dict(model.get("imbalance") or {})
        if str(imbalance.get("strategy")) != "scale_pos_weight":
            raise ConfigError("imbalance.strategy 必须是冻结值 scale_pos_weight")
        if str(imbalance.get("formula")) != "train_negative_count / train_positive_count":
            raise ConfigError("imbalance.formula 必须是冻结的 train 派生公式")

        determinism = dict(raw["determinism"])
        if float(determinism.get("absolute_tolerance", 0.0)) <= 0.0:
            raise ConfigError("determinism.absolute_tolerance 必须为正数")

        importance = dict(raw["importance"])
        if list(importance.get("types") or []) != ["gain", "split"]:
            raise ConfigError("importance.types 必须是 [gain, split]（不做 SHAP / Permutation / PDP）")
        if str(importance.get("aggregation")) != "source_feature":
            raise ConfigError("importance.aggregation 必须是 source_feature")

        outputs_raw = dict(raw["paths"]["outputs"])
        if not allow_external_paths:
            for key, value in outputs_raw.items():
                _assert_project_relative(value, f"paths.outputs.{key}")

        modeling = ModelingConfig.from_yaml(resolve_path(str(cw003["config_path"])))

        return cls(
            work_package=str(raw["work_package"]),
            seed=int(raw["seed"]),
            cw003=cw003,
            model=model,
            determinism=determinism,
            importance=importance,
            outputs={key: resolve_path(value) for key, value in outputs_raw.items()},
            limits=dict(raw["limits"]),
            modeling=modeling,
            config_path=config_path,
            config_sha256=config_sha256,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, *, allow_external_paths: bool = False) -> LightGbmConfig:
        config_path = resolve_path(path)
        if not config_path.is_file():
            raise ConfigError(f"配置文件不存在：{config_path}")
        raw_bytes = config_path.read_bytes()
        raw = yaml.safe_load(raw_bytes.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件顶层必须是映射：{config_path}")
        return cls.from_mapping(
            raw,
            config_path=config_path,
            config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            allow_external_paths=allow_external_paths,
        )


# --- 前置门禁 -----------------------------------------------------------


def sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def verify_prerequisite_gates(config: LightGbmConfig) -> dict[str, Any]:
    """CW-004 开工门禁：CW-003 状态、配置指纹、特征契约指纹、正式产物与数据门禁。

    Raises:
        PrerequisiteGateError: 任一条件不满足（应据此置为 blocked 并停止）。
        DataGateError: CW-002/CW-003 数据门禁不通过（原样向上抛出）。
    """
    failures: list[str] = []
    cw003 = config.cw003

    # 1. 前置协议任务状态
    status_path = resolve_path(str(cw003["prerequisite_task_status"]))
    status_evidence: dict[str, Any] = {"relative_path": config.relative(status_path)}
    if not status_path.is_file():
        failures.append("缺少 CW-003 协议任务 status.json")
    else:
        payload = _load_json(status_path)
        actual = str(payload.get("status"))
        status_evidence.update(
            {
                "task_id": payload.get("task_id"),
                "status": actual,
                "required_status": str(cw003["prerequisite_required_status"]),
            }
        )
        if actual != str(cw003["prerequisite_required_status"]):
            failures.append(
                f"CW-003 协议任务状态为 {actual}，必须为 {cw003['prerequisite_required_status']}"
            )

    # 2. CW-003 建模配置指纹（数据路径、切分、允许清单、预处理、阈值网格的来源）
    cw003_config = config.modeling
    config_evidence = {
        "relative_path": config.relative(cw003_config.config_path),
        "sha256": cw003_config.config_sha256,
        "expected_sha256": str(cw003["expected_config_sha256"]),
    }
    config_evidence["match"] = (
        cw003_config.config_sha256 == str(cw003["expected_config_sha256"])
    )
    if not config_evidence["match"]:
        failures.append(
            "CW-003 配置 SHA-256 与门禁期望不符："
            f"{cw003_config.config_sha256} != {cw003['expected_config_sha256']}"
        )

    # 3. 特征契约指纹与计数
    manifest_path = resolve_path(str(cw003["feature_manifest_path"]))
    manifest_evidence: dict[str, Any] = {"relative_path": config.relative(manifest_path)}
    manifest: dict[str, Any] = {}
    if not manifest_path.is_file():
        failures.append("缺少 CW-003 feature_manifest.json")
    else:
        digest = sha256_of_file(manifest_path)
        manifest_evidence.update(
            {
                "sha256": digest,
                "expected_sha256": str(cw003["expected_feature_manifest_sha256"]),
                "match": digest == str(cw003["expected_feature_manifest_sha256"]),
            }
        )
        if not manifest_evidence["match"]:
            failures.append(
                f"feature_manifest.json SHA-256 与门禁期望不符：{digest}"
            )
        manifest = _load_json(manifest_path)
        kept = manifest.get("kept", {})
        numeric_count = len(kept.get("numeric", []))
        categorical_count = len(kept.get("categorical", []))
        manifest_evidence.update(
            {
                "kept_numeric": numeric_count,
                "kept_categorical": categorical_count,
                "transformed_dimension": manifest.get("transformed_dimension"),
                "output_format": manifest.get("output_format"),
            }
        )
        if numeric_count != int(cw003["expected_kept_numeric"]):
            failures.append(f"kept 数值列 {numeric_count} != {cw003['expected_kept_numeric']}")
        if categorical_count != int(cw003["expected_kept_categorical"]):
            failures.append(
                f"kept 类别列 {categorical_count} != {cw003['expected_kept_categorical']}"
            )
        if int(manifest.get("transformed_dimension", -1)) != int(
            cw003["expected_transformed_dimension"]
        ):
            failures.append("feature_manifest 的 transformed_dimension 与门禁期望不符")
        if str(manifest.get("output_format")) != str(cw003["expected_output_format"]):
            failures.append("feature_manifest 的 output_format 与门禁期望不符")

    # 4. CW-003 正式产物（只校验存在性并记录来源指纹；不重新评估其 test）
    report_path = resolve_path(str(cw003["report_path"]))
    report_evidence: dict[str, Any] = {"relative_path": config.relative(report_path)}
    if not report_path.is_file():
        failures.append("缺少 CW-003 report.json（三基线来源）")
    else:
        report_evidence["sha256"] = sha256_of_file(report_path)

    artifact_evidence: list[dict[str, Any]] = []
    for relative in cw003["required_artifacts"]:
        path = resolve_path(str(relative))
        artifact_evidence.append(
            {"relative_path": config.relative(path), "exists": path.is_file()}
        )
        if not path.is_file():
            failures.append(f"缺少 CW-003 正式产物：{config.relative(path)}")

    if failures:
        raise PrerequisiteGateError(
            "CW-004 前置门禁未通过，任务应置为 blocked；不得替换文件、重算切分或降低门禁。\n"
            + "\n".join(f"  - {item}" for item in failures)
        )

    # 5. 数据门禁（复用 CW-003 的实现；指纹、清单、边界任一不符即抛 DataGateError）
    data_gate = verify_data_gate(cw003_config)

    return {
        "prerequisite_task": status_evidence,
        "cw003_config": config_evidence,
        "cw003_feature_manifest": manifest_evidence,
        "cw003_report": report_evidence,
        "cw003_required_artifacts": artifact_evidence,
        "data_gate": data_gate,
    }


# --- 特征契约核对 -------------------------------------------------------


def assert_feature_contract(
    config: LightGbmConfig, decision: FilterDecision, manifest: dict[str, Any]
) -> dict[str, Any]:
    """逐项核对重算的 kept 列与已验收 feature manifest 完全一致。

    Raises:
        FeatureContractError: 列名顺序、数量或来源不一致。
    """
    expected_numeric = list(manifest["kept"]["numeric"])
    expected_categorical = list(manifest["kept"]["categorical"])
    problems: list[str] = []
    if decision.numeric != expected_numeric:
        problems.append(
            f"数值列不一致：重算 {len(decision.numeric)} 列 vs manifest "
            f"{len(expected_numeric)} 列"
        )
    if decision.categorical != expected_categorical:
        problems.append(
            f"类别列不一致：重算 {len(decision.categorical)} 列 vs manifest "
            f"{len(expected_categorical)} 列"
        )
    if problems:
        raise FeatureContractError(
            "重算的特征清单与 CW-003 已验收 feature_manifest 不一致：\n"
            + "\n".join(f"  - {item}" for item in problems)
        )
    return {
        "kept_numeric": len(decision.numeric),
        "kept_categorical": len(decision.categorical),
        "kept_total": len(decision.kept),
        "output_format": str(config.cw003["expected_output_format"]),
        "matches_cw003_manifest": True,
        "order_matches": True,
    }


def _read_cw003_manifest(config: LightGbmConfig) -> dict[str, Any]:
    return _load_json(resolve_path(str(config.cw003["feature_manifest_path"])))


# --- 候选运行 -----------------------------------------------------------


@dataclass
class CandidateRun:
    """一次 train/validation 候选运行的完整证据。**不含 test 的任何内容。**"""

    label: str
    kept_numeric: list[str]
    kept_categorical: list[str]
    transformed_feature_names: list[str]
    transformed_dimension: int
    output_format: str
    scale_pos_weight: float
    best_iteration: int
    validation_scores: np.ndarray
    frozen: dict[str, Any]
    validation_confusion: dict[str, Any]
    aggregated_gain: dict[str, float]
    aggregated_split: dict[str, int]
    gain_by_dimension: np.ndarray
    split_by_dimension: np.ndarray
    importance_conservation: dict[str, Any]
    early_stopping: dict[str, Any]
    model_params: dict[str, Any]
    preprocessor: Any = None
    model: Any = None
    timings: dict[str, float] = field(default_factory=dict)

    def comparison_payload(self) -> dict[str, Any]:
        """抽取确定性比较所需的最小字段（便于测试直接扰动）。"""
        return {
            "kept_numeric": list(self.kept_numeric),
            "kept_categorical": list(self.kept_categorical),
            "transformed_dimension": int(self.transformed_dimension),
            "output_format": str(self.output_format),
            "scale_pos_weight": float(self.scale_pos_weight),
            "best_iteration": int(self.best_iteration),
            "validation_scores": np.asarray(self.validation_scores, dtype=float),
            "frozen_thresholds": _threshold_payload(self.frozen),
            "validation_confusion_matrix": dict(self.validation_confusion),
            "aggregated_gain_importance": dict(self.aggregated_gain),
            "aggregated_split_importance": dict(self.aggregated_split),
        }


def _threshold_payload(frozen: dict[str, Any]) -> dict[str, Any]:
    """只取确定性比较关心的阈值字段（去掉曲线对象）。"""
    return {
        "selected_f1_threshold": float(frozen["selected_f1_threshold"]),
        "validation_f1": float(frozen["validation_f1"]),
        "validation_precision": float(frozen["validation_precision"]),
        "validation_recall": float(frozen["validation_recall"]),
        "cost_scenarios": {
            name: {
                "fp_cost": float(payload["fp_cost"]),
                "fn_cost": float(payload["fn_cost"]),
                "selected_threshold": float(payload["selected_threshold"]),
                "validation_total_cost": float(payload["validation_total_cost"]),
            }
            for name, payload in frozen["cost_scenarios"].items()
        },
    }


def run_candidate(
    label: str,
    config: LightGbmConfig,
    decision: FilterDecision,
    assembled: Any,
    tracker: HoldoutAccessTracker,
    *,
    metric_note: str | None = None,
) -> CandidateRun:
    """执行一次候选运行：fit 预处理（仅 train）+ fit LightGBM + validation 早停 + 冻结阈值。

    **全程不访问 test**：唯一的数据来源是 train 与 validation 两个分区。
    """
    cw003 = config.modeling
    feature_columns = list(decision.kept)
    assert_no_forbidden_features(feature_columns, cw003.forbidden_features)

    # ``assembled.features`` 返回的是列子集副本；分区原帧仍留在 ``assembled`` 中，
    # 因此用完即可释放副本，纯粹为了压低峰值内存，不改变任何语义。
    train_frame = assembled.features(PARTITION_TRAIN, feature_columns)
    validation_frame = assembled.features(PARTITION_VALIDATION, feature_columns)
    train_target = assembled.target(PARTITION_TRAIN, tracker=tracker)
    validation_target = assembled.target(PARTITION_VALIDATION, tracker=tracker)

    timings: dict[str, float] = {}

    # --- 预处理：只在 train 上 fit ---
    phase = time.perf_counter()
    preprocessor = build_preprocessor(cw003, decision)
    train_matrix = preprocessor.fit_transform(train_frame)
    if not sp.issparse(train_matrix) or not sp.isspmatrix_csr(train_matrix):
        raise FeatureContractError("预处理输出不是 CSR 稀疏矩阵：预处理语义已被改变")
    timings["preprocess_fit_transform"] = round(time.perf_counter() - phase, 3)
    del train_frame
    gc.collect()

    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    assert_no_forbidden_transformed_names(feature_names, cw003.forbidden_features)
    if len(feature_names) != config.expected_transformed_dimension:
        raise FeatureContractError(
            f"转换后维度 {len(feature_names)} != 配置声明的已验收契约 "
            f"{config.expected_transformed_dimension}"
        )

    phase = time.perf_counter()
    validation_matrix = preprocessor.transform(validation_frame)
    if not sp.issparse(validation_matrix):
        raise FeatureContractError("validation 变换输出不是稀疏矩阵")
    timings["preprocess_transform_validation"] = round(time.perf_counter() - phase, 3)
    del validation_frame
    gc.collect()

    # --- 类别不平衡权重：只由 train 标签计算 ---
    scale_pos_weight = compute_scale_pos_weight(train_target)

    # --- 拟合 + validation 早停 ---
    early = config.early_stopping
    estimator = build_lgbm_estimator(config.params, scale_pos_weight=scale_pos_weight)
    diagnostics = fit_lgbm_with_early_stopping(
        estimator,
        train_matrix,
        train_target,
        validation_matrix,
        validation_target,
        metric=str(early["metric"]),
        stopping_rounds=int(early["stopping_rounds"]),
        first_metric_only=bool(early["first_metric_only"]),
        api_notes=[str(item) for item in config.model.get("api_compatibility_notes", [])],
    )
    timings["lightgbm_fit"] = round(diagnostics.fit_seconds, 3)

    phase = time.perf_counter()
    validation_scores = np.asarray(estimator.predict_proba(validation_matrix))[:, 1]
    timings["lightgbm_predict_validation"] = round(time.perf_counter() - phase, 3)

    # --- 阈值只在 validation 上冻结（复用 CW-003 的 99 网格、tie-break 与成本场景）---
    phase = time.perf_counter()
    frozen = freeze_thresholds(
        cw003, validation_target, validation_scores, model_key=LGBM_KEY
    )
    timings["validation_threshold_freeze"] = round(time.perf_counter() - phase, 3)
    confusion = metrics_at_threshold(
        validation_target,
        validation_scores,
        threshold=float(frozen["selected_f1_threshold"]),
    ).to_dict()

    # --- 重要性：gain / split 聚合回源特征 ---
    phase = time.perf_counter()
    dim_sources = transformed_dim_sources(preprocessor, decision.numeric, decision.categorical)
    if len(dim_sources) != len(feature_names):
        raise FeatureContractError(
            f"维度映射长度 {len(dim_sources)} != 转换维度 {len(feature_names)}"
        )
    gain_values = booster_importance(estimator, "gain")
    split_values = booster_importance(estimator, "split")
    aggregated_gain = aggregate_by_source(dim_sources, gain_values)
    aggregated_split_raw = aggregate_by_source(dim_sources, split_values)
    conservation = {
        "gain": assert_conserved(float(np.sum(gain_values)), aggregated_gain, label="gain"),
        "split": assert_conserved(float(np.sum(split_values)), aggregated_split_raw, label="split"),
    }
    timings["importance_aggregation"] = round(time.perf_counter() - phase, 3)

    # 训练矩阵在拟合后即可释放：候选 B 会重新构造，test 阶段也不需要。
    del train_matrix, validation_matrix
    gc.collect()

    if metric_note:
        diagnostics.api_notes.append(metric_note)

    return CandidateRun(
        label=label,
        kept_numeric=list(decision.numeric),
        kept_categorical=list(decision.categorical),
        transformed_feature_names=feature_names,
        transformed_dimension=len(feature_names),
        output_format="sparse_csr",
        scale_pos_weight=scale_pos_weight,
        best_iteration=diagnostics.best_iteration,
        validation_scores=np.asarray(validation_scores, dtype=float),
        frozen=frozen,
        validation_confusion=confusion,
        aggregated_gain=aggregated_gain,
        aggregated_split={name: int(round(value)) for name, value in aggregated_split_raw.items()},
        gain_by_dimension=np.asarray(gain_values, dtype=float),
        split_by_dimension=np.asarray(split_values, dtype=float),
        importance_conservation=conservation,
        early_stopping=diagnostics.to_dict(),
        model_params=describe_lightgbm(estimator),
        preprocessor=preprocessor,
        model=estimator,
        timings=timings,
    )


# --- 确定性比较 ---------------------------------------------------------


def _compare_float_arrays(
    left: np.ndarray, right: np.ndarray, *, atol: float, rtol: float
) -> dict[str, Any]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        return {
            "match": False,
            "reason": f"形状不一致：{left.shape} vs {right.shape}",
            "max_absolute_difference": None,
        }
    if left.size == 0:
        return {"match": True, "max_absolute_difference": 0.0}
    difference = np.abs(left - right)
    limit = atol + rtol * np.abs(right)
    return {
        "match": bool(np.all(difference <= limit)),
        "max_absolute_difference": float(np.max(difference)),
        "n_elements": int(left.size),
        "n_beyond_tolerance": int(np.sum(difference > limit)),
    }


def _compare_float_maps(
    left: dict[str, float], right: dict[str, float], *, atol: float, rtol: float
) -> dict[str, Any]:
    if set(left) != set(right):
        return {
            "match": False,
            "reason": "键集合不一致",
            "left_only": sorted(set(left) - set(right))[:5],
            "right_only": sorted(set(right) - set(left))[:5],
        }
    names = sorted(left)
    left_values = np.asarray([left[name] for name in names], dtype=float)
    right_values = np.asarray([right[name] for name in names], dtype=float)
    result = _compare_float_arrays(left_values, right_values, atol=atol, rtol=rtol)
    result["n_keys"] = len(names)
    return result


def compare_determinism(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    *,
    absolute_tolerance: float,
    relative_tolerance: float = 0.0,
) -> dict[str, Any]:
    """比较候选运行 A/B 的确定性证据。

    比较项：kept 列、转换维度、输出格式、``scale_pos_weight``、``best_iteration``、
    validation 分数、冻结阈值、validation 混淆矩阵与聚合后的 gain/split 重要性。

    Returns:
        逐项结果与 ``passed``。任一项超出容差即 ``passed=False``，调用方必须据此失败并
        **不开放 test**。
    """
    atol = float(absolute_tolerance)
    rtol = float(relative_tolerance)
    items: dict[str, Any] = {}

    for key in ("kept_numeric", "kept_categorical"):
        left, right = list(payload_a[key]), list(payload_b[key])
        entry: dict[str, Any] = {
            "match": left == right,
            "n_a": len(left),
            "n_b": len(right),
        }
        if left != right:
            entry["first_difference"] = next(
                (
                    {"index": index, "a": one, "b": other}
                    for index, (one, other) in enumerate(zip(left, right))
                    if one != other
                ),
                {"index": min(len(left), len(right)), "a": None, "b": None},
            )
        items[key] = entry
    items["output_format"] = {
        "match": payload_a["output_format"] == payload_b["output_format"],
        "a": payload_a["output_format"],
        "b": payload_b["output_format"],
    }
    items["transformed_dimension"] = {
        "match": int(payload_a["transformed_dimension"]) == int(payload_b["transformed_dimension"]),
        "a": int(payload_a["transformed_dimension"]),
        "b": int(payload_b["transformed_dimension"]),
    }
    items["best_iteration"] = {
        "match": int(payload_a["best_iteration"]) == int(payload_b["best_iteration"]),
        "a": int(payload_a["best_iteration"]),
        "b": int(payload_b["best_iteration"]),
    }
    for key in ("scale_pos_weight",):
        difference = abs(float(payload_a[key]) - float(payload_b[key]))
        items[key] = {
            "match": difference <= atol,
            "a": float(payload_a[key]),
            "b": float(payload_b[key]),
            "max_absolute_difference": difference,
        }

    items["validation_scores"] = _compare_float_arrays(
        payload_a["validation_scores"], payload_b["validation_scores"], atol=atol, rtol=rtol
    )

    # 冻结阈值：阈值本身是网格上的离散值，必须完全相等；派生浮点指标按容差比较。
    frozen_items: dict[str, Any] = {}
    thresholds_match = True
    for field in ("selected_f1_threshold",):
        same = float(payload_a["frozen_thresholds"][field]) == float(
            payload_b["frozen_thresholds"][field]
        )
        thresholds_match &= same
        frozen_items[field] = {
            "match": same,
            "a": float(payload_a["frozen_thresholds"][field]),
            "b": float(payload_b["frozen_thresholds"][field]),
        }
    for field in ("validation_f1", "validation_precision", "validation_recall"):
        difference = abs(
            float(payload_a["frozen_thresholds"][field]) - float(payload_b["frozen_thresholds"][field])
        )
        thresholds_match &= difference <= atol
        frozen_items[field] = {
            "match": difference <= atol,
            "max_absolute_difference": difference,
        }
    scenarios_a = payload_a["frozen_thresholds"]["cost_scenarios"]
    scenarios_b = payload_b["frozen_thresholds"]["cost_scenarios"]
    scenario_items: dict[str, Any] = {}
    if set(scenarios_a) != set(scenarios_b):
        thresholds_match = False
        scenario_items["_keys"] = {"match": False, "reason": "成本场景集合不一致"}
    else:
        for name in sorted(scenarios_a):
            left, right = scenarios_a[name], scenarios_b[name]
            same_threshold = float(left["selected_threshold"]) == float(right["selected_threshold"])
            cost_difference = abs(
                float(left["validation_total_cost"]) - float(right["validation_total_cost"])
            )
            scenario_items[name] = {
                "match": same_threshold and cost_difference <= atol,
                "a_threshold": float(left["selected_threshold"]),
                "b_threshold": float(right["selected_threshold"]),
                "max_absolute_difference": cost_difference,
            }
            thresholds_match &= scenario_items[name]["match"]
    frozen_items["cost_scenarios"] = scenario_items
    items["frozen_thresholds"] = {"match": bool(thresholds_match), "detail": frozen_items}

    # validation 混淆矩阵：整数计数，必须完全相等。
    confusion_a = payload_a["validation_confusion_matrix"]
    confusion_b = payload_b["validation_confusion_matrix"]
    confusion_items = {}
    confusion_match = set(confusion_a) == set(confusion_b)
    for key in sorted(set(confusion_a) & set(confusion_b)):
        value_a, value_b = confusion_a[key], confusion_b[key]
        if isinstance(value_a, float) or isinstance(value_b, float):
            same = abs(float(value_a) - float(value_b)) <= atol
        else:
            same = int(value_a) == int(value_b)
        confusion_items[key] = {"match": same, "a": value_a, "b": value_b}
        confusion_match &= same
    items["validation_confusion_matrix"] = {
        "match": bool(confusion_match),
        "detail": confusion_items,
    }

    items["aggregated_gain_importance"] = _compare_float_maps(
        payload_a["aggregated_gain_importance"],
        payload_b["aggregated_gain_importance"],
        atol=atol,
        rtol=rtol,
    )
    items["aggregated_split_importance"] = _compare_float_maps(
        {name: float(value) for name, value in payload_a["aggregated_split_importance"].items()},
        {name: float(value) for name, value in payload_b["aggregated_split_importance"].items()},
        atol=atol,
        rtol=rtol,
    )

    failed = sorted(name for name, payload in items.items() if not payload.get("match"))
    return {
        "tolerance": {"absolute": atol, "relative": rtol},
        "compared_items": sorted(items),
        "items": items,
        "failed_items": failed,
        "passed": not failed,
        "test_opened": False,
    }


# --- 三基线来源读取 -----------------------------------------------------


def read_cw003_baselines(config: LightGbmConfig, gates: dict[str, Any]) -> dict[str, Any]:
    """从已验收的 CW-003 ``report.json`` 读取三基线结果并记录来源指纹。

    **不重新评估**任何基线的 test：本函数只做文件读取与字段抽取。
    """
    report_path = resolve_path(str(config.cw003["report_path"]))
    report = _load_json(report_path)
    baselines: dict[str, Any] = {}
    for key in config.baseline_keys:
        if key not in report.get("baselines", {}):
            raise PrerequisiteGateError(f"CW-003 report.json 缺少基线 {key}")
        payload = report["baselines"][key]
        baselines[key] = {
            "type": payload.get("type"),
            "params": payload.get("params"),
            "fit_seconds": payload.get("fit_seconds"),
            "predict_seconds": payload.get("predict_seconds"),
            "thresholds": payload.get("thresholds"),
            "test": payload.get("test"),
        }
    return {
        "source_work_package": "CW-003",
        "source_report_relative_path": config.relative(report_path),
        "source_report_sha256": gates["cw003_report"]["sha256"],
        "source_report_generated_at": report.get("generated_at"),
        "re_evaluated": False,
        "note": (
            "三基线指标直接读取自 CW-003 已验收 report.json，本工作包未重新评估它们的 test；"
            "因此四模型对比是**跨来源汇总**，不是同一轮重训。"
        ),
        "baselines": baselines,
    }


# --- 四模型对比 ---------------------------------------------------------


def build_comparison_frame(
    config: LightGbmConfig,
    cw003_source: dict[str, Any],
    lightgbm_entry: dict[str, Any],
    *,
    lightgbm_train_seconds: float,
) -> pd.DataFrame:
    """四模型汇总表（无逐行数据）。"""
    rows: list[dict[str, Any]] = []
    for key in config.baseline_keys:
        payload = cw003_source["baselines"][key]
        metrics = payload["test"]["metrics"]
        rows.append(
            {
                "model_key": key,
                "source_work_package": "CW-003",
                "source_report_sha256": cw003_source["source_report_sha256"],
                "metrics_source": "cw003_report_json_read_only",
                "model_family": payload["type"],
                "frozen_threshold": metrics["threshold"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "predicted_positive_rate": metrics["predicted_positive_rate"],
                "train_seconds": payload["fit_seconds"],
                "train_seconds_source": "cw003_report_json_read_only",
                "complexity_note": _complexity_note(key),
            }
        )
    metrics = lightgbm_entry["metrics"]
    rows.append(
        {
            "model_key": LGBM_KEY,
            "source_work_package": WORK_PACKAGE,
            "source_report_sha256": config.config_sha256,
            "metrics_source": "cw004_single_shot_test_evaluation",
            "model_family": "LGBMClassifier",
            "frozen_threshold": metrics["threshold"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": metrics["false_positive_rate"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "predicted_positive_rate": metrics["predicted_positive_rate"],
            "train_seconds": round(float(lightgbm_train_seconds), 3),
            "train_seconds_source": "cw004_candidate_run_b",
            "complexity_note": _complexity_note(LGBM_KEY),
        }
    )
    return pd.DataFrame(rows)


def _complexity_note(key: str) -> str:
    return {
        "rule_tail_anomaly": "无参数透明规则（两个训练分位数阈值），不做拟合。",
        "dummy": "常数先验预测器，无特征、无参数估计。",
        "logistic_regression": "线性模型：源特征 → 稀疏线性组合，无特征交互。",
        "lightgbm": "梯度提升树（1000 棵上限、31 叶、深度不限），非线性与特征交互。",
    }.get(key, "未登记")


# --- 主流程 -------------------------------------------------------------


@dataclass
class LightGbmTrainingResult:
    """一次 CW-004 真实数据实验的产物位置与核心结论。"""

    report: dict[str, Any]
    report_json: Path
    report_markdown: Path
    threshold_curve_csv: Path
    comparison_csv: Path
    feature_importance_csv: Path
    bundle_path: Path


def run_lightgbm_training(
    config: LightGbmConfig,
    *,
    chunk_size: int = ASSEMBLY_CHUNK_SIZE,
) -> LightGbmTrainingResult:
    """执行完整 CW-004 实验。

    Raises:
        PrerequisiteGateError: 前置门禁不通过（应据此置为 blocked）。
        DataGateError: 数据门禁不通过。
        FeatureContractError: 特征契约与 CW-003 已验收 manifest 不一致。
        DeterminismError: 候选 A/B 未通过确定性比较（**不开放 test**）。
        LightGbmSpecError: LightGBM 缺失或规格不合法（应按协议置为 blocked）。
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}

    phase = time.perf_counter()
    gates = verify_prerequisite_gates(config)
    timings["prerequisite_gates"] = round(time.perf_counter() - phase, 3)

    cw003 = config.modeling
    manifest = _read_cw003_manifest(config)

    phase = time.perf_counter()
    lookup = load_partition_lookup(cw003)
    stats = accumulate_train_stats(cw003, lookup, chunk_size=chunk_size)
    decision = decide_columns(cw003, stats)
    timings["train_only_column_decisions"] = round(time.perf_counter() - phase, 3)

    contract = assert_feature_contract(config, decision, manifest)
    if contract["kept_total"] != config.expected_source_features:
        raise FeatureContractError(
            f"kept 列合计 {contract['kept_total']} != 配置声明的 "
            f"{config.expected_source_features}"
        )

    phase = time.perf_counter()
    assembled = assemble_partitions(cw003, lookup, decision, chunk_size=chunk_size)
    invariants = assert_partition_invariants(cw003, assembled)
    timings["assemble_partitions"] = round(time.perf_counter() - phase, 3)
    del lookup, stats
    gc.collect()

    tracker = HoldoutAccessTracker()

    # --- 候选运行 A（不访问 test）---
    run_a = run_candidate("A", config, decision, assembled, tracker)
    payload_a = run_a.comparison_payload()
    timings["candidate_run_a_seconds"] = round(
        sum(run_a.timings.values()), 3
    )
    timings["candidate_run_a_fit_seconds"] = run_a.timings.get("lightgbm_fit", 0.0)
    # 释放候选 A 的预处理器与模型：报告只需要它的确定性证据，候选 B 会重新拟合一份。
    run_a.preprocessor = None
    run_a.model = None
    gc.collect()

    # --- 候选运行 B（不访问 test）---
    run_b = run_candidate("B", config, decision, assembled, tracker)
    payload_b = run_b.comparison_payload()
    timings["candidate_run_b_seconds"] = round(sum(run_b.timings.values()), 3)

    determinism = compare_determinism(
        payload_a,
        payload_b,
        absolute_tolerance=config.absolute_tolerance,
        relative_tolerance=config.relative_tolerance,
    )
    if not determinism["passed"]:
        raise DeterminismError(
            "候选运行 A/B 未通过确定性比较，已停止且**未**开放 test。"
            f"失败项：{determinism['failed_items']}。"
            "不得放宽容差或以任何方式先评估 test 再补记录。"
        )

    # --- 冻结后才允许接触 test ---
    tracker.freeze(
        reason=(
            "候选运行 A/B 的确定性比较在声明容差内通过；LightGBM 模型、"
            f"best_iteration={run_b.best_iteration} 与全部 F1/成本阈值已冻结；"
            "此后只进行一次受控 test 评估"
        )
    )

    phase = time.perf_counter()
    # test 特征只能经受控入口取得：frozen 检查与访问记账都在 features() 内部、
    # 且发生在返回任何数据之前。这里不存在"先取数据、再手工补日志"的路径。
    test_features = assembled.features(
        PARTITION_TEST, list(decision.kept), tracker=tracker
    )
    test_target = assembled.target(PARTITION_TEST, tracker=tracker)
    test_metrics_summary = _validate_test_labels_once(cw003, test_target)
    test_matrix = run_b.preprocessor.transform(test_features)
    if not sp.issparse(test_matrix):
        raise FeatureContractError("test 变换输出不是稀疏矩阵")
    test_scores = np.asarray(run_b.model.predict_proba(test_matrix))[:, 1]
    lightgbm_test = _evaluate_frozen_with_auc(
        cw003, test_target, test_scores, run_b.frozen, model_key=LGBM_KEY
    )
    timings["frozen_test_evaluation"] = round(time.perf_counter() - phase, 3)
    del test_matrix, test_features
    gc.collect()

    cw003_source = read_cw003_baselines(config, gates)
    comparison = build_comparison_frame(
        config,
        cw003_source,
        lightgbm_test,
        lightgbm_train_seconds=run_b.timings.get("lightgbm_fit", 0.0),
    )

    importance_frame_all = build_importance_frame(
        source_names=list(decision.kept),
        per_dimension_sources=transformed_dim_sources(
            run_b.preprocessor, decision.numeric, decision.categorical
        ),
        gain_values=run_b.gain_by_dimension,
        split_values=run_b.split_by_dimension,
        kinds={
            **{name: "numeric" for name in decision.numeric},
            **{name: "categorical" for name in decision.categorical},
        },
    )

    bundle = LightGbmBundle(
        preprocessor=run_b.preprocessor,
        model=run_b.model,
        best_iteration=run_b.best_iteration,
        source_features=list(decision.kept),
        numeric_features=list(decision.numeric),
        categorical_features=list(decision.categorical),
        transformed_feature_names=list(run_b.transformed_feature_names),
        scale_pos_weight=run_b.scale_pos_weight,
        frozen_thresholds=_threshold_payload(run_b.frozen),
        fingerprints={
            "cw004_config_sha256": config.config_sha256,
            "cw003_config_sha256": gates["cw003_config"]["sha256"],
            "cw003_feature_manifest_sha256": gates["cw003_feature_manifest"]["sha256"],
            "cw003_report_sha256": gates["cw003_report"]["sha256"],
            "train_transaction_sha256": gates["data_gate"]["fingerprints"][
                "train_transaction.csv"
            ]["sha256"],
            "train_identity_sha256": gates["data_gate"]["fingerprints"]["train_identity.csv"][
                "sha256"
            ],
            "split_algorithm_version": gates["data_gate"]["split_algorithm_version"],
        },
        model_params=dict(config.params),
        config_sha256=config.config_sha256,
        notes=[
            "分数（score）不是已校准的欺诈概率：scale_pos_weight 会系统性抬高正类分数，"
            "本轮不做概率校准。",
            "identity_present 由装配阶段派生，但 remainder='drop' 使其不进入模型；"
            "推理契约只要求 199 个真实源特征。",
            "bundle 不含任何原始训练/验证/测试数据，也不含逐行分数。",
        ],
    )
    bundle.validate()

    report = _build_report(
        config=config,
        gates=gates,
        contract=contract,
        invariants=invariants,
        decision=decision,
        manifest=manifest,
        run_a=run_a,
        run_b=run_b,
        determinism=determinism,
        tracker=tracker,
        test_partition_summary=test_metrics_summary,
        lightgbm_test=lightgbm_test,
        cw003_source=cw003_source,
        comparison=comparison,
        importance_frame=importance_frame_all,
        bundle=bundle,
        timings=timings,
        total_seconds=time.perf_counter() - started,
        chunk_size=int(chunk_size),
    )

    outputs = _write_artifacts(
        config, report, run_b, importance_frame_all, comparison, bundle
    )
    return LightGbmTrainingResult(report=report, **outputs)


def _validate_test_labels_once(config: ModelingConfig, labels: np.ndarray) -> dict[str, Any]:
    """**冻结之后**、在**已经读取到**的 test 标签上做合法性校验。

    与 CW-003 的 ``validate_test_partition`` 的区别：本函数不自己调用
    ``assembled.target``，因此不会产生第二次 labels 访问日志。CW-004 要求生命周期日志
    中**恰好一次** features 与**恰好一次** labels 访问。
    """
    values = set(np.unique(labels).tolist())
    if not values <= {0, 1}:
        raise ModelingContractError(f"test 分区目标列取值非法：{sorted(values)}")
    expected = config.expected_split_sizes[PARTITION_TEST]
    if labels.size != expected:
        raise ModelingContractError(f"test 分区标签长度 {labels.size} != 期望 {expected}")
    return {
        "rows": int(labels.size),
        "label_rate": float(labels.mean()),
        "positive_count": int(labels.sum()),
        "read_after_freeze": True,
        "single_read": True,
    }


def _evaluate_frozen_with_auc(
    config: ModelingConfig,
    test_target: np.ndarray,
    test_scores: np.ndarray,
    frozen: dict[str, Any],
    *,
    model_key: str,
) -> dict[str, Any]:
    """冻结阈值下的 test 评估，并补齐 ROC-AUC / PR-AUC。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ai_detector.ieee_training import evaluate_frozen_thresholds

    if np.unique(test_target).size < 2:
        raise ModelingContractError("测试分区只有单一类别，ROC-AUC / PR-AUC 无定义")
    payload = evaluate_frozen_thresholds(
        config, test_target, test_scores, frozen, model_key=model_key
    )
    payload["metrics"]["roc_auc"] = float(roc_auc_score(test_target, test_scores))
    payload["metrics"]["pr_auc"] = float(average_precision_score(test_target, test_scores))
    payload["score_semantics"] = "模型分数（score），不是已校准概率"
    return payload


# --- 报告 ---------------------------------------------------------------


def _candidate_summary(run: CandidateRun) -> dict[str, Any]:
    """候选运行的可序列化摘要（**不含 test 的任何内容**）。"""
    scores = np.asarray(run.validation_scores, dtype=float)
    top_gain = sorted(run.aggregated_gain.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "label": run.label,
        "partition_scope": [PARTITION_TRAIN, PARTITION_VALIDATION],
        "test_accessed": False,
        "kept_numeric": list(run.kept_numeric),
        "kept_categorical": list(run.kept_categorical),
        "kept_total": len(run.kept_numeric) + len(run.kept_categorical),
        "transformed_dimension": int(run.transformed_dimension),
        "output_format": run.output_format,
        "scale_pos_weight": float(run.scale_pos_weight),
        "scale_pos_weight_source": "train_negative_count / train_positive_count（只用 train 标签）",
        "best_iteration": int(run.best_iteration),
        "early_stopping": dict(run.early_stopping),
        "validation_score_summary": {
            "n": int(scores.size),
            "min": float(scores.min()) if scores.size else None,
            "max": float(scores.max()) if scores.size else None,
            "mean": float(scores.mean()) if scores.size else None,
        },
        "frozen_thresholds": _threshold_payload(run.frozen),
        "validation_confusion_matrix": dict(run.validation_confusion),
        "importance_conservation": dict(run.importance_conservation),
        "top_gain_source_features": [
            {"source_feature": name, "aggregated_gain": float(value)} for name, value in top_gain
        ],
        "timings_seconds": dict(run.timings),
        "model_params": dict(run.model_params),
    }


def _build_report(
    *,
    config: LightGbmConfig,
    gates: dict[str, Any],
    contract: dict[str, Any],
    invariants: dict[str, Any],
    decision: FilterDecision,
    manifest: dict[str, Any],
    run_a: CandidateRun,
    run_b: CandidateRun,
    determinism: dict[str, Any],
    tracker: HoldoutAccessTracker,
    test_partition_summary: dict[str, Any],
    lightgbm_test: dict[str, Any],
    cw003_source: dict[str, Any],
    comparison: pd.DataFrame,
    importance_frame: pd.DataFrame,
    bundle: LightGbmBundle,
    timings: dict[str, float],
    total_seconds: float,
    chunk_size: int,
) -> dict[str, Any]:
    cw003 = config.modeling
    grid = cw003.threshold_grid()
    environment = environment_versions()
    environment["lightgbm"] = lightgbm_version()

    report: dict[str, Any] = {
        "work_package": WORK_PACKAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "seed": config.seed,
        "dataset": {
            "id": cw003.dataset_id,
            "boundaries": gates["data_gate"]["boundaries"],
            "expected_total_rows": cw003.expected_total_rows,
        },
        "config": {
            "relative_path": config.relative(config.config_path),
            "config_sha256": config.config_sha256,
        },
        "prerequisite_gates": {
            "prerequisite_task": gates["prerequisite_task"],
            "cw003_config": gates["cw003_config"],
            "cw003_feature_manifest": gates["cw003_feature_manifest"],
            "cw003_report": gates["cw003_report"],
            "cw003_required_artifacts": gates["cw003_required_artifacts"],
            "all_passed": True,
            "note": (
                "任一前置门禁不满足即置为 blocked；本报告只记录通过与指纹，"
                "不重新评估 CW-003 三类基线的 test。"
            ),
        },
        "data": {
            "fingerprints": gates["data_gate"]["fingerprints"],
            "split_manifest": gates["data_gate"]["split_manifest"],
            "split_algorithm_version": gates["data_gate"]["split_algorithm_version"],
            "boundary_recomputed": False,
        },
        "partitions": {
            PARTITION_TRAIN: {
                "rows": invariants["train_rows"],
                "label_rate": invariants["label_rates"][PARTITION_TRAIN],
            },
            PARTITION_VALIDATION: {
                "rows": invariants["validation_rows"],
                "label_rate": invariants["label_rates"][PARTITION_VALIDATION],
            },
            PARTITION_TEST: {
                "rows": invariants["test_rows"],
                "label_rate": test_partition_summary["label_rate"],
                "positive_count": test_partition_summary["positive_count"],
                "label_rate_computed_after_freeze": True,
            },
            "invariants": {
                "ids_disjoint": invariants["ids_disjoint"],
                "ids_complete": invariants["ids_complete"],
            },
        },
        "features": {
            "contract": {
                **contract,
                "expected_source_features": config.expected_source_features,
                "expected_transformed_dimension": config.expected_transformed_dimension,
                "cw003_manifest_transformed_dimension": manifest.get("transformed_dimension"),
                "cw003_manifest_output_format": manifest.get("output_format"),
                "recomputed_from_train_only": True,
            },
            "decision": decision.to_dict(),
            "excluded_breakdown": _excluded_breakdown(decision),
            "forbidden_features": list(cw003.forbidden_features),
            "identity_present_consumed_by_model": False,
            "identity_present_note": (
                "identity_present 由装配阶段派生并由 CW-003 的 remainder='drop' 丢弃，"
                "不进入模型；本工作包不改变该语义，推理契约只要求 199 个真实源特征。"
            ),
        },
        "model": {
            "type": config.model["type"],
            "params": dict(config.params),
            "param_source": str(config.model.get("param_source", "")),
            "hyperparameter_search": False,
            "imbalance": dict(config.imbalance),
            "early_stopping": dict(config.early_stopping),
            "api_compatibility_notes": [
                str(item) for item in config.model.get("api_compatibility_notes", [])
            ],
            "lightgbm_version": lightgbm_version(),
            "device": "cpu",
            "n_jobs": int(config.params["n_jobs"]),
        },
        "candidate_runs": {
            "design": {
                "n_runs": 2,
                "partition_scope": [PARTITION_TRAIN, PARTITION_VALIDATION],
                "test_accessed": False,
                "note": (
                    "A/B 两次候选运行只读 train/validation；确定性比较通过后才冻结并开放一次 test。"
                ),
            },
            "A": _candidate_summary(run_a),
            "B": _candidate_summary(run_b),
            "determinism": determinism,
        },
        "thresholds": {
            "source_partition": PARTITION_VALIDATION,
            "grid_size": len(grid),
            "grid_spec": dict(cw003.thresholds["grid"]),
            "f1_tie_break": list(cw003.thresholds["f1_tie_break"]),
            "cost_tie_break": list(cw003.thresholds["cost_tie_break"]),
            "metric": str(cw003.thresholds["metric"]),
            "frozen": _threshold_payload(run_b.frozen),
            "curve_rows": len(grid),
            "curve_written_to": config.relative(config.outputs["threshold_curve_csv"]),
        },
        "evaluation": {
            "evaluation_partition": str(config.limits.get("evaluation_partition", PARTITION_TEST)),
            "single_shot_test": bool(config.limits.get("single_shot_test", True)),
            "validation_used_for": [
                "early_stopping",
                "threshold_curve",
                "f1_threshold",
                "cost_thresholds",
            ],
            "test_used_for": ["single_frozen_evaluation"],
            "test_evaluation_history": test_evaluation_history(config),
            "frozen_artifacts": {
                "model": "LGBMClassifier",
                "best_iteration": int(run_b.best_iteration),
                "scale_pos_weight": float(run_b.scale_pos_weight),
                "all_thresholds_frozen": True,
            },
            # 由 HoldoutAccessTracker 在真实执行阶段产生，不是事后手工追加的说明。
            "test_access": tracker.to_dict(),
            "pre_freeze_structural_only": {
                "description": (
                    "阈值冻结之前只对 test 做结构性检查（连接键集合、分区互斥、不重不漏）；"
                    "test 标签在装配阶段即被封存，冻结前不可读"
                ),
                "test_labels_read_before_freeze": invariants["test_labels_read_before_freeze"],
                "label_rates_computed_for": invariants["label_rates_computed_for"],
                "candidate_runs_accessed_test": False,
            },
            "lifecycle_requirements": {
                "first_event_is_freeze": (
                    tracker.entries[0]["event"] == "thresholds_frozen" if tracker.entries else False
                ),
                "n_feature_accesses_after_freeze": sum(
                    1
                    for entry in tracker.entries
                    if entry.get("event") == "test_access" and entry.get("what") == "features"
                ),
                "n_label_accesses_after_freeze": sum(
                    1
                    for entry in tracker.entries
                    if entry.get("event") == "test_access" and entry.get("what") == "labels"
                ),
                "report_generation_accessed_test": False,
                "importance_computation_accessed_test": False,
                "bundle_save_accessed_test": False,
            },
        },
        "test_evaluation": lightgbm_test,
        "cw003_baseline_source": cw003_source,
        "comparison": comparison.to_dict(orient="records"),
        "feature_importance": {
            "types": ["gain", "split"],
            "aggregation": "source_feature",
            "n_source_features": int(len(importance_frame)),
            "top_n_in_report": IMPORTANCE_TOP_N,
            "top_source_features": importance_frame.head(IMPORTANCE_TOP_N).to_dict(orient="records"),
            "conservation": run_b.importance_conservation,
            "aggregation_tolerance": {
                "absolute": float(CONSERVATION_ATOL),
                "relative": float(CONSERVATION_RTOL),
            },
            "full_table_written_to": config.relative(config.outputs["feature_importance_csv"]),
            "category_values_exposed": False,
            "forbidden_analyses": list(FORBIDDEN_ANALYSES),
            "interpretation_limits": [
                "gain/split 只描述模型在训练数据上如何使用这些列来降低损失，**不是**因果关系、"
                "业务含义或合规解释。",
                "匿名列（V*/C*/D*/M*/id_*）不得据此被赋予业务语义。",
                "类别列只以源列名出现，不输出任何 one-hot 类别取值或完整类别集合。",
                "两组候选运行的聚合重要性必须一致，否则确定性比较失败且不开放 test。",
            ],
        },
        "inference_bundle": {
            **bundle.describe(),
            "relative_path": config.relative(config.outputs["bundle"]),
        },
        "resources": {
            "peak_memory_mib": peak_memory_mib(),
            "peak_memory_measurement": "进程峰值 RSS（Windows: PeakWorkingSetSize）",
            "total_seconds": round(float(total_seconds), 3),
            "assembly_chunk_size": int(chunk_size),
            "note": "分块大小只影响内存与耗时，不影响任何结论。",
        },
        "timings_seconds": dict(timings),
        "total_seconds": round(float(total_seconds), 3),
        "limitations": _limitations(),
        "environment": environment,
    }
    # 早停的纯事实描述（机器可读副本），与 report.md 中的同一句保持一致。
    report["candidate_runs"]["early_stopping_observation"] = early_stopping_note(report)
    return report


def _excluded_breakdown(decision: FilterDecision) -> dict[str, int]:
    from ai_detector.features.ieee_cis import EXCLUSION_REASONS

    breakdown: dict[str, int] = {code: 0 for code in EXCLUSION_REASONS}
    for item in decision.excluded:
        breakdown[item["reason_code"]] = breakdown.get(item["reason_code"], 0) + 1
    return breakdown


def _limitations() -> list[str]:
    return [
        "IEEE-CIS 是匿名化的公开竞赛数据，不能视为真实银行生产数据；报告不设置任何效果门槛，"
        "也不将结果外推为生产效果。",
        "模型输出是**模型分数**，不是已校准的欺诈概率：scale_pos_weight 会系统性抬高正类分数，"
        "本轮不做概率校准。",
        "三基线指标读取自已验收的 CW-003 report.json，本工作包未重新评估它们的 test；"
        "四模型对比是跨来源汇总，不是同一轮重训。",
        "gain/split 重要性只反映模型在训练数据上的使用方式，不是因果关系、业务含义或合规解释。",
        "匿名列是否包含未知代理或时间邻近泄漏风险未解决；本轮只登记风险，不声称已排除。",
        "参数在第一次真实训练前即冻结，因此没有做任何超参数搜索；LightGBM 未达到更优效果属于预期"
        "范围内的结果，不是流程失败。",
        "相对成本比例是研究假设，不是真实货币损失，也不代表任何机构的风险偏好。",
        "数值列统一按数值处理，未区分「数值编码的类别」与「连续量」；这是建模简化，不是语义断言。",
        "本工作包不做 SHAP、Permutation Importance、PDP、概率校准、服务化、数据库、Docker、CI、"
        "RAG 或 LLM，也不进入 P2/CW-005。",
    ]


def archived_prior_runs(report_dir: Path) -> list[str]:
    """列出报告目录下已归档的历史运行（目录名以 ``_`` 开头）。

    只读取目录名，不读取任何内容，也不接触 test 数据。用于在报告里**如实披露**
    "本次不是任务历史上唯一一次完整运行"。
    """
    if not report_dir.is_dir():
        return []
    return sorted(
        child.name
        for child in report_dir.iterdir()
        if child.is_dir() and child.name.startswith("_")
    )


def test_evaluation_history(config: LightGbmConfig) -> dict[str, Any]:
    """test 评估历史的披露块。

    ``single_shot_test`` 是**单次运行内**的语义：每一次完整运行都恰好评估一次 test，
    但它**不**表示整个任务历史只评估过一次 test。历史归档运行在这里逐条列出。
    """
    archives = archived_prior_runs(config.outputs["report_dir"])
    return {
        "per_complete_run": (
            "本工作包的每一次完整运行都在**该次运行内部**恰好评估一次 test："
            "冻结后只读一次 test 特征、一次 test 标签，随后不再访问。"
        ),
        "single_shot_scope": "single_shot_test 指单次运行内的语义，不代表整个任务历史只评估过一次 test。",
        "complete_runs_in_this_invocation": 1,
        "archived_prior_runs": archives,
        "archived_prior_runs_count": len(archives),
        "archived_prior_runs_location": config.relative(config.outputs["report_dir"]),
        "note": (
            "历史各次完整运行的原因、命令与结果见 WORK_LOG.md；归档目录只保存当时的报告产物，"
            "不被覆盖。任何一次运行都没有用 test 结果反过来调整参数、特征或阈值。"
        ),
    }


def early_stopping_note(report: dict[str, Any]) -> str:
    """对早停结果作一句**纯事实**描述，避免读者把 best_iteration 误读成模型质量。

    只陈述报告里已经存在的数字（最优轮、实际轮数、保留树数、上限与是否触发），
    不做因果判断，也不据此提出任何参数调整建议——参数在训练前已冻结。
    """
    early = report["candidate_runs"]["B"]["early_stopping"]
    return (
        f"早停在 validation 的 {early['metric']} 上于第 {early['best_iteration']} 轮取得最优值 "
        f"{early['best_score']:.10f}，跑满 {early['rounds_run']} 轮（= 最优轮 + stopping_rounds "
        f"{early['stopping_rounds']}）后在 {early['n_estimators_cap']} 轮上限之前停止，"
        f"最终保留 {early['trees_kept']} 棵树。该指标与排序类指标（ROC-AUC / PR-AUC）不是同一个量，"
        "因此“最优轮很小”只说明**该早停指标**很早就停止改善，不构成对模型质量的判断；"
        "参数在第一次真实训练前已冻结，本工作包未据此调整任何参数。"
    )


def _render_markdown(report: dict[str, Any]) -> str:
    gates = report["prerequisite_gates"]
    partitions = report["partitions"]
    model = report["model"]
    early = report["candidate_runs"]["B"]["early_stopping"]
    determinism = report["candidate_runs"]["determinism"]
    frozen = report["thresholds"]["frozen"]
    metrics = report["test_evaluation"]["metrics"]

    lines = [
        f"# CW-004 LightGBM 挑战者报告（{report['work_package']}）",
        "",
        f"- 生成时间（UTC）：{report['generated_at']}",
        f"- 数据集：{report['dataset']['id']}（IEEE-CIS Fraud Detection 公开匿名竞赛数据）",
        f"- 评估分区：{report['evaluation']['evaluation_partition']}，全部阈值冻结后**只评估一次**",
        "- 本报告不设置任何效果门槛，也不代表真实银行生产效果。",
        "",
        "## 前置门禁与来源指纹",
        "",
        "| 项目 | 来源 | SHA-256 | 通过 |",
        "| --- | --- | --- | --- |",
        f"| CW-003 协议任务 | `{gates['prerequisite_task']['relative_path']}` | — | "
        f"{gates['prerequisite_task']['status']} |",
        f"| CW-003 建模配置 | `{gates['cw003_config']['relative_path']}` | "
        f"`{gates['cw003_config']['sha256']}` | {gates['cw003_config']['match']} |",
        f"| CW-003 特征契约 | `{gates['cw003_feature_manifest']['relative_path']}` | "
        f"`{gates['cw003_feature_manifest']['sha256']}` | "
        f"{gates['cw003_feature_manifest']['match']} |",
        f"| CW-003 三基线报告 | `{gates['cw003_report']['relative_path']}` | "
        f"`{gates['cw003_report']['sha256']}` | 只读 |",
        f"| CW-004 本配置 | `{report['config']['relative_path']}` | "
        f"`{report['config']['config_sha256']}` | — |",
        "",
        "| 分区 | 行数 | 标签率 |",
        "| --- | --- | --- |",
    ]
    for name in (PARTITION_TRAIN, PARTITION_VALIDATION, PARTITION_TEST):
        lines.append(
            f"| {name} | {partitions[name]['rows']:,} | {partitions[name]['label_rate']:.4%} |"
        )

    lines.extend(
        [
            "",
            "## 特征契约（复用 CW-003，不重新定义）",
            "",
            f"- 源特征：{report['features']['contract']['kept_numeric']} 数值 + "
            f"{report['features']['contract']['kept_categorical']} 类别 = "
            f"{report['features']['contract']['kept_total']} 列，与已验收 manifest "
            f"**逐项一致**：{report['features']['contract']['matches_cw003_manifest']}。",
            f"- 转换后维度：{report['features']['contract']['expected_transformed_dimension']}"
            f"（{report['features']['contract']['output_format']}），稀疏矩阵全程不稠密化。",
            "- 目标列 `isFraud`、连接键 `TransactionID` 与原始时间列 `TransactionDT` 通过硬校验"
            "排除在源特征、转换后特征与模型输入之外。",
            "",
            "## 冻结的模型参数",
            "",
            "| 参数 | 值 |",
            "| --- | --- |",
        ]
    )
    for key, value in model["params"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            f"- LightGBM 版本：`{model['lightgbm_version']}`；设备 CPU；`n_jobs={model['n_jobs']}`；"
            "**未做任何超参数搜索**。",
            f"- 类别不平衡：`scale_pos_weight = {report['candidate_runs']['B']['scale_pos_weight']:.6f}`"
            "（负例数 / 正例数，只用 train 标签计算）。",
            f"- 早停：metric `{early['metric']}`、stopping_rounds {early['stopping_rounds']}、"
            f"评估集为 validation（**只读 validation**）；`best_iteration = "
            f"{early['best_iteration']}`、实际跑 {early['rounds_run']} 轮、保留 "
            f"{early['trees_kept']} 棵树（上限 {early['n_estimators_cap']}），"
            f"触发早停：{early['early_stopping_fired']}。",
            "- 模型分数（score）**不是**已校准概率；本轮不做概率校准。",
            f"- 早停观察：`{early_stopping_note(report)}`",
            "",
            "## 两阶段确定性与 test 生命周期",
            "",
            f"- 候选 A/B 的分区范围：{report['candidate_runs']['design']['partition_scope']}；"
            f"候选运行期间访问 test：{report['candidate_runs']['design']['test_accessed']}"
            "（必须为 False）。",
            f"- 比较容差：绝对 `{determinism['tolerance']['absolute']}`、相对 "
            f"`{determinism['tolerance']['relative']}`；比较项 "
            f"{len(determinism['compared_items'])} 个，失败项 {determinism['failed_items'] or '无'}。",
            "- test 生命周期门禁：test 特征与标签只能经**接收追踪器**的受控入口读取，"
            "冻结检查与访问记账发生在返回任何数据**之前**；`assembled['test']` 整体取出被拒绝。",
            f"- test 生命周期日志：冻结事件为第一条记录 "
            f"`{report['evaluation']['lifecycle_requirements']['first_event_is_freeze']}`；"
            f"冻结后 features 访问 "
            f"{report['evaluation']['lifecycle_requirements']['n_feature_accesses_after_freeze']} 次、"
            f"labels 访问 "
            f"{report['evaluation']['lifecycle_requirements']['n_label_accesses_after_freeze']} 次。",
            f"- test 评估历史：{report['evaluation']['test_evaluation_history']['per_complete_run']}"
            f"{report['evaluation']['test_evaluation_history']['single_shot_scope']}"
            f"本报告目录下另有 {report['evaluation']['test_evaluation_history']['archived_prior_runs_count']} "
            "个历史归档运行："
            f"{report['evaluation']['test_evaluation_history']['archived_prior_runs']}。",
            "",
            "## 测试集一次性评估（阈值冻结之后）",
            "",
            "| 阈值 | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | 预测正例率 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
            f"| {metrics['threshold']:.2f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
            f"{metrics['false_positive_rate']:.4f} | {metrics['predicted_positive_rate']:.4f} |",
            "",
            f"- 混淆矩阵：TP {metrics['true_positive']:,} / FP {metrics['false_positive']:,} / "
            f"TN {metrics['true_negative']:,} / FN {metrics['false_negative']:,}",
            "",
            "### 相对成本结果（研究用相对单位，非货币）",
            "",
            "| 场景 | FP:FN | 冻结阈值 | 测试总成本 | 每笔平均成本 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, payload in report["test_evaluation"]["cost_scenarios"].items():
        lines.append(
            f"| {name} | {payload['fp_cost']:.0f}:{payload['fn_cost']:.0f} | "
            f"{payload['frozen_threshold']:.2f} | {payload['test_total_cost']:,.0f} | "
            f"{payload['test_average_cost']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 四模型对比（跨来源汇总，不是同一轮重训）",
            "",
            "| 模型 | 来源 | 阈值 | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | 训练耗时(s) | 复杂度 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["comparison"]:
        lines.append(
            f"| {row['model_key']} | {row['source_work_package']} | {row['frozen_threshold']:.2f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['false_positive_rate']:.4f} | "
            f"{row['train_seconds']} | {row['complexity_note']} |"
        )
    lines.extend(
        [
            "",
            f"- CW-003 三基线只从已验收 `{report['cw003_baseline_source']['source_report_relative_path']}` "
            "读取，**未重新评估**它们的 test；来源指纹 "
            f"`{report['cw003_baseline_source']['source_report_sha256']}`。",
            "- 数值差异只作描述，不构成因果解释，也不外推到生产或真实银行。",
            "",
            "## 聚合特征重要性（gain / split，聚合回源特征）",
            "",
            "| 排名 | 源特征 | 类型 | 聚合 gain | 聚合 split | gain 占比 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["feature_importance"]["top_source_features"][:10]:
        lines.append(
            f"| {row['rank']} | `{row['source_feature']}` | {row['kind']} | "
            f"{row['aggregated_gain']:.4f} | {row['aggregated_split']} | {row['gain_share']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- 全部 {report['feature_importance']['n_source_features']} 个源特征写入 "
            f"`{report['feature_importance']['full_table_written_to']}`；完整 Top "
            f"{report['feature_importance']['top_n_in_report']} 在 report.json 中。",
            "- 守恒校验：gain 绝对差 "
            f"{report['feature_importance']['conservation']['gain']['absolute_difference']:.3e}、"
            "split 绝对差 "
            f"{report['feature_importance']['conservation']['split']['absolute_difference']:.3e}。",
            "- 不含任何 one-hot 类别取值、完整类别集合或逐行贡献。",
        ]
    )
    for item in report["feature_importance"]["interpretation_limits"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 推理 bundle",
            "",
            f"- 路径：`{report['inference_bundle']['relative_path']}`；schema "
            f"`{report['inference_bundle']['schema_version']}`；best_iteration "
            f"{report['inference_bundle']['best_iteration']}。",
            f"- 源特征 {report['inference_bundle']['n_source_features']} 个、转换维度 "
            f"{report['inference_bundle']['n_transformed_features']} 个；缺列或多列都会立即失败。",
            f"- 不含原始数据：`contains_raw_data = "
            f"{report['inference_bundle']['contains_raw_data']}`。",
            "",
            "## 资源与耗时",
            "",
            f"- 峰值内存：{report['resources']['peak_memory_mib']} MiB"
            f"（{report['resources']['peak_memory_measurement']}）",
            f"- 总耗时：{report['total_seconds']} s",
            "",
            "| 阶段 | 秒 |",
            "| --- | --- |",
        ]
    )
    for key, value in report["timings_seconds"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## 风险与限制", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 环境版本", "", "| 组件 | 版本 |", "| --- | --- |"])
    for key, value in report["environment"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "> 本报告基于公开匿名竞赛数据，**不是**生产风控系统，也不代表任何真实银行的线上效果。"
            "报告未设置任何「效果必须达到某值」的门槛。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(
    config: LightGbmConfig,
    report: dict[str, Any],
    run_b: CandidateRun,
    importance_frame: pd.DataFrame,
    comparison: pd.DataFrame,
    bundle: LightGbmBundle,
) -> dict[str, Any]:
    outputs = config.outputs
    outputs["report_dir"].mkdir(parents=True, exist_ok=True)
    outputs["model_dir"].mkdir(parents=True, exist_ok=True)

    outputs["report_json"].write_text(json_dumps(report), encoding="utf-8")
    outputs["report_markdown"].write_text(_render_markdown(report), encoding="utf-8")

    curve = curve_to_frame(run_b.frozen["_curve"])
    curve.insert(0, "model_key", LGBM_KEY)
    curve.insert(1, "partition", PARTITION_VALIDATION)
    curve.to_csv(outputs["threshold_curve_csv"], index=False)

    comparison.to_csv(outputs["comparison_csv"], index=False)
    importance_frame.to_csv(outputs["feature_importance_csv"], index=False)

    save_bundle(bundle, outputs["bundle"])

    return {
        "report_json": outputs["report_json"],
        "report_markdown": outputs["report_markdown"],
        "threshold_curve_csv": outputs["threshold_curve_csv"],
        "comparison_csv": outputs["comparison_csv"],
        "feature_importance_csv": outputs["feature_importance_csv"],
        "bundle_path": outputs["bundle"],
    }
