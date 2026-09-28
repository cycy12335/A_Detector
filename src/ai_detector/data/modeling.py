"""CW-003 建模数据装配：数据门禁、按需读取、左连接、复用 CW-002 切分。

## 纪律

- **分区只用 CW-002 的行级清单**（``split_manifest_rows.csv``）。本模块不重新推导边界、
  不做随机切分、不按行号切分。清单与原始表的 ``TransactionID`` 必须一一对应。
- **只读取配置声明的列**。433 列全表不会被加载；数值矩阵以 ``float32`` 装配。
- **训练期统计只累计 train 分区**，用于决定列的去留；validation/test 只应用已冻结的决策。

## 内存策略（本机可用内存约 4 GiB，实测峰值记入 WORK_LOG）

1. 第一遍：只读允许清单列，按分块累计 **train 分区**的缺失计数与类别取值集合，
   只保留聚合量，不驻留数据。
2. 决定保留列之后，第二遍：再按分块读取**保留列**，按分区追加到各自的列表，
   最后按分区各做一次 ``concat``。峰值约为"单次全量装配"的一半。
3. 数值列统一转 ``float32``；类别列保持字符串，交由稀疏 one-hot 处理。

## 派生字段

``identity_present`` 表示该交易的 ``TransactionID`` 是否命中 identity 表。它可能携带连接结构
或采集差异信息，属于**潜在代理变量**：本模块在配置登记时同时登记该风险，报告中一并列出。
"""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

from ai_detector.config import PROJECT_ROOT, ConfigError, resolve_path
from ai_detector.data.fingerprint import report_path
from ai_detector.data.temporal_split import PARTITIONS

if TYPE_CHECKING:  # 仅用于类型标注：features 层依赖本模块，运行时导入会形成环。
    from ai_detector.features.ieee_cis import FilterDecision

WORK_PACKAGE = "CW-003"

#: 装配时的分块大小（行）。只影响内存与耗时，不影响任何结论。
ASSEMBLY_CHUNK_SIZE = 50_000

PARTITION_TRAIN = "train"
PARTITION_VALIDATION = "validation"
PARTITION_TEST = "test"


class DataGateError(RuntimeError):
    """开工数据门禁不通过。出现即应阻塞，不得自行替换文件或重算切分。"""


class ModelingContractError(ValueError):
    """装配结果违反数据契约。"""


@dataclass(frozen=True)
class ModelingConfig:
    """``configs/ieee_cis_baseline.yaml`` 的类型化表示。"""

    work_package: str
    seed: int
    dataset_id: str
    expected_fingerprints: dict[str, str]
    split_algorithm_version: str
    expected_split_sizes: dict[str, int]
    expected_boundaries: dict[str, float]
    expected_total_rows: int
    paths: dict[str, Path]
    outputs: dict[str, Path]
    target: str
    join_key: str
    split_time: str
    forbidden_features: tuple[str, ...]
    derived: dict[str, Any]
    train_only_filters: dict[str, Any]
    preprocessing: dict[str, Any]
    allowlist: dict[str, list[dict[str, str]]]
    reason_texts: dict[str, str]
    rule_baseline: dict[str, Any]
    models: dict[str, Any]
    thresholds: dict[str, Any]
    limits: dict[str, Any]
    config_path: Path
    config_sha256: str = ""

    # --- 路径 -----------------------------------------------------------

    def is_within_project_root(self, path: str | Path) -> bool:
        try:
            Path(path).resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            return False
        return True

    def relative(self, path: str | Path) -> str:
        """报告用路径：根目录内给相对路径，否则退化为仅文件名（绝不写绝对路径）。"""
        return report_path(path, PROJECT_ROOT)[0]

    # --- 便捷访问 -------------------------------------------------------

    def allowlist_for(self, source_file: str) -> list[dict[str, str]]:
        return self.allowlist[source_file]

    def allowlist_names(self) -> list[str]:
        return [entry["name"] for entries in self.allowlist.values() for entry in entries]

    def expected_columns_for(self, source_file: str) -> list[str]:
        """某个来源文件的允许清单列名（不含连接键）。"""
        return [entry["name"] for entry in self.allowlist_for(source_file)]

    def rule_fields(self) -> list[str]:
        return [str(item["name"]) for item in self.rule_baseline["fields"]]

    def threshold_grid(self) -> tuple[float, ...]:
        """把声明式网格确定性展开为显式阈值序列。"""
        spec = self.thresholds["grid"]
        start = float(spec["start"])
        stop = float(spec["stop"])
        step = float(spec["step"])
        if step <= 0:
            raise ConfigError("thresholds.grid.step 必须为正数")
        count = int(round((stop - start) / step)) + 1
        grid = tuple(round(start + index * step, 10) for index in range(count))
        if not grid or any(not 0.0 < value < 1.0 for value in grid):
            raise ConfigError("展开后的阈值网格必须全部位于开区间 (0, 1)")
        return grid

    def cost_scenarios(self) -> list[dict[str, Any]]:
        return list(self.thresholds["cost_scenarios"])

    # --- 加载 -----------------------------------------------------------

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, config_path: Path, config_sha256: str = "") -> ModelingConfig:
        required = (
            "work_package",
            "seed",
            "dataset",
            "paths",
            "columns",
            "forbidden_features",
            "train_only_filters",
            "preprocessing",
            "feature_allowlist",
            "rule_baseline",
            "models",
            "thresholds",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ConfigError("CW-003 配置缺少字段：" + ", ".join(missing))

        columns = raw["columns"]
        for key, expected in (("target", "isFraud"), ("join_key", "TransactionID"), ("split_time", "TransactionDT")):
            if str(columns[key]) != expected:
                raise ConfigError(
                    f"columns.{key} 必须为 {expected}（CW-002 已固定的数据契约），实际为 {columns[key]}"
                )

        forbidden = tuple(str(value) for value in raw["forbidden_features"])
        for key in ("target", "join_key", "split_time"):
            if str(columns[key]) not in forbidden:
                raise ConfigError(
                    f"columns.{key}={columns[key]} 未列入 forbidden_features；"
                    "禁止字段清单必须完整覆盖目标列、连接键与时间列"
                )

        # 允许清单不得包含任何禁止字段——这是配置层的硬校验。
        allowlist = {
            source: list(payload["columns"])
            for source, payload in raw["feature_allowlist"]["sources"].items()
        }
        offenders = sorted(
            {entry["name"] for entries in allowlist.values() for entry in entries} & set(forbidden)
        )
        if offenders:
            raise ConfigError(f"特征允许清单包含禁止字段：{offenders}")

        # 每个来源文件的声明计数必须与清单一致。
        for source, payload in raw["feature_allowlist"]["sources"].items():
            declared = int(payload["declared_total"])
            if declared != len(allowlist[source]):
                raise ConfigError(
                    f"{source} 的 declared_total={declared} 与实际列数 {len(allowlist[source])} 不符"
                )
            declared_numeric = int(payload["declared_numeric"])
            actual_numeric = sum(1 for e in allowlist[source] if e["kind"] == "numeric")
            if declared_numeric != actual_numeric:
                raise ConfigError(
                    f"{source} 的 declared_numeric={declared_numeric} 与实际 {actual_numeric} 不符"
                )

        paths_raw = raw["paths"]
        outputs_raw = paths_raw["outputs"]
        return cls(
            work_package=str(raw["work_package"]),
            seed=int(raw["seed"]),
            dataset_id=str(raw["dataset"]["id"]),
            expected_fingerprints={
                str(k): str(v) for k, v in raw["dataset"]["expected_fingerprints"].items()
            },
            split_algorithm_version=str(raw["dataset"]["split_algorithm_version"]),
            expected_split_sizes={
                str(k): int(v) for k, v in raw["dataset"]["expected_split_sizes"].items()
            },
            expected_boundaries={
                str(k): float(v) for k, v in raw["dataset"]["expected_boundaries"].items()
            },
            expected_total_rows=int(raw["dataset"]["expected_total_rows"]),
            paths={
                key: resolve_path(value)
                for key, value in paths_raw.items()
                if key != "outputs"
            },
            outputs={key: resolve_path(value) for key, value in outputs_raw.items()},
            target=str(columns["target"]),
            join_key=str(columns["join_key"]),
            split_time=str(columns["split_time"]),
            forbidden_features=forbidden,
            derived=dict(raw.get("derived", {})),
            train_only_filters=dict(raw["train_only_filters"]),
            preprocessing=dict(raw["preprocessing"]),
            allowlist=allowlist,
            reason_texts={
                str(k): str(v) for k, v in raw["feature_allowlist"]["reason_texts"].items()
            },
            rule_baseline=dict(raw["rule_baseline"]),
            models=dict(raw["models"]),
            thresholds=dict(raw["thresholds"]),
            limits=dict(raw.get("limits", {})),
            config_path=config_path,
            config_sha256=config_sha256,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelingConfig:
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
        )


# --- 数据门禁 -----------------------------------------------------------


def sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data_gate(config: ModelingConfig) -> dict[str, Any]:
    """开工数据门禁：文件、指纹、CW-002 清单与审计产物必须全部一致。

    任一条件不满足即抛出 :class:`DataGateError`，调用方必须把任务置为阻塞并停止，
    不得自行下载、替换文件、重算边界或用 fixture 冒充真实实验。

    Returns:
        门禁证据（指纹、行数、边界、分区规模），供报告记录。
    """
    failures: list[str] = []

    raw_dir = config.paths["transaction"].parent
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, expected in config.expected_fingerprints.items():
        path = raw_dir / name
        if not path.is_file():
            failures.append(f"缺少原始文件：{name}")
            continue
        actual = sha256_of_file(path)
        fingerprints[name] = {
            "relative_path": config.relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
            "match": actual == expected,
        }
        if actual != expected:
            failures.append(f"{name} 的 SHA-256 与 CW-002 记录不符：{actual} != {expected}")

    manifest_path = config.paths["split_manifest_rows"]
    manifest_evidence: dict[str, Any] = {}
    if not manifest_path.is_file():
        failures.append("缺少 CW-002 行级切分清单 split_manifest_rows.csv")
    else:
        rows = pd.read_csv(manifest_path)
        sizes = {str(k): int(v) for k, v in rows["partition"].value_counts().items()}
        manifest_evidence = {
            "relative_path": config.relative(manifest_path),
            "n_rows": int(len(rows)),
            "unique_join_keys": bool(rows[config.join_key].is_unique),
            "sizes": sizes,
        }
        if len(rows) != config.expected_total_rows:
            failures.append(f"清单行数 {len(rows)} != {config.expected_total_rows}")
        if not rows[config.join_key].is_unique:
            failures.append("清单中的连接键存在重复")
        for part, expected in config.expected_split_sizes.items():
            if sizes.get(part) != expected:
                failures.append(f"分区 {part} 规模 {sizes.get(part)} != {expected}")

    split_json_path = config.paths["cw002_split_manifest_json"]
    if not split_json_path.is_file():
        failures.append("缺少 CW-002 split_manifest.json")
    else:
        payload = yaml.safe_load(split_json_path.read_text(encoding="utf-8"))
        if str(payload.get("algorithm_version")) != config.split_algorithm_version:
            failures.append(
                f"切分算法版本 {payload.get('algorithm_version')} != {config.split_algorithm_version}"
            )
        for key, expected in config.expected_boundaries.items():
            actual = float(payload.get("boundaries", {}).get(key, float("nan")))
            if actual != expected:
                failures.append(f"切分边界 {key}={actual} != {expected}")

    audit_path = config.paths["cw002_audit_json"]
    if not audit_path.is_file():
        failures.append("缺少 CW-002 audit.json")
    else:
        payload = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
        recorded = {name: info["sha256"] for name, info in payload.get("files", {}).items()}
        for name, expected in config.expected_fingerprints.items():
            if recorded.get(name) != expected:
                failures.append(f"CW-002 审计报告记录的 {name} 指纹与门禁期望不符")

    if failures:
        raise DataGateError(
            "CW-003 开工数据门禁未通过，任务应置为 blocked；不得替换文件或重算切分。\n"
            + "\n".join(f"  - {item}" for item in failures)
        )

    return {
        "fingerprints": fingerprints,
        "split_manifest": manifest_evidence,
        "boundaries": dict(config.expected_boundaries),
        "split_algorithm_version": config.split_algorithm_version,
    }


# --- 分区与读取 ---------------------------------------------------------


def load_partition_lookup(config: ModelingConfig) -> pd.Series:
    """从 CW-002 行级清单加载 ``TransactionID -> partition`` 映射。

    Raises:
        DataGateError: 清单缺失、分区名非法或连接键重复。
    """
    path = config.paths["split_manifest_rows"]
    if not path.is_file():
        raise DataGateError(f"缺少 CW-002 行级切分清单：{config.relative(path)}")
    rows = pd.read_csv(path, usecols=[config.join_key, "partition"])
    if rows[config.join_key].duplicated().any():
        raise DataGateError("CW-002 清单中的连接键存在重复，无法作为唯一分区来源")
    unknown = set(rows["partition"].unique()) - set(PARTITIONS)
    if unknown:
        raise DataGateError(f"CW-002 清单出现未知分区名：{sorted(unknown)}")
    return rows.set_index(config.join_key)["partition"]


def iter_required_columns(
    config: ModelingConfig,
    *,
    columns_by_file: dict[str, list[str]],
    chunk_size: int = ASSEMBLY_CHUNK_SIZE,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """按来源文件分块读取所需列。只读取声明列，不加载全表。"""
    for source_key, source_file in (("transaction", "train_transaction.csv"), ("identity", "train_identity.csv")):
        path = config.paths[source_key]
        columns = list(columns_by_file.get(source_file, []))
        if not columns:
            continue
        wanted = columns if config.join_key in columns else [config.join_key, *columns]
        reader = pd.read_csv(path, usecols=wanted, chunksize=chunk_size, low_memory=True)
        for chunk in reader:
            yield source_file, chunk


def load_identity_frame(config: ModelingConfig) -> pd.DataFrame:
    """读取 identity 的允许清单列，并校验连接键唯一（防止一对多放大交易行数）。

    Raises:
        ModelingContractError: identity 缺少连接键或存在重复键。
    """
    columns = config.expected_columns_for("train_identity.csv")
    path = config.paths["identity"]
    frame = pd.read_csv(path, usecols=[config.join_key, *columns])
    if config.join_key not in frame.columns:
        raise ModelingContractError("train_identity.csv 缺少连接键")
    if frame[config.join_key].isna().any():
        raise ModelingContractError("train_identity.csv 的连接键存在空值")
    if frame[config.join_key].duplicated().any():
        raise ModelingContractError(
            "train_identity.csv 的连接键存在重复：左连接会放大交易行数，必须先拦截"
        )
    return frame


def align_chunk_with_lookup(
    config: ModelingConfig, lookup: pd.Series, chunk: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """给分块附加分区列；出现清单之外的连接键即报错。"""
    partitions = chunk[config.join_key].map(lookup)
    if partitions.isna().any():
        missing = int(partitions.isna().sum())
        sample = chunk.loc[partitions.isna(), config.join_key].head(3).tolist()
        raise ModelingContractError(
            f"有 {missing} 条交易的连接键不在 CW-002 清单中，示例：{sample}；"
            "分区归属必须完全来自清单，不得自行推导"
        )
    return chunk, partitions.to_numpy(dtype=object)


def forbidden_columns_present(columns: list[str], forbidden: tuple[str, ...]) -> list[str]:
    """返回 ``columns`` 中混入的禁止字段（按被禁顺序）。"""
    return [name for name in forbidden if name in columns]


# --- 装配 ---------------------------------------------------------------


class HoldoutLeakageError(RuntimeError):
    """在阈值冻结之前尝试读取 test 分区的标签或特征内容。"""


@dataclass
class HoldoutAccessTracker:
    """记录并约束 test 分区的访问。

    设计目标是让"test 只评估一次"成为**可执行的不变量**，而不是一句声明：

    - 装配阶段就把 test 的标签从 test 分区中剥离（见 :class:`AssembledData`），
      因此冻结前的代码路径物理上拿不到 test 标签。
    - 取用 test 标签或特征必须经过 :meth:`read`；未冻结时直接抛
      :class:`HoldoutLeakageError`。
    - 访问日志由 :meth:`read` 在真实的执行阶段产生，而不是事后手工追加说明。

    读取训练/验证分区不经过本追踪器，也不产生日志。
    """

    entries: list[dict[str, Any]] = field(default_factory=list)
    frozen: bool = False
    frozen_reason: str | None = None

    def freeze(self, *, reason: str) -> None:
        """标记所有阈值已冻结；此后才允许访问 test。"""
        if self.frozen:
            raise HoldoutLeakageError("test 访问追踪器已被冻结过，重复冻结说明流程有误")
        self.frozen = True
        self.frozen_reason = reason
        self.entries.append(
            {
                "sequence": len(self.entries) + 1,
                "event": "thresholds_frozen",
                "reason": reason,
                "test_access_allowed_after_this_point": True,
            }
        )

    def read(self, *, what: str, purpose: str, phase: str) -> None:
        """登记一次 test 访问；未冻结时抛错。

        Raises:
            HoldoutLeakageError: 在阈值冻结之前访问 test。
        """
        if not self.frozen:
            raise HoldoutLeakageError(
                f"在阈值冻结之前尝试访问 test 分区的{what}（阶段：{phase}）；"
                "test 只在所有阈值冻结后评估一次，冻结前只允许结构性检查（如连接键集合）"
            )
        self.entries.append(
            {
                "sequence": len(self.entries) + 1,
                "event": "test_access",
                "phase": phase,
                "what": what,
                "purpose": purpose,
                "after_freeze": True,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen": self.frozen,
            "frozen_reason": self.frozen_reason,
            "entries": list(self.entries),
            "n_test_accesses": sum(1 for e in self.entries if e.get("event") == "test_access"),
        }


@dataclass
class AssembledData:
    """装配结果：三个分区的特征 + **封存的 test 标签**。

    ## test 分区只有受控入口，没有"顺手取一下"的路径

    对 ``test`` 分区：

    - ``target()`` 必须经**已冻结**的追踪器；test 标签在装配阶段就已被剥离封存，
      冻结前的代码路径物理上拿不到。
    - ``features()`` 必须经**已冻结**的追踪器；检查与记账发生**在返回任何数据之前**，
      因此"冻结前读不到 test 特征"与"读到了就一定有日志"同时成立。
    - ``__getitem__``（``assembled["test"]``）**直接拒绝**，不存在整体取出 test 的入口。
    - 只读元信息走 :meth:`structural_ids` / :meth:`structural_columns` /
      :meth:`structural_row_count`，它们只暴露连接键、列名与行数，不暴露任何取值。

    存储字典 ``_frames`` 是内部容器（装配簿记与内存释放用），不是读取入口。
    """

    _frames: dict[str, pd.DataFrame]
    target_column: str
    join_key_column: str
    _sealed_test_target: np.ndarray | None = None

    # --- 内部存储（不作为读取入口）--------------------------------------

    def release(self, partition: str) -> None:
        """释放某分区的帧（内存优化）。释放 train/validation 不影响任何纪律。"""
        self._frames.pop(partition, None)

    # --- 领取入口 -------------------------------------------------------

    def __getitem__(self, partition: str) -> pd.DataFrame:
        """取分区整帧。**test 分区不允许整体取出**，必须走受控入口。

        Raises:
            HoldoutLeakageError: 对 test 分区调用。
        """
        if partition == PARTITION_TEST:
            raise HoldoutLeakageError(
                "test 分区不得经 assembled['test'] 整体取出：特征请用 "
                "features('test', columns, tracker=...)，标签请用 target('test', tracker=...)，"
                "只读元信息请用 structural_ids / structural_columns / structural_row_count。"
            )
        return self._frames[partition]

    def features(
        self,
        partition: str,
        columns: list[str],
        *,
        tracker: HoldoutAccessTracker | None = None,
    ) -> pd.DataFrame:
        """取某分区的特征子集（不含目标列）。

        对 test 分区，``tracker`` **必须**提供且必须**已冻结**：
        :meth:`HoldoutAccessTracker.read` 在切片之前执行，因此冻结前访问会抛
        :class:`HoldoutLeakageError` **且不返回任何数据**；成功读取时由这里（而不是调用方）
        自动产生访问日志，"读到了"与"有日志"不可能分开。

        Raises:
            HoldoutLeakageError: test 分区缺少 tracker，或 tracker 尚未冻结。
            ModelingContractError: 分区缺少请求的列。
        """
        frame = self._frames[partition]
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ModelingContractError(f"分区 {partition} 缺少特征列：{missing[:5]}")
        if partition == PARTITION_TEST:
            if tracker is None:
                raise HoldoutLeakageError(
                    "读取 test 分区特征必须提供生命周期追踪器："
                    "features('test', columns, tracker=tracker)。"
                    "没有追踪器就没有冻结检查与访问记账，因此该入口拒绝返回数据。"
                )
            # 先检查 + 记账，再切片：数据在门禁通过之前不会被物化。
            tracker.read(
                what="features", purpose="single_shot_evaluation", phase="post_freeze_evaluation"
            )
        return frame.loc[:, columns]

    def target(self, partition: str, *, tracker: HoldoutAccessTracker) -> np.ndarray:
        """取某分区的标签。

        训练/验证分区直接返回；**test 分区必须经由已冻结的追踪器**，否则抛
        :class:`HoldoutLeakageError`。
        """
        if partition != PARTITION_TEST:
            return self._frames[partition][self.target_column].to_numpy(dtype=int)
        tracker.read(
            what="labels", purpose="single_shot_evaluation", phase="post_freeze_evaluation"
        )
        if self._sealed_test_target is None:
            raise ModelingContractError("test 标签未被封存，装配流程有误")
        return self._sealed_test_target

    def label_rate(self, partition: str, *, tracker: HoldoutAccessTracker) -> float:
        """标签率。test 同样需要已冻结的追踪器。"""
        values = self.target(partition, tracker=tracker)
        return float(np.mean(values)) if values.size else 0.0

    # --- 只读元信息（冻结前允许，不含任何取值）--------------------------

    def structural_ids(self, partition: str) -> set:
        """某分区的连接键集合。这是冻结前允许的 test 访问（纯结构性）。"""
        return set(self._frames[partition][self.join_key_column])

    def structural_columns(self, partition: str) -> list[str]:
        """某分区的列名。只暴露 schema，不暴露任何取值。"""
        return list(self._frames[partition].columns)

    def structural_row_count(self, partition: str) -> int:
        """某分区的行数。纯结构性，冻结前允许。"""
        return int(len(self._frames[partition]))


def split_kept_by_source(
    config: ModelingConfig, decision: "FilterDecision"
) -> dict[str, list[str]]:
    """把保留列按来源文件拆开，用于分别读取 transaction 与 identity。"""
    by_source: dict[str, list[str]] = {name: [] for name in config.allowlist}
    for name in decision.kept:
        by_source[decision.stats[name].source_file].append(name)
    return by_source


def assemble_partitions(
    config: ModelingConfig,
    lookup: pd.Series,
    decision: "FilterDecision",
    *,
    chunk_size: int = ASSEMBLY_CHUNK_SIZE,
) -> AssembledData:
    """按分区装配建模数据：transaction 为主表，identity 仅左连接。

    返回 :class:`AssembledData`；其中 test 分区的目标列被**剥离并封存**，只能通过
    已冻结的 :class:`HoldoutAccessTracker` 取出。

    保证：
    - 只读取保留列 + 连接键 + 目标列，不加载全表。
    - identity 以**唯一键**左连接，不删交易、不放大行数。
    - 分区归属完全来自 CW-002 清单。

    Raises:
        ModelingContractError: 行数不变量或禁止字段校验失败。
    """
    offenders = forbidden_columns_present(decision.kept, config.forbidden_features)
    if offenders:
        raise ModelingContractError(f"特征清单中出现禁止字段：{offenders}")

    by_source = split_kept_by_source(config, decision)
    identity_columns = by_source.get("train_identity.csv", [])
    transaction_columns = by_source.get("train_transaction.csv", [])
    identity_present_enabled = bool(
        config.derived.get("identity_present", {}).get("enabled")
    )

    identity_frame = load_identity_frame(config)
    keep_columns = [config.join_key, *identity_columns]
    identity_frame = identity_frame.loc[:, list(dict.fromkeys(keep_columns))]

    numeric_columns = set(decision.numeric)
    buckets: dict[str, list[pd.DataFrame]] = {
        name: [] for name in (PARTITION_TRAIN, PARTITION_VALIDATION, PARTITION_TEST)
    }
    counts = {name: 0 for name in buckets}

    usecols = [config.join_key, config.target, *transaction_columns]
    identity_keys = identity_frame[config.join_key]
    for raw_chunk in pd.read_csv(
        config.paths["transaction"], usecols=usecols, chunksize=chunk_size, low_memory=True
    ):
        chunk, partitions = align_chunk_with_lookup(config, lookup, raw_chunk)

        if identity_columns:
            chunk = chunk.merge(identity_frame, on=config.join_key, how="left")
            if len(chunk) != len(raw_chunk):
                raise ModelingContractError(
                    "identity 左连接改变了交易行数："
                    f"{len(raw_chunk)} -> {len(chunk)}，说明 identity 连接键不唯一"
                )

        # 派生字段与类型转换都按"整块"完成：逐列赋值会把宽表打散成上千个块，
        # 既严重拖慢后续操作，也显著抬高内存（pandas 会给出 PerformanceWarning）。
        if identity_present_enabled:
            chunk = pd.concat(
                [
                    chunk,
                    chunk[config.join_key]
                    .isin(identity_keys)
                    .astype("int8")
                    .rename("identity_present"),
                ],
                axis=1,
            )

        numeric_present = [name for name in numeric_columns if name in chunk.columns]
        if numeric_present:
            chunk[numeric_present] = (
                chunk[numeric_present].apply(pd.to_numeric, errors="coerce").astype("float32")
            )

        for partition in (PARTITION_TRAIN, PARTITION_VALIDATION, PARTITION_TEST):
            selected = chunk.loc[partitions == partition]
            if selected.empty:
                continue
            buckets[partition].append(selected.copy())
            counts[partition] += len(selected)
        del chunk, raw_chunk

    frames = {
        name: (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=[config.join_key, config.target])
        )
        for name, parts in buckets.items()
    }
    for parts in buckets.values():
        parts.clear()
    del buckets
    gc.collect()

    total = sum(counts.values())
    if total != config.expected_total_rows:
        raise ModelingContractError(
            f"装配后总行数 {total} != 期望 {config.expected_total_rows}（不重不漏校验失败）"
        )
    for name, expected in config.expected_split_sizes.items():
        if counts[name] != expected:
            raise ModelingContractError(f"分区 {name} 行数 {counts[name]} != 期望 {expected}")

    # 把 test 标签从 test 分区中剥离并封存：冻结前的代码路径因此**物理上拿不到** test 标签，
    # "test 只在所有阈值冻结后评估一次"成为结构性保证，而不是靠约定或事后手工记日志。
    test_frame = frames[PARTITION_TEST]
    if config.target not in test_frame.columns:
        raise ModelingContractError("test 分区缺少目标列，无法封存")
    sealed_test_target = test_frame[config.target].to_numpy(dtype=np.int64)
    frames[PARTITION_TEST] = test_frame.drop(columns=[config.target])

    return AssembledData(
        _frames=frames,
        target_column=config.target,
        join_key_column=config.join_key,
        _sealed_test_target=sealed_test_target,
    )


def assert_partition_invariants(
    config: ModelingConfig,
    assembled: AssembledData,
    *,
    label_partitions: tuple[str, ...] = (PARTITION_TRAIN, PARTITION_VALIDATION),
) -> dict[str, Any]:
    """校验分区不变量。

    **本函数在阈值冻结之前调用，因此绝不读取 test 标签。**

    - 对三个分区只做结构性检查：连接键集合、两两不相交、合计不重不漏；
    - 标签取值合法性与标签率只对 ``label_partitions``（默认 train、validation）计算；
      test 的标签合法性与标签率由冻结后的 :func:`validate_test_partition` 负责。

    Raises:
        ModelingContractError: 任一不变量被破坏，或误把 test 放进 ``label_partitions``。
    """
    partitions = (PARTITION_TRAIN, PARTITION_VALIDATION, PARTITION_TEST)
    id_sets = {name: assembled.structural_ids(name) for name in partitions}
    for left in range(len(partitions)):
        for right in range(left + 1, len(partitions)):
            shared = id_sets[partitions[left]] & id_sets[partitions[right]]
            if shared:
                raise ModelingContractError(
                    f"分区 {partitions[left]} 与 {partitions[right]} 的 ID 存在交集，"
                    f"示例：{sorted(shared)[:3]}"
                )

    total = sum(len(values) for values in id_sets.values())
    if total != config.expected_total_rows:
        raise ModelingContractError(
            f"三个分区的唯一 ID 合计 {total} != {config.expected_total_rows}（不重不漏校验失败）"
        )

    label_rates: dict[str, float] = {}
    for name in label_partitions:
        if name == PARTITION_TEST:
            raise ModelingContractError(
                "assert_partition_invariants 不得在冻结前读取 test 标签；"
                "test 的标签校验请调用 validate_test_partition()"
            )
        labels = assembled[name][config.target]
        values = set(pd.unique(labels.dropna()))
        if not values <= {0, 1}:
            raise ModelingContractError(f"分区 {name} 的目标列取值非法：{sorted(values)}")
        label_rates[name] = float(labels.mean())

    if not label_rates:
        raise ModelingContractError("label_partitions 不能为空")

    return {
        "train_rows": assembled.structural_row_count(PARTITION_TRAIN),
        "validation_rows": assembled.structural_row_count(PARTITION_VALIDATION),
        "test_rows": assembled.structural_row_count(PARTITION_TEST),
        "ids_disjoint": True,
        "ids_complete": True,
        "label_rates": label_rates,
        "label_rates_computed_for": list(label_partitions),
        "test_labels_read_before_freeze": False,
    }


def validate_test_partition(
    config: ModelingConfig,
    assembled: AssembledData,
    tracker: HoldoutAccessTracker,
) -> dict[str, Any]:
    """**冻结之后**才允许调用：校验 test 标签合法性并计算标签率。

    这是 test 标签在整个流程中第一次被读取，且必须经过追踪器——未冻结即抛
    :class:`HoldoutLeakageError`。因此"冻结前不读 test 标签"是可执行的不变量。

    Raises:
        HoldoutLeakageError: 在阈值冻结之前调用。
        ModelingContractError: test 标签非法或长度不符。
    """
    labels = assembled.target(PARTITION_TEST, tracker=tracker)
    values = set(np.unique(labels))
    if not values <= {0, 1}:
        raise ModelingContractError(f"test 分区目标列取值非法：{sorted(values)}")
    expected = config.expected_split_sizes[PARTITION_TEST]
    if labels.size != expected:
        raise ModelingContractError(
            f"test 分区标签长度 {labels.size} != 期望 {expected}"
        )
    return {
        "rows": int(labels.size),
        "label_rate": float(labels.mean()),
        "positive_count": int(labels.sum()),
        "read_after_freeze": True,
    }
