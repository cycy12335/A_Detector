"""IEEE-CIS 数据契约、文件发现、Schema 与连接审计。

## 内存策略

transaction 文件约 394 列、约 590k 行，一次性读入会占用大量内存。因此审计分两遍读取：

- **第一遍只读 3 个结构性列**（``TransactionID``、``TransactionDT``、``isFraud``），
  用于 Schema 校验、唯一性校验、连接审计和切分清单。这一遍的内存占用与列数无关。
- **第二遍读全部列**，分块喂给 :class:`~ai_detector.data.profiling.ChunkedProfiler`。

两遍都按 ``chunk_size`` 分块，且都只做聚合，不保留原始行。

## 为什么不用整表读入再校验

"先读进来再检查"在列数多时会直接把内存打满，而且失败时要等很久。分块 + 快速失败能在
第一时间给出可读错误，也避免用户在内存不足时反复试错。

## 边界

- 原始数据只读，绝不修改、不复制、不写入仓库范围。
- 报告只记录相对路径。
- 不读取、不记录任何凭据。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ai_detector.config import PROJECT_ROOT, ConfigError, resolve_path
from ai_detector.data.fingerprint import FileFingerprint, fingerprint_file, report_path
from ai_detector.data.profiling import ProfilingConfig

#: 结构性列：第一遍读取只需要这三列。
STRUCTURAL_COLUMNS: tuple[str, str, str] = ("TransactionID", "TransactionDT", "isFraud")


class MissingDataError(RuntimeError):
    """必需的原始数据文件缺失。"""


class DataContractError(ValueError):
    """数据违反 Schema 或连接契约。"""


@dataclass(frozen=True)
class DatasetMeta:
    """数据集来源与条款说明。"""

    dataset_id: str
    display_name: str
    version_note: str
    provider: str
    organizer: str
    official_data_url: str
    official_rules_url: str
    license_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "display_name": self.display_name,
            "version_note": self.version_note,
            "provider": self.provider,
            "organizer": self.organizer,
            "official_data_url": self.official_data_url,
            "official_rules_url": self.official_rules_url,
            "license_note": self.license_note,
        }


@dataclass(frozen=True)
class IeeeCisConfig:
    """``configs/ieee_cis.yaml`` 的类型化表示。"""

    meta: DatasetMeta
    raw_dir: Path
    transaction_path: Path
    identity_path: Path
    join_key: str
    target: str
    split_time: str
    train_ratio: float
    validation_ratio: float
    split_strategy: str
    forbidden_features: dict[str, str]
    anonymous_prefixes: tuple[str, ...]
    profiling: ProfilingConfig
    outputs: dict[str, Path] = field(default_factory=dict)

    # --- 路径 -----------------------------------------------------------

    def is_within_project_root(self, path: str | Path) -> bool:
        """路径是否位于项目根目录之内。"""
        try:
            Path(path).resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            return False
        return True

    def relative(self, path: str | Path) -> str:
        """返回写入报告用的路径。

        策略由 :func:`ai_detector.data.fingerprint.report_path` 单点实现：位于项目根目录
        之内时给出相对路径，否则退化为仅文件名（绝不写绝对路径）。调用方可用
        :meth:`is_within_project_root` 在报告中显式标注退化情况。
        """
        return report_path(path, PROJECT_ROOT)[0]

    @property
    def raw_dir_relative(self) -> str:
        return self.relative(self.raw_dir)

    def required_files(self) -> tuple[tuple[str, Path], ...]:
        """返回 ``(逻辑名, 路径)`` 序列，供缺失检查与报错信息使用。"""
        return (
            ("train_transaction.csv", self.transaction_path),
            ("train_identity.csv", self.identity_path),
        )

    # --- 加载 -----------------------------------------------------------

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> IeeeCisConfig:
        """从已解析的 YAML 映射构造配置，缺字段时立即报错。"""
        required = (
            "dataset",
            "files",
            "columns",
            "split",
            "forbidden_features",
            "profiling",
            "output",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ConfigError("IEEE-CIS 配置缺少字段：" + ", ".join(missing))

        dataset = raw["dataset"]
        columns = raw["columns"]
        files = raw["files"]
        split = raw["split"]
        profiling = raw["profiling"]
        output = raw["output"]

        raw_dir = resolve_path(files["raw_dir"])
        transaction_path = raw_dir / str(files["transactions"]["name"])
        identity_path = raw_dir / str(files["identity"]["name"])

        forbidden: dict[str, str] = {}
        for item in raw["forbidden_features"]:
            if "column" not in item or "reason" not in item:
                raise ConfigError("forbidden_features 每一项都必须包含 column 与 reason")
            forbidden[str(item["column"])] = str(item["reason"])

        for key in ("join_key", "target", "split_time"):
            if key not in columns:
                raise ConfigError(f"columns 段缺少 {key}")
        # 禁止特征必须覆盖结构性列，否则后续角色判定会出现漏洞。
        for key in ("join_key", "target", "split_time"):
            if str(columns[key]) not in forbidden:
                raise ConfigError(
                    f"columns.{key}={columns[key]} 未出现在 forbidden_features 中，"
                    "禁止特征清单必须完整覆盖目标列、连接键与时间列"
                )

        profiling_config = ProfilingConfig(
            chunk_size=int(profiling.get("chunk_size", 25000)),
            max_tracked_unique=int(profiling.get("max_tracked_unique", 5000)),
            max_total_tracked_unique=int(profiling.get("max_total_tracked_unique", 1_000_000)),
            quantile_sample_size=int(profiling.get("quantile_sample_size", 20000)),
            quantiles=tuple(float(value) for value in profiling.get("quantiles", [])),
            max_category_summaries=int(profiling.get("max_category_summaries", 10)),
            sample_seed=int(profiling.get("sample_seed", 42)),
        )

        return cls(
            meta=DatasetMeta(
                dataset_id=str(dataset["id"]),
                display_name=str(dataset["display_name"]),
                version_note=str(dataset["version_note"]),
                provider=str(dataset["provider"]),
                organizer=str(dataset["organizer"]),
                official_data_url=str(dataset["official_data_url"]),
                official_rules_url=str(dataset["official_rules_url"]),
                license_note=str(dataset["license_note"]),
            ),
            raw_dir=raw_dir,
            transaction_path=transaction_path,
            identity_path=identity_path,
            join_key=str(columns["join_key"]),
            target=str(columns["target"]),
            split_time=str(columns["split_time"]),
            train_ratio=float(split["train_ratio"]),
            validation_ratio=float(split["validation_ratio"]),
            split_strategy=str(split.get("strategy", "temporal_tie_safe")),
            forbidden_features=forbidden,
            anonymous_prefixes=tuple(str(p) for p in raw.get("anonymous_prefixes", [])),
            profiling=profiling_config,
            outputs={key: resolve_path(value) for key, value in output.items()},
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> IeeeCisConfig:
        config_path = resolve_path(path)
        if not config_path.is_file():
            raise ConfigError(f"配置文件不存在：{config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ConfigError(f"配置文件顶层必须是映射：{config_path}")
        return cls.from_mapping(raw)


# --- 文件发现 -----------------------------------------------------------


def check_required_files(config: IeeeCisConfig) -> None:
    """确认两个原始文件存在；缺失时一次性列出全部缺失项与放置路径。

    只检查存在性，不读取文件内容。

    Raises:
        MissingDataError: 任一必需文件缺失。
    """
    missing = [(name, path) for name, path in config.required_files() if not path.is_file()]
    if not missing:
        return

    lines = [
        "缺少必需的 IEEE-CIS 原始数据文件，无法执行审计。",
        f"期望目录（相对项目根目录）：{config.raw_dir_relative}",
        "缺失文件：",
    ]
    for name, path in missing:
        lines.append(f"  - {name}（期望位置：{config.relative(path.parent)}/{name}）")
    lines.extend(
        [
            "人工放置步骤：",
            "  1. 打开官方数据页并在页面同意竞赛规则与数据条款："
            f"{config.meta.official_data_url}",
            "  2. 手动下载 train_transaction.csv 与 train_identity.csv。",
            f"  3. 放入 {config.raw_dir_relative}，文件名保持原样。",
            "  4. 重新运行：python scripts/audit_ieee_cis.py --config configs/ieee_cis.yaml",
            "本项目不代替使用者同意条款，也不下载、不转发任何逐行数据。",
        ]
    )
    raise MissingDataError("\n".join(lines))


def fingerprint_required_files(config: IeeeCisConfig) -> dict[str, FileFingerprint]:
    """对两个原始文件做流式指纹，返回 ``逻辑名 -> 指纹``。

    哈希的读取块大小与列画像的 ``chunk_size`` 无关：前者是字节缓冲，用
    :data:`~ai_detector.data.fingerprint.DEFAULT_CHUNK_SIZE` 即可。
    """
    return {
        name: fingerprint_file(path, root=PROJECT_ROOT)
        for name, path in config.required_files()
    }


# --- 分块读取 -----------------------------------------------------------


def iter_chunks(
    path: str | Path,
    *,
    chunk_size: int,
    usecols: tuple[str, ...] | None = None,
) -> Iterator[pd.DataFrame]:
    """按块迭代读取 CSV。

    Args:
        path: CSV 路径。
        chunk_size: 每块行数。
        usecols: 只读取这些列；``None`` 表示读取全部列。

    Yields:
        数据框分块。
    """
    if chunk_size < 1:
        raise DataContractError(f"chunk_size 必须为正整数，实际为 {chunk_size}")
    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        usecols=list(usecols) if usecols else None,
        low_memory=True,
    )
    for chunk in reader:
        yield chunk


def read_header_columns(path: str | Path) -> list[str]:
    """只读取表头，返回列名列表（不加载数据行）。"""
    return list(pd.read_csv(path, nrows=0).columns)


# --- Schema 审计 --------------------------------------------------------


def assert_columns_present(actual: list[str], required: tuple[str, ...], *, file_label: str) -> None:
    """确认必需列全部存在。"""
    missing = [column for column in required if column not in actual]
    if missing:
        raise DataContractError(
            f"{file_label} 缺少必需列：{missing}；实际列数 {len(actual)}。"
            "请确认下载的是官方竞赛文件且未被改名或裁剪。"
        )


def assert_binary_target(values: pd.Series, *, target: str, file_label: str) -> None:
    """确认目标列非空且严格属于 ``{0, 1}``。

    刻意不做 ``astype(int)`` 这种静默截断：标签 ``0.3`` 会被截成 ``0``，把上游数据错误
    伪装成合法输入。这里与 ``evaluation.metrics`` 的严格校验保持一致，但没有跨层复用，
    以保持数据层不依赖评估层。
    """
    if values.isna().any():
        raise DataContractError(f"{file_label} 的 {target} 存在空值，无法作为标签")

    as_float = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(as_float).all():
        raise DataContractError(f"{file_label} 的 {target} 含非有限值（NaN 或 inf）")
    observed = np.unique(as_float)
    if not np.isin(observed, (0.0, 1.0)).all():
        preview = observed[:10].tolist()
        raise DataContractError(
            f"{file_label} 的 {target} 必须严格属于 {{0, 1}}，实际出现：{preview}"
            "（浮点标签不会被静默截断为 0/1）"
        )


def assert_time_finite(values: pd.Series, *, split_time: str, file_label: str) -> None:
    """确认时间列非空且可转为有限数值。"""
    if values.isna().any():
        raise DataContractError(f"{file_label} 的 {split_time} 存在空值，无法用于时间切分")
    as_float = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(as_float).all():
        raise DataContractError(
            f"{file_label} 的 {split_time} 含非有限值（NaN 或 inf），无法用于时间切分"
        )


@dataclass
class TransactionScan:
    """第一遍扫描 transaction 文件的聚合结果。"""

    n_rows: int = 0
    n_null_keys: int = 0
    n_duplicate_key_rows: int = 0
    n_null_time: int = 0
    n_non_finite_time: int = 0
    n_null_target: int = 0
    n_invalid_target: int = 0
    observed_target_values: set[float] = field(default_factory=set)
    min_time: float | None = None
    max_time: float | None = None
    columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_null_join_key": self.n_null_keys,
            "n_duplicate_join_key_rows": self.n_duplicate_key_rows,
            "n_null_split_time": self.n_null_time,
            "n_non_finite_split_time": self.n_non_finite_time,
            "n_null_target": self.n_null_target,
            "n_invalid_target": self.n_invalid_target,
            "observed_target_values": sorted(self.observed_target_values),
            "min_split_time": self.min_time,
            "max_split_time": self.max_time,
            "n_columns": len(self.columns),
        }


def scan_transactions(config: IeeeCisConfig) -> tuple[TransactionScan, np.ndarray, np.ndarray, np.ndarray]:
    """分块扫描 transaction 文件，做 Schema 校验并收集切分所需的三列。

    只读取 ``TransactionID``、``TransactionDT``、``isFraud`` 三列，因此内存占用与
    文件宽度无关。

    Returns:
        ``(scan, transaction_ids, transaction_dt, is_fraud)``；三个数组按文件原始顺序。

    Raises:
        DataContractError: 缺列、键重复/为空、时间非法或标签非法。
    """
    header = read_header_columns(config.transaction_path)
    assert_columns_present(header, STRUCTURAL_COLUMNS, file_label="train_transaction.csv")

    scan = TransactionScan(columns=list(header))
    seen_keys: set[int] = set()
    id_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    for chunk in iter_chunks(
        config.transaction_path,
        chunk_size=config.profiling.chunk_size,
        usecols=STRUCTURAL_COLUMNS,
    ):
        scan.n_rows += len(chunk)
        keys = chunk[config.join_key]
        times = chunk[config.split_time]
        targets = chunk[config.target]

        scan.n_null_keys += int(keys.isna().sum())
        if keys.isna().any():
            raise DataContractError(
                f"train_transaction.csv 的 {config.join_key} 存在空值，无法作为连接键"
            )

        key_values = pd.to_numeric(keys, errors="coerce")
        if key_values.isna().any():
            raise DataContractError(
                f"train_transaction.csv 的 {config.join_key} 无法全部解析为数值"
            )
        key_array = key_values.to_numpy(dtype=np.int64)
        if len(np.unique(key_array)) != len(key_array):
            scan.n_duplicate_key_rows += len(key_array) - len(np.unique(key_array))
            raise DataContractError(
                f"train_transaction.csv 的 {config.join_key} 在同一分块内重复；"
                "连接键必须唯一，否则左连接会放大交易行数"
            )
        overlap = seen_keys.intersection(key_array.tolist())
        if overlap:
            raise DataContractError(
                f"train_transaction.csv 的 {config.join_key} 跨分块重复，示例：{sorted(overlap)[:5]}"
            )
        seen_keys.update(key_array.tolist())

        assert_time_finite(times, split_time=config.split_time, file_label="train_transaction.csv")
        assert_binary_target(targets, target=config.target, file_label="train_transaction.csv")

        time_array = pd.to_numeric(times, errors="coerce").to_numpy(dtype=float)
        target_array = pd.to_numeric(targets, errors="coerce").to_numpy(dtype=float)
        scan.observed_target_values.update(np.unique(target_array).tolist())
        chunk_min = float(time_array.min())
        chunk_max = float(time_array.max())
        scan.min_time = chunk_min if scan.min_time is None else min(scan.min_time, chunk_min)
        scan.max_time = chunk_max if scan.max_time is None else max(scan.max_time, chunk_max)

        id_parts.append(key_array)
        time_parts.append(time_array)
        target_parts.append(target_array.astype(np.int8))

    if scan.n_rows == 0:
        raise DataContractError("train_transaction.csv 没有任何数据行")

    return (
        scan,
        np.concatenate(id_parts),
        np.concatenate(time_parts),
        np.concatenate(target_parts),
    )


@dataclass
class JoinAudit:
    """transaction 与 identity 的连接审计结果。"""

    transaction_rows: int
    identity_rows: int
    identity_unique_keys: int
    matched_transaction_rows: int
    identity_coverage_rate: float
    orphan_identity_keys: int
    rows_after_left_join: int
    transaction_rows_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_rows": self.transaction_rows,
            "identity_rows": self.identity_rows,
            "identity_unique_keys": self.identity_unique_keys,
            "matched_transaction_rows": self.matched_transaction_rows,
            "identity_coverage_rate": self.identity_coverage_rate,
            "orphan_identity_keys": self.orphan_identity_keys,
            "rows_after_left_join": self.rows_after_left_join,
            "transaction_rows_preserved": self.transaction_rows_preserved,
        }


def audit_join(
    config: IeeeCisConfig,
    transaction_ids: np.ndarray,
) -> JoinAudit:
    """审计 identity 与 transaction 的连接关系。

    规则：

    - identity 的 ``TransactionID`` 必须非空且唯一；重复键必须报错，不允许一对多连接
      放大交易行数。
    - 以 transaction 为主表做左连接，**不得因为 identity 缺失而删除交易**。
    - 实际执行分块左连接并统计结果行数，而不是只在理论上断言。

    Raises:
        DataContractError: identity 缺列、键为空或重复。
    """
    header = read_header_columns(config.identity_path)
    assert_columns_present(header, (config.join_key,), file_label="train_identity.csv")

    identity_key_parts: list[np.ndarray] = []
    identity_rows = 0
    seen: set[int] = set()

    for chunk in iter_chunks(
        config.identity_path,
        chunk_size=config.profiling.chunk_size,
        usecols=(config.join_key,),
    ):
        identity_rows += len(chunk)
        keys = chunk[config.join_key]
        if keys.isna().any():
            raise DataContractError(
                f"train_identity.csv 的 {config.join_key} 存在空值，无法作为连接键"
            )
        key_values = pd.to_numeric(keys, errors="coerce")
        if key_values.isna().any():
            raise DataContractError(
                f"train_identity.csv 的 {config.join_key} 无法全部解析为数值"
            )
        key_array = key_values.to_numpy(dtype=np.int64)
        if len(np.unique(key_array)) != len(key_array):
            raise DataContractError(
                f"train_identity.csv 的 {config.join_key} 在同一分块内重复；"
                "ticket 级一对多连接会放大交易行数，必须先用主键唯一性校验拦截"
            )
        overlap = seen.intersection(key_array.tolist())
        if overlap:
            raise DataContractError(
                f"train_identity.csv 的 {config.join_key} 跨分块重复，"
                f"示例：{sorted(overlap)[:5]}"
            )
        seen.update(key_array.tolist())
        identity_key_parts.append(key_array)

    identity_keys = (
        np.concatenate(identity_key_parts) if identity_key_parts else np.empty(0, dtype=np.int64)
    )
    identity_key_set = set(identity_keys.tolist())
    transaction_key_set = set(transaction_ids.tolist())

    # 两侧键都已校验唯一，因此交集大小就是被 identity 覆盖的交易行数。
    matched_rows = len(identity_key_set & transaction_key_set)
    orphan = len(identity_key_set - transaction_key_set)

    # 分块执行真实左连接，统计连接后行数。只带连接键列，因此内存占用很小。
    identity_key_frame = pd.DataFrame({config.join_key: identity_keys})
    rows_after_join = 0
    for chunk in iter_chunks(
        config.transaction_path,
        chunk_size=config.profiling.chunk_size,
        usecols=(config.join_key,),
    ):
        merged = chunk.merge(identity_key_frame, on=config.join_key, how="left")
        rows_after_join += len(merged)

    transaction_rows = int(transaction_ids.size)
    return JoinAudit(
        transaction_rows=transaction_rows,
        identity_rows=identity_rows,
        identity_unique_keys=len(identity_key_set),
        matched_transaction_rows=matched_rows,
        identity_coverage_rate=(
            float(matched_rows / transaction_rows) if transaction_rows else 0.0
        ),
        orphan_identity_keys=orphan,
        rows_after_left_join=rows_after_join,
        transaction_rows_preserved=rows_after_join == transaction_rows,
    )
