"""时间并列安全的三分区切分清单。

与 `CW-001` 的位置切分不同，本模块服务于 IEEE-CIS 数据：同一个 ``TransactionDT``
可能对应多笔交易，按行号切分会把同一时刻拆到两个分区，造成边界处的时间穿越。

因此这里的边界是**时间值**而不是行号：

- 训练集：``TransactionDT < validation_start``
- 验证集：``validation_start <= TransactionDT < test_start``
- 测试集：``TransactionDT >= test_start``

代价是实际比例会因时间并列而偏离目标比例；**安全边界优先于精确比例**。

``TransactionDT`` 是相对时间偏移，参考起点未公开，因此本模块只处理原始数值，
不转换、不推断、不记录任何日历日期。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

SPLIT_ALGORITHM_VERSION = "temporal_tie_safe_v1"

TRAIN_PARTITION = "train"
VALIDATION_PARTITION = "validation"
TEST_PARTITION = "test"
PARTITIONS: tuple[str, str, str] = (TRAIN_PARTITION, VALIDATION_PARTITION, TEST_PARTITION)


class TemporalSplitError(ValueError):
    """切分无法满足约束时抛出（时间值不足、比例非法、分区会为空等）。"""


@dataclass(frozen=True)
class SplitBoundaries:
    """切分边界。``*_start`` 为 ``None`` 表示该分区从最小时间开始。"""

    validation_start: float
    test_start: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalSplitManifest:
    """切分清单：时间值到分区的确定性映射。"""

    algorithm_version: str
    train_ratio_target: float
    validation_ratio_target: float
    boundaries: SplitBoundaries
    n_rows: int
    sizes: dict[str, int]
    actual_ratios: dict[str, float]
    label_rates: dict[str, float]
    n_distinct_times: int
    tie_groups_at_boundary: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assign_partition(
    transaction_dt: pd.Series | np.ndarray,
    boundaries: SplitBoundaries,
) -> np.ndarray:
    """按边界把时间值映射为分区名数组。

    使用 ``searchsorted`` 语义：``dt < validation_start`` 为训练集。边界比较使用
    ``<`` 与 ``>=``，因此等于 ``validation_start`` 的交易全部进入验证集，不会跨分区。
    """
    values = np.asarray(transaction_dt, dtype=float)
    if not np.isfinite(values).all():
        raise TemporalSplitError("TransactionDT 含非有限值（NaN 或 inf），无法切分")

    partition = np.full(values.shape, TEST_PARTITION, dtype=object)
    partition[values < boundaries.test_start] = VALIDATION_PARTITION
    partition[values < boundaries.validation_start] = TRAIN_PARTITION
    return partition


def _closest_prefix_index(cumulative: np.ndarray, target: float, low: int, high: int) -> int:
    """在 ``[low, high]`` 中选出累计行数最接近 ``target`` 的前缀下标。

    并列时取较小下标（把更多行留给后面的分区），保证结果确定。

    Raises:
        TemporalSplitError: 可行区间为空。
    """
    if low > high:
        raise TemporalSplitError(
            "无法在为各分区保留足够时间点的前提下满足目标比例，请检查数据的时间分布"
        )
    candidates = np.arange(low, high + 1)
    distances = np.abs(cumulative[candidates] - target)
    return int(candidates[int(np.argmin(distances))])


def compute_boundaries(
    transaction_dt: pd.Series | np.ndarray,
    *,
    train_ratio: float,
    validation_ratio: float,
) -> SplitBoundaries:
    """由时间值和目标比例推导两个切分边界。

    算法：把时间值聚合为 ``(唯一时间, 该时间的行数)`` 并取前缀和。由于同一次交易的
    时间点不可拆分，边界只能落在**两个相邻唯一时间之间**，实际比例因此无法精确等于目标。
    在可行的前缀中选取累计行数最接近目标比例者，使偏差最小。

    约束：训练集、验证集、测试集各至少包含一个时间点。因此训练集前缀最多取到倒数第三个
    时间点，验证集前缀最多取到倒数第二个时间点。

    Raises:
        TemporalSplitError: 比例非法、时间值含非有限值、或唯一时间过少以至于无法切出
            三个非空分区。
    """
    if not 0.0 < train_ratio < 1.0:
        raise TemporalSplitError(f"train_ratio 必须位于 (0, 1)，实际为 {train_ratio}")
    if not 0.0 < validation_ratio < 1.0:
        raise TemporalSplitError(f"validation_ratio 必须位于 (0, 1)，实际为 {validation_ratio}")
    if train_ratio + validation_ratio >= 1.0:
        raise TemporalSplitError("train_ratio 与 validation_ratio 之和必须小于 1")

    values = np.asarray(transaction_dt, dtype=float)
    if values.size == 0:
        raise TemporalSplitError("TransactionDT 为空，无法切分")
    if not np.isfinite(values).all():
        raise TemporalSplitError("TransactionDT 含非有限值（NaN 或 inf），无法切分")

    unique_times, counts = np.unique(values, return_counts=True)
    n_rows = int(values.size)
    n_groups = int(unique_times.size)

    # 需要至少 3 个不同时间值，才能切出三个非空分区。
    if n_groups < 3:
        raise TemporalSplitError(
            f"唯一 TransactionDT 只有 {n_groups} 个，不足以切出三个非空分区"
        )

    cumulative = np.cumsum(counts)
    target_train = n_rows * train_ratio
    target_validation = n_rows * validation_ratio

    # 训练集取前 train_last+1 个时间点，至少给后面留两个时间点。
    train_last = _closest_prefix_index(cumulative, target_train, 0, n_groups - 3)
    # 验证集紧随其后，至少给测试集留一个时间点。
    validation_last = _closest_prefix_index(
        cumulative,
        cumulative[train_last] + target_validation,
        train_last + 1,
        n_groups - 2,
    )

    return SplitBoundaries(
        validation_start=float(unique_times[train_last + 1]),
        test_start=float(unique_times[validation_last + 1]),
    )


def build_split_manifest(
    transaction_ids: pd.Series,
    transaction_dt: pd.Series,
    is_fraud: pd.Series,
    *,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> tuple[TemporalSplitManifest, pd.DataFrame]:
    """构造切分清单并返回逐行分区表。

    先按 ``(TransactionDT, TransactionID)`` 稳定排序，保证相同输入必然得到相同结果，
    与输入原始顺序无关。

    Args:
        transaction_ids: 连接键列。
        transaction_dt: 相对时间偏移列。
        is_fraud: 目标列，只用于统计各分区标签率，不参与边界推导。
        train_ratio: 目标训练集比例。
        validation_ratio: 目标验证集比例。

    Returns:
        ``(manifest, row_partitions)``；后者含 ``TransactionID``、``TransactionDT``
        和 ``partition`` 三列，按时间和 ID 升序排列。

    Raises:
        TemporalSplitError: 长度不一致、ID 重复、时间非法或分区为空。
    """
    frame = pd.DataFrame(
        {
            "TransactionID": pd.Series(transaction_ids).reset_index(drop=True),
            "TransactionDT": pd.to_numeric(
                pd.Series(transaction_dt).reset_index(drop=True), errors="coerce"
            ),
            "isFraud": pd.to_numeric(
                pd.Series(is_fraud).reset_index(drop=True), errors="coerce"
            ),
        }
    )

    if len(frame) == 0:
        raise TemporalSplitError("输入为空，无法切分")
    if frame["TransactionID"].isna().any():
        raise TemporalSplitError("TransactionID 存在空值，无法切分")
    if not frame["TransactionID"].is_unique:
        raise TemporalSplitError("TransactionID 存在重复，无法建立唯一的分区清单")
    if frame["TransactionDT"].isna().any():
        raise TemporalSplitError("TransactionDT 存在空值或无法解析为数值，无法切分")

    # 稳定排序：先时间后 ID。相同输入在任何机器上都得到同一顺序。
    ordered = frame.sort_values(
        ["TransactionDT", "TransactionID"], kind="stable"
    ).reset_index(drop=True)

    boundaries = compute_boundaries(
        ordered["TransactionDT"],
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    ordered["partition"] = assign_partition(ordered["TransactionDT"], boundaries)

    sizes = {
        name: int((ordered["partition"] == name).sum()) for name in PARTITIONS
    }
    empty = [name for name, size in sizes.items() if size < 1]
    if empty:
        raise TemporalSplitError(f"分区为空：{empty}；请检查比例与时间分布")

    n_rows = len(ordered)
    label_rates = {
        name: float(ordered.loc[ordered["partition"] == name, "isFraud"].mean())
        for name in PARTITIONS
    }
    actual_ratios = {name: sizes[name] / n_rows for name in PARTITIONS}

    unique_times, counts = np.unique(ordered["TransactionDT"].to_numpy(dtype=float), return_counts=True)
    time_to_count = dict(zip(unique_times.tolist(), counts.tolist(), strict=True))
    tie_groups = {
        "validation_start": int(time_to_count.get(boundaries.validation_start, 0)),
        "test_start": int(time_to_count.get(boundaries.test_start, 0)),
    }

    manifest = TemporalSplitManifest(
        algorithm_version=SPLIT_ALGORITHM_VERSION,
        train_ratio_target=train_ratio,
        validation_ratio_target=validation_ratio,
        boundaries=boundaries,
        n_rows=n_rows,
        sizes=sizes,
        actual_ratios=actual_ratios,
        label_rates=label_rates,
        n_distinct_times=int(unique_times.size),
        tie_groups_at_boundary=tie_groups,
    )
    return manifest, ordered.loc[:, ["TransactionID", "TransactionDT", "partition"]]


def assert_split_is_consistent(row_partitions: pd.DataFrame) -> None:
    """校验切分不变量：不重、不漏、ID 两两不相交、同时间不跨分区。

    Raises:
        TemporalSplitError: 任一不变量被破坏。
    """
    required = {"TransactionID", "TransactionDT", "partition"}
    missing = required - set(row_partitions.columns)
    if missing:
        raise TemporalSplitError(f"分区表缺少列：{sorted(missing)}")

    if row_partitions["TransactionID"].duplicated().any():
        raise TemporalSplitError("分区表中 TransactionID 重复，切分不唯一")

    observed = set(row_partitions["partition"].unique())
    unexpected = observed - set(PARTITIONS)
    if unexpected:
        raise TemporalSplitError(f"出现未知分区名：{sorted(unexpected)}")

    # 同一个 TransactionDT 只能属于一个分区。
    times_per_partition = row_partitions.groupby("TransactionDT")["partition"].nunique()
    shared = times_per_partition[times_per_partition > 1]
    if not shared.empty:
        raise TemporalSplitError(
            f"有 {len(shared)} 个 TransactionDT 跨分区，违反时间并列保护"
        )

    # 时间顺序：训练集 < 验证集 < 测试集。
    order = row_partitions.groupby("partition")["TransactionDT"].agg(["min", "max"])
    for name in PARTITIONS:
        if name not in order.index:
            raise TemporalSplitError(f"分区 {name} 不存在")
    if order.loc[TRAIN_PARTITION, "max"] > order.loc[VALIDATION_PARTITION, "min"]:
        raise TemporalSplitError("训练集与验证集发生时间穿越")
    if order.loc[VALIDATION_PARTITION, "max"] > order.loc[TEST_PARTITION, "min"]:
        raise TemporalSplitError("验证集与测试集发生时间穿越")
