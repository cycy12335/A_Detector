"""聚合列画像与特征角色登记。

## 内存策略（与 `WORK_LOG.md` 记录一致）

IEEE-CIS 的 transaction 文件约 394 列，一次性全表读入会占用大量内存。因此本模块：

1. **分块处理**：由调用方按 ``chunk_size`` 逐块喂入，模块只保留聚合量。
2. **唯一值跟踪有双重上限**：每列上限 ``max_tracked_unique``，全局预算
   ``max_total_tracked_unique``。触顶后该列标记为**截断**，``distinct_count`` 只表示
   已跟踪到的下界，而不是真实唯一值数。宁可报告"下界 + 截断标记"，也不静默给出错误数字。
3. **分位数来自抽样**：数值列只保留至多 ``quantile_sample_size`` 个值的固定种子随机子样本，
   分位数因此是**近似值**，报告中必须标注。

## 输出边界

- 只输出聚合统计，**不输出任何逐行取值**。
- 类别列不输出原始取值（邮箱、设备串等可能直接识别个人），只输出计数类摘要。
- 报告只记录相对路径，不含用户主目录等机器专属信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai_detector.data.temporal_split import (
    PARTITIONS,
    SplitBoundaries,
    assign_partition,
)

ROLE_TARGET = "target"
ROLE_JOIN_KEY = "join_key"
ROLE_SPLIT_ONLY = "split_only"
ROLE_CANDIDATE_FEATURE = "candidate_feature"
ROLE_EXCLUDED = "excluded"

ALL_ROLES = (
    ROLE_TARGET,
    ROLE_JOIN_KEY,
    ROLE_SPLIT_ONLY,
    ROLE_CANDIDATE_FEATURE,
    ROLE_EXCLUDED,
)

#: 匿名列统一风险登记文本。不得为匿名列编造业务含义。
ANONYMOUS_RISK_NOTE = (
    "语义匿名，无法仅凭列名排除潜在代理或泄漏风险；本工作包不为其编造业务含义。"
)
#: 列名有业务暗示但未获确认时的登记文本。
SUGGESTIVE_NAME_NOTE = (
    "列名有业务暗示，但数据以匿名化形式发布，其真实含义无法从公开数据确认。"
)

#: identity 中未能匹配到任何交易的行所属的分区标签。这些行不属于任何切分分区，
#: 单独成桶而不是被丢弃或强行归入某个分区。
UNMATCHED_PARTITION = "unmatched_join_key"


@dataclass(frozen=True)
class ProfilingConfig:
    """列画像参数。"""

    chunk_size: int = 50000
    max_tracked_unique: int = 5000
    max_total_tracked_unique: int = 1_000_000
    quantile_sample_size: int = 20000
    quantiles: tuple[float, ...] = (0.01, 0.25, 0.5, 0.75, 0.99)
    max_category_summaries: int = 10
    sample_seed: int = 42

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size 必须为正整数")
        if self.max_tracked_unique < 1:
            raise ValueError("max_tracked_unique 必须为正整数")
        if self.quantile_sample_size < 1:
            raise ValueError("quantile_sample_size 必须为正整数")
        if any(not 0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError("quantiles 必须全部位于开区间 (0, 1)")


@dataclass
class ColumnProfile:
    """单列的聚合画像。"""

    column_name: str
    source_file: str
    original_dtype: str
    is_numeric: bool
    role: str
    role_reason: str
    semantic_note: str
    total_rows: int = 0
    non_null_count: int = 0
    null_rate: float = 0.0
    non_finite_count: int = 0
    distinct_count: int | None = None
    distinct_is_censored: bool = False
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    std: float | None = None
    quantiles: dict[str, float] = field(default_factory=dict)
    quantiles_are_approximate: bool = False
    quantile_sample_size: int = 0
    null_rate_by_partition: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "source_file": self.source_file,
            "original_dtype": self.original_dtype,
            "is_numeric": self.is_numeric,
            "role": self.role,
            "role_reason": self.role_reason,
            "semantic_note": self.semantic_note,
            "total_rows": self.total_rows,
            "non_null_count": self.non_null_count,
            "null_rate": self.null_rate,
            "non_finite_count": self.non_finite_count,
            "distinct_count": self.distinct_count,
            "distinct_is_censored": self.distinct_is_censored,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "quantiles": {key: value for key, value in self.quantiles.items()},
            "quantiles_are_approximate": self.quantiles_are_approximate,
            "quantile_sample_size": self.quantile_sample_size,
            "null_rate_by_partition": dict(self.null_rate_by_partition),
        }


def quantile_key(quantile: float) -> str:
    """把分位数映射为稳定的字段名，例如 ``0.25`` -> ``q25``。"""
    return f"q{round(quantile * 100):02d}"


def is_numeric_dtype(dtype: Any) -> bool:
    """数值列判定。布尔列按类别处理（IEEE-CIS 中布尔列没有业务含义）。"""
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


class _ColumnAccumulator:
    """单列的流式累加器。"""

    def __init__(
        self,
        name: str,
        dtype: Any,
        *,
        config: ProfilingConfig,
        numeric: bool,
        column_index: int,
    ) -> None:
        self.name = name
        self.original_dtype = str(dtype)
        self.numeric = numeric
        self.config = config
        # 每列独立的固定种子生成器。刻意不使用内置 hash()：字符串 hash 受
        # PYTHONHASHSEED 随机化影响，会让抽样结果在不同进程间不一致，破坏可复现性。
        self._sample_rng = np.random.default_rng(config.sample_seed + column_index)

        self.total_rows = 0
        self.non_null_count = 0
        self.non_finite_count = 0
        self._sum = 0.0
        self._sumsq = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None

        self._distinct: set[Any] | None = set()
        self.distinct_is_censored = False

        self._sample: np.ndarray | None = np.empty(0, dtype=float) if numeric else None

        # 分区标签动态发现：transaction 用三个切分分区，identity 还会出现"未能匹配连接键"
        # 这一桶，因此不能把分区集合写死成固定三个。
        self.partition_rows: dict[str, int] = {}
        self.partition_nulls: dict[str, int] = {}

    # --- 唯一值 ---------------------------------------------------------

    def trackable_for_unique(self, remaining_budget: int) -> bool:
        """是否还应继续为唯一值计数消耗集合内存。

        ``remaining_budget`` 是**剩余**的全局条目预算（不是已用量）。预算耗尽或本列已达
        每列上限时返回 ``False``，由调用方决定转为截断。
        """
        if self._distinct is None or self.distinct_is_censored:
            return False
        limit = min(self.config.max_tracked_unique, max(remaining_budget, 0))
        return len(self._distinct) < limit

    def tracked_count(self) -> int:
        """当前已跟踪到的唯一值个数；未跟踪时返回 0。"""
        return 0 if self._distinct is None else len(self._distinct)

    def add_unique(self, values: np.ndarray) -> None:
        if self._distinct is None:
            return
        self._distinct.update(values.tolist())
        if len(self._distinct) >= self.config.max_tracked_unique:
            # 达到每列上限：保留已跟踪到的下界，并标记为截断。
            self.distinct_is_censored = True

    def censor_unique(self) -> None:
        """停止跟踪并释放集合内存。

        ``distinct_count`` 会变为 ``None``（未跟踪），与"已跟踪到 N 个但可能更多"
        （``distinct_count=N`` 且 ``distinct_is_censored=True``）区分开。
        """
        if self._distinct is not None:
            self.distinct_is_censored = True
            self._distinct = None

    @property
    def tracked_unique(self) -> tuple[int | None, bool]:
        if self._distinct is None:
            return (None, True)
        return (len(self._distinct), self.distinct_is_censored)

    # --- 数值统计 -------------------------------------------------------

    def update(self, series: pd.Series, partition: np.ndarray) -> None:
        """累加一个分块。"""
        values = series.to_numpy()
        not_null = pd.notna(values)
        self.total_rows += len(values)
        self.non_null_count += int(not_null.sum())

        for name in np.unique(partition):
            in_partition = partition == name
            label = str(name)
            self.partition_rows[label] = self.partition_rows.get(label, 0) + int(in_partition.sum())
            self.partition_nulls[label] = self.partition_nulls.get(label, 0) + int(
                (~not_null & in_partition).sum()
            )

        if self.numeric:
            numeric_values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(numeric_values)
            self.non_finite_count += int((not_null & ~finite).sum())
            usable = numeric_values[finite]
            if usable.size:
                self._sum += float(usable.sum())
                self._sumsq += float(np.square(usable).sum())
                chunk_min = float(usable.min())
                chunk_max = float(usable.max())
                self.minimum = chunk_min if self.minimum is None else min(self.minimum, chunk_min)
                self.maximum = chunk_max if self.maximum is None else max(self.maximum, chunk_max)
                self._merge_sample(usable)

    def add_unique_from(self, series: pd.Series, rng: np.random.Generator) -> None:
        """从分块中抽取唯一值候选，避免对整列做集合运算。"""
        if self._distinct is None or self.distinct_is_censored:
            return
        values = series.dropna().to_numpy()
        if values.size == 0:
            return
        budget = self.config.max_tracked_unique - len(self._distinct)
        if budget <= 0:
            self.distinct_is_censored = True
            return
        if values.size > budget:
            values = values[rng.choice(values.size, size=budget, replace=False)]
        self.add_unique(values)

    def _merge_sample(self, values: np.ndarray) -> None:
        """把分块取值并入本列样本，超过上限时随机下采样并保持原有顺序。

        抽样使用本列固定的种子发生器，因此同一输入必然得到同一份样本。
        """
        cap = self.config.quantile_sample_size
        assert self._sample is not None
        if self._sample.size >= cap:
            return
        combined = np.concatenate([self._sample, values])
        if combined.size > cap:
            keep = self._sample_rng.choice(combined.size, size=cap, replace=False)
            keep.sort()  # 保持时间顺序，便于排查
            combined = combined[keep]
        self._sample = combined

    def numeric_summary(self) -> dict[str, Any]:
        non_null = self.non_null_count
        summary: dict[str, Any] = {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": None,
            "std": None,
            "quantiles": {},
            "quantiles_are_approximate": False,
            "quantile_sample_size": 0,
        }
        if not self.numeric or non_null == 0 or self._sample is None or self._sample.size == 0:
            return summary

        # 均值/标准差基于精确的累计和，不受抽样影响。
        mean = self._sum / non_null
        variance = max(self._sumsq / non_null - mean * mean, 0.0)
        summary["mean"] = mean
        summary["std"] = float(np.sqrt(variance))

        quantile_values = self._sample
        summary["quantile_sample_size"] = int(quantile_values.size)
        # 样本量小于列的非空数时，分位数是近似值。
        summary["quantiles_are_approximate"] = quantile_values.size < non_null
        for quantile in self.config.quantiles:
            summary["quantiles"][quantile_key(quantile)] = float(
                np.quantile(quantile_values, quantile)
            )
        return summary

    def to_profile(self, *, source_file: str, role: str, role_reason: str, semantic_note: str) -> ColumnProfile:
        distinct_count, censored = self.tracked_unique
        null_rate = (self.total_rows - self.non_null_count) / self.total_rows if self.total_rows else 0.0
        null_rate_by_partition = {
            name: (
                self.partition_nulls[name] / self.partition_rows[name]
                if self.partition_rows[name]
                else 0.0
            )
            for name in sorted(self.partition_rows)
        }
        summary = self.numeric_summary()
        return ColumnProfile(
            column_name=self.name,
            source_file=source_file,
            original_dtype=self.original_dtype,
            is_numeric=self.numeric,
            role=role,
            role_reason=role_reason,
            semantic_note=semantic_note,
            total_rows=self.total_rows,
            non_null_count=self.non_null_count,
            null_rate=null_rate,
            non_finite_count=self.non_finite_count,
            distinct_count=distinct_count,
            distinct_is_censored=censored,
            minimum=summary["minimum"],
            maximum=summary["maximum"],
            mean=summary["mean"],
            std=summary["std"],
            quantiles=summary["quantiles"],
            quantiles_are_approximate=summary["quantiles_are_approximate"],
            quantile_sample_size=summary["quantile_sample_size"],
            null_rate_by_partition=null_rate_by_partition,
        )


class ChunkedProfiler:
    """按分块累加整个文件的列画像。"""

    def __init__(
        self,
        *,
        source_file: str,
        config: ProfilingConfig,
        boundaries: SplitBoundaries | None = None,
        split_time_column: str | None = None,
        join_key_column: str | None = None,
        partition_map: dict[int, str] | None = None,
        unmatched_label: str = UNMATCHED_PARTITION,
    ) -> None:
        """构造画像器。

        分区来源有三种，按优先级：

        1. ``boundaries`` + ``split_time_column``：直接用时间列判定分区（transaction 文件）。
        2. ``partition_map`` + ``join_key_column``：按键查表判定分区（identity 文件没有时间列）。
           查不到键时归入 ``unmatched_label``，不丢弃、不臆测。
        3. 都没有：全部行归入第一个分区标签，只产出整体统计。
        """
        self.source_file = source_file
        self.config = config
        self.boundaries = boundaries
        self.split_time_column = split_time_column
        self.join_key_column = join_key_column
        self.partition_map = partition_map
        self.unmatched_label = unmatched_label
        self._accumulators: dict[str, _ColumnAccumulator] = {}
        self._column_order: list[str] = []
        self._rng = np.random.default_rng(config.sample_seed)
        self._unique_budget_exhausted = False
        # 增量维护已用预算，避免每个分块都遍历全部列累加一次。
        self._tracked_unique_total = 0

    def update(self, chunk: pd.DataFrame) -> None:
        """处理一个分块。列集合以第一次出现的分块为准。"""
        if not self._column_order:
            self._column_order = list(chunk.columns)
            for index, name in enumerate(self._column_order):
                self._accumulators[name] = _ColumnAccumulator(
                    name,
                    chunk[name].dtype,
                    config=self.config,
                    numeric=is_numeric_dtype(chunk[name].dtype),
                    column_index=index,
                )

        missing = [name for name in self._column_order if name not in chunk.columns]
        if missing:
            raise ValueError(f"分块缺少此前出现过的列：{missing[:5]}")

        partition = self._resolve_partition(chunk)

        for name in self._column_order:
            accumulator = self._accumulators[name]
            series = chunk[name]
            accumulator.update(series, partition)

            # 已达每列上限：停止累加但**保留**集合，使报告中仍能给出唯一值下界。
            if accumulator.distinct_is_censored:
                continue
            # 全局预算耗尽：必须释放集合内存，此时该列只能报告"未跟踪"。
            if self._unique_budget_exhausted:
                accumulator.censor_unique()
                continue

            remaining = self.config.max_total_tracked_unique - self._tracked_unique_total
            if accumulator.trackable_for_unique(remaining):
                before = accumulator.tracked_count()
                accumulator.add_unique_from(series, self._rng)
                self._tracked_unique_total += accumulator.tracked_count() - before
            else:
                accumulator.censor_unique()
                self._unique_budget_exhausted = True

    def _resolve_partition(self, chunk: pd.DataFrame) -> np.ndarray:
        """判定当前分块每行所属分区。"""
        if self.boundaries is not None and self.split_time_column is not None:
            return assign_partition(chunk[self.split_time_column], self.boundaries)
        if self.partition_map is not None and self.join_key_column is not None:
            keys = chunk[self.join_key_column]
            return np.array(
                [self.partition_map.get(int(key), self.unmatched_label) for key in keys],
                dtype=object,
            )
        return np.full(len(chunk), PARTITIONS[0], dtype=object)

    def build_profiles(
        self,
        *,
        join_key: str,
        target: str,
        split_time: str,
        forbidden: dict[str, str],
        anonymous_prefixes: tuple[str, ...],
    ) -> list[ColumnProfile]:
        """产出列画像并就地登记特征角色。

        返回顺序与列在文件中出现的顺序一致，便于与原始 Schema 对照。
        """
        profiles = [
            self._accumulators[name].to_profile(
                source_file=self.source_file,
                role=ROLE_CANDIDATE_FEATURE,
                role_reason="",
                semantic_note="",
            )
            for name in self._column_order
        ]
        return register_feature_roles(
            profiles,
            join_key=join_key,
            target=target,
            split_time=split_time,
            forbidden=forbidden,
            anonymous_prefixes=anonymous_prefixes,
        )

    @property
    def column_order(self) -> list[str]:
        return list(self._column_order)


def classify_role(
    column: str,
    *,
    join_key: str,
    target: str,
    split_time: str,
    forbidden: dict[str, str],
    all_missing_in_train: bool,
) -> tuple[str, str]:
    """决定列的角色与原因。

    顺序很重要：先判定目标/连接键/时间列（它们绝不能成为候选特征），再看显式禁止清单，
    然后才是"训练分区全缺失"这一数据驱动规则，最后才轮到候选特征。

    "训练分区全缺失"只依据**训练分区**判定：后续特征保留/删除决策不得使用验证集或测试集
    的统计，否则会把验证/测试分布泄漏进特征选择。
    """
    if column == target:
        return ROLE_TARGET, "目标列，只能作为标签。"
    if column == join_key:
        return ROLE_JOIN_KEY, "连接键/标识列，唯一值编码行身份，作为特征会泄漏行顺序与切分位置。"
    if column == split_time:
        return ROLE_SPLIT_ONLY, "原始时间列，仅用于时间切分；作为特征会把时间与分区边界泄漏给模型。"
    if column in forbidden:
        return ROLE_EXCLUDED, forbidden[column]
    if all_missing_in_train:
        return ROLE_EXCLUDED, "在训练分区中 100% 缺失，无可学习信息（仅依据训练分区判定）。"
    return ROLE_CANDIDATE_FEATURE, "允许进入后续工作包的特征评估，本工作包不对其做取舍。"


def semantic_note_for(column: str, anonymous_prefixes: tuple[str, ...]) -> str:
    """为列生成语义说明。匿名列一律登记风险，不编造含义。"""
    if any(column.startswith(prefix) for prefix in anonymous_prefixes):
        return ANONYMOUS_RISK_NOTE
    return SUGGESTIVE_NAME_NOTE


def register_feature_roles(
    profiles: list[ColumnProfile],
    *,
    join_key: str,
    target: str,
    split_time: str,
    forbidden: dict[str, str],
    anonymous_prefixes: tuple[str, ...],
) -> list[ColumnProfile]:
    """就地登记每列的角色、原因与语义说明。"""
    for profile in profiles:
        train_null_rate = profile.null_rate_by_partition.get(PARTITIONS[0], 0.0)
        role, reason = classify_role(
            profile.column_name,
            join_key=join_key,
            target=target,
            split_time=split_time,
            forbidden=forbidden,
            all_missing_in_train=profile.non_null_count == 0 or train_null_rate >= 1.0,
        )
        profile.role = role
        profile.role_reason = reason
        profile.semantic_note = semantic_note_for(profile.column_name, anonymous_prefixes)
    return profiles


def role_summary(profiles: list[ColumnProfile]) -> dict[str, list[str]]:
    """按角色汇总**去重后的列名**，便于快速核对禁止列未进入候选特征。

    同一个列名可能来自多个文件（例如 ``TransactionID`` 同时存在于 transaction 与
    identity），去重后按名称排序，避免汇总里出现重复项而被误读成登记错误。
    逐文件的完整登记请用 :func:`role_registry`。
    """
    summary: dict[str, list[str]] = {role: [] for role in ALL_ROLES}
    for profile in profiles:
        summary[profile.role].append(profile.column_name)
    return {role: sorted(set(names)) for role, names in summary.items()}


def role_registry(profiles: list[ColumnProfile]) -> list[dict[str, Any]]:
    """逐 ``(来源文件, 列)`` 的角色登记，保留来源维度。"""
    return [
        {
            "column_name": profile.column_name,
            "source_file": profile.source_file,
            "role": profile.role,
            "role_reason": profile.role_reason,
            "semantic_note": profile.semantic_note,
        }
        for profile in profiles
    ]
