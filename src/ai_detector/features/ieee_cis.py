"""CW-003 特征契约与训练期决策。

## 两类决策，边界必须清楚

1. **保留/排除列**：由训练分区的缺失率、常量性、类别基数决定（``train_only_filters``）。
   validation/test 只应用已冻结的结论，不参与决定。
2. **预处理参数**：中位数、众数、均值/方差、未知类别处理，全部在训练分区 ``fit``。

非训练分区的行若发生变化（扰动、替换），上述两项决策与最终特征清单都必须保持不变——
测试用"扰动 held-out 数据"来守住这条边界。

## 稀疏性

one-hot 编码器固定 ``sparse_output=True``；``ColumnTransformer`` 的 ``sparse_threshold``
被显式设为 ``1.0``，因此管线**永远不会**把 one-hot 块稠密化。

注意这个参数的**语义方向**：本仓库使用的 scikit-learn 版本里判定式是
``sparse_output_ = density < sparse_threshold``（严格小于），所以只有阈值 **1.0** 才会稳定
走稀疏分支；把它设为 ``0.0`` 反而会让 ``density < 0.0`` 恒为假、把 one-hot 稠密化。

代价是最终设计矩阵以 CSR 存储；在数值列占比很高时，CSR 的单位内存高于稠密，这一点会在报告中
如实记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_detector.data.modeling import (
    ModelingConfig,
    ModelingContractError,
    load_identity_frame,
)

KIND_NUMERIC = "numeric"
KIND_CATEGORICAL = "categorical"

#: 排除原因码 -> 说明，写入 feature_manifest。
EXCLUSION_REASONS = {
    "all_null_in_train": "训练分区 100% 缺失：插补后为常量，无可学习信息。",
    "constant_in_train": "训练分区取值恒定（唯一取值数 <= 1）：对模型无区分度。",
    "missing_rate_above_threshold": "训练分区缺失率超过配置阈值 max_missing_rate。",
    "cardinality_above_threshold": "训练分区类别基数超过配置阈值 max_cardinality（one-hot 会膨胀）。",
    "cardinality_below_threshold": "训练分区类别基数低于 min_cardinality：由常量规则处理。",
    "column_absent_in_source": "该列未在源文件中出现。",
}


@dataclass
class ColumnTrainStats:
    """单列在**训练分区**上的统计量。"""

    name: str
    source_file: str
    kind: str
    n_rows: int = 0
    n_missing: int = 0
    n_distinct: int = 0
    distinct_is_censored: bool = False
    _distinct_values: set[str] | None = field(default=None, repr=False)

    @property
    def missing_rate(self) -> float:
        return self.n_missing / self.n_rows if self.n_rows else 0.0

    @property
    def n_present(self) -> int:
        return self.n_rows - self.n_missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_file": self.source_file,
            "kind": self.kind,
            "train_rows": self.n_rows,
            "train_missing": self.n_missing,
            "train_missing_rate": self.missing_rate,
            "train_distinct": self.n_distinct,
            "train_distinct_is_censored": self.distinct_is_censored,
        }


@dataclass
class FilterDecision:
    """训练期列保留决策。"""

    numeric: list[str]
    categorical: list[str]
    excluded: list[dict[str, str]]
    stats: dict[str, ColumnTrainStats]

    @property
    def kept(self) -> list[str]:
        return [*self.numeric, *self.categorical]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept_numeric": len(self.numeric),
            "kept_categorical": len(self.categorical),
            "kept_total": len(self.kept),
            "excluded_total": len(self.excluded),
        }


def _track_distinct(column_stats: ColumnTrainStats, series: pd.Series, cap: int) -> None:
    """把分块中**真正新增**的取值并入该列的有界集合。

    关键点：跨分块的重复值**不得**消耗新增额度。旧实现用"本块唯一值个数 > 剩余额度"直接
    判定截断，会把一个真实只有 3 类、却反复出现在多个分块里的列误判为超过上限并错误排除。
    这里只在"确实出现了一个未见过的新值、而集合已达上限"时才标记截断。
    """
    if column_stats.distinct_is_censored:
        return
    seen = column_stats._distinct_values
    assert seen is not None
    values = pd.unique(series.dropna())
    if values.size == 0:
        return

    as_text = [str(value) for value in values]
    if len(seen) >= cap:
        # 集合已满：只要存在任何未见过的取值就截断；否则什么也不做。
        if any(text not in seen for text in as_text):
            column_stats.distinct_is_censored = True
        return

    for text in as_text:
        if text in seen:
            continue
        if len(seen) >= cap:
            column_stats.distinct_is_censored = True
            return
        seen.add(text)


def _iter_transaction_chunks(config: ModelingConfig, columns: list[str], chunk_size: int):
    """按分块读取 transaction 文件指定列（始终带连接键）。"""
    wanted = [config.join_key, *columns]
    for chunk in pd.read_csv(
        config.paths["transaction"], usecols=list(dict.fromkeys(wanted)),
        chunksize=chunk_size, low_memory=True,
    ):
        yield chunk


def accumulate_train_stats(
    config: ModelingConfig,
    lookup: pd.Series,
    *,
    chunk_size: int | None = None,
) -> dict[str, ColumnTrainStats]:
    """只累计 **训练分区** 的缺失计数、类别取值集合与唯一值数。

    ## 缺失率的分母必须与最终建模矩阵一致

    这是本模块最容易出错的地方：identity 只覆盖部分交易，**未匹配的行在左连接后的模型矩阵里
    就是缺失**。因此 identity 列的 ``n_rows`` 取**训练分区的全部交易数**，未匹配的连接键计入
    ``n_missing``。若只统计 identity 表中的匹配行，缺失率会被严重低估（真实数据上 24 个
    identity 列的真实缺失率是 72%–86%，而错误口径会报出 0%–49%），从而让本应被排除的列留在
    特征清单里。

    ## 内存

    类别取值集合按 ``max_cardinality + 1`` 封顶，且只在出现真正新值时增长；因此内存与类别
    基数无关，跨分块重复值也不会触发误判。

    Raises:
        ModelingContractError: 分块缺列、identity 连接键不唯一，或出现清单外的连接键。
    """
    from ai_detector.data.modeling import ASSEMBLY_CHUNK_SIZE, align_chunk_with_lookup

    cap = int(config.train_only_filters["categorical"]["max_cardinality"]) + 1
    size = chunk_size or ASSEMBLY_CHUNK_SIZE
    stats: dict[str, ColumnTrainStats] = {
        entry["name"]: ColumnTrainStats(
            name=entry["name"], source_file=source, kind=entry["kind"]
        )
        for source, entries in config.allowlist.items()
        for entry in entries
    }
    for column_stats in stats.values():
        column_stats._distinct_values = set()

    transaction_columns = config.expected_columns_for("train_transaction.csv")
    identity_columns = config.expected_columns_for("train_identity.csv")

    # --- 第一段：transaction 列（无需连接，分母即训练交易数）-----------------
    for chunk in _iter_transaction_chunks(config, transaction_columns, size):
        chunk, partitions = align_chunk_with_lookup(config, lookup, chunk)
        train_chunk = chunk.loc[partitions == "train"]
        if train_chunk.empty:
            continue
        for name in transaction_columns:
            if name not in train_chunk.columns:
                raise ModelingContractError(f"分块缺少允许清单列：{name}")
            column_stats = stats[name]
            series = train_chunk[name]
            column_stats.n_rows += len(series)
            column_stats.n_missing += int(series.isna().sum())
            _track_distinct(column_stats, series, cap)

    # --- 第二段：identity 列（左连接后统计，未匹配行计为缺失）---------------
    if identity_columns:
        identity_frame = load_identity_frame(config)
        identity_frame = identity_frame.loc[
            :, [config.join_key, *identity_columns]
        ]
        for chunk in _iter_transaction_chunks(config, [], size):
            chunk, partitions = align_chunk_with_lookup(config, lookup, chunk)
            merged = chunk.merge(identity_frame, on=config.join_key, how="left")
            if len(merged) != len(chunk):
                raise ModelingContractError(
                    "identity 左连接改变了交易行数，说明 identity 连接键不唯一"
                )
            train_chunk = merged.loc[partitions == "train"]
            if train_chunk.empty:
                continue
            for name in identity_columns:
                column_stats = stats[name]
                series = train_chunk[name]
                # 分母是训练交易总数；未匹配 identity 的行在此处就是 NaN。
                column_stats.n_rows += len(series)
                column_stats.n_missing += int(series.isna().sum())
                _track_distinct(column_stats, series, cap)

    for column_stats in stats.values():
        if column_stats._distinct_values is not None:
            column_stats.n_distinct = len(column_stats._distinct_values)
            if column_stats.distinct_is_censored:
                # 截断时把计数记为上限值，明确表示"至少这么多"。
                column_stats.n_distinct = cap
        column_stats._distinct_values = None

    return stats


def decide_columns(
    config: ModelingConfig, stats: dict[str, ColumnTrainStats]
) -> FilterDecision:
    """按训练期统计与配置规则决定保留哪些列。

    规则顺序（全部只依赖训练分区）：
    1. 训练分区全缺失 -> 排除。
    2. 训练分区常量（唯一取值 <= 1） -> 排除。数值列与类别列一视同仁。
    3. 训练分区缺失率 > ``max_missing_rate`` -> 排除。
    4. 类别列基数 > ``max_cardinality`` 或 < ``min_cardinality`` -> 排除。
    """
    filters = config.train_only_filters
    max_missing = float(filters["max_missing_rate"])
    cat_filters = filters["categorical"]
    max_cardinality = int(cat_filters["max_cardinality"])
    min_cardinality = int(cat_filters["min_cardinality"])

    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[dict[str, str]] = []

    for name in sorted(stats):
        column_stats = stats[name]
        reason: str | None = None

        if column_stats.n_present == 0:
            reason = "all_null_in_train"
        elif column_stats.n_distinct <= 1:
            reason = "constant_in_train"
        elif column_stats.missing_rate > max_missing:
            reason = "missing_rate_above_threshold"
        elif column_stats.kind == KIND_CATEGORICAL:
            if column_stats.distinct_is_censored or column_stats.n_distinct > max_cardinality:
                reason = "cardinality_above_threshold"
            elif column_stats.n_distinct < min_cardinality:
                reason = "cardinality_below_threshold"

        if reason is not None:
            excluded.append(
                {
                    "name": name,
                    "source_file": column_stats.source_file,
                    "kind": column_stats.kind,
                    "reason_code": reason,
                    "reason": EXCLUSION_REASONS[reason],
                    "train_missing_rate": f"{column_stats.missing_rate:.6f}",
                    "train_distinct": str(column_stats.n_distinct),
                }
            )
            continue

        (numeric if column_stats.kind == KIND_NUMERIC else categorical).append(name)

    return FilterDecision(
        numeric=numeric, categorical=categorical, excluded=excluded, stats=stats
    )


def build_preprocessor(config: ModelingConfig, decision: FilterDecision) -> ColumnTransformer:
    """构造数值（中位数插补 + 标准化）与类别（众数插补 + 稀疏 one-hot）分支。

    所有可学习参数只在 ``fit`` 时从训练分区估计，因此对 validation/test 只做 ``transform``。
    """
    numeric_spec = config.preprocessing["numeric"]
    categorical_spec = config.preprocessing["categorical"]

    numeric_branch = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=str(numeric_spec["impute_strategy"]),
                    add_indicator=bool(numeric_spec["add_missing_indicator"]),
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=str(categorical_spec["impute_strategy"]))),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown=str(categorical_spec["handle_unknown"]),
                    sparse_output=bool(categorical_spec["sparse_output"]),
                    dtype=np.float32,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if decision.numeric:
        transformers.append(("numeric", numeric_branch, list(decision.numeric)))
    if decision.categorical:
        transformers.append(("categorical", categorical_branch, list(decision.categorical)))
    if not transformers:
        raise ModelingContractError("没有任何列通过训练期筛选，无法构造预处理器")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        # 强制稀疏输出。注意语义：本仓库使用的 scikit-learn 版本里判定式是
        # `sparse_output_ = density < sparse_threshold`（严格小于），因此
        # **只有把阈值设成 1.0 才会始终走稀疏分支**；设成 0.0 会让 `density < 0.0`
        # 恒为假、从而把 one-hot 块稠密化——那正是本工作包禁止的做法。
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def build_feature_pipeline(config: ModelingConfig, decision: FilterDecision, model: Any) -> Pipeline:
    """预处理 + 模型组合为单一可持久化对象，保证推理复用训练期统计量。"""
    return Pipeline(
        steps=[("preprocessor", build_preprocessor(config, decision)), ("model", model)]
    )


def assert_no_forbidden_features(columns: list[str], forbidden: tuple[str, ...]) -> None:
    """硬校验：禁止字段不得出现在源特征清单中。

    Raises:
        ModelingContractError: 出现任何禁止字段。
    """
    offenders = [name for name in forbidden if name in columns]
    if offenders:
        raise ModelingContractError(f"特征清单中出现禁止字段：{offenders}")


def assert_no_forbidden_transformed_names(
    feature_names: list[str], forbidden: tuple[str, ...]
) -> None:
    """硬校验：禁止字段不得出现在转换后的特征名中。

    使用 ``verbose_feature_names_out`` 后，转换名形如 ``numeric__V1``、``categorical__M1_T``，
    因此按"以 ``__`` 分隔后的最后一段"以及原始子串两种方式检查，避免漏判。
    """
    offenders: list[str] = []
    for name in feature_names:
        tail = name.split("__", 1)[-1]
        for forbidden_name in forbidden:
            if tail == forbidden_name or tail.startswith(f"{forbidden_name}_"):
                offenders.append(name)
                break
    if offenders:
        raise ModelingContractError(f"转换后的特征名中出现禁止字段：{offenders[:5]}")
